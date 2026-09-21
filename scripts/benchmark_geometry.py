"""Lossless, hash-linked geometry exports from the *measured* benchmark fit.

No model, knot selection, or least-squares solver is invoked here. NPZ arrays
use ordinary numeric dtypes and can be loaded with ``allow_pickle=False``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from spline_fitting.data.point_cloud_io import interpolate_parameters_by_chord

GEOMETRY_SCHEMA_VERSION = "paired_measured_geometry_v1"


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def case_key(case: dict) -> str:
    identity = json.dumps([case["dataset"], case["sample_id"]], ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _tensor(value):
    return torch.as_tensor(value).detach().cpu().double()


def _json_write(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _arrays_write(path: Path, arrays: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: _tensor(value).numpy() for key, value in arrays.items()})
    temporary.replace(path)


def _safe_list(value):
    """Nonfinite partial-failure geometry is retained as NaN in NPZ, null in JSON."""
    array = _tensor(value).numpy()
    return np.where(np.isfinite(array), array, None).tolist()


def write_case_geometry(directory: Path, case: dict, *, fingerprint: str) -> dict:
    destination = directory / "geometry" / case_key(case)
    destination.mkdir(parents=True, exist_ok=True)
    document_path = destination / "case.json"
    arrays_path = destination / "case.npz"
    if document_path.exists():
        document = json.loads(document_path.read_text(encoding="utf-8"))
        if document["fingerprint"] != fingerprint or file_sha256(arrays_path) != document["arrays_sha256"]:
            raise ValueError(f"Existing case geometry is inconsistent: {document_path}")
        return {"path": document_path.relative_to(directory).as_posix(), "sha256": file_sha256(document_path)}
    points = _tensor(case["points"])
    arrays = {"input_points_normalized": points}
    has_transform = case.get("center") is not None and case.get("scale") is not None
    center = _tensor(case["center"]) if has_transform else None
    scale = _tensor(case["scale"]) if has_transform else None
    if has_transform:
        arrays.update(normalization_center=center, normalization_scale=scale,
                      input_points_original=points * scale + center)
    if case.get("reference") is not None:
        arrays["reference_points_normalized"] = case["reference"]
        arrays["reference_chord_parameters"] = case["reference_grid"]
        if case.get("reference_original") is not None:
            arrays["reference_points_original"] = case["reference_original"]
        elif has_transform:
            arrays["reference_points_original"] = _tensor(case["reference"]) * scale + center
    for name in ("source_internal_knots", "canonical_internal_knots", "source_parameters"):
        if case.get(name) is not None:
            arrays[name] = case[name]
    _arrays_write(arrays_path, arrays)
    document = {
        "schema_version": GEOMETRY_SCHEMA_VERSION, "fingerprint": fingerprint,
        **{key: case.get(key) for key in (
            "dataset", "sample_id", "group_id", "split", "source_kind", "source_note",
            "dataset_label", "source_k", "canonical_k", "source_dataset", "source_manifest",
            "source_manifest_record", "source_points_path", "source_points_sha256",
        )},
        "normalization": {
            "available": has_transform,
            "definition": "normalized = (original - center) / scalar_scale",
            "center": _safe_list(center) if has_transform else None,
            "scale": float(scale) if has_transform else None,
            "original_input_note": "Inverse transformed actual model inputs; may differ from pre-normalization resampling by floating-point rounding.",
        },
        "arrays_file": arrays_path.name, "arrays_sha256": file_sha256(arrays_path),
        "array_shapes": {name: list(_tensor(value).shape) for name, value in arrays.items()},
    }
    _json_write(document_path, document)
    return {"path": document_path.relative_to(directory).as_posix(), "sha256": file_sha256(document_path)}


def write_method_geometry(directory: Path, case: dict, row: dict, fit, parameters,
                          *, case_artifact: dict, fingerprint: str, dense_points: int) -> dict:
    """Capture even failed/unavailable rows; absent outputs remain genuinely absent."""
    destination = directory / "geometry" / case_key(case)
    method_key = hashlib.sha256(row["method"].encode()).hexdigest()[:20]
    document_path = destination / f"{method_key}.json"
    arrays_path = destination / f"{method_key}.npz"
    arrays, spline, export_error = {}, None, None
    if parameters is not None:
        arrays["sample_parameters"] = _tensor(parameters)
    if fit is not None:
        arrays.update(internal_knots=fit.internal_knots, full_knot_vector=fit.knot_vector,
                      control_points_normalized=fit.control_points)
        spline = {"degree": int(fit.degree), "internal_knots": _safe_list(fit.internal_knots),
                  "full_knot_vector": _safe_list(fit.knot_vector),
                  "control_points_normalized": _safe_list(fit.control_points),
                  "sample_parameters": _safe_list(parameters) if parameters is not None else None}
        try:
            if parameters is not None:
                predicted = fit.evaluate(_tensor(parameters))
                residuals = predicted - _tensor(case["points"])
                arrays.update(fitted_input_normalized=predicted, input_residual_vectors_normalized=residuals,
                              input_squared_residuals_normalized=residuals.square().sum(-1))
            dense_parameters = torch.linspace(0, 1, dense_points, dtype=torch.float64)
            arrays.update(dense_parameters=dense_parameters,
                          dense_curve_normalized=fit.evaluate(dense_parameters),
                          knot_positions_normalized=fit.evaluate(_tensor(fit.internal_knots)))
            if parameters is not None and case.get("reference") is not None:
                reference_parameters = interpolate_parameters_by_chord(
                    torch.linspace(0, 1, parameters.numel(), dtype=torch.float64),
                    _tensor(parameters), _tensor(case["reference_grid"]),
                )
                predicted = fit.evaluate(reference_parameters)
                residuals = predicted - _tensor(case["reference"])
                arrays.update(reference_parameters=reference_parameters,
                              fitted_reference_normalized=predicted,
                              reference_residual_vectors_normalized=residuals,
                              reference_squared_residuals_normalized=residuals.square().sum(-1))
        except (RuntimeError, ValueError) as error:
            export_error = f"{type(error).__name__}: {error}"
            if row["status"] == "ok":
                raise  # A successful fit must never silently have an incomplete export.
        if case.get("center") is not None and case.get("scale") is not None:
            center, scale = _tensor(case["center"]), _tensor(case["scale"])
            for name, value in list(arrays.items()):
                if name.endswith("_normalized"):
                    factor = scale.square() if "squared_residuals" in name else scale
                    arrays[name.removesuffix("_normalized") + "_original"] = _tensor(value) * factor + (
                        0 if "residual" in name else center
                    )
            spline["control_points_original"] = _safe_list(arrays["control_points_original"])
    _arrays_write(arrays_path, arrays)
    metric_names = ("mse", "max_squared_error", "reference_mse", "reference_max_squared_error")
    original_error_factor = float(_tensor(case["scale"]).square()) if case.get("scale") is not None else None
    document = {
        "schema_version": GEOMETRY_SCHEMA_VERSION, "fingerprint": fingerprint,
        "case_artifact": case_artifact, "measurement": dict(row), "spline": spline,
        "geometry_evaluation_error": export_error,
        "errors_normalized": {name: row.get(name) for name in metric_names},
        "errors_original": {name: row[name] * original_error_factor if row.get(name) is not None and original_error_factor is not None else None
                            for name in metric_names},
        "arrays_file": arrays_path.name, "arrays_sha256": file_sha256(arrays_path),
        "array_shapes": {name: list(_tensor(value).shape) for name, value in arrays.items()},
        "timing_scope": "Export, dense evaluation, metrics and plotting are excluded from successful method/network timings. The spline is from the final timed repetition; reported times are medians of all recorded timed repetitions.",
        "coordinate_error_scaling": "Original-coordinate squared errors = normalized squared errors * scalar_scale**2; reported benchmark pass/fail uses normalized input MSE only.",
    }
    _json_write(document_path, document)
    return {"path": document_path.relative_to(directory).as_posix(), "sha256": file_sha256(document_path)}


def load_geometry_artifact(directory: Path, artifact: dict) -> tuple[dict, dict]:
    """Verify both JSON and NPZ bytes before reusing saved measured geometry."""
    directory = directory.resolve()
    path = (directory / artifact["path"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Geometry artifact must remain inside benchmark directory")
    if file_sha256(path) != artifact["sha256"]:
        raise ValueError(f"Geometry JSON hash mismatch: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    arrays_path = (path.parent / document["arrays_file"]).resolve()
    if not arrays_path.is_relative_to(directory):
        raise ValueError("Geometry arrays must remain inside benchmark directory")
    if file_sha256(arrays_path) != document["arrays_sha256"]:
        raise ValueError(f"Geometry NPZ hash mismatch: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return document, arrays
