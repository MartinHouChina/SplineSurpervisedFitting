"""Delete redundant Ours knots from every case of a hash-verified saved benchmark.

Raw measurements and all other methods are copied unchanged. The original
common-insertion stage is archived, and reported total times combine the saved
historical measurement with freshly measured pruning, not a fresh end-to-end run.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_geometry import file_sha256, load_geometry_artifact, write_method_geometry
from benchmark_v15_datasets import fit_error_metrics
from plot_dual_error_comparison import _geometry, dual_pass, identity, pair_measurements
from spline_fitting.data.synthetic import bspline_basis_matrix
from spline_fitting.evaluation.bspline_inference import BSplineLeastSquaresFit
from spline_fitting.evaluation.dual_error_knot_pruning import prune_knots_to_dual_tolerance
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def protocol_fingerprint(metadata):
    """Canonical acquisition/protocol identity, independent of progress state."""
    payload = copy.deepcopy(metadata)
    payload.pop("fingerprint", None)
    payload.get("ours_post_pruning", {}).pop("complete", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _saved_solver_options(row, metadata):
    """Do not silently replace an unknown or regularized incoming LS contract."""
    diagnostics = row.get("repair_diagnostics")
    options = diagnostics.get("solver_options") if isinstance(diagnostics, dict) else None
    names = ("interpolate_endpoints", "smoothness_weight", "control_ridge")
    if not isinstance(options, dict) or any(name not in options for name in names):
        raise ValueError(f"Source Ours repair must declare endpoint and regularization solver_options: {identity(row)}")
    if not isinstance(options["interpolate_endpoints"], bool) or options["interpolate_endpoints"] != metadata["endpoint_convention"]["ours"]:
        raise ValueError(f"Source Ours solver_options disagree with endpoint_convention: {identity(row)}")
    for name in names[1:]:
        value = options[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Source Ours solver option {name} must be finite and nonnegative: {identity(row)}")
    return {name: options[name] for name in names}


def _copy_referenced_geometry(source_root, output_dir, report):
    """Copy only the exact JSON/NPZ dependency closure, never derived figures."""
    copied = {}

    def artifact(reference):
        source_path = (source_root / reference["path"]).resolve()
        target = (output_dir / reference["path"]).resolve()
        if not target.is_relative_to(output_dir):
            raise ValueError("Geometry references must be portable relative paths inside the output directory")
        key = source_path, target
        if key in copied:
            return copied[key]
        document, _ = load_geometry_artifact(source_root, reference)
        source_arrays = (source_path.parent / document["arrays_file"]).resolve()
        target_arrays = (target.parent / document["arrays_file"]).resolve()
        if not target_arrays.is_relative_to(output_dir):
            raise ValueError("Geometry array references must remain inside the output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target_arrays.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        shutil.copy2(source_arrays, target_arrays)
        copied[key] = document
        return document

    for row in report["raw_measurements"] + report["measurements"]:
        document = artifact(row["geometry_artifact"])
        reference = document["case_artifact"]
        if not (source_root / reference["path"]).is_file():
            first = Path(row["geometry_artifact"]["path"]).parts[0]
            if first in {"raw", "repaired"}:
                reference = dict(reference, path=(Path(first) / reference["path"]).as_posix())
        artifact(reference)


def _case_from_arrays(document, source):
    case = dict(document)
    for field, name in (
        ("points", "input_points_normalized"),
        ("reference", "reference_points_normalized"),
        ("reference_grid", "reference_chord_parameters"),
        ("reference_original", "reference_points_original"),
        ("center", "normalization_center"), ("scale", "normalization_scale"),
        ("source_internal_knots", "source_internal_knots"),
        ("canonical_internal_knots", "canonical_internal_knots"),
        ("source_parameters", "source_parameters"),
    ):
        case[field] = torch.as_tensor(source[name], dtype=torch.float64) if name in source else None
    case["input_points_sha256"] = hashlib.sha256(case["points"].numpy().tobytes()).hexdigest()
    return case


def restore_saved_fit(document, arrays, points):
    """Restore original control points without refitting or selecting any knots."""
    if not document.get("spline"):
        raise ValueError("A successful measurement must contain saved spline geometry")
    required = ("sample_parameters", "internal_knots", "full_knot_vector", "control_points_normalized")
    if any(name not in arrays for name in required):
        raise ValueError("A successful measurement is missing saved spline arrays")
    parameters, internal, knots, controls = (
        torch.as_tensor(arrays[name], dtype=torch.float64) for name in required
    )
    degree = int(document["spline"]["degree"])
    if any(not bool(torch.isfinite(value).all()) for value in (parameters, internal, knots, controls, points)):
        raise ValueError("Saved successful geometry contains nonfinite values")
    if parameters.ndim != 1 or parameters.numel() != points.shape[0] or controls.ndim != 2:
        raise ValueError("Saved spline dimensions are inconsistent")
    expected_knots = build_open_knot_vector(internal, degree)
    if not torch.equal(knots, expected_knots):
        raise ValueError("Saved full knot vector disagrees with internal knots")
    if controls.shape != (internal.numel() + degree + 1, points.shape[1]):
        raise ValueError("Saved control point count disagrees with knot vector")
    basis = bspline_basis_matrix(parameters, knots, degree, controls.shape[0])
    predicted = basis @ controls
    residual = (predicted - points).square()
    mse, coordinate_mse = residual.sum(-1).mean(), residual.mean()
    data_error = residual.sum()
    fit = BSplineLeastSquaresFit(
        degree=degree, internal_knots=internal, knot_vector=knots,
        basis_matrix=basis, control_points=controls, reconstructed_points=predicted,
        fit_mse=mse, fit_rmse=mse.sqrt(), coordinate_mse=coordinate_mse,
        coordinate_rmse=coordinate_mse.sqrt(), data_squared_error=data_error,
        smoothness_squared=controls.diff(n=2, dim=0).square().sum(),
        control_squared=controls.square().sum(), augmented_objective=data_error,
        solver_rank=None,
    )
    if "fitted_input_normalized" in arrays and not np.allclose(
            predicted.numpy(), arrays["fitted_input_normalized"], rtol=1e-10, atol=1e-12):
        raise ValueError("Saved control points do not reproduce saved fitted input")
    return fit, parameters


def _check_metrics(row, metrics):
    for field, measured in metrics.items():
        saved = row.get(field)
        if (saved is None) != (measured is None):
            raise ValueError(f"Saved {field} availability mismatch: {identity(row)}")
        if measured is not None and not math.isclose(saved, measured, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(f"Saved {field} disagrees with reconstructed spline: {identity(row)}")


def _audit_saved_geometry(source_root, report):
    """Check every requested method/case, not just cases eligible for deletion."""
    paired = pair_measurements(report)
    inputs = {}
    for pair in paired:
        for row in pair:
            case, source, arrays = _geometry(source_root, row)
            points = source["input_points_normalized"]
            key = identity(row)[:2]
            digest = hashlib.sha256(points.tobytes()).hexdigest()
            if row.get("input_points_sha256") not in (None, digest):
                raise ValueError(f"Saved input point hash mismatch: {identity(row)}")
            if key in inputs and inputs[key] != digest:
                raise ValueError(f"Methods or stages do not share identical input points: {key}")
            inputs[key] = digest
            if row.get("status") == "ok":
                residuals = arrays.get("input_squared_residuals_normalized")
                if residuals is None or not np.isfinite(residuals).all():
                    raise ValueError(f"Successful geometry lacks finite residuals: {identity(row)}")
                _check_metrics(row, dict(mse=float(residuals.mean()), max_squared_error=float(residuals.max())))
    declared = {(c["dataset"], str(c["sample_id"])): c for c in report["metadata"].get("cases", [])}
    for key, digest in inputs.items():
        if declared.get(key, {}).get("input_points_sha256") not in (None, digest):
            raise ValueError(f"Declared case input hash mismatch: {key}")
    return len(inputs)


def postprune_report(source_report, output_dir, *, min_internal_knots=0,
                     max_deletions=32, torch_num_threads=4):
    source_report, output_dir = Path(source_report).resolve(), Path(output_dir).resolve()
    source_root = source_report.parent
    if output_dir == source_root or output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError("Output must be a separate directory outside the source tree")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output_dir}")
    source = json.loads(source_report.read_text(encoding="utf-8"))
    metadata = source["metadata"]
    if metadata.get("ours_post_pruning", {}).get("enabled") or "pre_pruning_measurements" in source:
        raise ValueError("Source already includes pruning; use the original common-insertion report")
    if "ours" not in metadata["methods"]:
        raise ValueError("Source benchmark does not declare Ours")
    cap = metadata["max_internal_knots"]
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("Source knot cap must be a nonnegative integer")
    if not 0 <= min_internal_knots <= cap or max_deletions < 0 or torch_num_threads < 1:
        raise ValueError("Invalid pruning minimum, deletion budget, or thread count")
    if not isinstance(metadata.get("endpoint_convention", {}).get("ours"), bool):
        raise ValueError("Source must declare the original Ours endpoint convention")
    case_count = _audit_saved_geometry(source_root, source)
    solver_options = {identity(row): _saved_solver_options(row, metadata)
                      for row in source["measurements"] if row["method"] == "ours" and row["status"] == "ok"}
    source_hash = file_sha256(source_report)
    torch.set_num_threads(torch_num_threads)
    report = copy.deepcopy(source)
    protocol = report["metadata"]
    protocol["schema_version"] = "saved_common_dual_insertion_ours_post_pruning_v2"
    protocol["timing_protocol"] = (
        "Composite timing from saved historical base-method + common-insertion total_ms "
        "and freshly measured Ours pruning_ms on CPU; total_ms=pre_pruning_ms+pruning_ms "
        "for Ours. Other methods retain their original timings unchanged. No raw/network "
        "or insertion stage was rerun. Verification, metrics, export and plots are excluded."
    )
    protocol["ours_post_pruning"] = dict(
        enabled=True, method="ours", stage="after_common_dual_error_insertion",
        source_report=str(source_report), source_report_sha256=source_hash,
        source_snapshot="source_snapshot/" + source_report.name,
        source_timing_protocol=metadata.get("timing_protocol"),
        source_fingerprint=metadata.get("fingerprint"),
        acquisition_utc=datetime.now(timezone.utc).isoformat(),
        min_internal_knots=min_internal_knots, max_deletions=max_deletions,
        max_internal_knots=cap, mse_tolerance=metadata["mse_tolerance"],
        max_squared_error_tolerance=metadata["max_squared_error_tolerance"],
        interpolate_endpoints=metadata["endpoint_convention"]["ours"],
        solver_options_source="Replay each Ours row's repair_diagnostics.solver_options: interpolate_endpoints, smoothness_weight, control_ridge",
        parameters_updated=False, surviving_knots_relocated=False,
        all_source_cases_retained=True, case_count=case_count,
        selection="all requested cases; no result-dependent filtering",
        pruning_hardware=dict(device="cpu", torch_threads=torch_num_threads),
        timing_provenance=protocol["timing_protocol"],
        code_sha256={name: file_sha256(ROOT / name) for name in (
            "scripts/postprune_saved_comparison.py", "src/spline_fitting/evaluation/dual_error_knot_pruning.py")},
    )
    protocol["fingerprint_definition"] = (
        "SHA256 of UTF-8 JSON(metadata), excluding top-level fingerprint and "
        "ours_post_pruning.complete, sort_keys=True, separators=(',', ':'), "
        "ensure_ascii=True, allow_nan=False"
    )
    fingerprint = protocol_fingerprint(protocol)
    protocol["fingerprint"] = fingerprint
    # A full unmodified archive keeps JSON/NPZ hashes and original relative paths
    # usable even after the destination Ours geometry is overwritten below. Only
    # declared geometry dependencies enter the active output, so old figures,
    # spreadsheets, summaries and unrelated caches cannot masquerade as new ones.
    output_dir.mkdir(parents=True)
    _copy_referenced_geometry(source_root, output_dir, source)
    shutil.copytree(source_root, output_dir / "source_snapshot")
    report["pre_pruning_measurements"] = copy.deepcopy(source["measurements"])
    for row in report["pre_pruning_measurements"]:
        row["geometry_artifact"]["path"] = "source_snapshot/" + row["geometry_artifact"]["path"]
    # Do not publish a stale copied comparison.json while processing.
    report["measurements"] = []
    protocol["ours_post_pruning"]["complete"] = False
    output = output_dir / "comparison.json"
    _write_json(output, report)
    completed_ours = 0
    for before in source["measurements"]:
        final = copy.deepcopy(before)
        if before["method"] == "ours":
            completed_ours += 1
            final.update(pruning_ms=0., removed_knots_count=0,
                         pruning_termination="no_valid_initial_fit",
                         pre_pruning_k=before.get("final_k"), pre_pruning_mse=before.get("mse"),
                         pre_pruning_max_squared_error=before.get("max_squared_error"),
                         pre_pruning_ms=before.get("total_ms"))
            if before["status"] == "ok":
                case_document, source_arrays, arrays = _geometry(source_root, before)
                document, _ = load_geometry_artifact(source_root, before["geometry_artifact"])
                case = _case_from_arrays(case_document, source_arrays)
                fit, parameters = restore_saved_fit(document, arrays, case["points"])
                _check_metrics(before, fit_error_metrics(fit, parameters, case))
                if int(fit.internal_knots.numel()) != before["final_k"]:
                    raise ValueError(f"Saved knot count mismatch: {identity(before)}")
                result = prune_knots_to_dual_tolerance(
                    parameters, case["points"], fit,
                    mse_tolerance=metadata["mse_tolerance"],
                    max_squared_error_tolerance=metadata["max_squared_error_tolerance"],
                    max_internal_knots=cap, min_internal_knots=min_internal_knots,
                    max_deletions=max_deletions,
                    **solver_options[identity(before)],
                )
                updated = fit_error_metrics(result.final_fit, parameters, case)
                final.update(updated, final_k=int(result.final_fit.internal_knots.numel()),
                             knots=result.final_fit.internal_knots.tolist(),
                             pruning_ms=result.elapsed_pruning_ms,
                             removed_knots_count=len(result.removed_knots),
                             pruning_termination=result.termination,
                             pruning_diagnostics={k: v for k, v in vars(result).items()
                                                  if k not in {"initial_fit", "final_fit"}},
                             total_ms=before["total_ms"] + result.elapsed_pruning_ms)
                final["fit_pass"] = updated["mse"] <= metadata["mse_tolerance"]
                final["joint_pass"] = dual_pass(final, metadata)
                final["reference_pass"] = (updated["reference_mse"] <= metadata["mse_tolerance"]
                                            if updated["reference_mse"] is not None else None)
                final["reference_joint_pass"] = (
                    final["reference_pass"] and updated["reference_max_squared_error"] <= metadata["max_squared_error_tolerance"]
                    if updated["reference_mse"] is not None else None)
                if final["final_k"] > before["final_k"] or (dual_pass(before, metadata) and not final["joint_pass"]):
                    raise RuntimeError("Pruning increased knots or violated an originally satisfied bound")
                # Case files are not modified: baseline and raw hash references
                # continue to resolve to exactly their original source points.
                case_artifact = document["case_artifact"]
                if not (output_dir / case_artifact["path"]).is_file():
                    raise ValueError("Saved final case artifact must resolve at the original report root")
                final.pop("geometry_artifact", None)
                final["geometry_artifact"] = write_method_geometry(
                    output_dir, case, final, result.final_fit, parameters,
                    case_artifact=case_artifact, fingerprint=fingerprint,
                    dense_points=len(arrays.get("dense_parameters", np.empty(801))),
                    timing_scope=protocol["timing_protocol"],
                    pass_criterion=metadata.get("pass_definition"),
                )
            print(f"[{completed_ours}/{case_count}] {before['dataset']} {before['sample_id']}: "
                  f"K {before.get('final_k')} -> {final.get('final_k')}, "
                  f"dual {before.get('joint_pass')} -> {final.get('joint_pass')}, "
                  f"pruning {final['pruning_ms']:.2f} ms", flush=True)
        report["measurements"].append(final)
        if before["method"] == "ours":
            _write_json(output, report)
    pair_measurements(report)
    _audit_saved_geometry(output_dir, report)
    if file_sha256(source_report) != source_hash:
        raise RuntimeError("Source report changed during pruning")
    protocol["ours_post_pruning"]["complete"] = True
    if protocol_fingerprint(protocol) != fingerprint:
        raise RuntimeError("Immutable pruning protocol changed during execution")
    _write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=ROOT / "outputs/comparisons/historical_best_dual_error_20260922/comparison.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-internal-knots", type=int, default=0)
    parser.add_argument("--max-deletions", type=int, default=32)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    args = parser.parse_args(argv)
    postprune_report(args.source_report, args.output_dir,
                     min_internal_knots=args.min_internal_knots,
                     max_deletions=args.max_deletions, torch_num_threads=args.torch_num_threads)
    print(f"Saved {args.output_dir / 'comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
