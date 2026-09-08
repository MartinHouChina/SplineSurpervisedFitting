from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .point_cloud_io import normalize_ordered_point_cloud, resample_ordered_point_cloud


REAL_WORLD_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "sample_id",
        "source_dataset",
        "group_id",
        "split",
        "points_path",
        "num_points",
        "point_dim",
        "has_knot_labels",
        "metadata",
    }
)
REAL_WORLD_SPLITS = frozenset({"train", "val", "test"})


def _validate_manifest_record(
    record: Mapping[str, Any], *, line_number: int
) -> dict[str, Any]:
    missing = REAL_WORLD_MANIFEST_REQUIRED_FIELDS.difference(record)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"manifest line {line_number} is missing fields: {names}")

    normalized = dict(record)
    for field in ("sample_id", "source_dataset", "group_id", "points_path"):
        if not isinstance(normalized[field], str) or not normalized[field]:
            raise ValueError(
                f"manifest line {line_number} field {field!r} must be a non-empty string"
            )
    if normalized["split"] not in REAL_WORLD_SPLITS:
        raise ValueError(
            f"manifest line {line_number} has unsupported split {normalized['split']!r}"
        )
    if not isinstance(normalized["metadata"], Mapping):
        raise ValueError(f"manifest line {line_number} metadata must be an object")
    if not isinstance(normalized["has_knot_labels"], bool):
        raise ValueError(f"manifest line {line_number} has_knot_labels must be boolean")
    if int(normalized["num_points"]) < 2:
        raise ValueError(f"manifest line {line_number} num_points must be at least two")
    if int(normalized["point_dim"]) not in {2, 3}:
        raise ValueError(f"manifest line {line_number} point_dim must be 2 or 3")
    normalized["num_points"] = int(normalized["num_points"])
    normalized["point_dim"] = int(normalized["point_dim"])
    normalized["metadata"] = dict(normalized["metadata"])
    return normalized


def read_curve_manifest(
    path: str | Path,
    *,
    split: str | Sequence[str] | None = None,
    sources: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read and validate the shared JSONL manifest for real-world curves."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"curve manifest does not exist: {manifest_path}")

    requested_splits = _as_filter_set(split)
    requested_sources = _as_filter_set(sources)
    if requested_splits is not None:
        unsupported = requested_splits.difference(REAL_WORLD_SPLITS)
        if unsupported:
            raise ValueError(f"unsupported requested splits: {sorted(unsupported)}")

    records: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on manifest line {line_number}: {error.msg}"
                ) from error
            if not isinstance(raw, Mapping):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            record = _validate_manifest_record(raw, line_number=line_number)
            sample_id = record["sample_id"]
            if sample_id in all_ids:
                raise ValueError(f"duplicate manifest sample_id: {sample_id}")
            all_ids.add(sample_id)
            if requested_splits is not None and record["split"] not in requested_splits:
                continue
            if (
                requested_sources is not None
                and record["source_dataset"] not in requested_sources
            ):
                continue
            records.append(record)
    return records


def _as_filter_set(value: str | Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def write_curve_manifest(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write real-world curve records as JSONL."""

    manifest_path = Path(path)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    validated: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        item = _validate_manifest_record(record, line_number=line_number)
        if item["sample_id"] in sample_ids:
            raise ValueError(f"duplicate manifest sample_id: {item['sample_id']}")
        sample_ids.add(item["sample_id"])
        validated.append(item)

    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(manifest_path)
    return manifest_path


def resolve_points_path(manifest_path: str | Path, record: Mapping[str, Any]) -> Path:
    points_path = Path(str(record["points_path"]))
    if not points_path.is_absolute():
        points_path = Path(manifest_path).resolve().parent / points_path
    return points_path


def load_manifest_points(
    manifest_path: str | Path,
    record: Mapping[str, Any],
) -> torch.Tensor:
    """Load one raw, ordered point sequence referenced by a manifest row."""

    path = resolve_points_path(manifest_path, record)
    if not path.is_file():
        raise FileNotFoundError(f"manifest point file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "points" not in archive:
                raise ValueError(f"NPZ point file has no 'points' array: {path}")
            raw = archive["points"]
    elif suffix in {".csv", ".txt", ".xyz"}:
        raw = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            raw = raw.get("points", raw.get("point_cloud"))
    else:
        raise ValueError(f"unsupported manifest point format: {path}")

    points = torch.as_tensor(raw, dtype=torch.float32)
    expected_shape = (int(record["num_points"]), int(record["point_dim"]))
    if points.ndim != 2 or tuple(points.shape) != expected_shape:
        raise ValueError(
            f"manifest shape {expected_shape} does not match {path}: {tuple(points.shape)}"
        )
    if not torch.isfinite(points).all():
        raise ValueError(f"point file contains NaN or infinite values: {path}")
    return points.contiguous()


class RealWorldCurveDataset(Dataset):
    """Load manifest-backed curves for deployment or unlabeled adaptation.

    The dataset never manufactures knot labels. A fixed ``num_points`` resamples
    each ordered polyline on a uniform chord grid for network input. The original
    dense points remain accessible through :meth:`load_reference_points`.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str | Sequence[str] | None = None,
        sources: str | Sequence[str] | None = None,
        num_points: int | None = None,
        normalize: bool = True,
    ) -> None:
        if num_points is not None and num_points < 4:
            raise ValueError("num_points must be at least four for cubic fitting")
        self.manifest_path = Path(manifest)
        self.records = read_curve_manifest(
            self.manifest_path,
            split=split,
            sources=sources,
        )
        self.num_points = num_points
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int) -> dict[str, Any]:
        return dict(self.records[index])

    def load_reference_points(self, index: int) -> torch.Tensor:
        return load_manifest_points(self.manifest_path, self.records[index])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        record = self.records[index]
        reference = self.load_reference_points(index)
        points = reference
        if self.num_points is not None:
            points = resample_ordered_point_cloud(points, self.num_points)

        if self.normalize:
            normalized = normalize_ordered_point_cloud(points)
            model_points = normalized["points"]
            chord_params = normalized["chord_params"]
            center = normalized["center"]
            scale = normalized["scale"]
        else:
            model_points = points
            segment_lengths = (points[1:] - points[:-1]).norm(dim=-1)
            chord_params = torch.cat(
                [points.new_zeros(1), torch.cumsum(segment_lengths, dim=0)]
            )
            chord_params = chord_params / chord_params[-1].clamp_min(1e-8)
            center = points.new_zeros(points.shape[-1])
            scale = points.new_ones(())

        return {
            "points": model_points,
            "chord_params": chord_params,
            "center": center,
            "scale": scale,
            "sample_id": index,
            "curve_id": str(record["sample_id"]),
            "group_id": str(record["group_id"]),
            "source_dataset": str(record["source_dataset"]),
            "split": str(record["split"]),
            "has_knot_labels": bool(record["has_knot_labels"]),
        }
