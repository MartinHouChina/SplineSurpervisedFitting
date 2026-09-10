"""Deterministic synthetic/real mixtures without fabricated knot labels."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .real_world import RealWorldCurveDataset, read_curve_manifest, resolve_points_path
from .synthetic import SyntheticCubicBSplineDataset


EPOCH_SEED_STRIDE = 10_000_000


def _minimal_knot_count_target(
    sample: dict | None = None,
    *,
    certified: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a collate-safe minimal-knot count target and validity flag.

    Real curves and uncertified synthetic curves deliberately receive an
    invalid zero target: their minimum feasible knot count is not known.
    """
    valid = bool(
        certified
        and sample is not None
        and sample.get("source_minimality_certified", False)
    )
    if not valid:
        return torch.tensor(0, dtype=torch.long), torch.tensor(False)

    if "true_internal_knot_mask" in sample:
        count = sample["true_internal_knot_mask"].sum(dtype=torch.long)
    elif "source_internal_knot_count" in sample:
        count = torch.as_tensor(
            sample["source_internal_knot_count"], dtype=torch.long
        )
    else:
        raise RuntimeError(
            "certified synthetic sample is missing its internal-knot count"
        )
    return count.reshape(()), torch.tensor(True)


def _curve_record(
    points: torch.Tensor,
    source: str,
    *,
    max_internal_knots: int,
    synthetic_sample: dict | None = None,
    certified: bool = False,
) -> dict:
    count, valid = _minimal_knot_count_target(
        synthetic_sample,
        certified=certified,
    )
    if bool(valid):
        if synthetic_sample is None:
            raise RuntimeError("valid geometry target requires a synthetic sample")
        required = {
            "true_params",
            "true_internal_knots",
            "true_internal_knot_mask",
        }
        missing = sorted(required.difference(synthetic_sample))
        if missing:
            raise RuntimeError(
                "certified synthetic sample is missing geometry targets: "
                + ", ".join(missing)
            )
        target_params = synthetic_sample["true_params"]
        target_internal_knots = synthetic_sample["true_internal_knots"]
        target_internal_knot_mask = synthetic_sample["true_internal_knot_mask"]
        if target_params.shape != (points.shape[0],):
            raise RuntimeError("synthetic parameter target has an unexpected shape")
        if target_internal_knots.shape != (max_internal_knots,):
            raise RuntimeError("synthetic internal-knot target has an unexpected shape")
        if target_internal_knot_mask.shape != (max_internal_knots,):
            raise RuntimeError("synthetic internal-knot mask has an unexpected shape")
    else:
        target_params = points.new_zeros(points.shape[0])
        target_internal_knots = points.new_zeros(max_internal_knots)
        target_internal_knot_mask = torch.zeros(
            max_internal_knots,
            dtype=torch.bool,
        )
    return {
        "points": points,
        "source": source,
        "target_internal_knot_count": count,
        "target_internal_knot_count_valid": valid,
        "target_params": target_params,
        "target_internal_knots": target_internal_knots,
        "target_internal_knot_mask": target_internal_knot_mask,
        "target_geometry_valid": valid.detach().clone(),
    }


def grouped_indices(records: list[dict], limit: int, seed: int) -> list[int]:
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["group_id"]].append(index)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(limit, len(records)):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == min(limit, len(records)):
                    return selected
    return selected


def load_real_sources(manifests, *, num_points: int, point_dim: int, progress=None):
    """Validate writer/tile and identical-file splits across all manifests."""
    sources, provenance, group_splits, path_splits = [], [], {}, {}
    seen = set()
    for raw_path in manifests:
        path = Path(raw_path).resolve()
        if path in seen:
            raise ValueError(f"duplicate real manifest: {path}")
        seen.add(path)
        if progress:
            progress(f"  Reading manifest: {path}")
        records = read_curve_manifest(path)
        source_names = sorted({r["source_dataset"] for r in records})
        if not records:
            raise ValueError(f"empty manifest: {path}")
        for record_index, record in enumerate(records, 1):
            if record["point_dim"] != point_dim:
                raise ValueError(f"point dimension mismatch in {path}")
            group = (record["source_dataset"], record["group_id"])
            resolved = resolve_points_path(path, record).resolve()
            for mapping, key in ((group_splits, group), (path_splits, str(resolved))):
                previous = mapping.setdefault(key, record["split"])
                if previous != record["split"]:
                    raise ValueError(f"train/val/test leakage for {key}")
            if not resolved.is_file():
                raise FileNotFoundError(f"missing manifest point file: {resolved}")
            if progress and (record_index % 2000 == 0 or record_index == len(records)):
                progress(f"  Checking groups/files: {record_index}/{len(records)}")
        train = RealWorldCurveDataset(path, split="train", num_points=num_points)
        val = RealWorldCurveDataset(path, split="val", num_points=num_points)
        if not len(train) or not len(val):
            raise ValueError(f"manifest must have nonempty train and val splits: {path}")
        label = "+".join(source_names)
        if any(item[0] == label for item in sources):
            raise ValueError(f"duplicate source name across manifests: {label}")
        sources.append((label, train, val))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance.append(dict(name=label, manifest=str(path), manifest_sha256=digest,
                               train_size=len(train), val_size=len(val),
                               test_size=sum(r["split"] == "test" for r in records)))
        if progress:
            progress(f"  Ready {label}: train={len(train)}, val={len(val)} (test excluded from training)")
    return sources, provenance


class MixedTrainingCurves(Dataset):
    """Draw sources equally within the specified real-data fraction.

    A large UJI source cannot overwhelm smaller geography sources. The index
    and epoch determine source choice, real record and synthetic curve seed.
    """
    def __init__(self, config, real_sources=(), *, size=4000, seed=42,
                 real_fraction=0.5, epoch=0, resample=True):
        if size < 1 or size >= EPOCH_SEED_STRIDE:
            raise ValueError("training size must be positive and below the epoch seed stride")
        if not 0 <= real_fraction <= 1:
            raise ValueError("real_fraction must lie in [0,1]")
        self.size = size
        self.seed = seed + (epoch * EPOCH_SEED_STRIDE if resample else 0)
        self.real_sources = list(real_sources)
        self.real_fraction = real_fraction if self.real_sources else 0.0
        options = dict(config)
        self.certified_synthetic_targets = bool(
            options.get("certified_minimal_source", False)
        )
        options.update(
            return_ground_truth=False,
            cache_samples=False,
        )
        self.synthetic = SyntheticCubicBSplineDataset(size=size, seed=self.seed, **options)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = random.Random(self.seed + index + 2**40)
        if self.real_sources and rng.random() < self.real_fraction:
            label, train, _ = self.real_sources[rng.randrange(len(self.real_sources))]
            points = train[rng.randrange(len(train))]["points"]
            sample = None
        else:
            label = "Synthetic"
            sample = self.synthetic[index]
            points = sample["points"]
        return _curve_record(
            points,
            label,
            max_internal_knots=self.synthetic.max_internal_knots,
            synthetic_sample=sample,
            certified=self.certified_synthetic_targets,
        )


class ValidationCurves(Dataset):
    def __init__(self, config, real_sources=(), *, size=1000, seed=1_000_000,
                 real_per_source=100, synthetic_boundary_samples=0):
        if size < 1 or real_per_source < 1:
            raise ValueError("validation counts must be positive")
        if (
            isinstance(synthetic_boundary_samples, bool)
            or not isinstance(synthetic_boundary_samples, int)
            or synthetic_boundary_samples < 0
        ):
            raise ValueError(
                "synthetic_boundary_samples must be a non-negative integer"
            )
        options = dict(config)
        self.certified_synthetic_targets = bool(
            options.get("certified_minimal_source", False)
        )
        options.update(
            return_ground_truth=False,
            cache_samples=True,
        )
        boundary_size = min(synthetic_boundary_samples, size)
        random_size = max(size - boundary_size, 1)
        self.synthetic = SyntheticCubicBSplineDataset(
            size=random_size,
            seed=seed,
            **options,
        )
        self.synthetic_boundary = None
        self.entries = []
        if boundary_size:
            boundary_options = dict(options)
            boundary_options["min_control_points"] = options["max_control_points"]
            self.synthetic_boundary = SyntheticCubicBSplineDataset(
                size=boundary_size,
                seed=seed + 20_000_003,
                **boundary_options,
            )
            self.entries.extend(
                ("Synthetic", self.synthetic_boundary, i)
                for i in range(boundary_size)
            )
        self.entries.extend(
            ("Synthetic", self.synthetic, i)
            for i in range(size - boundary_size)
        )
        self.selected_real_ids = {}
        for source_index, (label, _, val) in enumerate(real_sources):
            chosen = grouped_indices(val.records, real_per_source, seed + source_index)
            self.entries.extend((label, val, i) for i in chosen)
            self.selected_real_ids[label] = [val.records[i]["sample_id"] for i in chosen]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        label, dataset, i = self.entries[index]
        sample = dataset[i]
        is_certified_synthetic = (
            label == "Synthetic" and self.certified_synthetic_targets
        )
        return _curve_record(
            sample["points"],
            label,
            max_internal_knots=self.synthetic.max_internal_knots,
            synthetic_sample=sample if is_certified_synthetic else None,
            certified=is_certified_synthetic,
        )
