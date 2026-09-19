"""Deterministic synthetic/real mixtures without fabricated knot labels."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .real_world import RealWorldCurveDataset, read_curve_manifest, resolve_points_path
from .synthetic import SyntheticCubicBSplineDataset, generate_sampling_parameters


EPOCH_SEED_STRIDE = 10_000_000
SHAPE_SYNTHETIC_SEED_OFFSET = 2_000_000_000_000
LOW_K_MIN = 4
LOW_K_MAX = 8


def _namespaced_seed(seed: int, namespace: str) -> int:
    """Derive a stable PyTorch seed without relying on large integer offsets."""
    digest = hashlib.sha256(f"v16:{namespace}:{seed}".encode("ascii")).digest()
    # The CPU generator aliases seeds that differ only above the low 32 bits.
    # Use those bits deliberately so each namespace gets an independent stream.
    return int.from_bytes(digest[:4], "little")


class _LowKCertifiedSyntheticView(Dataset):
    """Generate certified K=4..8 samples with full-range target padding."""

    def __init__(self, config: dict, *, size: int, seed: int) -> None:
        options = dict(config)
        if not options.get("certified_minimal_source", False):
            raise ValueError(
                "synthetic_simple_fraction requires certified_minimal_source=True"
            )
        degree = 3
        min_control_points = int(options.get("min_control_points", 5))
        max_control_points = int(options.get("max_control_points", 10))
        self.low_control_min = max(min_control_points, LOW_K_MIN + degree + 1)
        self.low_control_max = min(max_control_points, LOW_K_MAX + degree + 1)
        if self.low_control_min > self.low_control_max:
            raise ValueError(
                "synthetic_simple_fraction requires the configured source range "
                "to overlap internal K=4..8"
            )
        self.max_internal_knots = max_control_points - degree - 1
        options.update(
            min_control_points=self.low_control_min,
            max_control_points=self.low_control_max,
            return_ground_truth=False,
            cache_samples=False,
        )
        self.size = int(size)
        self.dataset = SyntheticCubicBSplineDataset(
            size=size,
            seed=seed,
            **options,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        sample = self.dataset[index]
        knots = sample["true_internal_knots"]
        if knots.numel() == self.max_internal_knots:
            return sample

        # The dedicated low-K generator pads to its own K<=8 capacity. Preserve
        # the historical mixed-dataset contract by extending only the compact
        # label tensors to the original configured source width.
        result = dict(sample)
        padded_knots = knots.new_zeros(self.max_internal_knots)
        padded_mask = torch.zeros(self.max_internal_knots, dtype=torch.bool)
        padded_knots[: knots.numel()] = knots
        padded_mask[: knots.numel()] = sample["true_internal_knot_mask"]
        result["true_internal_knots"] = padded_knots
        result["true_internal_knot_mask"] = padded_mask
        return result


class _CompactShapeSyntheticDataset(Dataset):
    """Deterministic, smooth procedural curves without fabricated knot labels."""

    _FAMILIES = ("shape_industrial", "shape_terrain", "shape_handwriting")

    def __init__(self, config: dict, *, size: int, seed: int) -> None:
        options = dict(config)
        self.size = int(size)
        self.seed = int(seed)
        self.num_points = int(options.get("num_points", 64))
        self.point_dim = int(options.get("point_dim", 2))
        self.sampling_nonuniformity = float(
            options.get("sampling_nonuniformity", 0.45)
        )
        self.dtype = options.get("dtype", torch.float32)
        if self.point_dim not in (2, 3):
            raise ValueError(
                "synthetic_shape_fraction supports only point_dim 2 or 3"
            )
        if self.num_points < 2:
            raise ValueError("num_points must be at least 2")

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _uniform(generator: torch.Generator, low: float, high: float) -> float:
        value = float(torch.rand((), generator=generator, dtype=torch.float64))
        return low + (high - low) * value

    def _base_curve(
        self,
        family: str,
        parameters: torch.Tensor,
        complexity: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        t = parameters
        phase = self._uniform(generator, -math.pi, math.pi)
        amplitude = self._uniform(generator, 0.75, 1.15)
        points = torch.zeros(self.num_points, self.point_dim, dtype=self.dtype)

        if family == "shape_industrial":
            # An open arc blended with one or two smooth inflections. This
            # resembles a designed profile without introducing sharp corners.
            span = math.pi * (0.55 + 0.08 * complexity)
            angle = span * (t - 0.5) + 0.2 * phase
            points[:, 0] = torch.sin(angle)
            points[:, 1] = amplitude * (
                0.72 * torch.cos(angle)
                + 0.08 * complexity * torch.sin(2.0 * math.pi * t + phase)
            )
        elif family == "shape_terrain":
            # Monotone progress with bounded multi-scale undulation: a compact
            # analogue of coastline/contour geometry.
            # Keep the highest case comfortably inside the configured K<=24
            # source regime while retaining distinct low/high frequencies.
            frequency = 1 + ((complexity + 1) // 2)
            points[:, 0] = 2.0 * t - 1.0
            points[:, 1] = amplitude * (
                0.42 * torch.sin(frequency * math.pi * t + phase)
                + 0.13 * torch.sin((frequency + 2) * math.pi * t - 0.5 * phase)
            )
        else:
            # A loop-and-tail open stroke. The sin(pi*t) envelope keeps the
            # endpoints distinct while varying the number of interior turns.
            turns = 0.80 + 0.10 * complexity
            angle = 2.0 * math.pi * turns * t + phase
            envelope = 0.25 + 0.75 * torch.sin(math.pi * t)
            points[:, 0] = 1.5 * (t - 0.5) + 0.24 * envelope * torch.sin(angle)
            points[:, 1] = amplitude * (
                0.36 * envelope * torch.cos(angle) + 0.12 * torch.sin(math.pi * t)
            )

        if self.point_dim == 3:
            depth_frequency = 1 + (complexity % 3)
            points[:, 2] = 0.22 * amplitude * torch.sin(
                depth_frequency * math.pi * t - 0.35 * phase
            )
        return points

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        parameters = generate_sampling_parameters(
            self.num_points,
            nonuniformity=self.sampling_nonuniformity,
            generator=generator,
            dtype=self.dtype,
        )
        family = self._FAMILIES[index % len(self._FAMILIES)]
        complexity = 1 + ((index // len(self._FAMILIES)) % 5)
        points = self._base_curve(family, parameters, complexity, generator)

        # Apply a deterministic random frame so the network cannot identify a
        # family from a fixed axis convention.
        frame = torch.randn(
            self.point_dim,
            self.point_dim,
            generator=generator,
            dtype=self.dtype,
        )
        orthogonal, _ = torch.linalg.qr(frame)
        points = points @ orthogonal.transpose(0, 1)
        center = points.mean(dim=0, keepdim=True)
        points = points - center
        scale = points.norm(dim=-1).amax().clamp_min(1e-8)
        points = points / scale
        return {
            "points": points,
            "synthetic_family": family,
        }


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
                 real_fraction=0.5, epoch=0, resample=True,
                 synthetic_simple_fraction=0.0,
                 synthetic_shape_fraction=0.0):
        if size < 1 or size >= EPOCH_SEED_STRIDE:
            raise ValueError("training size must be positive and below the epoch seed stride")
        if not 0 <= real_fraction <= 1:
            raise ValueError("real_fraction must lie in [0,1]")
        fractions = {
            "synthetic_simple_fraction": synthetic_simple_fraction,
            "synthetic_shape_fraction": synthetic_shape_fraction,
        }
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in fractions.values()
        ):
            raise ValueError("synthetic augmentation fractions must lie in [0,1]")
        if synthetic_simple_fraction + synthetic_shape_fraction > 1.0 + 1e-12:
            raise ValueError("synthetic augmentation fractions must sum to at most 1")
        self.size = size
        self.seed = seed + (epoch * EPOCH_SEED_STRIDE if resample else 0)
        self.real_sources = list(real_sources)
        self.real_fraction = real_fraction if self.real_sources else 0.0
        self.synthetic_simple_fraction = float(synthetic_simple_fraction)
        self.synthetic_shape_fraction = float(synthetic_shape_fraction)
        self.synthetic_augmentation_enabled = bool(
            self.synthetic_simple_fraction or self.synthetic_shape_fraction
        )
        options = dict(config)
        self.certified_synthetic_targets = bool(
            options.get("certified_minimal_source", False)
        )
        options.update(
            return_ground_truth=False,
            cache_samples=False,
        )
        self.synthetic = SyntheticCubicBSplineDataset(size=size, seed=self.seed, **options)
        self.simple_synthetic = None
        if self.synthetic_simple_fraction:
            self.simple_synthetic = _LowKCertifiedSyntheticView(
                config,
                size=size,
                seed=_namespaced_seed(self.seed, "certified-low-k"),
            )
        self.shape_synthetic = None
        if self.synthetic_shape_fraction:
            self.shape_synthetic = _CompactShapeSyntheticDataset(
                config,
                size=size,
                seed=self.seed + SHAPE_SYNTHETIC_SEED_OFFSET,
            )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = random.Random(self.seed + index + 2**40)
        if self.real_sources and rng.random() < self.real_fraction:
            label, train, _ = self.real_sources[rng.randrange(len(self.real_sources))]
            points = train[rng.randrange(len(train))]["points"]
            sample = None
            family = "real"
        else:
            label = "Synthetic"
            family = "historical"
            if self.synthetic_augmentation_enabled:
                # Family selection is independent of the real-source gate. In
                # particular, adding validation-only manifests with
                # real_fraction=0 must not perturb the synthetic train stream.
                family_roll = random.Random(
                    _namespaced_seed(self.seed + index, "synthetic-family")
                ).random()
                if family_roll < self.synthetic_simple_fraction:
                    if self.simple_synthetic is None:
                        raise RuntimeError("simple synthetic dataset was not initialized")
                    sample = self.simple_synthetic[index]
                    family = "simple"
                elif family_roll < (
                    self.synthetic_simple_fraction + self.synthetic_shape_fraction
                ):
                    if self.shape_synthetic is None:
                        raise RuntimeError("shape synthetic dataset was not initialized")
                    sample = self.shape_synthetic[index]
                    family = sample["synthetic_family"]
                else:
                    sample = self.synthetic[index]
            else:
                sample = self.synthetic[index]
            points = sample["points"]
        record = _curve_record(
            points,
            label,
            max_internal_knots=self.synthetic.max_internal_knots,
            synthetic_sample=(
                sample if family in {"historical", "simple"} else None
            ),
            certified=(
                self.certified_synthetic_targets
                and family in {"historical", "simple"}
            ),
        )
        if self.synthetic_augmentation_enabled:
            record["synthetic_family"] = family
        return record


class ValidationCurves(Dataset):
    def __init__(self, config, real_sources=(), *, size=1000, seed=1_000_000,
                 real_per_source=100):
        if size < 1 or real_per_source < 1:
            raise ValueError("validation counts must be positive")
        options = dict(config)
        self.certified_synthetic_targets = bool(
            options.get("certified_minimal_source", False)
        )
        options.update(
            return_ground_truth=False,
            cache_samples=True,
        )
        self.synthetic = SyntheticCubicBSplineDataset(size=size, seed=seed, **options)
        self.entries = [("Synthetic", self.synthetic, i) for i in range(size)]
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
