from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..evaluation.minimal_knot_pruning import prune_knots_to_rms_tolerance
from ..spline.bspline_deletion_teacher import single_knot_deletion_rmse_batch


_CACHE_FORMAT = "spline_fitting.one_shot_teacher"
_CACHE_VERSION = 2
_INPUT_NAMES = ("parameters", "points", "candidate_knots")
_LOSS_KEYS = (
    "teacher_retained_mask",
    "teacher_soft_keep_risk",
    "teacher_internal_knots",
    "teacher_internal_knot_mask",
    "teacher_count",
    "teacher_fit_rms",
    "teacher_threshold_satisfied",
    "teacher_deletion_order",
    "teacher_single_deletion_rms",
)


@dataclass(frozen=True)
class OneShotTeacherConfig:
    """Numerical contract shared by offline labels and hard deployment.

    ``temperature`` is dimensionless because the soft risk is computed from
    the final stopping state's leave-one-out RMS divided by
    ``error_tolerance``. Endpoint interpolation is intentionally mandatory:
    the teacher must judge full-curve coverage in exactly the same way as the
    deployment refitter.
    """

    error_tolerance: float
    temperature: float = 0.1
    min_internal_knots: int = 0
    degree: int = 3
    smoothness_weight: float = 1e-6
    control_ridge: float = 0.0
    interpolate_endpoints: bool = True
    rcond: float | None = None
    proposal_fingerprint: str = ""
    dataset_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.error_tolerance) or self.error_tolerance <= 0.0:
            raise ValueError("error_tolerance must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if isinstance(self.min_internal_knots, bool) or not isinstance(
            self.min_internal_knots, int
        ):
            raise TypeError("min_internal_knots must be an integer")
        if self.min_internal_knots < 0:
            raise ValueError("min_internal_knots must be non-negative")
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            raise TypeError("degree must be an integer")
        if self.degree < 1:
            raise ValueError("degree must be positive")
        if not math.isfinite(self.smoothness_weight) or self.smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must be finite and non-negative")
        if not math.isfinite(self.control_ridge) or self.control_ridge < 0.0:
            raise ValueError("control_ridge must be finite and non-negative")
        if self.interpolate_endpoints is not True:
            raise ValueError("one-shot teacher requires interpolate_endpoints=True")
        if self.rcond is not None and (
            not math.isfinite(self.rcond) or self.rcond < 0.0
        ):
            raise ValueError("rcond must be finite and non-negative or None")
        if not isinstance(self.proposal_fingerprint, str):
            raise TypeError("proposal_fingerprint must be a string")
        if not isinstance(self.dataset_fingerprint, str):
            raise TypeError("dataset_fingerprint must be a string")

    def as_dict(self) -> dict[str, int | float | bool | str | None]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> OneShotTeacherConfig:
        expected = {field.name for field in fields(cls)}
        supplied = set(values)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"invalid teacher config keys: missing={missing}, extra={extra}"
            )
        return cls(**dict(values))  # type: ignore[arg-type]

    def fingerprint(self) -> str:
        canonical = {
            "cache_format": _CACHE_FORMAT,
            "cache_version": _CACHE_VERSION,
            "config": self.as_dict(),
        }
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _normalize_sample_indices(
    sample_indices: torch.Tensor | Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    indices = torch.as_tensor(sample_indices, dtype=torch.long, device="cpu")
    if indices.ndim != 1 or indices.shape[0] != batch_size:
        raise ValueError(f"sample_indices must have shape [{batch_size}]")
    if torch.any(indices < 0):
        raise ValueError("sample_indices must be non-negative")
    if torch.unique(indices).numel() != batch_size:
        raise ValueError("sample_indices must be unique")
    # Metadata must not alias a caller-owned tensor: mutating a dataset's ID
    # array later must be detected rather than silently changing the cache key.
    return indices.clone().contiguous()


def _normalize_shapes(
    shapes: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    if set(shapes) != set(_INPUT_NAMES):
        missing = sorted(set(_INPUT_NAMES) - set(shapes))
        extra = sorted(set(shapes) - set(_INPUT_NAMES))
        raise ValueError(f"invalid input shape keys: missing={missing}, extra={extra}")
    normalized: dict[str, tuple[int, ...]] = {}
    expected_ranks = {"parameters": 2, "points": 3, "candidate_knots": 2}
    for name in _INPUT_NAMES:
        shape = tuple(int(value) for value in shapes[name])
        if len(shape) != expected_ranks[name] or any(value <= 0 for value in shape):
            raise ValueError(f"invalid {name} shape: {shape}")
        normalized[name] = shape
    batch_sizes = {shape[0] for shape in normalized.values()}
    if len(batch_sizes) != 1:
        raise ValueError("all input shapes must share a batch size")
    if normalized["parameters"][:2] != normalized["points"][:2]:
        raise ValueError("parameters and points shapes must share [B, M]")
    return normalized


@dataclass(frozen=True)
class OneShotTeacherBatch:
    """Dense, cacheable labels produced by hard RMS-constrained pruning."""

    sample_indices: torch.Tensor
    input_shapes: dict[str, tuple[int, ...]]
    config: OneShotTeacherConfig
    teacher_retained_mask: torch.Tensor
    teacher_soft_keep_risk: torch.Tensor
    teacher_internal_knots: torch.Tensor
    teacher_internal_knot_mask: torch.Tensor
    teacher_count: torch.Tensor
    teacher_fit_rms: torch.Tensor
    teacher_threshold_satisfied: torch.Tensor
    teacher_deletion_order: torch.Tensor
    teacher_single_deletion_rms: torch.Tensor

    def __post_init__(self) -> None:
        shapes = _normalize_shapes(self.input_shapes)
        batch_size, candidate_count = shapes["candidate_knots"]
        indices = _normalize_sample_indices(self.sample_indices, batch_size)
        object.__setattr__(self, "sample_indices", indices)
        object.__setattr__(self, "input_shapes", shapes)

        slot_shapes = {
            "teacher_retained_mask": self.teacher_retained_mask,
            "teacher_soft_keep_risk": self.teacher_soft_keep_risk,
            "teacher_internal_knots": self.teacher_internal_knots,
            "teacher_internal_knot_mask": self.teacher_internal_knot_mask,
            "teacher_deletion_order": self.teacher_deletion_order,
            "teacher_single_deletion_rms": self.teacher_single_deletion_rms,
        }
        for name, value in slot_shapes.items():
            if not isinstance(value, torch.Tensor) or value.shape != (
                batch_size,
                candidate_count,
            ):
                raise ValueError(
                    f"{name} must have shape [{batch_size}, {candidate_count}]"
                )
        sample_shapes = {
            "teacher_count": self.teacher_count,
            "teacher_fit_rms": self.teacher_fit_rms,
            "teacher_threshold_satisfied": self.teacher_threshold_satisfied,
        }
        for name, value in sample_shapes.items():
            if not isinstance(value, torch.Tensor) or value.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [{batch_size}]")

        if self.teacher_retained_mask.dtype != torch.bool:
            raise ValueError("teacher_retained_mask must be Boolean")
        if self.teacher_internal_knot_mask.dtype != torch.bool:
            raise ValueError("teacher_internal_knot_mask must be Boolean")
        if self.teacher_threshold_satisfied.dtype != torch.bool:
            raise ValueError("teacher_threshold_satisfied must be Boolean")
        if self.teacher_count.dtype != torch.long:
            raise ValueError("teacher_count must use torch.long")
        if self.teacher_deletion_order.dtype != torch.long:
            raise ValueError("teacher_deletion_order must use torch.long")

        floating = (
            self.teacher_soft_keep_risk,
            self.teacher_internal_knots,
            self.teacher_fit_rms,
            self.teacher_single_deletion_rms,
        )
        if not all(value.is_floating_point() for value in floating):
            raise ValueError(
                "teacher risks, knots and RMS values must be floating point"
            )
        if not all(torch.isfinite(value).all() for value in floating):
            raise ValueError("teacher floating-point tensors must be finite")
        if torch.any(
            (self.teacher_soft_keep_risk < 0.0) | (self.teacher_soft_keep_risk > 1.0)
        ):
            raise ValueError("teacher_soft_keep_risk must lie in [0, 1]")
        if torch.any((self.teacher_count < 0) | (self.teacher_count > candidate_count)):
            raise ValueError("teacher_count lies outside the candidate range")
        if not torch.equal(
            self.teacher_count.cpu(),
            self.teacher_retained_mask.sum(dim=-1).to(torch.long).cpu(),
        ):
            raise ValueError("teacher_count disagrees with teacher_retained_mask")
        if not torch.equal(
            self.teacher_count.cpu(),
            self.teacher_internal_knot_mask.sum(dim=-1).to(torch.long).cpu(),
        ):
            raise ValueError("teacher_count disagrees with teacher_internal_knot_mask")
        slots = torch.arange(candidate_count, device=self.teacher_count.device)
        expected_packed_mask = slots.unsqueeze(0) < self.teacher_count.unsqueeze(1)
        if not torch.equal(
            self.teacher_internal_knot_mask.cpu(), expected_packed_mask.cpu()
        ):
            raise ValueError("teacher_internal_knot_mask must be a packed prefix mask")
        valid_deletion = (self.teacher_deletion_order == -1) | (
            (self.teacher_deletion_order >= 0)
            & (self.teacher_deletion_order < candidate_count)
        )
        if not torch.all(valid_deletion):
            raise ValueError(
                "teacher_deletion_order contains an invalid original index"
            )

    @property
    def config_fingerprint(self) -> str:
        return self.config.fingerprint()

    def as_loss_kwargs(self) -> dict[str, torch.Tensor]:
        """Return only tensors that can be merged into a training batch."""
        return {name: getattr(self, name) for name in _LOSS_KEYS}

    def as_dict(self) -> dict[str, torch.Tensor]:
        return self.as_loss_kwargs()

    def to(self, device: torch.device | str) -> OneShotTeacherBatch:
        values = {
            name: value.to(device) for name, value in self.as_loss_kwargs().items()
        }
        return OneShotTeacherBatch(
            sample_indices=self.sample_indices,
            input_shapes=self.input_shapes,
            config=self.config,
            **values,
        )


def _validate_model_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_knots: torch.Tensor,
    config: OneShotTeacherConfig,
) -> None:
    if parameters.ndim != 2:
        raise ValueError("parameters must have shape [B, M]")
    if points.ndim != 3:
        raise ValueError("points must have shape [B, M, D]")
    if candidate_knots.ndim != 2:
        raise ValueError("candidate_knots must have shape [B, K]")
    if (
        parameters.shape[0] != points.shape[0]
        or parameters.shape[0] != candidate_knots.shape[0]
    ):
        raise ValueError("parameters, points and candidate_knots must share B")
    if parameters.shape[1] != points.shape[1]:
        raise ValueError("parameters and points must share M")
    if candidate_knots.shape[1] < 1:
        raise ValueError("candidate_knots must contain at least one knot")
    if config.min_internal_knots > candidate_knots.shape[1]:
        raise ValueError("min_internal_knots exceeds the candidate knot count")


@torch.no_grad()
def build_one_shot_teacher_batch(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_knots: torch.Tensor,
    *,
    sample_indices: torch.Tensor | Sequence[int],
    config: OneShotTeacherConfig,
) -> OneShotTeacherBatch:
    """Create hard subset labels and soft one-deletion risk labels for a batch."""
    _validate_model_inputs(parameters, points, candidate_knots, config)
    batch_size, candidate_count = candidate_knots.shape
    normalized_indices = _normalize_sample_indices(sample_indices, batch_size)

    # Keep the initial all-candidate deletion RMS as a diagnostic only. It is
    # not a valid final keep target: several redundant knots may each be safe
    # to delete from the full set even though at least one must ultimately
    # survive.
    single_deletion_rms = single_knot_deletion_rmse_batch(
        parameters,
        points,
        candidate_knots,
        degree=config.degree,
        smoothness_weight=config.smoothness_weight,
        control_ridge=config.control_ridge,
        interpolate_endpoints=True,
        rcond=config.rcond,
    )
    retained_mask = torch.zeros_like(candidate_knots, dtype=torch.bool)
    soft_keep_risk = torch.zeros_like(candidate_knots)
    packed_knots = torch.zeros_like(candidate_knots)
    packed_knot_mask = torch.zeros_like(candidate_knots, dtype=torch.bool)
    counts = torch.zeros(batch_size, dtype=torch.long, device=candidate_knots.device)
    final_rms = torch.zeros(batch_size, dtype=points.dtype, device=points.device)
    threshold_satisfied = torch.zeros(
        batch_size, dtype=torch.bool, device=points.device
    )
    deletion_order = torch.full(
        (batch_size, candidate_count),
        -1,
        dtype=torch.long,
        device=candidate_knots.device,
    )

    for batch_index in range(batch_size):
        result = prune_knots_to_rms_tolerance(
            parameters[batch_index],
            points[batch_index],
            candidate_knots[batch_index],
            error_tolerance=config.error_tolerance,
            min_internal_knots=config.min_internal_knots,
            degree=config.degree,
            smoothness_weight=config.smoothness_weight,
            control_ridge=config.control_ridge,
            interpolate_endpoints=True,
            rcond=config.rcond,
        )
        original_slots = list(range(candidate_count))
        accepted_original_slots: list[int] = []
        for step in result.accepted_steps:
            accepted_original_slots.append(original_slots.pop(step.removed_index))

        retained_mask[batch_index, original_slots] = True
        count = len(original_slots)
        counts[batch_index] = count
        if count:
            packed_knots[batch_index, :count] = candidate_knots[
                batch_index, original_slots
            ]
            packed_knot_mask[batch_index, :count] = True
        if accepted_original_slots:
            deletion_order[batch_index, : len(accepted_original_slots)] = torch.tensor(
                accepted_original_slots,
                dtype=torch.long,
                device=candidate_knots.device,
            )

        if original_slots:
            # Greedy pruning normally terminates with one rejected round. Its
            # RMS vector is the exact leave-one-out risk of each knot in the
            # final retained state. Map that current-state order back to the
            # immutable original candidate slots.
            rejected_step = (
                result.steps[-1]
                if result.steps and not result.steps[-1].accepted
                else None
            )
            if rejected_step is None:
                # Reaching min_internal_knots is the only normal stop without
                # a rejected round. Those slots are mandatory by configuration.
                final_keep_risk = candidate_knots.new_ones(len(original_slots))
            else:
                if rejected_step.all_candidate_rmse.shape != (len(original_slots),):
                    raise RuntimeError(
                        "final pruning rejection does not match retained slots"
                    )
                final_knots = candidate_knots[batch_index, original_slots]
                if not torch.equal(rejected_step.knots_before, final_knots):
                    raise RuntimeError(
                        "final pruning rejection lost original slot alignment"
                    )
                normalized_margin = (
                    rejected_step.all_candidate_rmse / config.error_tolerance - 1.0
                )
                final_keep_risk = torch.sigmoid(
                    normalized_margin / config.temperature
                ).clamp_min(0.5)
            soft_keep_risk[batch_index, original_slots] = final_keep_risk
        final_rms[batch_index] = result.final_fit.fit_rmse
        threshold_satisfied[batch_index] = result.threshold_satisfied

    return OneShotTeacherBatch(
        sample_indices=normalized_indices,
        input_shapes={
            "parameters": tuple(parameters.shape),
            "points": tuple(points.shape),
            "candidate_knots": tuple(candidate_knots.shape),
        },
        config=config,
        teacher_retained_mask=retained_mask,
        teacher_soft_keep_risk=soft_keep_risk,
        teacher_internal_knots=packed_knots,
        teacher_internal_knot_mask=packed_knot_mask,
        teacher_count=counts,
        teacher_fit_rms=final_rms,
        teacher_threshold_satisfied=threshold_satisfied,
        teacher_deletion_order=deletion_order,
        teacher_single_deletion_rms=single_deletion_rms,
    )


# A shorter verb is convenient in offline-cache generation scripts.
compute_one_shot_teacher_batch = build_one_shot_teacher_batch


def _cache_payload(batch: OneShotTeacherBatch) -> dict[str, object]:
    return {
        "format": _CACHE_FORMAT,
        "version": _CACHE_VERSION,
        "config": batch.config.as_dict(),
        "config_fingerprint": batch.config_fingerprint,
        "sample_indices": batch.sample_indices.detach().cpu(),
        "input_shapes": {
            name: list(shape) for name, shape in batch.input_shapes.items()
        },
        "labels": {
            name: value.detach().cpu() for name, value in batch.as_loss_kwargs().items()
        },
    }


def save_one_shot_teacher_cache(
    path: str | Path,
    batch: OneShotTeacherBatch,
) -> None:
    """Save a portable tensor-only ``.pt`` teacher cache."""
    destination = Path(path)
    if destination.suffix.lower() != ".pt":
        raise ValueError("one-shot teacher cache must use a .pt extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cache_payload(batch), destination)


def _load_tensor_payload(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("teacher cache root must be a mapping")
    return payload


def load_one_shot_teacher_cache(
    path: str | Path,
    *,
    expected_config: OneShotTeacherConfig,
    expected_sample_indices: torch.Tensor | Sequence[int],
    expected_input_shapes: Mapping[str, Sequence[int]],
    device: torch.device | str = "cpu",
) -> OneShotTeacherBatch:
    """Load a cache only after exact config, index, and input-shape checks."""
    source = Path(path)
    if source.suffix.lower() != ".pt":
        raise ValueError("one-shot teacher cache must use a .pt extension")
    payload = _load_tensor_payload(source)
    if payload.get("format") != _CACHE_FORMAT:
        raise ValueError("not a one-shot teacher cache")
    if payload.get("version") != _CACHE_VERSION:
        raise ValueError(
            f"unsupported teacher cache version: {payload.get('version')!r}"
        )

    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("teacher cache config is missing or invalid")
    cached_config = OneShotTeacherConfig.from_dict(raw_config)
    stored_fingerprint = payload.get("config_fingerprint")
    if stored_fingerprint != cached_config.fingerprint():
        raise ValueError("teacher cache config fingerprint is corrupted")
    if cached_config.fingerprint() != expected_config.fingerprint():
        raise ValueError("teacher cache config does not match expected_config")

    normalized_shapes = _normalize_shapes(expected_input_shapes)
    raw_shapes = payload.get("input_shapes")
    if not isinstance(raw_shapes, Mapping):
        raise ValueError("teacher cache input_shapes is missing or invalid")
    cached_shapes = _normalize_shapes(raw_shapes)  # type: ignore[arg-type]
    if cached_shapes != normalized_shapes:
        raise ValueError(
            f"teacher cache input shapes differ: cached={cached_shapes}, "
            f"expected={normalized_shapes}"
        )

    raw_indices = payload.get("sample_indices")
    if not isinstance(raw_indices, torch.Tensor):
        raise ValueError("teacher cache sample_indices is missing or invalid")
    expected_indices = _normalize_sample_indices(
        expected_sample_indices, normalized_shapes["parameters"][0]
    )
    cached_indices = _normalize_sample_indices(
        raw_indices, normalized_shapes["parameters"][0]
    )
    if not torch.equal(cached_indices, expected_indices):
        raise ValueError("teacher cache sample_indices do not match")

    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, Mapping):
        raise ValueError("teacher cache labels are missing or invalid")
    if set(raw_labels) != set(_LOSS_KEYS):
        missing = sorted(set(_LOSS_KEYS) - set(raw_labels))
        extra = sorted(set(raw_labels) - set(_LOSS_KEYS))
        raise ValueError(
            f"invalid teacher label keys: missing={missing}, extra={extra}"
        )
    labels: dict[str, torch.Tensor] = {}
    for name in _LOSS_KEYS:
        value = raw_labels[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"cached {name} is not a tensor")
        labels[name] = value.to(device)

    return OneShotTeacherBatch(
        sample_indices=cached_indices,
        input_shapes=cached_shapes,
        config=cached_config,
        **labels,
    )


class TeacherAugmentedDataset(Dataset):
    """Merge one cached teacher row into every base-dataset sample.

    Offline labels cannot follow a dataset whose population changes each
    epoch. A base dataset advertising ``resample_each_epoch=True`` is therefore
    rejected immediately. ``set_epoch`` is proxied for compatible datasets,
    while sample IDs are checked again on every access.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        teacher: OneShotTeacherBatch | str | Path,
        *,
        expected_config: OneShotTeacherConfig | None = None,
        expected_sample_indices: torch.Tensor | Sequence[int] | None = None,
        expected_input_shapes: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        if bool(getattr(base_dataset, "resample_each_epoch", False)):
            raise ValueError(
                "offline teacher cache cannot wrap a dataset with "
                "resample_each_epoch=True"
            )
        if isinstance(teacher, (str, Path)):
            if (
                expected_config is None
                or expected_sample_indices is None
                or expected_input_shapes is None
            ):
                raise ValueError(
                    "loading a teacher cache requires expected_config, "
                    "expected_sample_indices, and expected_input_shapes"
                )
            teacher = load_one_shot_teacher_cache(
                teacher,
                expected_config=expected_config,
                expected_sample_indices=expected_sample_indices,
                expected_input_shapes=expected_input_shapes,
            )
        if len(base_dataset) != teacher.sample_indices.numel():
            raise ValueError("base dataset length and teacher batch length differ")
        self.base_dataset = base_dataset
        self.teacher = teacher

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("TeacherAugmentedDataset requires mapping samples")
        if "sample_id" not in sample:
            raise KeyError("base dataset sample is missing sample_id")
        sample_id = int(torch.as_tensor(sample["sample_id"]).item())
        expected_id = int(self.teacher.sample_indices[index].item())
        if sample_id != expected_id:
            raise ValueError(
                f"base sample_id {sample_id} does not match cached ID {expected_id} "
                f"at row {index}"
            )
        result = dict(sample)
        for name, values in self.teacher.as_loss_kwargs().items():
            if name in result:
                raise KeyError(f"base sample already defines {name}")
            result[name] = values[index]
        return result

    def set_epoch(self, epoch: int) -> None:
        if bool(getattr(self.base_dataset, "resample_each_epoch", False)):
            raise RuntimeError(
                "offline teacher labels cannot be used after enabling resampling"
            )
        setter = getattr(self.base_dataset, "set_epoch", None)
        if setter is not None:
            setter(epoch)


__all__ = [
    "OneShotTeacherBatch",
    "OneShotTeacherConfig",
    "TeacherAugmentedDataset",
    "build_one_shot_teacher_batch",
    "compute_one_shot_teacher_batch",
    "load_one_shot_teacher_cache",
    "save_one_shot_teacher_cache",
]
