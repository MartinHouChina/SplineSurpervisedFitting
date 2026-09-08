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

from ..evaluation.bspline_inference import (
    BSplineLeastSquaresFit,
    refit_bspline_control_points,
)
from ..evaluation.hybrid_knot_search import refine_knot_positions
from ..evaluation.minimal_knot_pruning import (
    MinimalKnotPruningResult,
    prune_knots_to_rms_tolerance,
)
from ..spline.bspline_deletion_teacher import single_knot_deletion_rmse_batch


_CACHE_FORMAT = "spline_fitting.one_shot_teacher"
_CACHE_VERSION = 3
_LEGACY_CACHE_VERSION = 2
_INPUT_NAMES = ("parameters", "points", "candidate_knots")
_LOSS_KEYS = (
    "teacher_retained_mask",
    "teacher_soft_keep_risk",
    "teacher_internal_knots",
    "teacher_internal_knot_mask",
    "teacher_count",
    "teacher_fit_rms",
    "teacher_fit_mse",
    "teacher_threshold_satisfied",
    "teacher_deletion_order",
    "teacher_single_deletion_rms",
    "teacher_greedy_count",
    "teacher_greedy_fit_rms",
    "teacher_relocation_mean_abs",
    "teacher_relocation_max_abs",
    "teacher_extra_deleted_after_relocation",
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
    relocation_strategy: str = "none"
    relocation_rounds: int = 2
    relocation_sweeps: int = 1
    relocation_grid_size: int = 5
    relocation_restarts: int = 1
    relocation_min_gap: float = 1e-4
    relocation_max_shift: float = 0.15

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
        if self.relocation_strategy not in {"none", "delete_then_relax"}:
            raise ValueError(
                "relocation_strategy must be 'none' or 'delete_then_relax'"
            )
        integer_options = {
            "relocation_rounds": self.relocation_rounds,
            "relocation_sweeps": self.relocation_sweeps,
            "relocation_grid_size": self.relocation_grid_size,
            "relocation_restarts": self.relocation_restarts,
        }
        for name, value in integer_options.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.relocation_rounds < 1:
            raise ValueError("relocation_rounds must be positive")
        if self.relocation_sweeps < 1:
            raise ValueError("relocation_sweeps must be positive")
        if self.relocation_grid_size < 3 or self.relocation_grid_size % 2 == 0:
            raise ValueError("relocation_grid_size must be odd and at least 3")
        if self.relocation_restarts < 1:
            raise ValueError("relocation_restarts must be positive")
        if not math.isfinite(self.relocation_min_gap) or self.relocation_min_gap < 0.0:
            raise ValueError("relocation_min_gap must be finite and non-negative")
        if (
            not math.isfinite(self.relocation_max_shift)
            or not 0.0 < self.relocation_max_shift < 0.5
        ):
            raise ValueError("relocation_max_shift must lie in (0, 0.5)")

    def as_dict(self) -> dict[str, int | float | bool | str | None]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> OneShotTeacherConfig:
        expected = {field.name for field in fields(cls)}
        supplied = set(values)
        legacy = expected - {
            "relocation_strategy",
            "relocation_rounds",
            "relocation_sweeps",
            "relocation_grid_size",
            "relocation_restarts",
            "relocation_min_gap",
            "relocation_max_shift",
        }
        if supplied == legacy:
            return cls(**dict(values))  # type: ignore[arg-type]
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"invalid teacher config keys: missing={missing}, extra={extra}"
            )
        return cls(**dict(values))  # type: ignore[arg-type]

    def fingerprint(self) -> str:
        return _config_fingerprint(self.as_dict(), version=_CACHE_VERSION)


def _config_fingerprint(
    values: Mapping[str, object],
    *,
    version: int,
) -> str:
    canonical = {
        "cache_format": _CACHE_FORMAT,
        "cache_version": version,
        "config": dict(values),
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
    teacher_fit_mse: torch.Tensor
    teacher_threshold_satisfied: torch.Tensor
    teacher_deletion_order: torch.Tensor
    teacher_single_deletion_rms: torch.Tensor
    teacher_greedy_count: torch.Tensor
    teacher_greedy_fit_rms: torch.Tensor
    teacher_relocation_mean_abs: torch.Tensor
    teacher_relocation_max_abs: torch.Tensor
    teacher_extra_deleted_after_relocation: torch.Tensor

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
            "teacher_fit_mse": self.teacher_fit_mse,
            "teacher_threshold_satisfied": self.teacher_threshold_satisfied,
            "teacher_greedy_count": self.teacher_greedy_count,
            "teacher_greedy_fit_rms": self.teacher_greedy_fit_rms,
            "teacher_relocation_mean_abs": self.teacher_relocation_mean_abs,
            "teacher_relocation_max_abs": self.teacher_relocation_max_abs,
            "teacher_extra_deleted_after_relocation": (
                self.teacher_extra_deleted_after_relocation
            ),
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
        if self.teacher_greedy_count.dtype != torch.long:
            raise ValueError("teacher_greedy_count must use torch.long")
        if self.teacher_extra_deleted_after_relocation.dtype != torch.long:
            raise ValueError(
                "teacher_extra_deleted_after_relocation must use torch.long"
            )
        if self.teacher_deletion_order.dtype != torch.long:
            raise ValueError("teacher_deletion_order must use torch.long")

        floating = (
            self.teacher_soft_keep_risk,
            self.teacher_internal_knots,
            self.teacher_fit_rms,
            self.teacher_fit_mse,
            self.teacher_single_deletion_rms,
            self.teacher_greedy_fit_rms,
            self.teacher_relocation_mean_abs,
            self.teacher_relocation_max_abs,
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
        if torch.any(
            (self.teacher_greedy_count < self.teacher_count)
            | (self.teacher_greedy_count > candidate_count)
        ):
            raise ValueError("teacher_greedy_count lies outside the valid range")
        if not torch.equal(
            self.teacher_extra_deleted_after_relocation.cpu(),
            (self.teacher_greedy_count - self.teacher_count).cpu(),
        ):
            raise ValueError(
                "teacher_extra_deleted_after_relocation disagrees with counts"
            )
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
        if not torch.allclose(
            self.teacher_fit_mse.cpu(),
            self.teacher_fit_rms.square().cpu(),
            rtol=1e-5,
            atol=1e-12,
        ):
            raise ValueError("teacher_fit_mse must equal teacher_fit_rms squared")
        if torch.any(self.teacher_relocation_mean_abs < 0.0) or torch.any(
            self.teacher_relocation_max_abs < self.teacher_relocation_mean_abs
        ):
            raise ValueError("teacher relocation distances are inconsistent")
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


def _map_pruning_slots(
    result: MinimalKnotPruningResult,
    starting_slots: list[int],
) -> tuple[list[int], list[int]]:
    """Map current-state greedy removals back to immutable proposal slots."""
    retained_slots = list(starting_slots)
    removed_slots: list[int] = []
    for step in result.accepted_steps:
        removed_slots.append(retained_slots.pop(step.removed_index))
    if len(retained_slots) != result.final_count:
        raise RuntimeError("teacher pruning lost immutable proposal-slot alignment")
    return retained_slots, removed_slots


def _prune_teacher_state(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    slots: list[int],
    config: OneShotTeacherConfig,
) -> tuple[MinimalKnotPruningResult, list[int], list[int]]:
    result = prune_knots_to_rms_tolerance(
        parameters,
        points,
        knots,
        error_tolerance=config.error_tolerance,
        min_internal_knots=config.min_internal_knots,
        degree=config.degree,
        smoothness_weight=config.smoothness_weight,
        control_ridge=config.control_ridge,
        interpolate_endpoints=True,
        rcond=config.rcond,
    )
    retained_slots, removed_slots = _map_pruning_slots(result, slots)
    return result, retained_slots, removed_slots


def _project_relocated_knots(
    proposed: torch.Tensor,
    anchors: torch.Tensor,
    *,
    min_gap: float,
    max_shift: float,
) -> torch.Tensor:
    """Project ordered teacher knots into the student's reachable domain."""
    count = int(proposed.numel())
    if count == 0:
        return proposed
    gap = proposed.new_tensor(min_gap)
    index = torch.arange(count, device=proposed.device, dtype=proposed.dtype)
    lower = torch.maximum(
        anchors - max_shift,
        (index + 1.0) * gap,
    )
    upper = torch.minimum(
        anchors + max_shift,
        1.0 - (count - index) * gap,
    )
    for knot_index in range(1, count):
        lower[knot_index] = torch.maximum(
            lower[knot_index], lower[knot_index - 1] + gap
        )
    for knot_index in range(count - 2, -1, -1):
        upper[knot_index] = torch.minimum(
            upper[knot_index], upper[knot_index + 1] - gap
        )
    if torch.any(lower > upper):
        raise ValueError("teacher relocation constraints have no feasible solution")
    projected = torch.maximum(torch.minimum(proposed, upper), lower)
    for knot_index in range(1, count):
        projected[knot_index] = torch.maximum(
            projected[knot_index], projected[knot_index - 1] + gap
        )
    return projected


def _relax_teacher_state(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    anchors: torch.Tensor,
    config: OneShotTeacherConfig,
) -> BSplineLeastSquaresFit:
    relaxed = refine_knot_positions(
        parameters,
        points,
        knots,
        degree=config.degree,
        smoothness_weight=config.smoothness_weight,
        control_ridge=config.control_ridge,
        interpolate_endpoints=True,
        rcond=config.rcond,
        min_gap=config.relocation_min_gap,
        sweeps=config.relocation_sweeps,
        grid_size=config.relocation_grid_size,
        restarts=config.relocation_restarts,
    )
    projected = _project_relocated_knots(
        relaxed.final_fit.internal_knots,
        anchors,
        min_gap=config.relocation_min_gap,
        max_shift=config.relocation_max_shift,
    )
    if torch.equal(projected, relaxed.final_fit.internal_knots):
        return relaxed.final_fit
    projected_fit = refit_bspline_control_points(
        parameters,
        points,
        projected,
        degree=config.degree,
        smoothness_weight=config.smoothness_weight,
        control_ridge=config.control_ridge,
        interpolate_endpoints=True,
        rcond=config.rcond,
    )
    if float(projected_fit.fit_mse) <= float(relaxed.initial_fit.fit_mse):
        return projected_fit
    return relaxed.initial_fit


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
    final_mse = torch.zeros(batch_size, dtype=points.dtype, device=points.device)
    greedy_counts = torch.zeros(
        batch_size, dtype=torch.long, device=candidate_knots.device
    )
    greedy_rms = torch.zeros(batch_size, dtype=points.dtype, device=points.device)
    relocation_mean_abs = torch.zeros(
        batch_size, dtype=points.dtype, device=points.device
    )
    relocation_max_abs = torch.zeros(
        batch_size, dtype=points.dtype, device=points.device
    )
    extra_deleted = torch.zeros(
        batch_size, dtype=torch.long, device=candidate_knots.device
    )
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
        original_slots = list(range(candidate_count))
        result, current_slots, accepted_original_slots = _prune_teacher_state(
            parameters[batch_index],
            points[batch_index],
            candidate_knots[batch_index],
            original_slots,
            config,
        )
        greedy_count = len(current_slots)
        greedy_counts[batch_index] = greedy_count
        greedy_rms[batch_index] = result.final_fit.fit_rmse
        current_knots = result.final_internal_knots
        current_fit = result.final_fit

        if config.relocation_strategy == "delete_then_relax":
            needs_final_relax = False
            for _ in range(config.relocation_rounds):
                if not current_slots:
                    break
                current_fit = _relax_teacher_state(
                    parameters[batch_index],
                    points[batch_index],
                    current_knots,
                    candidate_knots[batch_index, current_slots],
                    config,
                )
                current_knots = current_fit.internal_knots
                if len(current_slots) <= config.min_internal_knots:
                    break

                relaxed_pruning, next_slots, newly_removed = _prune_teacher_state(
                    parameters[batch_index],
                    points[batch_index],
                    current_knots,
                    current_slots,
                    config,
                )
                accepted_original_slots.extend(newly_removed)
                current_slots = next_slots
                current_fit = relaxed_pruning.final_fit
                current_knots = relaxed_pruning.final_internal_knots
                needs_final_relax = bool(newly_removed)
                if not newly_removed:
                    break

            # A successful deletion in the last allowed re-pruning round
            # creates a new survivor set. Always relax that final set once,
            # without opening another deletion round, so packed labels never
            # degenerate into an unchanged subset of the previous boundary.
            if current_slots and needs_final_relax:
                current_fit = _relax_teacher_state(
                    parameters[batch_index],
                    points[batch_index],
                    current_knots,
                    candidate_knots[batch_index, current_slots],
                    config,
                )
                current_knots = current_fit.internal_knots

        retained_mask[batch_index, current_slots] = True
        count = len(current_slots)
        counts[batch_index] = count
        if count:
            packed_knots[batch_index, :count] = current_knots
            packed_knot_mask[batch_index, :count] = True
            shifts = (current_knots - candidate_knots[batch_index, current_slots]).abs()
            relocation_mean_abs[batch_index] = shifts.mean()
            relocation_max_abs[batch_index] = shifts.max()
        if accepted_original_slots:
            deletion_order[batch_index, : len(accepted_original_slots)] = torch.tensor(
                accepted_original_slots,
                dtype=torch.long,
                device=candidate_knots.device,
            )

        if current_slots:
            if len(current_slots) <= config.min_internal_knots:
                final_keep_risk = candidate_knots.new_ones(len(current_slots))
            else:
                final_leave_one_out = single_knot_deletion_rmse_batch(
                    parameters[batch_index : batch_index + 1],
                    points[batch_index : batch_index + 1],
                    current_knots.unsqueeze(0),
                    degree=config.degree,
                    smoothness_weight=config.smoothness_weight,
                    control_ridge=config.control_ridge,
                    interpolate_endpoints=True,
                    rcond=config.rcond,
                )[0]
                normalized_margin = final_leave_one_out / config.error_tolerance - 1.0
                final_keep_risk = torch.sigmoid(
                    normalized_margin / config.temperature
                ).clamp_min(0.5)
            soft_keep_risk[batch_index, current_slots] = final_keep_risk
        final_rms[batch_index] = current_fit.fit_rmse
        final_mse[batch_index] = current_fit.fit_mse
        threshold_satisfied[batch_index] = (
            float(current_fit.fit_rmse) <= config.error_tolerance
        )
        extra_deleted[batch_index] = greedy_count - count

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
        teacher_fit_mse=final_mse,
        teacher_threshold_satisfied=threshold_satisfied,
        teacher_deletion_order=deletion_order,
        teacher_single_deletion_rms=single_deletion_rms,
        teacher_greedy_count=greedy_counts,
        teacher_greedy_fit_rms=greedy_rms,
        teacher_relocation_mean_abs=relocation_mean_abs,
        teacher_relocation_max_abs=relocation_max_abs,
        teacher_extra_deleted_after_relocation=extra_deleted,
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
    version = payload.get("version")
    if version not in {_LEGACY_CACHE_VERSION, _CACHE_VERSION}:
        raise ValueError(f"unsupported teacher cache version: {version!r}")

    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("teacher cache config is missing or invalid")
    cached_config = OneShotTeacherConfig.from_dict(raw_config)
    stored_fingerprint = payload.get("config_fingerprint")
    expected_cached_fingerprint = _config_fingerprint(
        raw_config,
        version=int(version),
    )
    if stored_fingerprint != expected_cached_fingerprint:
        raise ValueError("teacher cache config fingerprint is corrupted")
    if cached_config != expected_config:
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
    legacy_loss_keys = {
        "teacher_retained_mask",
        "teacher_soft_keep_risk",
        "teacher_internal_knots",
        "teacher_internal_knot_mask",
        "teacher_count",
        "teacher_fit_rms",
        "teacher_threshold_satisfied",
        "teacher_deletion_order",
        "teacher_single_deletion_rms",
    }
    expected_label_keys = (
        legacy_loss_keys if version == _LEGACY_CACHE_VERSION else set(_LOSS_KEYS)
    )
    if set(raw_labels) != expected_label_keys:
        missing = sorted(expected_label_keys - set(raw_labels))
        extra = sorted(set(raw_labels) - expected_label_keys)
        raise ValueError(
            f"invalid teacher label keys: missing={missing}, extra={extra}"
        )
    labels: dict[str, torch.Tensor] = {}
    for name in expected_label_keys:
        value = raw_labels[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"cached {name} is not a tensor")
        labels[name] = value.to(device)
    if version == _LEGACY_CACHE_VERSION:
        count = labels["teacher_count"]
        fit_rms = labels["teacher_fit_rms"]
        labels.update(
            {
                "teacher_fit_mse": fit_rms.square(),
                "teacher_greedy_count": count.clone(),
                "teacher_greedy_fit_rms": fit_rms.clone(),
                "teacher_relocation_mean_abs": fit_rms.new_zeros(fit_rms.shape),
                "teacher_relocation_max_abs": fit_rms.new_zeros(fit_rms.shape),
                "teacher_extra_deleted_after_relocation": count.new_zeros(count.shape),
            }
        )

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
