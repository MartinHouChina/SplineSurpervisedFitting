from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import torch

from ..data.synthetic import bspline_basis_matrix
from .bspline_inference import (
    BSplineLeastSquaresFit,
    refit_bspline_control_points,
    second_difference_matrix,
)
from .hybrid_knot_search import refine_knot_positions
from .knot_diagnostics import build_open_knot_vector, point_fit_statistics


REFERENCE_CERTIFICATE_SCOPE = (
    "measured mean squared Euclidean error on the supplied normalized ordered "
    "reference points; not a continuous-curve or global-minimality guarantee"
)


@dataclass(frozen=True)
class ReferenceCertifiedFitResult:
    """Effect-first post-processing result with an auditable discrete certificate."""

    target_mse: float
    initial_fit: BSplineLeastSquaresFit
    final_fit: BSplineLeastSquaresFit
    status: str
    threshold_satisfied: bool
    certificate_valid: bool
    full_column_rank: bool | None
    expected_solver_rank: int
    maximum_internal_knots: int
    rank_safe_maximum_internal_knots: int
    supplied_initial_count: int
    sanitized_initial_count: int
    added_knot_count: int
    position_refinement_used: bool
    interpolation_fallback_attempted: bool
    interpolation_fallback_used: bool
    tolerance_polyline_fallback_attempted: bool
    tolerance_polyline_fallback_used: bool
    tolerance_polyline_vertex_count: int | None
    tolerance_polyline_measured_mse: float | None
    tolerance_polyline_insertion_count: int
    tolerance_polyline_compaction_removed_count: int
    tolerance_polyline_fallback_exceeded_adaptive_maximum: bool
    polyline_exact_fallback_attempted: bool
    polyline_exact_fallback_used: bool
    polyline_fallback_exceeded_adaptive_maximum: bool
    final_control_point_count: int
    maximum_internal_knot_multiplicity: int
    refit_count: int
    insertion_candidate_evaluation_count: int
    position_refit_count: int
    endpoint_max_distance: float
    failure_reason: str | None
    certification_scope: str = REFERENCE_CERTIFICATE_SCOPE
    continuous_curve_guaranteed: bool = False
    globally_minimal_knot_count_guaranteed: bool = False

    @property
    def initial_mse(self) -> float:
        return float(self.initial_fit.fit_mse)

    @property
    def final_mse(self) -> float:
        return float(self.final_fit.fit_mse)

    @property
    def final_count(self) -> int:
        return int(self.final_fit.internal_knots.numel())


@dataclass(frozen=True)
class PolylineSimplificationResult:
    retained_indices: torch.Tensor
    measured_mse: float
    insertion_count: int
    compaction_removed_count: int

    @property
    def retained_vertex_count(self) -> int:
        return int(self.retained_indices.numel())


def _polyline_interval_statistics(
    parameters: torch.Tensor,
    points: torch.Tensor,
    left: int,
    right: int,
) -> tuple[float, float, int | None]:
    """Return interval SSE, largest point error and its global index."""

    if right <= left:
        raise ValueError("polyline interval must have positive width")
    if right == left + 1:
        return 0.0, 0.0, None
    interval_parameters = parameters[left : right + 1]
    denominator = parameters[right] - parameters[left]
    alpha = ((interval_parameters - parameters[left]) / denominator).unsqueeze(-1)
    interpolated = points[left] + alpha * (points[right] - points[left])
    squared_errors = (interpolated - points[left : right + 1]).square().sum(dim=-1)
    interior_errors = squared_errors[1:-1]
    local_index = int(torch.argmax(interior_errors))
    split_index = left + 1 + local_index
    return (
        float(squared_errors.sum()),
        float(interior_errors[local_index]),
        split_index,
    )


@torch.no_grad()
def simplify_polyline_to_mse(
    parameters: torch.Tensor,
    points: torch.Tensor,
    target_mse: float,
    *,
    compact: bool = True,
) -> PolylineSimplificationResult:
    """Greedily simplify a parameterized polyline under measured point MSE.

    Starting from the two endpoints, the point of largest squared Euclidean
    interpolation error is inserted recursively.  The stopping condition is
    the *global* mean squared Euclidean error on every supplied point, not a
    per-point tolerance.  An optional deletion pass then removes redundant
    retained vertices while preserving that same measured bound.
    """

    if parameters.ndim != 1 or points.ndim != 2:
        raise ValueError("parameters/points must have shape [M] and [M,D]")
    if parameters.shape[0] != points.shape[0] or points.shape[0] < 2:
        raise ValueError("polyline simplification requires at least two paired points")
    if parameters.device != points.device or parameters.dtype != points.dtype:
        raise ValueError("parameters and points must share device and dtype")
    if not parameters.is_floating_point() or not points.is_floating_point():
        raise ValueError("parameters and points must be floating-point")
    if not bool(torch.isfinite(parameters).all() and torch.isfinite(points).all()):
        raise ValueError("parameters and points must be finite")
    if torch.any(parameters[1:] <= parameters[:-1]):
        raise ValueError("polyline simplification requires strictly increasing parameters")
    if not math.isfinite(target_mse) or target_mse < 0.0:
        raise ValueError("target_mse must be finite and non-negative")

    point_count = int(points.shape[0])
    allowed_sse = float(target_mse) * point_count
    initial_sse, initial_maximum, initial_split = _polyline_interval_statistics(
        parameters,
        points,
        0,
        point_count - 1,
    )
    total_sse = initial_sse
    retained = {0, point_count - 1}
    # Entries are deterministic under ties: maximum error first, then interval
    # endpoints.  The stored SSE avoids recomputing the interval being split.
    heap: list[tuple[float, int, int, int, float]] = []
    if initial_split is not None:
        heapq.heappush(
            heap,
            (-initial_maximum, 0, point_count - 1, initial_split, initial_sse),
        )
    insertion_count = 0
    while total_sse > allowed_sse and heap:
        _, left, right, split, old_sse = heapq.heappop(heap)
        retained.add(split)
        insertion_count += 1
        left_sse, left_maximum, left_split = _polyline_interval_statistics(
            parameters,
            points,
            left,
            split,
        )
        right_sse, right_maximum, right_split = _polyline_interval_statistics(
            parameters,
            points,
            split,
            right,
        )
        total_sse += left_sse + right_sse - old_sse
        # Accumulated subtraction can produce a tiny negative number.
        total_sse = max(0.0, total_sse)
        if left_split is not None:
            heapq.heappush(
                heap,
                (-left_maximum, left, split, left_split, left_sse),
            )
        if right_split is not None:
            heapq.heappush(
                heap,
                (-right_maximum, split, right, right_split, right_sse),
            )

    ordered = sorted(retained)
    removed_count = 0
    if compact and len(ordered) > 2:
        # Removing one retained vertex only changes the two incident intervals.
        # Choose the feasible removal with the smallest resulting global SSE and
        # repeat until no single deletion preserves the target.
        while len(ordered) > 2:
            best: tuple[float, int] | None = None
            for retained_position in range(1, len(ordered) - 1):
                left = ordered[retained_position - 1]
                middle = ordered[retained_position]
                right = ordered[retained_position + 1]
                left_sse, _, _ = _polyline_interval_statistics(
                    parameters,
                    points,
                    left,
                    middle,
                )
                right_sse, _, _ = _polyline_interval_statistics(
                    parameters,
                    points,
                    middle,
                    right,
                )
                merged_sse, _, _ = _polyline_interval_statistics(
                    parameters,
                    points,
                    left,
                    right,
                )
                proposed_sse = max(
                    0.0,
                    total_sse - left_sse - right_sse + merged_sse,
                )
                if proposed_sse <= allowed_sse and (
                    best is None or proposed_sse < best[0]
                ):
                    best = (proposed_sse, retained_position)
            if best is None:
                break
            total_sse, retained_position = best
            del ordered[retained_position]
            removed_count += 1

    # Recompute from the final segmentation so the reported value is not based
    # on accumulated heap arithmetic.
    total_sse = sum(
        _polyline_interval_statistics(parameters, points, left, right)[0]
        for left, right in zip(ordered[:-1], ordered[1:])
    )
    return PolylineSimplificationResult(
        retained_indices=torch.tensor(
            ordered,
            device=parameters.device,
            dtype=torch.long,
        ),
        measured_mse=total_sse / point_count,
        insertion_count=insertion_count,
        compaction_removed_count=removed_count,
    )


def _validate_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    initial_knots: torch.Tensor,
    candidate_knots: torch.Tensor | None,
    *,
    target_mse: float,
    maximum_internal_knots: int,
    degree: int,
    min_gap: float,
    position_sweeps: int,
    position_grid_size: int,
    position_restarts: int,
    joint_refine_every: int,
    insertion_candidates: int,
) -> None:
    if parameters.ndim != 1 or points.ndim != 2:
        raise ValueError("parameters/points must have shape [M] and [M,D]")
    if parameters.shape[0] != points.shape[0]:
        raise ValueError("parameters and points must share the reference point count")
    if points.shape[0] < degree + 1:
        raise ValueError("reference curve has too few points for the spline degree")
    if initial_knots.ndim != 1:
        raise ValueError("initial_knots must have shape [K]")
    if candidate_knots is not None and candidate_knots.ndim != 1:
        raise ValueError("candidate_knots must have shape [Kc]")
    tensors = [parameters, points, initial_knots]
    if candidate_knots is not None:
        tensors.append(candidate_knots)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all curve tensors must be floating-point")
    if not all(tensor.device == points.device for tensor in tensors):
        raise ValueError("all curve tensors must share a device")
    if not all(tensor.dtype == points.dtype for tensor in tensors):
        raise ValueError("all curve tensors must share a dtype")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("all curve tensors must be finite")
    if torch.any(parameters[1:] < parameters[:-1]):
        raise ValueError("reference parameters must be non-decreasing")
    tolerance = 32.0 * torch.finfo(parameters.dtype).eps
    if abs(float(parameters[0])) > tolerance or abs(float(parameters[-1]) - 1.0) > tolerance:
        raise ValueError("reference parameters must span [0,1]")
    if not math.isfinite(target_mse) or target_mse < 0.0:
        raise ValueError("target_mse must be finite and non-negative")
    if isinstance(maximum_internal_knots, bool) or maximum_internal_knots < 0:
        raise ValueError("maximum_internal_knots must be a non-negative integer")
    if degree < 1:
        raise ValueError("degree must be positive")
    if not math.isfinite(min_gap) or min_gap < 0.0:
        raise ValueError("min_gap must be finite and non-negative")
    if position_sweeps < 0 or position_restarts < 1:
        raise ValueError("position sweeps/restarts must be non-negative/positive")
    if position_grid_size < 3 or position_grid_size % 2 == 0:
        raise ValueError("position_grid_size must be an odd integer of at least 3")
    if joint_refine_every < 1 or insertion_candidates < 1:
        raise ValueError("joint_refine_every and insertion_candidates must be positive")


def _sanitize_knots(
    knots: torch.Tensor,
    *,
    min_gap: float,
    maximum_count: int | None = None,
) -> torch.Tensor:
    if knots.numel() == 0 or maximum_count == 0:
        return knots.new_empty(0)
    machine_gap = 32.0 * torch.finfo(knots.dtype).eps
    required_gap = max(min_gap, machine_gap)
    ordered = torch.sort(knots[(knots > required_gap) & (knots < 1.0 - required_gap)]).values
    retained: list[torch.Tensor] = []
    previous = -math.inf
    for value in ordered:
        scalar = float(value)
        if scalar - previous >= required_gap:
            retained.append(value)
            previous = scalar
    result = torch.stack(retained) if retained else knots.new_empty(0)
    if maximum_count is not None and result.numel() > maximum_count:
        indices = torch.linspace(
            0,
            result.numel() - 1,
            maximum_count,
            device=result.device,
            dtype=result.dtype,
        ).round().to(torch.long)
        result = result[indices]
    return result


def _fit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    *,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
) -> BSplineLeastSquaresFit:
    return refit_bspline_control_points(
        parameters,
        points,
        knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=True,
    )


def _certificate_state(
    fit: BSplineLeastSquaresFit,
    points: torch.Tensor,
    *,
    target_mse: float,
    constructive_exact: bool = False,
) -> tuple[bool, bool, bool | None, int, float]:
    measured = (fit.reconstructed_points - points).square().sum(dim=-1).mean()
    expected_rank = int(fit.internal_knots.numel()) + fit.degree + 1
    full_rank = None if constructive_exact else fit.solver_rank == expected_rank
    endpoint_max = float(
        torch.stack(
            [
                (fit.reconstructed_points[0] - points[0]).norm(),
                (fit.reconstructed_points[-1] - points[-1]).norm(),
            ]
        ).max()
    )
    threshold_satisfied = bool(torch.isfinite(measured)) and float(measured) <= target_mse
    endpoint_tolerance = max(1e-10, math.sqrt(max(target_mse, 0.0)) * 1e-6)
    certificate_valid = (
        threshold_satisfied
        and (constructive_exact or full_rank is True)
        and endpoint_max <= endpoint_tolerance
    )
    return (
        threshold_satisfied,
        certificate_valid,
        full_rank,
        expected_rank,
        endpoint_max,
    )


def _averaged_interpolation_knots(
    parameters: torch.Tensor,
    degree: int,
) -> torch.Tensor | None:
    """Return the standard averaging-method knot vector for square interpolation."""

    if torch.any(parameters[1:] <= parameters[:-1]):
        return None
    knot_count = int(parameters.numel()) - degree - 1
    if knot_count <= 0:
        return parameters.new_empty(0)
    return torch.stack(
        [parameters[index : index + degree].mean() for index in range(1, knot_count + 1)]
    )


def _maximum_knot_multiplicity(knots: torch.Tensor) -> int:
    if knots.numel() == 0:
        return 0
    _, counts = torch.unique_consecutive(torch.sort(knots).values, return_counts=True)
    return int(counts.max())


def exact_polyline_bspline_fit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    *,
    degree: int = 3,
    evaluation_parameters: torch.Tensor | None = None,
    evaluation_points: torch.Tensor | None = None,
) -> BSplineLeastSquaresFit:
    """Represent every supplied polyline segment exactly as a Bezier span.

    Every interior chord parameter has multiplicity ``degree`` (C0 joining).
    Segment controls lie at equal fractions of its straight edge, hence the
    resulting degree-``p`` Bezier span is exactly linear.  This represents the
    supplied polyline, including its corners; it says nothing about an unknown
    physical curve between observations.
    """

    if (evaluation_parameters is None) != (evaluation_points is None):
        raise ValueError(
            "evaluation_parameters and evaluation_points must be provided together"
        )
    if parameters.ndim != 1 or points.ndim != 2:
        raise ValueError("parameters/points must have shape [M] and [M,D]")
    if parameters.shape[0] != points.shape[0]:
        raise ValueError("parameters and points must share the point count")
    if points.shape[0] < 2 or degree < 1:
        raise ValueError("polyline fitting needs at least two points and degree >= 1")
    if parameters.device != points.device or parameters.dtype != points.dtype:
        raise ValueError("parameters and points must share device and dtype")
    if torch.any(parameters[1:] <= parameters[:-1]):
        raise ValueError("exact polyline fallback requires strictly increasing parameters")
    tolerance = 32.0 * torch.finfo(parameters.dtype).eps
    if abs(float(parameters[0])) > tolerance or abs(float(parameters[-1]) - 1.0) > tolerance:
        raise ValueError("polyline parameters must span [0,1]")

    if evaluation_parameters is None:
        evaluated_parameters = parameters
        evaluated_points = points
    else:
        assert evaluation_points is not None
        evaluated_parameters = evaluation_parameters
        evaluated_points = evaluation_points
        if evaluated_parameters.ndim != 1 or evaluated_points.ndim != 2:
            raise ValueError(
                "evaluation parameters/points must have shape [Me] and [Me,D]"
            )
        if evaluated_parameters.shape[0] != evaluated_points.shape[0]:
            raise ValueError("evaluation parameters and points must share a point count")
        if evaluated_points.shape[1] != points.shape[1]:
            raise ValueError("construction and evaluation point dimensions must match")
        if (
            evaluated_parameters.device != points.device
            or evaluated_points.device != points.device
            or evaluated_parameters.dtype != points.dtype
            or evaluated_points.dtype != points.dtype
        ):
            raise ValueError("construction and evaluation tensors must share device/dtype")
        if not bool(
            torch.isfinite(evaluated_parameters).all()
            and torch.isfinite(evaluated_points).all()
        ):
            raise ValueError("evaluation parameters and points must be finite")
        if torch.any(evaluated_parameters[1:] < evaluated_parameters[:-1]):
            raise ValueError("evaluation parameters must be non-decreasing")
        if torch.any((evaluated_parameters < 0.0) | (evaluated_parameters > 1.0)):
            raise ValueError("evaluation parameters must lie in [0,1]")

    internal_knots = parameters[1:-1].repeat_interleave(degree)
    knot_vector = build_open_knot_vector(internal_knots, degree)
    controls: list[torch.Tensor] = [points[0]]
    for segment_index in range(points.shape[0] - 1):
        start = points[segment_index]
        delta = points[segment_index + 1] - start
        controls.extend(
            start + (step / degree) * delta for step in range(1, degree + 1)
        )
    control_points = torch.stack(controls)
    basis = bspline_basis_matrix(
        evaluated_parameters,
        knot_vector,
        degree,
        num_control_points=control_points.shape[0],
    )
    reconstructed = basis @ control_points
    statistics = point_fit_statistics(
        reconstructed.unsqueeze(0),
        evaluated_points.unsqueeze(0),
    )
    difference = second_difference_matrix(
        control_points.shape[0],
        device=points.device,
        dtype=points.dtype,
    )
    data_squared_error = (reconstructed - evaluated_points).square().sum()
    smoothness_squared = (difference @ control_points).square().sum()
    control_squared = control_points.square().sum()
    return BSplineLeastSquaresFit(
        degree=degree,
        internal_knots=internal_knots.detach().clone(),
        knot_vector=knot_vector.detach().clone(),
        basis_matrix=basis.detach().clone(),
        control_points=control_points.detach().clone(),
        reconstructed_points=reconstructed.detach().clone(),
        fit_mse=statistics["fit_mse"][0].detach().clone(),
        fit_rmse=statistics["fit_rmse"][0].detach().clone(),
        coordinate_mse=statistics["coordinate_mse"][0].detach().clone(),
        coordinate_rmse=statistics["coordinate_rmse"][0].detach().clone(),
        data_squared_error=data_squared_error.detach().clone(),
        smoothness_squared=smoothness_squared.detach().clone(),
        control_squared=control_squared.detach().clone(),
        augmented_objective=data_squared_error.detach().clone(),
        solver_rank=None,
    )


def _is_legal_insertion(value: torch.Tensor, knots: torch.Tensor, min_gap: float) -> bool:
    machine_gap = 32.0 * torch.finfo(knots.dtype).eps
    required_gap = max(min_gap, machine_gap)
    scalar = float(value)
    if not required_gap <= scalar <= 1.0 - required_gap:
        return False
    if knots.numel() and bool(torch.any(torch.abs(knots - value) < required_gap)):
        return False
    return True


def _build_insertion_pool(
    parameters: torch.Tensor,
    points: torch.Tensor,
    fit: BSplineLeastSquaresFit,
    candidate_knots: torch.Tensor,
    *,
    min_gap: float,
    maximum_candidates: int,
) -> list[torch.Tensor]:
    knots = fit.internal_knots
    residual_sq = (fit.reconstructed_points - points).square().sum(dim=-1)
    order = torch.argsort(residual_sq, descending=True, stable=True)
    raw: list[torch.Tensor] = []
    # Residual peaks and their adjacent parameter midpoints focus added
    # degrees of freedom near measured geometric failures.
    peak_count = max(1, maximum_candidates // 4)
    peak_indices = order[:peak_count]
    for index_tensor in peak_indices:
        index = int(index_tensor)
        raw.append(parameters[index])
    # Unselected network proposals remain useful geometric hints and are
    # evaluated jointly with error-driven insertions rather than discarded.
    raw.extend(candidate_knots.unbind())
    for index_tensor in peak_indices:
        index = int(index_tensor)
        if index > 0:
            raw.append(0.5 * (parameters[index - 1] + parameters[index]))
        if index + 1 < parameters.numel():
            raw.append(0.5 * (parameters[index] + parameters[index + 1]))
    # Gap midpoints prevent a concentrated network proposal from leaving an
    # entire parameter interval unreachable.
    boundaries = torch.cat([parameters.new_zeros(1), knots, parameters.new_ones(1)])
    gap_order = torch.argsort(boundaries[1:] - boundaries[:-1], descending=True)
    raw.extend(
        0.5 * (boundaries[index] + boundaries[index + 1])
        for index in gap_order[:maximum_candidates].tolist()
    )

    accepted: list[torch.Tensor] = []
    deduplication_tolerance = max(min_gap * 0.5, 64.0 * torch.finfo(parameters.dtype).eps)
    for value in raw:
        if not _is_legal_insertion(value, knots, min_gap):
            continue
        if any(abs(float(value - previous)) < deduplication_tolerance for previous in accepted):
            continue
        accepted.append(value.detach().clone())
        if len(accepted) >= maximum_candidates:
            break
    return accepted


@torch.no_grad()
def certify_reference_mse(
    parameters: torch.Tensor,
    points: torch.Tensor,
    initial_knots: torch.Tensor,
    *,
    candidate_knots: torch.Tensor | None = None,
    target_mse: float = 1e-5,
    maximum_internal_knots: int = 64,
    degree: int = 3,
    smoothness_weight: float = 0.0,
    control_ridge: float = 0.0,
    min_gap: float = 1e-5,
    position_sweeps: int = 1,
    position_grid_size: int = 5,
    position_restarts: int = 2,
    joint_refine_every: int = 4,
    insertion_candidates: int = 24,
    interpolation_fallback: bool = True,
    tolerance_polyline_fallback: bool = True,
    tolerance_polyline_compact: bool = True,
    polyline_exact_fallback: bool = True,
) -> ReferenceCertifiedFitResult:
    """Repair a spline until the supplied normalized reference MSE is observed.

    ``maximum_internal_knots=0`` means the rank-safe interpolation limit
    ``M-degree-1``.  The routine uses CPU-friendly float64 exact refits,
    relocation and adaptive insertion, but certifies only the finite reference
    samples supplied by the caller.  Failure to meet the target is returned as
    an explicit status rather than hidden behind a nominal guarantee.
    """

    _validate_inputs(
        parameters,
        points,
        initial_knots,
        candidate_knots,
        target_mse=target_mse,
        maximum_internal_knots=maximum_internal_knots,
        degree=degree,
        min_gap=min_gap,
        position_sweeps=position_sweeps,
        position_grid_size=position_grid_size,
        position_restarts=position_restarts,
        joint_refine_every=joint_refine_every,
        insertion_candidates=insertion_candidates,
    )
    if smoothness_weight < 0.0 or control_ridge < 0.0:
        raise ValueError("regularization weights must be non-negative")

    rank_safe_maximum = max(0, int(points.shape[0]) - degree - 1)
    effective_maximum = (
        rank_safe_maximum
        if maximum_internal_knots == 0
        else min(maximum_internal_knots, rank_safe_maximum)
    )
    supplied_initial_count = int(initial_knots.numel())
    current_knots = _sanitize_knots(
        initial_knots,
        min_gap=min_gap,
        maximum_count=effective_maximum,
    )
    sanitized_initial_count = int(current_knots.numel())
    all_candidates = _sanitize_knots(
        candidate_knots if candidate_knots is not None else initial_knots,
        min_gap=min_gap,
    )

    refit_count = 1
    insertion_evaluations = 0
    position_refits = 0
    position_refinement_used = False
    interpolation_attempted = False
    interpolation_used = False
    tolerance_polyline_attempted = False
    tolerance_polyline_used = False
    tolerance_polyline_vertex_count: int | None = None
    tolerance_polyline_measured_mse: float | None = None
    tolerance_polyline_insertions = 0
    tolerance_polyline_compaction_removed = 0
    polyline_attempted = False
    polyline_used = False
    additions = 0
    current_fit = _fit(
        parameters,
        points,
        current_knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
    )

    def state(
        fit: BSplineLeastSquaresFit,
    ) -> tuple[bool, bool, bool | None, int, float]:
        return _certificate_state(
            fit,
            points,
            target_mse=target_mse,
            constructive_exact=(tolerance_polyline_used or polyline_used)
            and fit is current_fit,
        )

    def result(status: str, failure_reason: str | None = None) -> ReferenceCertifiedFitResult:
        satisfied, valid, full_rank, expected_rank, endpoint_max = state(current_fit)
        if satisfied and not valid and failure_reason is None:
            failure_reason = "target_mse_met_but_numerical_certificate_is_not_full_rank"
        return ReferenceCertifiedFitResult(
            target_mse=float(target_mse),
            initial_fit=initial_fit,
            final_fit=current_fit,
            status=status,
            threshold_satisfied=satisfied,
            certificate_valid=valid,
            full_column_rank=full_rank,
            expected_solver_rank=expected_rank,
            maximum_internal_knots=effective_maximum,
            rank_safe_maximum_internal_knots=rank_safe_maximum,
            supplied_initial_count=supplied_initial_count,
            sanitized_initial_count=sanitized_initial_count,
            added_knot_count=additions,
            position_refinement_used=position_refinement_used,
            interpolation_fallback_attempted=interpolation_attempted,
            interpolation_fallback_used=interpolation_used,
            tolerance_polyline_fallback_attempted=tolerance_polyline_attempted,
            tolerance_polyline_fallback_used=tolerance_polyline_used,
            tolerance_polyline_vertex_count=tolerance_polyline_vertex_count,
            tolerance_polyline_measured_mse=tolerance_polyline_measured_mse,
            tolerance_polyline_insertion_count=tolerance_polyline_insertions,
            tolerance_polyline_compaction_removed_count=(
                tolerance_polyline_compaction_removed
            ),
            tolerance_polyline_fallback_exceeded_adaptive_maximum=(
                tolerance_polyline_used
                and int(current_fit.internal_knots.numel()) > effective_maximum
            ),
            polyline_exact_fallback_attempted=polyline_attempted,
            polyline_exact_fallback_used=polyline_used,
            polyline_fallback_exceeded_adaptive_maximum=(
                polyline_used
                and int(current_fit.internal_knots.numel()) > effective_maximum
            ),
            final_control_point_count=int(current_fit.control_points.shape[0]),
            maximum_internal_knot_multiplicity=_maximum_knot_multiplicity(
                current_fit.internal_knots
            ),
            refit_count=refit_count,
            insertion_candidate_evaluation_count=insertion_evaluations,
            position_refit_count=position_refits,
            endpoint_max_distance=endpoint_max,
            failure_reason=failure_reason,
        )

    initial_fit = current_fit
    satisfied, valid, *_ = state(current_fit)
    if satisfied and valid:
        return result("initial_reference_refit_pass")

    if current_knots.numel() and position_sweeps:
        refined = refine_knot_positions(
            parameters,
            points,
            current_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            min_gap=min_gap,
            sweeps=position_sweeps,
            grid_size=position_grid_size,
            restarts=position_restarts,
        )
        refit_count += refined.refit_count
        position_refits += refined.refit_count
        position_refinement_used = True
        if refined.final_mse < float(current_fit.fit_mse):
            current_fit = refined.final_fit
            current_knots = current_fit.internal_knots
        satisfied, valid, *_ = state(current_fit)
        if satisfied and valid:
            return result("joint_position_refinement_pass")

    interpolation_knots = _averaged_interpolation_knots(parameters, degree)
    if (
        interpolation_fallback
        and interpolation_knots is not None
        and interpolation_knots.numel() <= effective_maximum
        and (
            interpolation_knots.numel() == 0
            or _sanitize_knots(interpolation_knots, min_gap=min_gap).numel()
            == interpolation_knots.numel()
        )
    ):
        interpolation_attempted = True
        interpolation_fit = _fit(
            parameters,
            points,
            interpolation_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
        )
        refit_count += 1
        if float(interpolation_fit.fit_mse) < float(current_fit.fit_mse):
            current_fit = interpolation_fit
            current_knots = interpolation_knots
        satisfied, valid, *_ = state(current_fit)
        if satisfied and valid:
            additions = max(0, int(current_knots.numel()) - sanitized_initial_count)
            interpolation_used = True
            return result("interpolation_fallback_pass")

    additions_since_refinement = 0
    stalled = False
    while int(current_knots.numel()) < effective_maximum:
        pool = _build_insertion_pool(
            parameters,
            points,
            current_fit,
            all_candidates,
            min_gap=min_gap,
            maximum_candidates=insertion_candidates,
        )
        if not pool:
            stalled = True
            break
        best_fit: BSplineLeastSquaresFit | None = None
        best_knots: torch.Tensor | None = None
        for candidate in pool:
            trial_knots = torch.sort(
                torch.cat([current_knots, candidate.reshape(1)])
            ).values
            trial_fit = _fit(
                parameters,
                points,
                trial_knots,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
            )
            refit_count += 1
            insertion_evaluations += 1
            if best_fit is None or float(trial_fit.fit_mse) < float(best_fit.fit_mse):
                best_fit = trial_fit
                best_knots = trial_knots
        assert best_fit is not None and best_knots is not None
        current_fit = best_fit
        current_knots = best_knots
        additions += 1
        additions_since_refinement += 1
        satisfied, valid, *_ = state(current_fit)
        if satisfied and valid:
            return result("adaptive_insertion_pass")

        should_refine = (
            position_sweeps > 0
            and (
                additions_since_refinement >= joint_refine_every
                or int(current_knots.numel()) == effective_maximum
            )
        )
        if should_refine:
            refined = refine_knot_positions(
                parameters,
                points,
                current_knots,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                min_gap=min_gap,
                sweeps=position_sweeps,
                grid_size=position_grid_size,
                restarts=position_restarts,
            )
            refit_count += refined.refit_count
            position_refits += refined.refit_count
            position_refinement_used = True
            additions_since_refinement = 0
            if refined.final_mse < float(current_fit.fit_mse):
                current_fit = refined.final_fit
                current_knots = current_fit.internal_knots
            satisfied, valid, *_ = state(current_fit)
            if satisfied and valid:
                return result("adaptive_insertion_and_relocation_pass")

    satisfied, valid, *_ = state(current_fit)
    if tolerance_polyline_fallback:
        tolerance_polyline_attempted = True
        try:
            simplification = simplify_polyline_to_mse(
                parameters,
                points,
                target_mse,
                compact=tolerance_polyline_compact,
            )
            retained = simplification.retained_indices
            tolerance_polyline_vertex_count = simplification.retained_vertex_count
            tolerance_polyline_measured_mse = simplification.measured_mse
            tolerance_polyline_insertions = simplification.insertion_count
            tolerance_polyline_compaction_removed = (
                simplification.compaction_removed_count
            )
            tolerance_fit = exact_polyline_bspline_fit(
                parameters[retained],
                points[retained],
                degree=degree,
                evaluation_parameters=parameters,
                evaluation_points=points,
            )
        except ValueError:
            tolerance_fit = None
        if tolerance_fit is not None:
            tolerance_satisfied, tolerance_valid, *_ = _certificate_state(
                tolerance_fit,
                points,
                target_mse=target_mse,
                constructive_exact=True,
            )
            if tolerance_satisfied and tolerance_valid:
                current_fit = tolerance_fit
                current_knots = current_fit.internal_knots
                tolerance_polyline_used = True
                additions = max(
                    0,
                    int(current_knots.numel()) - sanitized_initial_count,
                )
                return result("tolerance_polyline_fallback_pass")
            if float(tolerance_fit.fit_mse) < float(current_fit.fit_mse):
                current_fit = tolerance_fit
                current_knots = current_fit.internal_knots

    satisfied, valid, *_ = state(current_fit)
    if polyline_exact_fallback:
        polyline_attempted = True
        try:
            polyline_fit = exact_polyline_bspline_fit(
                parameters,
                points,
                degree=degree,
            )
        except ValueError:
            polyline_fit = None
        if polyline_fit is not None:
            if float(polyline_fit.fit_mse) <= float(current_fit.fit_mse):
                current_fit = polyline_fit
                current_knots = current_fit.internal_knots
            polyline_used = current_fit is polyline_fit
            additions = max(0, int(current_knots.numel()) - sanitized_initial_count)
            satisfied, valid, *_ = state(current_fit)
            if satisfied and valid:
                return result("polyline_exact_fallback_pass")
    if satisfied and not valid:
        return result("target_met_without_numerical_certificate")
    if stalled:
        return result(
            "not_reached_within_limits",
            "no_legal_insertion_candidate_remained",
        )
    return result(
        "not_reached_within_limits",
        "maximum_rank_safe_internal_knot_limit_reached",
    )


__all__ = [
    "PolylineSimplificationResult",
    "REFERENCE_CERTIFICATE_SCOPE",
    "ReferenceCertifiedFitResult",
    "certify_reference_mse",
    "exact_polyline_bspline_fit",
    "simplify_polyline_to_mse",
]
