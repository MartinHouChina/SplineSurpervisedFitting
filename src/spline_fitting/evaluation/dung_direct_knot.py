"""Auditable serial adaptation of Dung and Tjahjowidodo (PLOS ONE, 2017).

The paper first bisects an ordered data set into the longest consecutive
pieces that a single degree-``p`` B-spline can fit below a maximum Euclidean
error, then relocates every boundary by fitting a local two-piece B-spline.
This module implements that serial path with simple interior knots only.  It
does not implement the paper's parallel split/join/shift procedure or its
multiple-knot continuity classifier; those limits are recorded in every
result rather than hidden behind the paper citation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .gradient_knot_pruning import chord_length_parameters


_ADAPTATION_LIMITS = (
    "serial bisection only; parallel split/join/shift is not implemented",
    "simple interior knots only; multiplicity and continuity classification are omitted",
    "default native max-error epsilon is sqrt(mse_tolerance), which is a unit conversion rather than an equivalent constraint",
    "capacity overflow is handled by deterministic uniform coarse-break subsampling before relocation",
    "guarded float64 least squares and finite-difference Gauss-Newton replace MATLAB numerical edge behavior",
)


@dataclass(frozen=True)
class DungDirectKnotResult:
    """Fitted curve and diagnostics for the disclosed serial adaptation."""

    degree: int
    mse_tolerance: float
    native_max_error_tolerance: float
    max_internal_knots: int
    scan_intervals: int
    optimization_iterations: int
    parameters: torch.Tensor
    segments: tuple[tuple[int, int], ...]
    segment_max_errors: tuple[float, ...]
    coarse_break_indices: tuple[int, ...]
    coarse_breaks: torch.Tensor
    proposed_internal_knot_count: int
    knots: torch.Tensor
    native_fit: BSplineLeastSquaresFit
    native_max_error: float
    final_fit: BSplineLeastSquaresFit
    final_max_error: float
    threshold_satisfied: bool
    scan_evaluations: int
    optimization_evaluations: int
    single_piece_evaluations: int
    capacity_exceeded: bool
    capacity_handling: str
    adaptation_limits: tuple[str, ...]
    diagnostics: dict[str, object]
    elapsed_seconds: float


def _validate_inputs(
    points: torch.Tensor,
    *,
    degree: int,
    mse_tolerance: float,
    max_internal_knots: int,
    max_error: float | None,
    scan_intervals: int,
    optimization_iterations: int,
) -> float:
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if (
        isinstance(max_internal_knots, bool)
        or not isinstance(max_internal_knots, int)
        or max_internal_knots < 0
    ):
        raise ValueError("max_internal_knots must be a non-negative integer")
    if (
        isinstance(scan_intervals, bool)
        or not isinstance(scan_intervals, int)
        or scan_intervals < 1
    ):
        raise ValueError("scan_intervals must be a positive integer")
    if (
        isinstance(optimization_iterations, bool)
        or not isinstance(optimization_iterations, int)
        or optimization_iterations < 0
    ):
        raise ValueError("optimization_iterations must be a non-negative integer")
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a CPU float64 torch.Tensor")
    if points.device.type != "cpu" or points.dtype != torch.float64:
        raise ValueError("points must be a CPU float64 torch.Tensor")
    if points.ndim != 2 or points.shape[0] < degree + 1 or points.shape[1] < 1:
        raise ValueError("points must have shape [M,D] with M >= degree + 1")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")
    epsilon = math.sqrt(mse_tolerance) if max_error is None else max_error
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("max_error must be finite and non-negative")
    return float(epsilon)


def _normalized_interval_parameters(parameters: torch.Tensor) -> torch.Tensor:
    width = parameters[-1] - parameters[0]
    if float(width) <= torch.finfo(parameters.dtype).eps:
        return torch.linspace(
            0.0,
            1.0,
            parameters.numel(),
            dtype=parameters.dtype,
            device=parameters.device,
        )
    result = (parameters - parameters[0]) / width
    result[0], result[-1] = 0.0, 1.0
    return result


def _native_refit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    *,
    degree: int,
) -> tuple[BSplineLeastSquaresFit, torch.Tensor, float]:
    fit = refit_bspline_control_points(
        parameters,
        points,
        knots,
        degree=degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=False,
    )
    residuals = (fit.reconstructed_points - points).norm(dim=-1)
    maximum = float(residuals.max()) if residuals.numel() else 0.0
    return fit, residuals, maximum


def _fit_single_piece(
    parameters: torch.Tensor,
    points: torch.Tensor,
    start: int,
    stop: int,
    *,
    degree: int,
) -> float:
    local_parameters = _normalized_interval_parameters(parameters[start : stop + 1])
    _, _, maximum = _native_refit(
        local_parameters,
        points[start : stop + 1],
        local_parameters.new_empty(0),
        degree=degree,
    )
    return maximum


def _serial_bisection(
    parameters: torch.Tensor,
    points: torch.Tensor,
    *,
    degree: int,
    epsilon: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...], int]:
    """Find longest feasible consecutive polynomial pieces from left to right."""

    sample_count = int(points.shape[0])
    minimum_samples = degree + 1
    segments: list[tuple[int, int]] = []
    errors: list[float] = []
    evaluations = 0
    start = 0

    while start < sample_count:
        remaining = sample_count - start
        if remaining <= minimum_samples:
            # A tiny terminal tail cannot define a new full-rank degree-p
            # piece.  Attach it to the previous piece (or use the only piece)
            # and record the actually measured native error.
            if segments:
                previous_start, _ = segments[-1]
                maximum = _fit_single_piece(
                    parameters,
                    points,
                    previous_start,
                    sample_count - 1,
                    degree=degree,
                )
                evaluations += 1
                segments[-1] = (previous_start, sample_count - 1)
                errors[-1] = maximum
            else:
                maximum = _fit_single_piece(
                    parameters, points, 0, sample_count - 1, degree=degree
                )
                evaluations += 1
                segments.append((0, sample_count - 1))
                errors.append(maximum)
            break

        full_error = _fit_single_piece(
            parameters, points, start, sample_count - 1, degree=degree
        )
        evaluations += 1
        if full_error <= epsilon:
            segments.append((start, sample_count - 1))
            errors.append(full_error)
            break

        lowest = start + degree
        # Leave enough observations to define one final degree-p piece.
        highest = sample_count - minimum_samples - 1
        if highest < lowest:
            segments.append((start, sample_count - 1))
            errors.append(full_error)
            break

        low_error = _fit_single_piece(
            parameters, points, start, lowest, degree=degree
        )
        evaluations += 1
        feasible = lowest
        feasible_error = low_error
        if highest > lowest:
            high_error = _fit_single_piece(
                parameters, points, start, highest, degree=degree
            )
            evaluations += 1
            if high_error <= epsilon:
                feasible, feasible_error = highest, high_error
            else:
                failing = highest
                while failing - feasible > 1:
                    middle = (feasible + failing) // 2
                    middle_error = _fit_single_piece(
                        parameters, points, start, middle, degree=degree
                    )
                    evaluations += 1
                    if middle_error <= epsilon:
                        feasible, feasible_error = middle, middle_error
                    else:
                        failing = middle

        segments.append((start, feasible))
        errors.append(feasible_error)
        start = feasible + 1

    return tuple(segments), tuple(errors), evaluations


def _uniform_capacity_indices(count: int, capacity: int) -> tuple[int, ...]:
    if capacity >= count:
        return tuple(range(count))
    if capacity == 0:
        return ()
    # Midpoints of ``capacity`` equal index bins give deterministic coverage
    # without privileging the first or last portion of the curve.
    return tuple(((2 * index + 1) * count) // (2 * capacity) for index in range(capacity))


def _local_search_range(
    parameters: torch.Tensor,
    left_segment: tuple[int, int],
    right_segment: tuple[int, int],
    *,
    degree: int,
) -> tuple[float, float, int, int]:
    a, b = left_segment
    c, d = right_segment
    left_extra = max(0, min(b - a - degree, 3) + degree - 1)
    right_extra = max(0, min(d - c - degree, 3) + degree - 1)
    lower_index = max(a, b - left_extra)
    upper_index = min(d, c + right_extra)
    pair_start, pair_stop = a, d
    pair_width = parameters[pair_stop] - parameters[pair_start]
    if float(pair_width) <= torch.finfo(parameters.dtype).eps:
        return 0.25, 0.75, pair_start, pair_stop
    lower = float((parameters[lower_index] - parameters[pair_start]) / pair_width)
    upper = float((parameters[upper_index] - parameters[pair_start]) / pair_width)
    interior = max(float(torch.finfo(parameters.dtype).eps) * 64.0, 1e-12)
    lower = min(max(lower, interior), 1.0 - interior)
    upper = min(max(upper, interior), 1.0 - interior)
    if upper <= lower:
        boundary = float((parameters[b] - parameters[pair_start]) / pair_width)
        radius = max(interior * 8.0, 1e-6)
        lower = max(interior, boundary - radius)
        upper = min(1.0 - interior, boundary + radius)
    return lower, upper, pair_start, pair_stop


def _relocate_simple_boundary(
    parameters: torch.Tensor,
    points: torch.Tensor,
    left_segment: tuple[int, int],
    right_segment: tuple[int, int],
    *,
    degree: int,
    scan_intervals: int,
    optimization_iterations: int,
) -> tuple[float, int, int, bool]:
    lower, upper, pair_start, pair_stop = _local_search_range(
        parameters, left_segment, right_segment, degree=degree
    )
    pair_parameters = _normalized_interval_parameters(
        parameters[pair_start : pair_stop + 1]
    )
    pair_points = points[pair_start : pair_stop + 1]
    cache: dict[float, tuple[torch.Tensor, float, float]] = {}
    scan_evaluations = 0
    optimization_evaluations = 0
    fallback_used = False

    def evaluate(location: float, *, scan: bool) -> tuple[torch.Tensor, float, float]:
        nonlocal scan_evaluations, optimization_evaluations
        location = min(max(float(location), lower), upper)
        if location not in cache:
            _, residuals, maximum = _native_refit(
                pair_parameters,
                pair_points,
                pair_parameters.new_tensor([location]),
                degree=degree,
            )
            squared_objective = float(torch.dot(residuals, residuals))
            cache[location] = (residuals, maximum, squared_objective)
            if scan:
                scan_evaluations += 1
            else:
                optimization_evaluations += 1
        return cache[location]

    scan_locations = torch.linspace(
        lower,
        upper,
        scan_intervals + 1,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    for location in scan_locations.tolist():
        evaluate(location, scan=True)
    current = min(
        cache,
        key=lambda location: (cache[location][1], cache[location][2], location),
    )
    previous_step: float | None = None
    stable_steps = 0
    finite_difference = math.sqrt(torch.finfo(parameters.dtype).eps)

    for _ in range(optimization_iterations):
        residuals, _, _ = evaluate(current, scan=False)
        shifted = min(current + finite_difference, upper)
        if shifted == current:
            shifted = max(current - finite_difference, lower)
        actual_step = shifted - current
        if actual_step == 0.0:
            fallback_used = True
            break
        shifted_residuals, _, _ = evaluate(shifted, scan=False)
        jacobian = (shifted_residuals - residuals) / actual_step
        denominator = float(torch.dot(jacobian, jacobian))
        numerator = float(torch.dot(jacobian, residuals))
        if (
            not math.isfinite(denominator)
            or not math.isfinite(numerator)
            or denominator <= torch.finfo(parameters.dtype).eps
        ):
            fallback_used = True
            break
        step = numerator / denominator
        if previous_step is not None and previous_step * step < 0.0:
            step *= 0.5
        updated = min(max(current - step, lower), upper)
        evaluate(updated, scan=False)
        if abs(updated - current) < 1e-12:
            stable_steps += 1
        else:
            stable_steps = 0
        current = updated
        previous_step = step
        if stable_steps >= 2:
            break

    best = min(
        cache,
        key=lambda location: (cache[location][1], cache[location][2], location),
    )
    pair_width = parameters[pair_stop] - parameters[pair_start]
    if float(pair_width) <= torch.finfo(parameters.dtype).eps:
        global_knot = 0.5 * (
            float(parameters[left_segment[1]])
            + float(parameters[right_segment[0]])
        )
    else:
        global_knot = float(parameters[pair_start] + best * pair_width)
    interior = max(float(torch.finfo(parameters.dtype).eps) * 64.0, 1e-12)
    global_knot = min(max(global_knot, interior), 1.0 - interior)
    return global_knot, scan_evaluations, optimization_evaluations, fallback_used


@torch.no_grad()
def fit_dung_direct_knots(
    points: torch.Tensor,
    *,
    degree: int = 3,
    mse_tolerance: float,
    max_internal_knots: int,
    max_error: float | None = None,
    scan_intervals: int = 10,
    optimization_iterations: int = 10,
) -> DungDirectKnotResult:
    """Fit one normalized ordered curve with the disclosed serial adaptation.

    ``mse_tolerance`` controls only the common reported MSE.  The native Dung
    segmentation criterion is ``max_i ||Q_i-S(t_i)||_2 <= epsilon``; when
    ``max_error`` is omitted, ``epsilon=sqrt(mse_tolerance)`` puts both values
    in compatible length units but does not make the two constraints
    mathematically equivalent.
    """

    epsilon = _validate_inputs(
        points,
        degree=degree,
        mse_tolerance=mse_tolerance,
        max_internal_knots=max_internal_knots,
        max_error=max_error,
        scan_intervals=scan_intervals,
        optimization_iterations=optimization_iterations,
    )
    started = time.perf_counter()
    observed = points.detach()
    parameters = chord_length_parameters(observed)
    segments, segment_errors, single_piece_evaluations = _serial_bisection(
        parameters,
        observed,
        degree=degree,
        epsilon=epsilon,
    )
    coarse_break_indices = tuple(segment[1] for segment in segments[:-1])
    if coarse_break_indices:
        coarse_breaks = torch.stack(
            [
                0.5 * (parameters[index] + parameters[index + 1])
                for index in coarse_break_indices
            ]
        )
    else:
        coarse_breaks = parameters.new_empty(0)

    proposed_count = len(coarse_break_indices)
    capacity_exceeded = proposed_count > max_internal_knots
    retained_breaks = _uniform_capacity_indices(proposed_count, max_internal_knots)
    capacity_handling = (
        "uniform_coarse_break_subsample"
        if capacity_exceeded
        else "none"
    )
    relocated: list[float] = []
    scan_evaluations = 0
    optimization_evaluations = 0
    fallback_count = 0
    for break_index in retained_breaks:
        knot, scans, optimizations, fallback = _relocate_simple_boundary(
            parameters,
            observed,
            segments[break_index],
            segments[break_index + 1],
            degree=degree,
            scan_intervals=scan_intervals,
            optimization_iterations=optimization_iterations,
        )
        relocated.append(knot)
        scan_evaluations += scans
        optimization_evaluations += optimizations
        fallback_count += int(fallback)

    knots = parameters.new_tensor(sorted(relocated))
    native_fit, _, native_maximum = _native_refit(
        parameters, observed, knots, degree=degree
    )
    final_fit = refit_bspline_control_points(
        parameters,
        observed,
        knots,
        degree=degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )
    final_residuals = (final_fit.reconstructed_points - observed).norm(dim=-1)
    final_maximum = float(final_residuals.max()) if final_residuals.numel() else 0.0
    diagnostics: dict[str, object] = {
        "paper": "Dung and Tjahjowidodo, PLOS ONE 2017, doi:10.1371/journal.pone.0173857",
        "variant": "serial_bisection_simple_knot_adaptation",
        "parameterization": "chord_length",
        "native_control_norm": "maximum Euclidean point residual",
        "native_max_error_tolerance": epsilon,
        "native_final_max_error": native_maximum,
        "native_final_mse": float(native_fit.fit_mse),
        "common_final_max_error": final_maximum,
        "common_final_mse": float(final_fit.fit_mse),
        "segments": segments,
        "segment_max_errors": segment_errors,
        "coarse_break_indices": coarse_break_indices,
        "proposed_internal_knot_count": proposed_count,
        "retained_internal_knot_count": len(relocated),
        "scan_evaluations": scan_evaluations,
        "optimization_evaluations": optimization_evaluations,
        "single_piece_evaluations": single_piece_evaluations,
        "exact_refit_count": (
            single_piece_evaluations
            + scan_evaluations
            + optimization_evaluations
            + 2
        ),
        "gauss_newton_fallback_count": fallback_count,
        "capacity_exceeded": capacity_exceeded,
        "capacity_handling": capacity_handling,
        "reported_refit": "unregularized endpoint-constrained standard B-spline least squares",
        "adaptation_limits": _ADAPTATION_LIMITS,
    }
    return DungDirectKnotResult(
        degree=degree,
        mse_tolerance=float(mse_tolerance),
        native_max_error_tolerance=epsilon,
        max_internal_knots=max_internal_knots,
        scan_intervals=scan_intervals,
        optimization_iterations=optimization_iterations,
        parameters=parameters.detach().clone(),
        segments=segments,
        segment_max_errors=segment_errors,
        coarse_break_indices=coarse_break_indices,
        coarse_breaks=coarse_breaks.detach().clone(),
        proposed_internal_knot_count=proposed_count,
        knots=knots.detach().clone(),
        native_fit=native_fit,
        native_max_error=native_maximum,
        final_fit=final_fit,
        final_max_error=final_maximum,
        threshold_satisfied=float(final_fit.fit_mse) <= mse_tolerance,
        scan_evaluations=scan_evaluations,
        optimization_evaluations=optimization_evaluations,
        single_piece_evaluations=single_piece_evaluations,
        capacity_exceeded=capacity_exceeded,
        capacity_handling=capacity_handling,
        adaptation_limits=_ADAPTATION_LIMITS,
        diagnostics=diagnostics,
        elapsed_seconds=time.perf_counter() - started,
    )
