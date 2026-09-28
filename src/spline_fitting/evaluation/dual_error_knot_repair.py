"""Shared, additive-only error repair for already fitted B-spline curves.

This is an explicitly separate comparison wrapper, not part of any published
baseline. It measures errors on the supplied observations at unchanged
parameters. It neither uses reference labels nor removes, relocates, or replaces
existing knots. Meeting both tolerances is checked, never guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points


@dataclass(frozen=True)
class DualErrorKnotRepairResult:
    initial_fit: BSplineLeastSquaresFit
    final_fit: BSplineLeastSquaresFit
    before_mse: float
    after_mse: float
    before_max_squared_error: float
    after_max_squared_error: float
    before_pass: bool
    after_pass: bool
    initial_knot_count: int
    final_knot_count: int
    inserted_knots: tuple[float, ...]
    trace: tuple[dict[str, Any], ...]
    refit_count: int
    elapsed_repair_ms: float
    termination: str
    solver_options: dict[str, Any]


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _validate_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    fit: BSplineLeastSquaresFit,
    max_internal_knots: int,
) -> None:
    if parameters.ndim != 1 or points.ndim != 2:
        raise ValueError("parameters and points must have shapes [N] and [N, D]")
    if points.shape[0] != parameters.numel() or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError("at least two matching observations with D >= 1 are required")
    for name, tensor in (
        ("parameters", parameters), ("points", points),
        ("initial internal knots", fit.internal_knots),
        ("initial full knot vector", fit.knot_vector),
        ("initial control points", fit.control_points),
    ):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float64:
            raise ValueError(f"{name} must be a CPU float64 tensor")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
    if bool(torch.any(parameters[1:] < parameters[:-1])):
        raise ValueError("parameters must be non-decreasing")
    if float(parameters[0]) != 0.0 or float(parameters[-1]) != 1.0:
        raise ValueError("parameters must span the normalized interval [0, 1]")
    if not isinstance(max_internal_knots, int) or isinstance(max_internal_knots, bool) or max_internal_knots < 0:
        raise ValueError("max_internal_knots must be a non-negative integer")
    if not isinstance(fit.degree, int) or fit.degree < 1:
        raise ValueError("initial fit degree must be a positive integer")
    knots = fit.internal_knots
    if knots.ndim != 1 or bool(torch.any((knots <= 0) | (knots >= 1))):
        raise ValueError("initial internal knots must be one-dimensional and inside (0, 1)")
    if bool(torch.any(knots[1:] < knots[:-1])):
        raise ValueError("initial internal knots must be non-decreasing")
    if knots.numel() > max_internal_knots:
        raise ValueError("initial knot count exceeds max_internal_knots; repair never deletes knots")
    if knots.numel() and int(torch.unique_consecutive(knots, return_counts=True)[1].max()) > fit.degree + 1:
        raise ValueError("initial knot multiplicity exceeds degree + 1")
    expected_full = torch.cat((torch.zeros(fit.degree + 1, dtype=torch.float64), knots,
                               torch.ones(fit.degree + 1, dtype=torch.float64)))
    if not torch.equal(fit.knot_vector, expected_full):
        raise ValueError("initial fit must use an open-clamped full knot vector consistent with its internal knots")
    if fit.control_points.shape != (knots.numel() + fit.degree + 1, points.shape[1]):
        raise ValueError("initial control point shape is inconsistent with its knots and observations")


def _errors(fit: BSplineLeastSquaresFit, parameters: torch.Tensor, points: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    # Re-evaluate rather than trusting cached fit_mse or reconstructed_points.
    squared = (fit.evaluate(parameters) - points).square().sum(dim=1)
    if not bool(torch.isfinite(squared).all()):
        raise ValueError("fit evaluation produced non-finite squared errors")
    return float(squared.mean()), float(squared.max()), squared


def _candidate_locations(
    parameters: torch.Tensor,
    squared_errors: torch.Tensor,
    knots: torch.Tensor,
    limit: int,
    separation: float,
) -> list[float]:
    """Deterministic residual peaks and high-residual knot-span centers."""
    t = parameters.tolist()
    errors = squared_errors.tolist()
    old = knots.tolist()
    proposed: list[tuple[float, float]] = []
    peaks = sorted(range(len(t)), key=lambda i: (-errors[i], i))[: max(8, limit)]
    for index in peaks:
        proposed.append((errors[index], t[index]))
        if index > 0:
            proposed.append((0.9 * errors[index], (t[index - 1] + t[index]) * 0.5))
        if index + 1 < len(t):
            proposed.append((0.9 * errors[index], (t[index] + t[index + 1]) * 0.5))
    boundaries = [0.0, *old, 1.0]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left <= 2 * separation:
            continue
        local = [error for parameter, error in zip(t, errors) if left <= parameter <= right]
        if local:
            proposed.append((0.95 * max(local), 0.5 * (left + right)))
    candidates: list[float] = []
    for _, position in sorted(proposed, key=lambda item: (-item[0], item[1])):
        if position <= separation or position >= 1.0 - separation:
            continue
        if any(abs(position - existing) <= separation for existing in old):
            continue
        if any(abs(position - existing) <= separation for existing in candidates):
            continue
        candidates.append(position)
        if len(candidates) >= limit:
            break
    return candidates


@torch.no_grad()
def repair_knots_to_dual_tolerance(
    parameters: torch.Tensor,
    points: torch.Tensor,
    initial_fit: BSplineLeastSquaresFit,
    *,
    mse_tolerance: float = 5e-5,
    max_squared_error_tolerance: float = 5e-4,
    max_internal_knots: int = 32,
    interpolate_endpoints: bool = True,
    smoothness_weight: float = 0.0,
    control_ridge: float = 0.0,
    candidates_per_iteration: int = 12,
    min_knot_separation: float = 1e-8,
) -> DualErrorKnotRepairResult:
    """Insert knots until both observed errors pass, or the search is exhausted.

    Existing passing fits are returned unchanged, by identity, with no refit.
    Each trial inserts exactly one *simple* knot and refits only control points.
    The endpoint and regularization settings are explicit: callers should use
    the original method's convention. Initial knot multiplicities are preserved.

    Trials are ordered by (dual feasibility, worst normalized constraint,
    normalized MSE, normalized maximum error). If a later step is worse, search
    may continue from it to explore larger nested spaces, but the returned fit
    is the best observed state. This bounded search is not a minimum-knot proof.
    All errors are squared Euclidean observation residuals, without square roots.
    """
    started = time.perf_counter()
    _positive_finite("mse_tolerance", mse_tolerance)
    _positive_finite("max_squared_error_tolerance", max_squared_error_tolerance)
    _positive_finite("min_knot_separation", min_knot_separation)
    if min_knot_separation >= 0.5:
        raise ValueError("min_knot_separation must be less than 0.5")
    for name, value in (("smoothness_weight", smoothness_weight), ("control_ridge", control_ridge)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not isinstance(candidates_per_iteration, int) or isinstance(candidates_per_iteration, bool) or candidates_per_iteration < 1:
        raise ValueError("candidates_per_iteration must be a positive integer")
    if not isinstance(interpolate_endpoints, bool):
        raise ValueError("interpolate_endpoints must be Boolean")
    _validate_inputs(parameters, points, initial_fit, max_internal_knots)
    before_mse, before_peak, initial_squared = _errors(initial_fit, parameters, points)

    def passed(mse: float, peak: float) -> bool:
        return mse <= mse_tolerance and peak <= max_squared_error_tolerance

    def quality(mse: float, peak: float) -> tuple[float, ...]:
        return (0.0 if passed(mse, peak) else 1.0,
                max(mse / mse_tolerance, peak / max_squared_error_tolerance),
                mse / mse_tolerance, peak / max_squared_error_tolerance)

    endpoint_peak = float(initial_squared[[0, -1]].max())
    options: dict[str, Any] = {
        "interpolate_endpoints": interpolate_endpoints,
        "smoothness_weight": smoothness_weight,
        "control_ridge": control_ridge,
        "solver": "refit_bspline_control_points: CPU float64 gelsd",
        "parameters_updated": False,
        "existing_knots_deleted_or_relocated": False,
        "initial_fit_preserved_until_insertion": True,
        "initial_endpoint_max_squared_error": endpoint_peak,
        "initial_endpoint_interpolation_mismatch": interpolate_endpoints and endpoint_peak > 1e-24,
        "mse_tolerance": mse_tolerance,
        "max_squared_error_tolerance": max_squared_error_tolerance,
        "max_internal_knots": max_internal_knots,
        "candidates_per_iteration": candidates_per_iteration,
        "min_knot_separation": min_knot_separation,
    }
    best_fit = current_fit = initial_fit
    best_mse, best_peak = before_mse, before_peak
    current_squared = initial_squared
    path_insertions: list[float] = []
    best_insertions: tuple[float, ...] = ()
    trace: list[dict[str, Any]] = []
    refit_count = 0
    termination = "already_satisfied" if passed(before_mse, before_peak) else "max_internal_knots_reached"
    while not passed(best_mse, best_peak) and current_fit.internal_knots.numel() < max_internal_knots:
        candidates = _candidate_locations(parameters, current_squared, current_fit.internal_knots,
                                          candidates_per_iteration, min_knot_separation)
        if not candidates:
            termination = "no_valid_insertion_candidates"
            break
        best_trial: tuple[tuple[float, ...], BSplineLeastSquaresFit, float, float, torch.Tensor, float, int] | None = None
        for position in candidates:
            trial_knots = torch.sort(torch.cat((current_fit.internal_knots,
                                                torch.tensor([position], dtype=torch.float64)))).values
            entry: dict[str, Any] = {
                "iteration": len(path_insertions) + 1,
                "inserted_knot": position,
                "internal_knot_count": int(trial_knots.numel()),
                "accepted": False,
            }
            refit_count += 1
            try:
                fit = refit_bspline_control_points(
                    parameters, points, trial_knots, degree=initial_fit.degree,
                    smoothness_weight=smoothness_weight, control_ridge=control_ridge,
                    interpolate_endpoints=interpolate_endpoints,
                )
                mse, peak, squared = _errors(fit, parameters, points)
                rank = quality(mse, peak)
                entry.update(mse=mse, max_squared_error=peak, dual_pass=passed(mse, peak),
                             normalized_worst_error=rank[1], status="ok")
                if best_trial is None or rank < best_trial[0]:
                    best_trial = (rank, fit, mse, peak, squared, position, len(trace))
            except (RuntimeError, ValueError) as exc:
                entry.update(status="refit_failed", error=str(exc))
            trace.append(entry)
        if best_trial is None:
            termination = "all_candidate_refits_failed"
            break
        rank, current_fit, mse, peak, current_squared, position, trace_index = best_trial
        trace[trace_index]["accepted"] = True
        path_insertions.append(position)
        if rank < quality(best_mse, best_peak):
            best_fit, best_mse, best_peak = current_fit, mse, peak
            best_insertions = tuple(path_insertions)
            trace[trace_index]["best_observed"] = True
        else:
            trace[trace_index]["best_observed"] = False
        if passed(best_mse, best_peak):
            termination = "dual_tolerance_met"
            break
    return DualErrorKnotRepairResult(
        initial_fit=initial_fit, final_fit=best_fit,
        before_mse=before_mse, after_mse=best_mse,
        before_max_squared_error=before_peak, after_max_squared_error=best_peak,
        before_pass=passed(before_mse, before_peak), after_pass=passed(best_mse, best_peak),
        initial_knot_count=int(initial_fit.internal_knots.numel()),
        final_knot_count=int(best_fit.internal_knots.numel()),
        inserted_knots=best_insertions, trace=tuple(trace), refit_count=refit_count,
        elapsed_repair_ms=(time.perf_counter() - started) * 1000.0,
        termination=termination, solver_options=options,
    )
