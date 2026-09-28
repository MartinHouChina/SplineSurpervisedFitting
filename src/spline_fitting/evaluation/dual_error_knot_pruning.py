"""Observation-error-checked deletion of redundant deployed B-spline knots.

Parameters and surviving knot values are fixed. Each individual deletion is
checked by an exact production control-point refit; only dual-feasible trials
are accepted. This is a greedy single-deletion search, not a global minimum-
knot guarantee or a guarantee about errors between observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points


@dataclass(frozen=True)
class DualErrorKnotPruningResult:
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
    removed_knots: tuple[float, ...]
    trace: tuple[dict[str, Any], ...]
    refit_count: int
    elapsed_pruning_ms: float
    termination: str
    solver_options: dict[str, Any]


def _nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


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
    if not isinstance(fit.degree, int) or isinstance(fit.degree, bool) or fit.degree < 1:
        raise ValueError("initial fit degree must be a positive integer")
    knots = fit.internal_knots
    if knots.ndim != 1 or bool(torch.any((knots <= 0) | (knots >= 1))):
        raise ValueError("initial internal knots must be one-dimensional and inside (0, 1)")
    if bool(torch.any(knots[1:] < knots[:-1])):
        raise ValueError("initial internal knots must be non-decreasing")
    if knots.numel() > max_internal_knots:
        raise ValueError("initial knot count exceeds max_internal_knots")
    if knots.numel() and int(torch.unique_consecutive(knots, return_counts=True)[1].max()) > fit.degree + 1:
        raise ValueError("initial knot multiplicity exceeds degree + 1")
    expected_full = torch.cat((torch.zeros(fit.degree + 1, dtype=torch.float64), knots,
                               torch.ones(fit.degree + 1, dtype=torch.float64)))
    if not torch.equal(fit.knot_vector, expected_full):
        raise ValueError("initial fit must use an open-clamped full knot vector consistent with its internal knots")
    if fit.control_points.shape != (knots.numel() + fit.degree + 1, points.shape[1]):
        raise ValueError("initial control point shape is inconsistent with its knots and observations")


def _errors(
    fit: BSplineLeastSquaresFit, parameters: torch.Tensor, points: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    # Cached fit fields may describe other observations: evaluate the actual curve.
    squared = (fit.evaluate(parameters) - points).square().sum(dim=1)
    if not bool(torch.isfinite(squared).all()):
        raise ValueError("fit evaluation produced non-finite squared errors")
    return float(squared.mean()), float(squared.max()), squared


@torch.no_grad()
def prune_knots_to_dual_tolerance(
    parameters: torch.Tensor,
    points: torch.Tensor,
    initial_fit: BSplineLeastSquaresFit,
    *,
    mse_tolerance: float = 5e-5,
    max_squared_error_tolerance: float = 5e-4,
    max_internal_knots: int = 32,
    min_internal_knots: int = 0,
    max_deletions: int = 32,
    interpolate_endpoints: bool = True,
    smoothness_weight: float = 0.0,
    control_ridge: float = 0.0,
) -> DualErrorKnotPruningResult:
    """Greedily delete redundant knots while both observed-error bounds pass.

    An initially infeasible fit is returned unchanged. Each round tests *all*
    individual knot occurrences with CPU float64 ``gelsd`` refits and selects
    the feasible trial with smallest (worst normalized error, normalized MSE,
    normalized maximum error), breaking exact ties by original index. In
    particular, a failed maximum-error check does not end the candidate scan:
    maximum residuals need not be monotone when a least-squares space changes.

    Every accepted step satisfies both bounds, although errors may increase
    within those bounds. Repeated knots are removed one occurrence at a time;
    survivors and parameters are never moved, and knots are never inserted.
    If no deletion is accepted, ``final_fit is initial_fit``. The explicit
    endpoint/regularization options should match the incoming fit's convention.
    """
    started = time.perf_counter()
    for name, value in (("mse_tolerance", mse_tolerance),
                        ("max_squared_error_tolerance", max_squared_error_tolerance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (("max_internal_knots", max_internal_knots),
                        ("min_internal_knots", min_internal_knots),
                        ("max_deletions", max_deletions)):
        _nonnegative_integer(name, value)
    if min_internal_knots > max_internal_knots:
        raise ValueError("min_internal_knots must not exceed max_internal_knots")
    for name, value in (("smoothness_weight", smoothness_weight), ("control_ridge", control_ridge)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not isinstance(interpolate_endpoints, bool):
        raise ValueError("interpolate_endpoints must be Boolean")
    _validate_inputs(parameters, points, initial_fit, max_internal_knots)
    before_mse, before_peak, initial_squared = _errors(initial_fit, parameters, points)

    def passed(mse: float, peak: float) -> bool:
        return mse <= mse_tolerance and peak <= max_squared_error_tolerance

    def quality(mse: float, peak: float) -> tuple[float, float, float]:
        return (max(mse / mse_tolerance, peak / max_squared_error_tolerance),
                mse / mse_tolerance, peak / max_squared_error_tolerance)

    endpoint_peak = float(initial_squared[[0, -1]].max())
    options: dict[str, Any] = {
        "interpolate_endpoints": interpolate_endpoints,
        "smoothness_weight": smoothness_weight,
        "control_ridge": control_ridge,
        "solver": "refit_bspline_control_points: CPU float64 gelsd",
        "parameters_updated": False,
        "surviving_knots_relocated": False,
        "knots_inserted": False,
        "initial_fit_preserved_until_deletion": True,
        "initial_endpoint_max_squared_error": endpoint_peak,
        "initial_endpoint_interpolation_mismatch": interpolate_endpoints and endpoint_peak > 1e-24,
        "mse_tolerance": mse_tolerance,
        "max_squared_error_tolerance": max_squared_error_tolerance,
        "max_internal_knots": max_internal_knots,
        "min_internal_knots": min_internal_knots,
        "max_deletions": max_deletions,
        "candidate_search": "all single-knot occurrences per iteration",
        "selection": "feasible minimum (normalized worst error, MSE, maximum error, index)",
        "error_scope": "squared Euclidean residuals at supplied observation parameters",
    }
    current_fit = initial_fit
    current_mse, current_peak = before_mse, before_peak
    removed: list[float] = []
    trace: list[dict[str, Any]] = []
    refit_count = 0
    termination = "initially_infeasible"
    while passed(current_mse, current_peak):
        if current_fit.internal_knots.numel() <= min_internal_knots:
            termination = "min_internal_knots_reached"
            break
        if len(removed) >= max_deletions:
            termination = "max_deletions_reached"
            break
        best_trial: tuple[tuple[float, float, float], BSplineLeastSquaresFit, float, float, float, int] | None = None
        successful_refits = 0
        current_knots = current_fit.internal_knots
        for index in range(current_knots.numel()):
            position = float(current_knots[index])
            trial_knots = torch.cat((current_knots[:index], current_knots[index + 1:]))
            entry: dict[str, Any] = {
                "iteration": len(removed) + 1,
                "removed_knot_index": index,
                "removed_knot": position,
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
                mse, peak, _ = _errors(fit, parameters, points)
                rank = quality(mse, peak)
                feasible = passed(mse, peak)
                successful_refits += 1
                entry.update(mse=mse, max_squared_error=peak, dual_pass=feasible,
                             normalized_worst_error=rank[0], status="ok")
                if feasible and (best_trial is None or rank < best_trial[0]):
                    best_trial = (rank, fit, mse, peak, position, len(trace))
            except (RuntimeError, ValueError) as exc:
                entry.update(status="refit_failed", error=str(exc))
            trace.append(entry)
        if best_trial is None:
            termination = ("all_candidate_refits_failed" if successful_refits == 0
                           else "no_feasible_single_knot_deletion")
            break
        _, current_fit, current_mse, current_peak, position, trace_index = best_trial
        trace[trace_index]["accepted"] = True
        removed.append(position)

    return DualErrorKnotPruningResult(
        initial_fit=initial_fit, final_fit=current_fit,
        before_mse=before_mse, after_mse=current_mse,
        before_max_squared_error=before_peak, after_max_squared_error=current_peak,
        before_pass=passed(before_mse, before_peak), after_pass=passed(current_mse, current_peak),
        initial_knot_count=int(initial_fit.internal_knots.numel()),
        final_knot_count=int(current_fit.internal_knots.numel()),
        removed_knots=tuple(removed), trace=tuple(trace), refit_count=refit_count,
        elapsed_pruning_ms=(time.perf_counter() - started) * 1000.0,
        termination=termination, solver_options=options,
    )
