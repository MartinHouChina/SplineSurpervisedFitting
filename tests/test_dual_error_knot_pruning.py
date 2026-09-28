from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation import dual_error_knot_pruning as pruning


def _fit(t, points, knots=(), *, endpoints=True, smoothness=0.0, ridge=0.0):
    return refit_bspline_control_points(
        t, points, torch.tensor(knots, dtype=torch.float64),
        smoothness_weight=smoothness, control_ridge=ridge,
        interpolate_endpoints=endpoints,
    )


def _polynomial():
    t = torch.linspace(0, 1, 65, dtype=torch.float64)
    return t, torch.stack((t, t**3 - t**2), dim=1)


def _errors(fit, t, points):
    squared = (fit.evaluate(t) - points).square().sum(dim=1)
    return float(squared.mean()), float(squared.max())


def test_polynomial_prunes_to_zero_with_exhaustive_exact_refits():
    t, points = _polynomial()
    initial = _fit(t, points, (0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9))
    old_t, old_points = t.clone(), points.clone()
    old_knots, old_control = initial.internal_knots.clone(), initial.control_points.clone()
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial)
    assert result.initial_fit is initial and result.final_fit is not initial
    assert result.before_pass and result.after_pass
    assert result.initial_knot_count == 7 and result.final_knot_count == 0
    assert len(result.removed_knots) == 7
    assert result.refit_count == sum(range(1, 8)) == len(result.trace)
    assert result.termination == "min_internal_knots_reached"
    assert result.final_fit.control_points.shape == (4, 2)
    assert torch.equal(t, old_t) and torch.equal(points, old_points)
    assert torch.equal(initial.internal_knots, old_knots)
    assert torch.equal(initial.control_points, old_control)
    assert torch.equal(result.final_fit.control_points[[0, -1]], points[[0, -1]])
    assert (result.after_mse, result.after_max_squared_error) == _errors(result.final_fit, t, points)
    assert result.elapsed_pruning_ms >= 0
    for iteration in range(1, 8):
        entries = [entry for entry in result.trace if entry["iteration"] == iteration]
        assert len(entries) == 8 - iteration
        accepted = [entry for entry in entries if entry["accepted"]]
        assert len(accepted) == 1 and accepted[0]["dual_pass"]
        assert accepted[0]["mse"] <= 5e-5
        assert accepted[0]["max_squared_error"] <= 5e-4


@pytest.mark.parametrize("limiting_error", ["mse", "max_squared_error"])
def test_either_bound_independently_prevents_a_deletion(limiting_error):
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    points = torch.stack((t, (t - 0.5).clamp_min(0).pow(3)), dim=1)
    initial = _fit(t, points, (0.5,))
    deleted = _fit(t, points)
    mse, peak = _errors(deleted, t, points)
    options = dict(mse_tolerance=mse * 2, max_squared_error_tolerance=peak * 2)
    options[f"{limiting_error}_tolerance"] /= 4
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial, **options)
    assert result.before_pass and result.after_pass
    assert result.final_fit is initial and result.removed_knots == ()
    assert result.refit_count == 1 and not result.trace[0]["accepted"]
    assert not result.trace[0]["dual_pass"]
    assert result.trace[0][limiting_error] > options[f"{limiting_error}_tolerance"]
    assert result.termination == "no_feasible_single_knot_deletion"


def test_all_candidates_checked_after_peak_failure_and_best_feasible_selected():
    t = torch.linspace(0, 1, 65, dtype=torch.float64)
    points = torch.stack((t, 0.15 * torch.sin(6 * torch.pi * t)), dim=1)
    knots = (0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9)
    initial = _fit(t, points, knots)
    deletion_errors = [_errors(_fit(t, points, knots[:i] + knots[i + 1:]), t, points)
                       for i in range(len(knots))]
    # The final candidate is feasible although the first fails its peak bound.
    peak_limit = (deletion_errors[0][1] + deletion_errors[-1][1]) / 2
    assert _errors(initial, t, points)[1] < peak_limit
    assert deletion_errors[-1][1] < peak_limit < deletion_errors[0][1]
    result = pruning.prune_knots_to_dual_tolerance(
        t, points, initial, mse_tolerance=1.0,
        max_squared_error_tolerance=peak_limit, max_deletions=1,
    )
    assert result.refit_count == len(knots)
    assert not result.trace[0]["dual_pass"]
    assert result.trace[-1]["dual_pass"] and result.trace[-1]["accepted"]
    assert result.removed_knots == (knots[-1],)
    assert result.termination == "max_deletions_reached"
    assert result.after_pass
    # Threshold-preserving pruning is allowed to increase either error.
    assert result.after_mse > result.before_mse
    assert result.after_max_squared_error > result.before_max_squared_error


def test_initially_infeasible_fit_is_identity_and_stale_metrics_ignored():
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    points = torch.stack((t, torch.zeros_like(t)), dim=1)
    points[50, 1] = 0.06
    initial = _fit(t, points, (0.2, 0.8))
    initial = replace(initial, fit_mse=torch.tensor(0.0, dtype=torch.float64),
                      reconstructed_points=points.clone())
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial)
    assert result.before_mse <= 5e-5 and result.before_max_squared_error > 5e-4
    assert not result.before_pass and not result.after_pass
    assert result.initial_fit is initial and result.final_fit is initial
    assert result.refit_count == 0 and result.trace == () and result.removed_knots == ()
    assert result.termination == "initially_infeasible"
    assert (result.before_mse, result.before_max_squared_error) == _errors(initial, t, points)


def test_multiplicities_survivor_subset_and_minimum_count():
    t, points = _polynomial()
    initial = _fit(t, points, (0.2, 0.2, 0.2, 0.7, 0.7))
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial, min_internal_knots=2)
    before = Counter(initial.internal_knots.tolist())
    after = Counter(result.final_fit.internal_knots.tolist())
    assert result.final_knot_count == 2 and len(result.removed_knots) == 3
    assert after + Counter(result.removed_knots) == before
    assert all(after[value] <= before[value] for value in after)
    assert torch.all(result.final_fit.internal_knots[1:] >= result.final_fit.internal_knots[:-1])
    assert result.refit_count == 5 + 4 + 3
    assert result.termination == "min_internal_knots_reached"


@pytest.mark.parametrize("options, termination", [
    ({"max_deletions": 0}, "max_deletions_reached"),
    ({"min_internal_knots": 2}, "min_internal_knots_reached"),
])
def test_zero_budget_or_count_floor_is_exact_noop(options, termination):
    t, points = _polynomial()
    initial = _fit(t, points, (0.2, 0.7))
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial, **options)
    assert result.final_fit is initial and result.after_pass
    assert result.refit_count == 0 and result.trace == ()
    assert result.termination == termination


def test_initial_polynomial_without_knots_is_legal():
    t, points = _polynomial()
    initial = _fit(t, points)
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial, max_internal_knots=0)
    assert result.final_fit is initial and result.refit_count == 0
    assert result.termination == "min_internal_knots_reached"


def test_failed_refit_is_recorded_and_other_candidates_are_still_checked(monkeypatch):
    t, points = _polynomial()
    initial = _fit(t, points, (0.2, 0.5, 0.7))
    original_refit = pruning.refit_bspline_control_points
    calls = []

    def sometimes_failed(parameters, observations, knots, **kwargs):
        calls.append((parameters.clone(), knots.clone(), kwargs))
        if len(calls) == 1:
            raise RuntimeError("deliberate trial failure")
        return original_refit(parameters, observations, knots, **kwargs)

    monkeypatch.setattr(pruning, "refit_bspline_control_points", sometimes_failed)
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial, max_deletions=1)
    assert len(calls) == result.refit_count == 3
    assert result.trace[0]["status"] == "refit_failed"
    assert "deliberate trial failure" in result.trace[0]["error"]
    assert not result.trace[0]["accepted"]
    assert result.final_knot_count == 2 and result.after_pass
    assert sum(entry["accepted"] for entry in result.trace) == 1
    assert all(torch.equal(parameters, t) for parameters, _, _ in calls)
    assert all(knots.numel() == 2 for _, knots, _ in calls)


def test_all_failed_refits_preserve_original_identity(monkeypatch):
    t, points = _polynomial()
    initial = _fit(t, points, (0.2, 0.5, 0.7))

    def failed(*args, **kwargs):
        raise ValueError("deliberate trial failure")

    monkeypatch.setattr(pruning, "refit_bspline_control_points", failed)
    result = pruning.prune_knots_to_dual_tolerance(t, points, initial)
    assert result.final_fit is initial and result.after_pass
    assert result.refit_count == 3 and result.removed_knots == ()
    assert all(entry["status"] == "refit_failed" for entry in result.trace)
    assert result.termination == "all_candidate_refits_failed"


def test_solver_options_are_forwarded_and_unconstrained_endpoints_preserved(monkeypatch):
    t, points = _polynomial()
    points[0, 1] += 0.01
    initial = _fit(t, points, (0.2, 0.5, 0.7), endpoints=False, smoothness=1e-7, ridge=2e-7)
    original_refit = pruning.refit_bspline_control_points
    options = []

    def record_options(*args, **kwargs):
        options.append(kwargs)
        return original_refit(*args, **kwargs)

    monkeypatch.setattr(pruning, "refit_bspline_control_points", record_options)
    result = pruning.prune_knots_to_dual_tolerance(
        t, points, initial, interpolate_endpoints=False,
        smoothness_weight=1e-7, control_ridge=2e-7, max_deletions=1,
    )
    assert result.final_knot_count == 2
    assert all(option == dict(degree=3, interpolate_endpoints=False,
                              smoothness_weight=1e-7, control_ridge=2e-7) for option in options)
    assert result.solver_options["interpolate_endpoints"] is False
    assert result.solver_options["initial_endpoint_interpolation_mismatch"] is False
    assert not torch.equal(result.final_fit.control_points[[0, -1]], points[[0, -1]])


@pytest.mark.parametrize("overrides", [
    {"mse_tolerance": 0}, {"mse_tolerance": float("nan")},
    {"max_squared_error_tolerance": float("inf")},
    {"max_internal_knots": -1}, {"max_internal_knots": 2.5},
    {"max_internal_knots": True}, {"min_internal_knots": -1},
    {"min_internal_knots": 33}, {"max_deletions": -1},
    {"max_deletions": 1.5}, {"max_deletions": False},
    {"interpolate_endpoints": 1}, {"smoothness_weight": float("nan")},
    {"control_ridge": -1},
])
def test_invalid_options_are_rejected(overrides):
    t, points = _polynomial()
    with pytest.raises(ValueError):
        pruning.prune_knots_to_dual_tolerance(t, points, _fit(t, points), **overrides)


def test_invalid_observations_and_initial_fit_are_rejected():
    t, points = _polynomial()
    initial = _fit(t, points, (0.2, 0.7))
    with pytest.raises(ValueError, match="exceeds"):
        pruning.prune_knots_to_dual_tolerance(t, points, initial, max_internal_knots=1)
    with pytest.raises(ValueError, match="float64"):
        pruning.prune_knots_to_dual_tolerance(t.float(), points, initial)
    invalid = points.clone()
    invalid[4, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pruning.prune_knots_to_dual_tolerance(t, invalid, initial)
    with pytest.raises(ValueError, match="non-decreasing"):
        pruning.prune_knots_to_dual_tolerance(t.flip(0), points, initial)
    with pytest.raises(ValueError, match="matching"):
        pruning.prune_knots_to_dual_tolerance(t[:1], points[:1], initial)
    with pytest.raises(ValueError, match="normalized interval"):
        pruning.prune_knots_to_dual_tolerance(t * 0.9, points, initial)
    with pytest.raises(ValueError, match="open-clamped"):
        pruning.prune_knots_to_dual_tolerance(t, points, replace(initial, knot_vector=initial.knot_vector + 0.01))
    with pytest.raises(ValueError, match="control point shape"):
        pruning.prune_knots_to_dual_tolerance(t, points, replace(initial, control_points=initial.control_points[:-1]))
