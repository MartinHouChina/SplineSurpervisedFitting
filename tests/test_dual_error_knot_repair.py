from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.dual_error_knot_repair import repair_knots_to_dual_tolerance


def _fit(t, points, knots=(), *, endpoints=True):
    return refit_bspline_control_points(
        t, points, torch.tensor(knots, dtype=torch.float64),
        smoothness_weight=0.0, interpolate_endpoints=endpoints,
    )


def _curve():
    t = torch.linspace(0, 1, 65, dtype=torch.float64)
    return t, torch.stack((t, 0.15 * torch.sin(6 * torch.pi * t)), dim=1)


def test_already_satisfied_is_exact_noop_including_cached_fields():
    t = torch.linspace(0, 1, 41, dtype=torch.float64)
    points = torch.stack((t, t**3 - t**2), dim=1)
    initial = _fit(t, points)
    result = repair_knots_to_dual_tolerance(t, points, initial)
    assert result.final_fit is initial
    assert result.initial_fit is initial
    assert result.before_pass and result.after_pass
    assert result.refit_count == 0
    assert result.inserted_knots == () and result.trace == ()
    assert result.termination == "already_satisfied"


def test_cap_failure_reports_actual_errors_and_keeps_original_knots():
    t, points = _curve()
    initial = _fit(t, points, (0.2, 0.2, 0.7))
    old_knots = initial.internal_knots.clone()
    original_t = t.clone()
    result = repair_knots_to_dual_tolerance(
        t, points, initial, max_internal_knots=5,
        mse_tolerance=1e-30, max_squared_error_tolerance=1e-29,
        candidates_per_iteration=4,
    )
    assert result.final_knot_count <= 5
    assert result.refit_count <= 8
    assert not result.after_pass
    assert result.termination == "max_internal_knots_reached"
    assert torch.equal(t, original_t)
    assert torch.equal(initial.internal_knots, old_knots)
    for value in old_knots.unique():
        assert int((result.final_fit.internal_knots == value).sum()) >= int((old_knots == value).sum())
    assert torch.all(result.final_fit.internal_knots[1:] >= result.final_fit.internal_knots[:-1])
    squared = (result.final_fit.evaluate(t) - points).square().sum(dim=1)
    assert result.after_mse == float(squared.mean())
    assert result.after_max_squared_error == float(squared.max())
    assert result.after_pass == (float(squared.mean()) <= 1e-30 and float(squared.max()) <= 1e-29)
    assert torch.allclose(result.final_fit.control_points[[0, -1]], points[[0, -1]], atol=0, rtol=0)
    assert result.final_knot_count - result.initial_knot_count == len(result.inserted_knots)


def test_mse_pass_but_peak_failure_triggers_repair_and_ignores_stale_metrics():
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    points = torch.stack((t, torch.zeros_like(t)), dim=1)
    points[50, 1] = 0.06
    initial = _fit(t, points)
    # The fit's cached scalar must not determine pass status.
    initial = replace(initial, fit_mse=torch.tensor(0.0, dtype=torch.float64))
    result = repair_knots_to_dual_tolerance(
        t, points, initial, max_internal_knots=3, candidates_per_iteration=4,
    )
    assert result.before_mse <= 5e-5
    assert result.before_max_squared_error > 5e-4
    assert not result.before_pass
    assert result.refit_count > 0
    assert result.after_pass == (result.after_mse <= 5e-5 and result.after_max_squared_error <= 5e-4)


def test_cap_zero_and_constant_curves():
    t = torch.linspace(0, 1, 17, dtype=torch.float64)
    points = torch.full((17, 2), 2.0, dtype=torch.float64)
    initial = _fit(t, points)
    result = repair_knots_to_dual_tolerance(t, points, initial, max_internal_knots=0)
    assert result.final_fit is initial and result.after_pass
    t, points = _curve()
    initial = _fit(t, points)
    result = repair_knots_to_dual_tolerance(t, points, initial, max_internal_knots=0)
    assert result.final_fit is initial and not result.after_pass
    assert result.refit_count == 0 and result.termination == "max_internal_knots_reached"


def test_residual_insertion_can_reach_both_tolerances_and_is_deterministic():
    t, points = _curve()
    initial = _fit(t, points)
    kwargs = dict(max_internal_knots=12, candidates_per_iteration=6)
    first = repair_knots_to_dual_tolerance(t, points, initial, **kwargs)
    second = repair_knots_to_dual_tolerance(t, points, initial, **kwargs)
    assert first.after_pass
    assert first.termination == "dual_tolerance_met"
    assert first.inserted_knots == second.inserted_knots
    assert first.after_mse == pytest.approx(second.after_mse, abs=1e-14)
    assert first.after_max_squared_error == pytest.approx(second.after_max_squared_error, abs=1e-14)
    assert all(0 < knot < 1 for knot in first.inserted_knots)
    assert first.after_mse < first.before_mse
    assert sum(entry["accepted"] for entry in first.trace) >= len(first.inserted_knots)


def test_unconstrained_endpoint_policy_is_explicit_and_not_silently_overridden():
    t, points = _curve()
    points[0, 1] = 0.2
    initial = _fit(t, points, endpoints=False)
    result = repair_knots_to_dual_tolerance(
        t, points, initial, interpolate_endpoints=False, max_internal_knots=2,
        mse_tolerance=1e-12, max_squared_error_tolerance=1e-11,
        candidates_per_iteration=3,
    )
    assert result.solver_options["interpolate_endpoints"] is False
    assert result.solver_options["initial_endpoint_interpolation_mismatch"] is False
    assert not torch.equal(result.final_fit.control_points[[0, -1]], points[[0, -1]])
    constrained = repair_knots_to_dual_tolerance(
        t, points, initial, interpolate_endpoints=True, max_internal_knots=2,
        mse_tolerance=1e-12, max_squared_error_tolerance=1e-11,
        candidates_per_iteration=3,
    )
    assert constrained.solver_options["initial_endpoint_interpolation_mismatch"] is True


@pytest.mark.parametrize("overrides", [
    {"mse_tolerance": 0}, {"mse_tolerance": float("nan")},
    {"max_squared_error_tolerance": float("inf")},
    {"max_internal_knots": -1}, {"max_internal_knots": 2.5},
    {"candidates_per_iteration": 0}, {"min_knot_separation": 0.5},
    {"smoothness_weight": float("nan")}, {"control_ridge": -1},
])
def test_invalid_options_are_rejected(overrides):
    t, points = _curve()
    with pytest.raises(ValueError):
        repair_knots_to_dual_tolerance(t, points, _fit(t, points), **overrides)


def test_invalid_inputs_and_initial_overcapacity_are_rejected():
    t, points = _curve()
    initial = _fit(t, points, (0.5,))
    with pytest.raises(ValueError, match="exceeds"):
        repair_knots_to_dual_tolerance(t, points, initial, max_internal_knots=0)
    with pytest.raises(ValueError, match="float64"):
        repair_knots_to_dual_tolerance(t.float(), points, initial)
    invalid = points.clone()
    invalid[4, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        repair_knots_to_dual_tolerance(t, invalid, initial)
    with pytest.raises(ValueError, match="non-decreasing"):
        repair_knots_to_dual_tolerance(t.flip(0), points, initial)
    with pytest.raises(ValueError, match="matching"):
        repair_knots_to_dual_tolerance(t[:1], points[:1], initial)
