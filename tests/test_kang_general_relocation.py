"""Paper Algorithm 1/3 control-flow regression, not a CVX equivalence claim."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spline_fitting.evaluation import sparse_knot_paper as kang


def _input():
    parameters = torch.linspace(0, 1, 31, dtype=torch.float64)
    points = torch.stack((parameters, torch.sin(7 * parameters)), dim=-1)
    knots = torch.linspace(0, 1, 10, dtype=torch.float64)[1:-1]
    return parameters, points, knots


def _run(parameters, points, knots, **kwargs):
    return kang._relocate_general_intervals(
        parameters, points, knots, 3, tolerance=1e-6,
        max_iterations=1, rho=1e4, admm_max_iterations=8,
        admm_tolerance=1e-6, bisection_iterations=1, **kwargs,
    )


def test_general_interval_test_can_reject_every_merge_without_cluster_shortcut(monkeypatch):
    parameters, points, knots = _input()
    calls = []
    fixed_error = float(kang._refit_mse(parameters, points, knots, 3))

    def solve(basis, points, jumps, tolerance, **kwargs):
        index = len(calls)
        calls.append(tolerance)
        values = points.new_full((jumps.shape[0], points.shape[1]), 2.0)
        values[index + 1] = 0.0
        return SimpleNamespace(jump_vectors=values), 7

    monkeypatch.setattr(kang, "_select_constrained_sparse_state", solve)
    output, refits, diagnostics = _run(parameters, points, knots)
    torch.testing.assert_close(output, knots, rtol=0, atol=0)
    assert calls == [fixed_error] * 7
    assert refits == 1
    assert diagnostics["candidate_interval_sparse_solves"] == 7
    assert diagnostics["candidate_interval_sparse_iterations"] == 49
    assert diagnostics["candidate_intervals_merged"] == 0
    assert diagnostics["candidate_intervals_rejected"] == 7
    assert diagnostics["narrowing_reached_tolerance"]


def test_general_ties_accept_and_budget_truncation_is_reported(monkeypatch):
    parameters, points, knots = _input()
    calls = []

    def solve(basis, points, jumps, tolerance, **kwargs):
        calls.append(tolerance)
        return SimpleNamespace(jump_vectors=points.new_ones((jumps.shape[0], 2))), 11

    monkeypatch.setattr(kang, "_select_constrained_sparse_state", solve)
    output, refits, diagnostics = _run(parameters, points, knots)
    # Algorithm 1 visits pairs in sequence; a single connected eight-knot run
    # must not be replaced unconditionally by one or two representatives.
    assert output.numel() == 4
    assert len(set(calls)) == 1  # Algorithm 1 Err stays fixed across merges.
    assert diagnostics["fixed_initial_active_ls_mse"] == calls[0]
    assert diagnostics["candidate_interval_sparse_solves"] == 4
    assert diagnostics["candidate_interval_sparse_iterations"] == 44
    assert diagnostics["candidate_intervals_merged"] == 4
    assert diagnostics["narrowing_budget_truncated_intervals"] == 4
    assert diagnostics["narrowing_iterations"] == 4
    assert not diagnostics["narrowing_reached_tolerance"]
    assert refits == 9


def test_infeasible_fixed_error_is_not_silently_relaxed(monkeypatch):
    parameters, points, knots = _input()
    def fail(*args, **kwargs):
        raise ValueError("infeasible constrained problem")
    monkeypatch.setattr(kang, "_select_constrained_sparse_state", fail)
    with pytest.raises(kang.KangIntervalSolveError, match="no non-paper fallback or tolerance relaxation") as raised:
        _run(parameters, points, knots)
    diagnostics = raised.value.solver_diagnostics
    assert not diagnostics["algorithm_completed"]
    assert diagnostics["candidate_interval_sparse_solves_completed"] == 0
    assert diagnostics["trial_unconstrained_minimum_mse"] >= 0
    assert "trial_minimum_minus_fixed_error" in diagnostics


def test_general_continuous_active_run_reports_real_resolve_diagnostics():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        parameters, points, _ = _input()
        result = kang.fit_sparse_knots_paper(
            parameters, points, initial_internal_knot_count=12,
            data_tolerance=1e-8, admm_max_iterations=20,
            lambda_bisection_iterations=1, relocation_max_iterations=2,
            relocation_algorithm="general",
        )
    finally:
        torch.set_num_threads(previous)
    assert result.active_count >= 8
    assert result.knots.numel() >= result.active_count // 2
    d = result.relocation_diagnostics
    assert d["candidate_interval_sparse_solves"] > 0
    assert d["candidate_interval_sparse_iterations"] > 0
    assert d["fixed_initial_active_ls_mse"] >= 0
    assert d["narrowing_iteration_budget"] == 2
    assert not result.repair_used
