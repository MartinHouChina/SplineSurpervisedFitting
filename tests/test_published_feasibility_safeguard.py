"""Regression tests for explicitly disclosed, capacity-bounded comparison repair."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation import published_baselines as baselines
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.gradient_knot_pruning import chord_length_parameters


METHODS = (
    "dung_direct_knot_2017_adaptation",
    "kang_sparse_2015_adaptation",
    "luo_linf_de_2022_adaptation",
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def multiscale_curve():
    t = torch.linspace(0.0, 1.0, 96, dtype=torch.float64)
    points = torch.stack((t, 0.18 * torch.sin(4 * torch.pi * t)
                          + 0.045 * torch.sin(17 * torch.pi * t)), dim=-1)
    return (points - points.amin(0)) / (points.amax(0) - points.amin(0)).max()


def full_options():
    return dict(mse_tolerance=5e-5, max_internal_knots=64, paper_initial_knots=64,
                paper_admm_iterations=400, paper_lambda_bisections=8,
                paper_relocation_iterations=8, dung_scan_intervals=10,
                dung_optimization_iterations=10, luo_eta=0.5,
                luo_de_population=10, luo_de_iterations=50)


def tiny_options(capacity, tolerance):
    return dict(mse_tolerance=tolerance, max_internal_knots=capacity,
                paper_initial_knots=capacity, paper_admm_iterations=8,
                paper_lambda_bisections=1, paper_relocation_iterations=1,
                dung_scan_intervals=2, dung_optimization_iterations=1,
                luo_de_population=5, luo_de_iterations=1)


@pytest.mark.parametrize("method", METHODS)
def test_full_overnight_budget_repairs_multiscale_failure_within_same_capacity(method):
    points = multiscale_curve()
    options = full_options()
    native = baselines.run_published_baseline(
        method, points, published_feasibility_safeguard=False, **options,
    )
    repaired = baselines.run_published_baseline(method, points, **options)
    d = repaired.diagnostics
    assert float(repaired.fit.fit_mse) <= options["mse_tolerance"]
    assert repaired.fit.internal_knots.numel() <= 64
    torch.testing.assert_close(repaired.parameters, native.parameters)
    torch.testing.assert_close(repaired.fit.reconstructed_points[[0, -1]], points[[0, -1]])
    assert float(repaired.fit.fit_mse) <= float(native.fit.fit_mse) + 1e-14
    assert d["comparison_feasibility_native_k"] == native.fit.internal_knots.numel()
    assert d["comparison_feasibility_native_mse"] == pytest.approx(float(native.fit.fit_mse))
    assert d["comparison_feasibility_final_mse"] == float(repaired.fit.fit_mse)
    assert d["comparison_feasibility_final_k"] == repaired.fit.internal_knots.numel()
    assert "not part of the cited method" in d["comparison_feasibility_safeguard_role"]
    assert d["comparison_method_label"] == "threshold-safe adaptation"
    if method == "kang_sparse_2015_adaptation":
        # This was a single 62-knot active run collapsed to one knot at d473159.
        assert max(d["cluster_sizes"]) > 4
        assert not d["algorithm4_cluster_relocation_applied"]
        assert native.fit.internal_knots.numel() == d["active_internal_knot_count"]
        assert not d["comparison_feasibility_safeguard_used"]
        torch.testing.assert_close(repaired.fit.internal_knots, native.fit.internal_knots)
    else:
        assert float(native.fit.fit_mse) > options["mse_tolerance"]
        assert d["comparison_feasibility_safeguard_used"]
        assert d["comparison_feasibility_refit_count"] > 0
        assert d["comparison_feasibility_added_knots"]


@pytest.mark.parametrize("method", METHODS)
def test_line_with_zero_capacity_is_unchanged_and_needs_no_repair(method):
    t = torch.linspace(0.0, 1.0, 25, dtype=torch.float64)
    points = torch.stack((t, 0.3 * t + 0.1), dim=-1)
    options = tiny_options(0, 5e-5)
    native = baselines.run_published_baseline(
        method, points, published_feasibility_safeguard=False, **options,
    )
    repaired = baselines.run_published_baseline(method, points, **options)
    assert repaired.fit.internal_knots.numel() == 0
    torch.testing.assert_close(repaired.fit.control_points, native.fit.control_points, atol=0, rtol=0)
    assert repaired.diagnostics["comparison_feasibility_status"] == "native_feasible"
    assert not repaired.diagnostics["comparison_feasibility_safeguard_attempted"]
    assert repaired.diagnostics["comparison_feasibility_refit_count"] == 0


@pytest.mark.parametrize("method", METHODS)
def test_insufficient_capacity_remains_honestly_infeasible_and_never_worsens(method):
    t = torch.linspace(0.0, 1.0, 57, dtype=torch.float64)
    points = torch.stack((t, 0.25 * torch.sin(6 * torch.pi * t)), dim=-1)
    options = tiny_options(1, 1e-8)
    native = baselines.run_published_baseline(
        method, points, published_feasibility_safeguard=False, **options,
    )
    repaired = baselines.run_published_baseline(method, points, **options)
    assert float(repaired.fit.fit_mse) > 1e-8
    assert float(repaired.fit.fit_mse) <= float(native.fit.fit_mse) + 1e-14
    assert repaired.fit.internal_knots.numel() <= 1
    assert not repaired.diagnostics["threshold_satisfied"]
    assert not repaired.diagnostics["comparison_feasibility_threshold_satisfied"]
    assert repaired.diagnostics["comparison_feasibility_status"] == "budget_exhausted_without_feasible_fit"


def test_safeguard_retains_best_result_even_if_later_refits_worsen(monkeypatch):
    points = multiscale_curve()
    parameters = chord_length_parameters(points)
    native = refit_bspline_control_points(
        parameters, points, parameters.new_empty(0), interpolate_endpoints=True,
    )
    original_refit = baselines.refit_bspline_control_points
    calls = []

    def unstable_refit(*args, **kwargs):
        fit = original_refit(*args, **kwargs)
        calls.append(fit)
        # Only the first augmentation round improves the fit; later numerical
        # solves and the uniform fallback are deliberately worse.
        factor = 0.8 if len(calls) <= 3 else 1.2
        return replace(fit, fit_mse=native.fit_mse * factor)

    monkeypatch.setattr(baselines, "refit_bspline_control_points", unstable_refit)
    fit, d = baselines._common_mse_feasibility_safeguard(
        parameters, points, native, preferred_candidates=parameters.new_empty(0),
        max_internal_knots=3, degree=3, mse_tolerance=1e-12,
    )
    assert float(fit.fit_mse) == pytest.approx(float(native.fit_mse) * 0.8)
    assert fit.internal_knots.numel() == 1
    assert d["comparison_feasibility_refit_count"] == len(calls)
    assert tuple(fit.internal_knots.tolist()) == d["comparison_feasibility_final_knots"]


def test_safeguard_uniform_fallback_can_restore_feasibility_at_full_capacity():
    from spline_fitting.data.synthetic import bspline_basis_matrix
    from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector

    parameters = torch.linspace(0, 1, 80, dtype=torch.float64)
    uniform = parameters.new_tensor([1 / 3, 2 / 3])
    controls = parameters.new_tensor([[0, 0], [0.1, 0.9], [0.3, -0.8],
                                     [0.6, 0.8], [0.9, -0.7], [1, 0]])
    points = bspline_basis_matrix(parameters, build_open_knot_vector(uniform, 3), 3, 6) @ controls
    native = refit_bspline_control_points(
        parameters, points, parameters.new_tensor([0.05, 0.1]), interpolate_endpoints=True,
    )
    fit, d = baselines._common_mse_feasibility_safeguard(
        parameters, points, native, preferred_candidates=parameters.new_empty(0),
        max_internal_knots=2, degree=3, mse_tolerance=1e-10,
    )
    assert float(native.fit_mse) > 1e-10
    assert float(fit.fit_mse) <= 1e-10
    torch.testing.assert_close(fit.internal_knots, uniform)
    assert d["comparison_feasibility_final_source"] == "uniform_capacity_fallback"
    assert d["comparison_feasibility_refit_count"] == 1


def test_safeguard_refits_are_inside_end_to_end_timer(monkeypatch):
    events = []
    original = baselines._common_mse_feasibility_safeguard

    def clock():
        events.append("start" if not events else "stop")
        return 1.0 if len(events) == 1 else 3.0

    def repair(*args, **kwargs):
        events.append("repair")
        return original(*args, **kwargs)

    monkeypatch.setattr(baselines, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(baselines, "_common_mse_feasibility_safeguard", repair)
    result = baselines.run_published_baseline(
        "luo_linf_de_2022_adaptation", multiscale_curve(), **tiny_options(28, 1e-4),
    )
    assert events[0] == "start" and events[-2:] == ["repair", "stop"]
    assert result.elapsed_ms == 2000.0


def test_dung_native_and_common_max_error_flags_are_not_conflated(monkeypatch):
    original = baselines.fit_dung_direct_knots

    def different_native_and_common(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, native_max_error=0.0, final_max_error=1.0)

    monkeypatch.setattr(baselines, "fit_dung_direct_knots", different_native_and_common)
    result = baselines.run_published_baseline(
        "dung_direct_knot_2017_adaptation", multiscale_curve(), **tiny_options(28, 1e-4),
    )
    assert result.diagnostics["threshold_satisfied_native_max_error"]
    assert not result.diagnostics["threshold_satisfied_common_max_error_before_safeguard"]
