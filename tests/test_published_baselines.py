from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation import published_baselines as baselines  # noqa: E402
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)


def _points() -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, 25, dtype=torch.float64)
    return torch.stack([t, 0.15 * torch.sin(2.0 * torch.pi * t)], dim=-1)


def _run(method: str, **kwargs: object) -> baselines.PublishedBaselineResult:
    options = dict(
        mse_tolerance=1e-4,
        max_internal_knots=2,
        gradient_steps=1,
        paper_initial_knots=3,
        paper_admm_iterations=5,
        paper_lambda_bisections=1,
        paper_relocation_iterations=1,
        liang_dense_knots=4,
        liang_initial_knots=1,
        liang_feature_samples=33,
        dung_scan_intervals=2,
        dung_optimization_iterations=1,
        luo_de_population=5,
        luo_de_iterations=1,
    )
    options.update(kwargs)
    return baselines.run_published_baseline(method, _points(), **options)


@pytest.mark.parametrize("method", baselines.COMPARISON_BASELINE_METHODS)
def test_baselines_report_common_endpoint_refit_and_euclidean_mse(method: str) -> None:
    result = _run(method)
    points = _points()
    fit = result.fit
    reference = refit_bspline_control_points(
        result.parameters,
        points,
        fit.internal_knots,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )

    assert result.method == method
    assert result.elapsed_ms > 0.0 and math.isfinite(result.elapsed_ms)
    torch.testing.assert_close(fit.fit_mse, reference.fit_mse, atol=1e-14, rtol=1e-10)
    expected_mse = (fit.reconstructed_points - points).square().sum(dim=-1).mean()
    torch.testing.assert_close(fit.fit_mse, expected_mse)
    torch.testing.assert_close(fit.reconstructed_points[[0, -1]], points[[0, -1]])
    assert result.diagnostics["network_used"] is False
    assert result.diagnostics["threshold_satisfied"] == (float(expected_mse) <= 1e-4)
    if method == "kang_sparse_2015_adaptation":
        assert result.diagnostics["non_paper_feasibility_repair_enabled"] is False
        assert result.diagnostics["paper_feasibility_repair_used"] is False
        assert "native_final_fit_mse_without_endpoint_constraint" in result.diagnostics
        assert result.diagnostics["native_endpoint_constrained_mse"] == float(fit.fit_mse)
    knot_cap = (
        3
        if method
        in {"kang_sparse_2015_adaptation", "luo_linf_de_2022_adaptation"}
        else 2
    )
    assert 0 <= fit.internal_knots.numel() <= knot_cap


def test_repository_numerical_control_is_not_mislabeled_as_published() -> None:
    assert "uniform_gradient_pruning" in baselines.NUMERICAL_BASELINE_METHODS
    assert "uniform_gradient_pruning" in baselines.COMPARISON_BASELINE_METHODS
    assert "uniform_gradient_pruning" not in baselines.PUBLISHED_ADAPTATION_METHODS
    assert baselines.PUBLISHED_BASELINE_METHODS == baselines.PUBLISHED_ADAPTATION_METHODS


def test_timing_encloses_parameterization_search_and_common_final_refit(monkeypatch) -> None:
    events: list[str] = []

    def clock() -> float:
        if not events:
            events.append("start")
            return 10.0
        events.append("stop")
        return 13.0

    monkeypatch.setattr(baselines, "time", SimpleNamespace(perf_counter=clock))
    original_chord = baselines.chord_length_parameters
    original_sparse = baselines.fit_sparse_knots_paper
    original_refit = baselines.refit_bspline_control_points

    def chord(*args, **kwargs):
        events.append("parameterization")
        return original_chord(*args, **kwargs)

    def sparse(*args, **kwargs):
        events.append("search")
        assert kwargs["feasibility_repair"] is False
        return original_sparse(*args, **kwargs)

    def refit(*args, **kwargs):
        events.append("common_refit")
        return original_refit(*args, **kwargs)

    monkeypatch.setattr(baselines, "chord_length_parameters", chord)
    monkeypatch.setattr(baselines, "fit_sparse_knots_paper", sparse)
    monkeypatch.setattr(baselines, "refit_bspline_control_points", refit)
    result = _run("kang_sparse_2015_adaptation")

    assert events == ["start", "parameterization", "search", "common_refit", "stop"]
    assert result.elapsed_ms == 3000.0


def test_gradient_baseline_still_optimizes_inside_inference_mode() -> None:
    with torch.inference_mode():
        result = _run("uniform_gradient_pruning")
    assert result.diagnostics["variable_projection_evaluations"] > 0
    assert torch.isfinite(result.fit.fit_mse)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mse_tolerance": -1.0},
        {"mse_tolerance": float("nan")},
        {"max_internal_knots": -1},
        {"max_internal_knots": True},
        {"degree": 0},
        {"gradient_steps": -1},
        {"paper_initial_knots": -1},
        {"paper_admm_iterations": 0},
        {"paper_lambda_bisections": 0},
        {"paper_relocation_iterations": 0},
        {"park_shape_weight": -0.1},
        {"park_shape_weight": float("nan")},
        {"liang_dense_knots": -1},
        {"liang_initial_knots": -1},
        {"liang_curvature_weight": 1.1},
        {"liang_feature_samples": 2},
        {"dung_max_error": -1.0},
        {"dung_max_error": float("nan")},
        {"dung_scan_intervals": 0},
        {"dung_optimization_iterations": -1},
        {"luo_eta": -0.1},
        {"luo_de_population": 4},
        {"luo_de_iterations": -1},
        {"luo_seed": -1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _run("uniform_gradient_pruning", **kwargs)


def test_liang_initial_knots_cannot_exceed_dispatcher_cap() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _run("liang_feature_iki_2017_adaptation", liang_initial_knots=3)


def test_unknown_method_and_wrong_numeric_dtype_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        _run("not-a-method")
    with pytest.raises(ValueError, match="CPU float64"):
        baselines.run_published_baseline(
            "yeh_feature_cdf_2020", _points().float(), mse_tolerance=1e-4
        )
