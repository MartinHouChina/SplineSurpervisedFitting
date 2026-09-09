from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)
from spline_fitting.evaluation.park_dominant_point import (  # noqa: E402
    _knots_from_dominant_parameters,
    _local_curvature_maxima,
    fit_park_dominant_points,
)


def _points() -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    return torch.stack([t, 0.2 * torch.sin(3.0 * torch.pi * t)], dim=-1)


def test_eq4_averages_degree_consecutive_dominant_parameters() -> None:
    parameters = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    knots, clamp_count = _knots_from_dominant_parameters(
        parameters, [0, 1, 3, 5, 7, 8], degree=3
    )
    expected = torch.tensor(
        [
            (1.0 + 3.0 + 5.0) / 24.0,
            (3.0 + 5.0 + 7.0) / 24.0,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(knots, expected)
    assert clamp_count == 0


def test_lcm_seeds_use_strict_local_maxima_and_mean_over_four_bound() -> None:
    curvatures = torch.tensor([0.0, 2.0, 1.0, 5.0, 1.0, 0.0], dtype=torch.float64)
    # Both maxima exceed mean(curvature)/4 and are sorted by significance.
    assert _local_curvature_maxima(curvatures) == [3, 1]


def test_result_uses_common_refit_and_audits_metric_deviations() -> None:
    points = _points()
    result = fit_park_dominant_points(
        points,
        mse_tolerance=1e-5,
        max_internal_knots=5,
    )
    reference = refit_bspline_control_points(
        result.parameters,
        points,
        result.knots,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )

    torch.testing.assert_close(result.final_fit.fit_mse, reference.fit_mse)
    torch.testing.assert_close(
        result.final_fit.reconstructed_points[[0, -1]], points[[0, -1]]
    )
    assert result.refit_count == len(result.scanned_knot_counts)
    assert result.refit_count == len(result.scanned_mse)
    assert result.refit_count == len(result.scanned_max_parameter_residual)
    assert result.scanned_knot_counts == tuple(range(result.K + 1))
    assert 0 <= result.K <= 5
    assert result.diagnostics["network_used"] is False
    assert "maximum-distance" in str(result.diagnostics["distance_metric_deviation"])
    assert "orthogonal-distance" in str(result.diagnostics["distance_metric_deviation"])
    assert result.diagnostics["threshold_satisfied"] == result.threshold_satisfied


def test_loose_tolerance_returns_zero_internal_knots() -> None:
    result = fit_park_dominant_points(
        _points(), mse_tolerance=1.0, max_internal_knots=6
    )
    assert result.K == 0
    assert result.scanned_knot_counts == (0,)
    assert result.refit_count == 1


def test_identical_points_are_deterministic_and_finite() -> None:
    points = torch.zeros((12, 2), dtype=torch.float64)
    first = fit_park_dominant_points(
        points, mse_tolerance=0.0, max_internal_knots=3
    )
    second = fit_park_dominant_points(
        points, mse_tolerance=0.0, max_internal_knots=3
    )
    assert torch.isfinite(first.final_fit.fit_mse)
    torch.testing.assert_close(first.knots, second.knots)
    torch.testing.assert_close(
        first.parameters, torch.linspace(0.0, 1.0, 12, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "points, kwargs, error",
    [
        (_points().float(), {}, "CPU float64"),
        (_points(), {"degree": 0}, "degree"),
        (_points(), {"mse_tolerance": -1.0}, "mse_tolerance"),
        (_points(), {"max_internal_knots": -1}, "max_internal_knots"),
        (_points(), {"shape_weight": 1.1}, "shape_weight"),
    ],
)
def test_invalid_inputs_are_rejected(
    points: torch.Tensor, kwargs: dict[str, object], error: str
) -> None:
    options: dict[str, object] = {
        "mse_tolerance": 1e-4,
        "max_internal_knots": 3,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=error):
        fit_park_dominant_points(points, **options)
