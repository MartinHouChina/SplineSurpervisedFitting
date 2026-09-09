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
from spline_fitting.evaluation.liang_feature_iki import (  # noqa: E402
    fit_liang_feature_iki,
    normalized_arc_curvature_feature,
    place_feature_quantile_knots,
)


def _bending_curve(count: int = 81, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    parameter = torch.linspace(0.0, 1.0, count, dtype=dtype)
    return torch.stack(
        [
            parameter,
            0.18 * torch.sin(3.0 * torch.pi * parameter)
            + 0.04 * torch.sin(9.0 * torch.pi * parameter),
        ],
        dim=-1,
    )


def test_feature_cdf_is_monotone_normalized_and_quantile_invertible() -> None:
    parameter = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    quarter_circle = torch.stack(
        [torch.cos(0.5 * torch.pi * parameter), torch.sin(0.5 * torch.pi * parameter)],
        dim=-1,
    )
    arc, curvature, feature = normalized_arc_curvature_feature(
        parameter,
        quarter_circle,
        curvature_weight=0.5,
    )
    knots = place_feature_quantile_knots(parameter, feature, 5)

    for cumulative in (arc, curvature, feature):
        assert torch.isfinite(cumulative).all()
        assert torch.all(cumulative[1:] >= cumulative[:-1])
    assert arc[0] == 0.0 and arc[-1] == 1.0
    assert curvature[0] == 0.0 and curvature[-1] == 1.0
    assert feature[0] == 0.0 and feature[-1] == 1.0
    torch.testing.assert_close(
        knots,
        torch.arange(1, 6, dtype=torch.float64) / 6.0,
        # The unsigned vertex turns are split over adjacent sample intervals,
        # so the finite-grid approximation is close to, but not exactly, the
        # continuous constant-curvature quantiles.
        atol=2e-3,
        rtol=0.0,
    )


def test_zero_curvature_falls_back_to_arc_mass() -> None:
    parameter = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    line = torch.stack([2.0 * parameter, torch.zeros_like(parameter)], dim=-1)
    arc, curvature, feature = normalized_arc_curvature_feature(
        parameter,
        line,
        curvature_weight=1.0,
    )

    torch.testing.assert_close(arc, parameter)
    torch.testing.assert_close(curvature, torch.zeros_like(parameter))
    torch.testing.assert_close(feature, arc)


def test_liang_iki_is_auditable_and_uses_common_final_refit() -> None:
    points = _bending_curve()
    result = fit_liang_feature_iki(
        points,
        degree=3,
        dense_internal_knot_count=12,
        feature_sample_count=257,
        initial_internal_knot_count=1,
        max_internal_knots=5,
        mse_tolerance=0.0,
    )

    assert result.parameters.dtype == torch.float64
    assert result.final_fit.control_points.dtype == torch.float64
    assert result.dense_count == 12
    assert result.dense_knots.numel() == 12
    assert result.initial_knots.numel() == 1
    assert result.inserted_knots.numel() == 4
    assert result.K == 5
    assert result.refit_count == 1 + len(result.scanned_mse)
    assert result.iki_refit_count == len(result.scanned_mse)
    assert result.scanned_internal_knot_counts == (1, 2, 3, 4, 5)
    assert len(result.insertion_spans) == result.inserted_knots.numel()
    assert len(result.scanned_worst_point_indices) == len(result.scanned_mse)
    assert all(
        later <= earlier + 1e-12
        for earlier, later in zip(result.scanned_mse, result.scanned_mse[1:])
    )

    for inserted, (left, right) in zip(
        result.inserted_knots.tolist(), result.insertion_spans
    ):
        assert inserted == pytest.approx(0.5 * (left + right), abs=1e-15)

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
    torch.testing.assert_close(result.final_fit.control_points, reference.control_points)
    torch.testing.assert_close(
        result.final_fit.reconstructed_points[[0, -1]], points[[0, -1]]
    )
    torch.testing.assert_close(
        result.final_fit.augmented_objective,
        result.final_fit.data_squared_error,
    )


def test_large_tolerance_stops_at_initial_feature_knots() -> None:
    result = fit_liang_feature_iki(
        _bending_curve(),
        mse_tolerance=1.0,
        max_internal_knots=7,
        dense_internal_knot_count=10,
        initial_internal_knot_count=2,
        feature_sample_count=129,
    )

    assert result.threshold_satisfied
    assert result.inserted_knots.numel() == 0
    assert result.scanned_internal_knot_counts == (2,)
    assert result.refit_count == 2
    torch.testing.assert_close(result.knots, result.initial_knots)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mse_tolerance": -1.0, "max_internal_knots": 4}, "mse_tolerance"),
        ({"mse_tolerance": 1e-4, "max_internal_knots": -1}, "max_internal_knots"),
        (
            {
                "mse_tolerance": 1e-4,
                "max_internal_knots": 1,
                "initial_internal_knot_count": 2,
            },
            "must not exceed",
        ),
        (
            {
                "mse_tolerance": 1e-4,
                "max_internal_knots": 4,
                "curvature_weight": 1.1,
            },
            "curvature_weight",
        ),
    ],
)
def test_invalid_liang_configuration_is_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_liang_feature_iki(_bending_curve(), **kwargs)


def test_non_float64_points_are_rejected() -> None:
    with pytest.raises(ValueError, match="CPU float64"):
        fit_liang_feature_iki(
            _bending_curve(dtype=torch.float32),
            mse_tolerance=1e-4,
            max_internal_knots=4,
        )
