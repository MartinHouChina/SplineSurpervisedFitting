from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve
from spline_fitting.evaluation.feature_cdf_knot_placement import (
    finite_difference_feature,
    fit_feature_cdf_to_tolerance,
    place_feature_cdf_knots,
)
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector


def test_feature_and_placed_knots_are_finite_ordered_and_bounded() -> None:
    parameters = torch.linspace(0.0, 1.0, 81, dtype=torch.float64)
    points = torch.stack([parameters, torch.sin(4.0 * torch.pi * parameters)], dim=-1)
    locations, feature = finite_difference_feature(
        parameters, points, derivative_order=4
    )
    knots = place_feature_cdf_knots(locations, feature, 12)

    assert locations.shape == feature.shape
    assert torch.isfinite(feature).all()
    assert torch.all(knots > 0.0) and torch.all(knots < 1.0)
    assert torch.all(knots[1:] > knots[:-1])


def test_cardinality_scan_finds_a_feasible_feature_cdf_fit() -> None:
    parameters = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    true_knots = torch.tensor([0.25, 0.58, 0.81], dtype=torch.float64)
    controls = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.7],
            [0.3, -0.2],
            [0.5, 0.9],
            [0.7, -0.4],
            [0.9, 0.5],
            [1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    points = evaluate_bspline_curve(
        parameters,
        controls,
        build_open_knot_vector(true_knots, degree=3),
        degree=3,
    )
    result = fit_feature_cdf_to_tolerance(
        parameters,
        points,
        mse_tolerance=5e-5,
        max_internal_knots=20,
    )

    assert result.threshold_satisfied
    assert 0 <= result.knots.numel() <= 20
    assert result.refit_count == len(result.scanned_counts)
    assert float(result.final_fit.fit_mse) <= 5e-5
