from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve  # noqa: E402
from spline_fitting.evaluation.gradient_knot_pruning import (  # noqa: E402
    gradient_knot_pruning_baseline,
)
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector  # noqa: E402


DTYPE = torch.float64


def test_cubic_polynomial_curve_prunes_all_knots() -> None:
    # A straight line is a cubic polynomial curve under its chord-length
    # parameterization, so the cubic Bezier (K=0) fit is exact.
    parameter = torch.linspace(0.0, 1.0, 41, dtype=DTYPE)
    points = torch.stack([parameter, 0.2 + 2.0 * parameter], dim=-1)

    result = gradient_knot_pruning_baseline(
        points,
        Kmax=4,
        mse_tolerance=1e-20,
        optimization_steps=3,
    )

    assert result.K == 0
    assert result.threshold_satisfied
    assert result.final_internal_knots.numel() == 0
    assert result.final_fit.fit_mse.dtype == torch.float64
    assert result.accepted_deletions == 4


def test_final_knots_are_ordered_and_respect_minimum_gap() -> None:
    parameter = torch.linspace(0.0, 1.0, 81, dtype=DTYPE)
    controls = torch.tensor(
        [[0.0, 0.0], [0.1, 0.8], [0.3, -0.7], [0.6, 0.8], [0.85, -0.6], [1.0, 0.0]],
        dtype=DTYPE,
    )
    points = evaluate_bspline_curve(
        parameter,
        controls,
        build_open_knot_vector(torch.tensor([0.32, 0.70], dtype=DTYPE), 3),
        3,
    )
    min_gap = 0.025

    result = gradient_knot_pruning_baseline(
        points,
        max_internal_knots=4,
        mse_tolerance=0.0025,
        min_gap=min_gap,
        optimization_steps=12,
        learning_rate=0.03,
    )

    boundaries = torch.cat(
        [result.knots.new_zeros(1), result.knots, result.knots.new_ones(1)]
    )
    assert torch.all(boundaries[1:] - boundaries[:-1] >= min_gap - 1e-12)
    assert result.threshold_satisfied
    assert result.refit_count > 1
    assert result.evaluation_count == result.gradient_steps
    assert result.total_search_time >= 0.0


def test_gradient_relocation_moves_surviving_knot_positions() -> None:
    parameter = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)
    controls = torch.tensor(
        [[0.0, 0.0], [0.1, 0.8], [0.3, -0.7], [0.6, 0.8], [0.85, -0.6], [1.0, 0.0]],
        dtype=DTYPE,
    )
    points = evaluate_bspline_curve(
        parameter,
        controls,
        build_open_knot_vector(torch.tensor([0.29, 0.71], dtype=DTYPE), 3),
        3,
    )

    result = gradient_knot_pruning_baseline(
        points,
        max_internal_knots=3,
        mse_tolerance=0.01,
        optimization_steps=25,
        learning_rate=0.03,
    )

    first = result.steps[0]
    assert first.accepted
    assert any(first.candidate_location_updates_accepted)
    selected_knots = first.candidate_updated_knots[first.selected_index]
    uniform_survivor = torch.tensor([0.5, 0.75], dtype=DTYPE)
    assert not torch.allclose(selected_knots, uniform_survivor)
    assert result.gradient_steps > 0
    assert result.threshold_satisfied


def test_full_uniform_kmax_is_relocated_before_any_deletion() -> None:
    parameter = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)
    controls = torch.tensor(
        [[0.0, 0.0], [0.1, 0.9], [0.35, -0.8], [0.7, 0.7], [1.0, 0.0]],
        dtype=DTYPE,
    )
    points = evaluate_bspline_curve(
        parameter,
        controls,
        build_open_knot_vector(torch.tensor([0.27], dtype=DTYPE), 3),
        3,
    )

    result = gradient_knot_pruning_baseline(
        points,
        max_internal_knots=1,
        min_internal_knots=1,
        mse_tolerance=1.0,
        optimization_steps=30,
        learning_rate=0.03,
    )

    assert result.initial_location_update_accepted
    assert float(result.initial_relocated_fit.fit_mse) <= float(
        result.initial_fit.fit_mse
    )
    assert not torch.allclose(
        result.initial_relocated_fit.internal_knots,
        result.initial_fit.internal_knots,
    )
    assert result.final_fit is result.initial_relocated_fit
