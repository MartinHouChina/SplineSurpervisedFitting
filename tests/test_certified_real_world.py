from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import (  # noqa: E402
    bspline_basis_matrix,
)
from spline_fitting.evaluation.certified_real_world import (  # noqa: E402
    certify_reference_mse,
    exact_polyline_bspline_fit,
    simplify_polyline_to_mse,
)
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector  # noqa: E402


def _zigzag(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    points = torch.stack(
        [parameters, 0.4 * torch.sin(7.0 * torch.pi * parameters)], dim=-1
    )
    return parameters, points


def test_exact_polyline_fallback_represents_edges_not_only_vertices() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [0.25, 0.8], [0.8, -0.2], [1.0, 0.4]],
        dtype=torch.float64,
    )
    lengths = (points[1:] - points[:-1]).norm(dim=-1)
    parameters = torch.cat(
        [torch.zeros(1, dtype=points.dtype), torch.cumsum(lengths / lengths.sum(), 0)]
    )
    fit = exact_polyline_bspline_fit(parameters, points, degree=3)

    queries = []
    expected = []
    for index in range(points.shape[0] - 1):
        for fraction in (0.2, 0.5, 0.8):
            queries.append(
                (1.0 - fraction) * parameters[index]
                + fraction * parameters[index + 1]
            )
            expected.append(
                (1.0 - fraction) * points[index] + fraction * points[index + 1]
            )
    torch.testing.assert_close(
        fit.evaluate(torch.stack(queries)),
        torch.stack(expected),
        atol=2e-12,
        rtol=0.0,
    )
    assert fit.internal_knots.numel() == 3 * (points.shape[0] - 2)
    assert fit.control_points.shape[0] == 3 * (points.shape[0] - 1) + 1


def test_interpolation_level_fallback_is_measured_and_full_rank() -> None:
    parameters, points = _zigzag(12)
    result = certify_reference_mse(
        parameters,
        points,
        parameters.new_empty(0),
        target_mse=1e-20,
        maximum_internal_knots=8,
        position_sweeps=0,
    )

    assert result.status == "interpolation_fallback_pass"
    assert result.certificate_valid
    assert result.full_column_rank is True
    assert result.final_count == 8
    assert result.final_mse <= 1e-20
    assert not result.continuous_curve_guaranteed
    assert not result.globally_minimal_knot_count_guaranteed


def test_polyline_fallback_is_explicit_and_may_exceed_adaptive_limit() -> None:
    parameters, points = _zigzag(10)
    result = certify_reference_mse(
        parameters,
        points,
        parameters.new_empty(0),
        target_mse=1e-20,
        maximum_internal_knots=1,
        position_sweeps=0,
        insertion_candidates=3,
        interpolation_fallback=False,
        tolerance_polyline_fallback=False,
        polyline_exact_fallback=True,
    )

    assert result.status == "polyline_exact_fallback_pass"
    assert result.certificate_valid
    assert result.full_column_rank is None
    assert result.polyline_exact_fallback_attempted
    assert result.polyline_exact_fallback_used
    assert result.polyline_fallback_exceeded_adaptive_maximum
    assert result.final_count == 3 * (points.shape[0] - 2)
    assert result.final_control_point_count == 3 * (points.shape[0] - 1) + 1
    assert result.maximum_internal_knot_multiplicity == 3
    assert result.final_mse <= 1e-20


def test_disabled_fallback_reports_target_not_reached_within_limits() -> None:
    parameters, points = _zigzag(20)
    result = certify_reference_mse(
        parameters,
        points,
        parameters.new_empty(0),
        target_mse=1e-16,
        maximum_internal_knots=1,
        position_sweeps=0,
        interpolation_fallback=False,
        tolerance_polyline_fallback=False,
        polyline_exact_fallback=False,
    )

    # With only one internal knot this oscillatory curve cannot meet the bound.
    assert result.status == "not_reached_within_limits"
    assert not result.threshold_satisfied
    assert not result.certificate_valid
    assert result.failure_reason is not None


def test_joint_position_refinement_never_worsens_reference_mse() -> None:
    parameters = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    true_knots = torch.tensor([0.75], dtype=torch.float64)
    knot_vector = build_open_knot_vector(true_knots, degree=3)
    control = torch.tensor(
        [[0.0, 0.0], [0.2, 0.9], [0.5, -0.7], [0.8, 0.8], [1.0, 0.0]],
        dtype=torch.float64,
    )
    points = bspline_basis_matrix(parameters, knot_vector, 3, 5) @ control
    result = certify_reference_mse(
        parameters,
        points,
        torch.tensor([0.25], dtype=torch.float64),
        target_mse=0.0,
        maximum_internal_knots=1,
        position_sweeps=2,
        position_grid_size=5,
        position_restarts=2,
        interpolation_fallback=False,
        tolerance_polyline_fallback=False,
        polyline_exact_fallback=False,
    )

    assert result.position_refinement_used
    assert result.final_mse <= result.initial_mse + 1e-15


def test_tolerance_polyline_is_certified_and_smaller_than_full_fallback() -> None:
    parameters = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    points = torch.stack(
        [
            parameters,
            0.18 * torch.sin(4.0 * torch.pi * parameters)
            + 0.03 * torch.sin(13.0 * torch.pi * parameters),
        ],
        dim=-1,
    )
    target_mse = 1e-5
    simplified = simplify_polyline_to_mse(
        parameters,
        points,
        target_mse,
        compact=True,
    )

    assert simplified.measured_mse <= target_mse
    assert 2 < simplified.retained_vertex_count < points.shape[0] // 4
    retained = simplified.retained_indices
    fit = exact_polyline_bspline_fit(
        parameters[retained],
        points[retained],
        degree=3,
        evaluation_parameters=parameters,
        evaluation_points=points,
    )
    assert float(fit.fit_mse) <= target_mse
    assert fit.internal_knots.numel() == 3 * (simplified.retained_vertex_count - 2)

    result = certify_reference_mse(
        parameters,
        points,
        parameters.new_empty(0),
        target_mse=target_mse,
        maximum_internal_knots=1,
        position_sweeps=0,
        insertion_candidates=3,
        interpolation_fallback=False,
        tolerance_polyline_fallback=True,
        polyline_exact_fallback=True,
    )
    assert result.status == "tolerance_polyline_fallback_pass"
    assert result.certificate_valid
    assert result.tolerance_polyline_fallback_attempted
    assert result.tolerance_polyline_fallback_used
    assert not result.polyline_exact_fallback_attempted
    assert result.tolerance_polyline_vertex_count == simplified.retained_vertex_count
    assert result.final_count < 3 * (points.shape[0] - 2)
