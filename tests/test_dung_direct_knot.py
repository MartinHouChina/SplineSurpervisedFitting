from __future__ import annotations

# ruff: noqa: E402

import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.dung_direct_knot import fit_dung_direct_knots


def _oscillating_curve(sample_count: int = 49) -> torch.Tensor:
    parameter = torch.linspace(0.0, 1.0, sample_count, dtype=torch.float64)
    return torch.stack(
        [parameter, 0.25 * torch.sin(6.0 * torch.pi * parameter)], dim=-1
    )


def test_straight_curve_uses_native_max_error_and_common_endpoint_refit() -> None:
    parameter = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    points = torch.stack([parameter, 2.0 * parameter - 0.3], dim=-1)
    result = fit_dung_direct_knots(
        points,
        mse_tolerance=4e-6,
        max_internal_knots=5,
        scan_intervals=3,
        optimization_iterations=2,
    )

    assert result.native_max_error_tolerance == pytest.approx(math.sqrt(4e-6))
    assert result.segments == ((0, 16),)
    assert result.coarse_break_indices == ()
    assert result.knots.numel() == 0
    assert result.scan_evaluations == result.optimization_evaluations == 0
    assert result.capacity_handling == "none"
    assert result.diagnostics["native_control_norm"] == "maximum Euclidean point residual"
    torch.testing.assert_close(result.final_fit.reconstructed_points[0], points[0])
    torch.testing.assert_close(result.final_fit.reconstructed_points[-1], points[-1])


def test_serial_bisection_and_simple_knot_relocation_are_deterministic() -> None:
    points = _oscillating_curve()
    options = dict(
        mse_tolerance=1e-6,
        max_error=2e-3,
        max_internal_knots=12,
        scan_intervals=3,
        optimization_iterations=2,
    )
    first = fit_dung_direct_knots(points, **options)
    second = fit_dung_direct_knots(points, **options)

    assert len(first.segments) > 1
    assert first.proposed_internal_knot_count == len(first.segments) - 1
    assert first.scan_evaluations == (3 + 1) * first.knots.numel()
    assert first.optimization_evaluations >= first.knots.numel()
    assert torch.all(first.knots > 0.0) and torch.all(first.knots < 1.0)
    assert torch.all(first.knots[1:] >= first.knots[:-1])
    assert first.segments == second.segments
    assert first.coarse_break_indices == second.coarse_break_indices
    assert first.scan_evaluations == second.scan_evaluations
    assert first.optimization_evaluations == second.optimization_evaluations
    torch.testing.assert_close(first.parameters, second.parameters, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.knots, second.knots, rtol=0.0, atol=0.0)


def test_capacity_overflow_is_explicit_and_uniformly_subsampled() -> None:
    result = fit_dung_direct_knots(
        _oscillating_curve(57),
        mse_tolerance=1e-8,
        max_error=1e-4,
        max_internal_knots=1,
        scan_intervals=2,
        optimization_iterations=1,
    )

    assert result.proposed_internal_knot_count > 1
    assert result.capacity_exceeded
    assert result.capacity_handling == "uniform_coarse_break_subsample"
    assert result.knots.numel() == 1
    assert result.diagnostics["retained_internal_knot_count"] == 1
    assert "capacity overflow" in " ".join(result.adaptation_limits)
    assert torch.isfinite(result.final_fit.fit_mse)


@pytest.mark.parametrize(
    ("points", "kwargs", "message"),
    [
        (torch.zeros(8, 2), {}, "CPU float64"),
        (
            torch.zeros(3, 2, dtype=torch.float64),
            {},
            "degree",
        ),
        (
            torch.zeros(8, 2, dtype=torch.float64),
            {"max_internal_knots": -1},
            "max_internal_knots",
        ),
        (
            torch.zeros(8, 2, dtype=torch.float64),
            {"scan_intervals": 0},
            "scan_intervals",
        ),
    ],
)
def test_invalid_inputs_are_rejected(points, kwargs, message) -> None:
    options = dict(mse_tolerance=1e-5, max_internal_knots=2)
    options.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        fit_dung_direct_knots(points, **options)
