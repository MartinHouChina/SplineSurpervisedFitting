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
from spline_fitting.losses import (  # noqa: E402
    differentiable_hard_gated_bspline_fit,
)


def test_differentiable_refit_matches_production_refit_and_has_gradients() -> None:
    base_parameters = torch.linspace(0.0, 1.0, 48)
    points = torch.stack(
        [base_parameters, 0.25 * torch.sin(5.0 * base_parameters)], dim=-1
    )
    parameters = base_parameters.unsqueeze(0).clone().requires_grad_()
    candidates = torch.tensor(
        [[0.18, 0.37, 0.61, 0.82]], dtype=torch.float32, requires_grad=True
    )
    keep_mask = torch.tensor([[True, False, True, True]])

    result = differentiable_hard_gated_bspline_fit(
        parameters,
        points.unsqueeze(0),
        candidates,
        keep_mask,
        degree=3,
        smoothness_weight=1e-6,
        solver_jitter=1e-8,
    )
    production = refit_bspline_control_points(
        base_parameters,
        points,
        candidates.detach()[0, keep_mask[0]],
        degree=3,
        smoothness_weight=1e-6,
        interpolate_endpoints=True,
    )

    assert result["fit_mse"].item() == pytest.approx(
        production.fit_mse.item(), rel=2e-3, abs=1e-8
    )
    result["fit_mse"].backward()
    assert parameters.grad is not None
    assert float(parameters.grad.abs().sum()) > 0.0
    assert candidates.grad is not None
    assert float(candidates.grad[keep_mask].abs().sum()) > 0.0
    assert float(candidates.grad[~keep_mask].abs().sum()) == 0.0


def test_differentiable_refit_supports_zero_internal_knots() -> None:
    parameters = torch.linspace(0.0, 1.0, 24).unsqueeze(0)
    points = torch.stack(
        [parameters[0], parameters[0].square()], dim=-1
    ).unsqueeze(0)
    result = differentiable_hard_gated_bspline_fit(
        parameters,
        points,
        torch.tensor([[0.25, 0.75]]),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    assert torch.isfinite(result["fit_mse"])
    assert result["retained_count"].item() == 0.0
