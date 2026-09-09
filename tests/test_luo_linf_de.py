from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.luo_linf_de import (  # noqa: E402
    _prox_linf_rows,
    fit_luo_linf_de,
)


def _curve() -> tuple[torch.Tensor, torch.Tensor]:
    parameters = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    points = torch.stack(
        [parameters, 0.12 * torch.sin(2.0 * torch.pi * parameters)], dim=-1
    )
    return parameters, points


def test_linf_prox_obeys_moreau_decomposition() -> None:
    values = torch.tensor(
        [[3.0, -1.0], [0.2, -0.1], [-2.0, 2.0]], dtype=torch.float64
    )
    prox = _prox_linf_rows(values, 1.0)
    l1_projection = values - prox

    assert torch.all(l1_projection.abs().sum(dim=-1) <= 1.0 + 1e-12)
    torch.testing.assert_close(prox[1], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(prox[0], torch.tensor([2.0, -1.0], dtype=torch.float64))


def test_luo_two_stage_adaptation_is_deterministic_and_refits_endpoints() -> None:
    parameters, points = _curve()
    options = dict(
        degree=3,
        initial_internal_knot_count=6,
        mse_tolerance=1e-4,
        admm_max_iterations=8,
        lambda_bisection_iterations=1,
        de_population=5,
        de_iterations=1,
        seed=17,
    )
    first = fit_luo_linf_de(parameters, points, **options)
    second = fit_luo_linf_de(parameters, points, **options)

    torch.testing.assert_close(first.knots, second.knots)
    torch.testing.assert_close(first.final_fit.fit_mse, second.final_fit.fit_mse)
    torch.testing.assert_close(
        first.final_fit.reconstructed_points[[0, -1]], points[[0, -1]]
    )
    assert first.de_evaluations == 10
    assert first.candidate_knots.numel() == first.optimized_knots.numel()
    assert first.de_final_max_error <= first.de_initial_max_error + 1e-12


def test_luo_rejects_too_small_de_population() -> None:
    parameters, points = _curve()
    try:
        fit_luo_linf_de(
            parameters,
            points,
            initial_internal_knot_count=4,
            mse_tolerance=1e-4,
            de_population=4,
        )
    except ValueError as error:
        assert "at least 5" in str(error)
    else:
        raise AssertionError("expected a ValueError")
