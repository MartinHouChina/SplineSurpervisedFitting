from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.spline.differentiable_solver import (  # noqa: E402
    coefficient_drop_objective_delta,
    solve_coefficients,
    solve_coefficients_and_drop_objective_delta,
)


def test_combined_solver_defaults_to_exact_legacy_composition() -> None:
    torch.manual_seed(21)
    design = torch.randn(3, 20, 7, dtype=torch.float64)
    points = torch.randn(3, 20, 2, dtype=torch.float64)

    expected_solver = solve_coefficients(
        design,
        points,
        degree=3,
        lambda_poly=1e-3,
        lambda_knot=2e-3,
    )
    expected_delta = coefficient_drop_objective_delta(
        expected_solver["coefficients"],
        expected_solver["normal_matrix"],
        first_column=4,
    )
    actual = solve_coefficients_and_drop_objective_delta(
        design,
        points,
        degree=3,
        lambda_poly=1e-3,
        lambda_knot=2e-3,
        first_column=4,
    )

    torch.testing.assert_close(
        actual["coefficients"], expected_solver["coefficients"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual["normal_matrix"], expected_solver["normal_matrix"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual["drop_objective_delta"], expected_delta, rtol=0.0, atol=0.0
    )
    assert not actual["factorization_used_cholesky"].any()
    assert actual["factorization_used_legacy"].all()


def test_cholesky_reuse_matches_legacy_on_well_conditioned_system() -> None:
    torch.manual_seed(22)
    design = torch.randn(4, 32, 9, dtype=torch.float64)
    points = torch.randn(4, 32, 3, dtype=torch.float64)
    expected = solve_coefficients_and_drop_objective_delta(
        design,
        points,
        degree=3,
        lambda_poly=0.2,
        lambda_knot=0.2,
        first_column=4,
    )
    actual = solve_coefficients_and_drop_objective_delta(
        design,
        points,
        degree=3,
        lambda_poly=0.2,
        lambda_knot=0.2,
        first_column=4,
        backend="cholesky_reuse",
    )

    assert actual["factorization_used_cholesky"].all()
    assert not actual["factorization_used_legacy"].any()
    torch.testing.assert_close(actual["coefficients"], expected["coefficients"])
    torch.testing.assert_close(
        actual["drop_objective_delta"], expected["drop_objective_delta"]
    )


def test_cholesky_reuse_falls_back_per_batch_element() -> None:
    # With lambda=-1, the first Gram matrix remains positive definite while
    # the zero-design second matrix is negative definite.  The latter is still
    # invertible, so it is a deterministic exercise of the LU fallback.
    design = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    points = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ],
        dtype=torch.float64,
    )
    expected = solve_coefficients_and_drop_objective_delta(
        design,
        points,
        degree=1,
        lambda_poly=-1.0,
        lambda_knot=0.0,
        jitter=0.0,
    )
    actual = solve_coefficients_and_drop_objective_delta(
        design,
        points,
        degree=1,
        lambda_poly=-1.0,
        lambda_knot=0.0,
        jitter=0.0,
        backend="cholesky_reuse",
    )

    torch.testing.assert_close(actual["coefficients"], expected["coefficients"])
    torch.testing.assert_close(
        actual["drop_objective_delta"], expected["drop_objective_delta"]
    )
    torch.testing.assert_close(
        actual["factorization_used_cholesky"], torch.tensor([True, False])
    )
    torch.testing.assert_close(
        actual["factorization_used_legacy"], torch.tensor([False, True])
    )


def test_cholesky_reuse_has_finite_gradients() -> None:
    torch.manual_seed(23)
    design = torch.randn(2, 16, 6, dtype=torch.float64, requires_grad=True)
    points = torch.randn(2, 16, 2, dtype=torch.float64, requires_grad=True)
    output = solve_coefficients_and_drop_objective_delta(
        design,
        points,
        degree=3,
        lambda_poly=0.1,
        lambda_knot=0.1,
        backend="cholesky_reuse",
    )

    (
        output["coefficients"].square().mean() + output["drop_objective_delta"].mean()
    ).backward()

    assert design.grad is not None and torch.isfinite(design.grad).all()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_combined_solver_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        solve_coefficients_and_drop_objective_delta(
            torch.ones(1, 4, 2),
            torch.ones(1, 4, 1),
            degree=1,
            backend="unknown",
        )
