from __future__ import annotations

import torch


def coefficient_drop_objective_delta(
    coefficients: torch.Tensor,
    normal_matrix: torch.Tensor,
    *,
    first_column: int = 0,
) -> torch.Tensor:
    """Return the exact quadratic-objective increase from dropping each column.

    For a ridge quadratic with optimum ``D*`` and Hessian/normal matrix ``A``,
    constraining coefficient row ``j`` to zero while re-optimizing every other
    row increases the objective by

    ``sum_d D*[j,d]^2 / (A^{-1})[j,j]``.

    The returned tensor has shape ``[B, C-first_column]``.  This is more
    informative than a raw coefficient magnitude because it accounts for
    column scaling and correlations through the inverse normal matrix.
    """
    if coefficients.ndim != 3 or normal_matrix.ndim != 3:
        raise ValueError("coefficients and normal_matrix must be rank-3 tensors")
    if normal_matrix.shape[-1] != normal_matrix.shape[-2]:
        raise ValueError("normal_matrix must be square")
    if coefficients.shape[:2] != normal_matrix.shape[:2]:
        raise ValueError("coefficients and normal_matrix must share [B, C]")
    if not 0 <= first_column <= coefficients.shape[1]:
        raise ValueError("first_column is outside the coefficient range")

    columns = normal_matrix.shape[-1]
    identity = torch.eye(
        columns,
        device=normal_matrix.device,
        dtype=normal_matrix.dtype,
    ).unsqueeze(0).expand(normal_matrix.shape[0], -1, -1)
    inverse = torch.linalg.solve(normal_matrix, identity)
    inverse_diagonal = torch.diagonal(inverse, dim1=-2, dim2=-1)
    numerator = coefficients.pow(2).sum(dim=-1)
    tiny = torch.finfo(normal_matrix.dtype).tiny
    delta = numerator / inverse_diagonal.clamp_min(tiny)
    return delta[:, first_column:]


def build_regularization_matrix(
    batch_size: int,
    degree: int,
    num_knots: int,
    lambda_poly: float,
    lambda_knot: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    diagonal = torch.cat(
        [
            torch.full((degree + 1,), lambda_poly, device=device, dtype=dtype),
            torch.full((num_knots,), lambda_knot, device=device, dtype=dtype),
        ]
    )
    return torch.diag(diagonal).unsqueeze(0).expand(batch_size, -1, -1)


def solve_coefficients(
    design_matrix: torch.Tensor,
    points: torch.Tensor,
    degree: int,
    lambda_poly: float = 1e-6,
    lambda_knot: float = 1e-4,
    jitter: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Differentiable ridge solve for D*=[c;d]."""
    if design_matrix.ndim != 3 or points.ndim != 3:
        raise ValueError("design_matrix and points must be batched rank-3 tensors")
    if design_matrix.shape[:2] != points.shape[:2]:
        raise ValueError("design_matrix and points must share [B, M]")

    batch_size = design_matrix.shape[0]
    num_columns = design_matrix.shape[-1]
    num_knots = num_columns - (degree + 1)
    phi_t = design_matrix.transpose(-1, -2)
    regularization = build_regularization_matrix(
        batch_size,
        degree,
        num_knots,
        lambda_poly,
        lambda_knot,
        design_matrix.device,
        design_matrix.dtype,
    )
    identity = torch.eye(num_columns, device=design_matrix.device, dtype=design_matrix.dtype)
    normal_matrix = phi_t @ design_matrix + regularization + jitter * identity.unsqueeze(0)
    right_hand_side = phi_t @ points
    coefficients = torch.linalg.solve(normal_matrix, right_hand_side)

    return {
        "coefficients": coefficients,
        "polynomial_coefficients": coefficients[:, : degree + 1],
        "increment_coefficients": coefficients[:, degree + 1 :],
        "normal_matrix": normal_matrix,
    }
