from __future__ import annotations

import math

import torch


def _falling_factorial(n: int, order: int) -> float:
    if order > n:
        return 0.0
    return float(math.factorial(n) // math.factorial(n - order))


def build_derivative_design_matrix(
    params: torch.Tensor,
    internal_knots: torch.Tensor,
    activity: torch.Tensor,
    degree: int,
    order: int = 1,
    eps: float = 1e-6,
    gate_transform: str = "sqrt",
) -> torch.Tensor:
    """Build the design matrix of the requested curve derivative."""
    if order < 0:
        raise ValueError("order must be non-negative")
    if gate_transform not in {"sqrt", "direct"}:
        raise ValueError("gate_transform must be 'sqrt' or 'direct'")
    if order == 0:
        from .truncated_power_basis import build_design_matrix

        return build_design_matrix(
            params,
            internal_knots,
            activity,
            degree,
            eps,
            gate_transform=gate_transform,
        )["design_matrix"]

    polynomial_columns = []
    for power in range(degree + 1):
        factor = _falling_factorial(power, order)
        if factor == 0.0:
            polynomial_columns.append(torch.zeros_like(params))
        else:
            polynomial_columns.append(factor * params.pow(power - order))
    polynomial = torch.stack(polynomial_columns, dim=-1)

    difference = params.unsqueeze(-1) - internal_knots.unsqueeze(-2)
    if order > degree:
        increment = torch.zeros(
            *difference.shape,
            device=difference.device,
            dtype=difference.dtype,
        )
    else:
        factor = _falling_factorial(degree, order)
        increment = factor * torch.relu(difference).pow(degree - order)
    if gate_transform == "sqrt":
        gate = torch.sqrt(activity.clamp_min(0.0) + eps).unsqueeze(-2)
    else:
        gate = activity.unsqueeze(-2)
    return torch.cat([polynomial, increment * gate], dim=-1)


def evaluate_derivative(
    params: torch.Tensor,
    internal_knots: torch.Tensor,
    activity: torch.Tensor,
    coefficients: torch.Tensor,
    degree: int,
    order: int = 1,
    eps: float = 1e-6,
    gate_transform: str = "sqrt",
) -> torch.Tensor:
    matrix = build_derivative_design_matrix(
        params,
        internal_knots,
        activity,
        degree,
        order,
        eps,
        gate_transform=gate_transform,
    )
    return matrix @ coefficients
