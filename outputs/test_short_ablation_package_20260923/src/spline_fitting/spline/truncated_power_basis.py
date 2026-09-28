from __future__ import annotations

import torch


def build_polynomial_basis(params: torch.Tensor, degree: int) -> torch.Tensor:
    """Return V=[1,t,...,t^p] with shape [B,M,p+1]."""
    if params.ndim != 2:
        raise ValueError("params must have shape [B, M]")
    return torch.stack([params.pow(power) for power in range(degree + 1)], dim=-1)


def build_truncated_power_basis(
    params: torch.Tensor,
    internal_knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Return H_ij=(t_i-u_j)_+^p with shape [B,M,K]."""
    if internal_knots.ndim != 2:
        raise ValueError("internal_knots must have shape [B, K]")
    difference = params.unsqueeze(-1) - internal_knots.unsqueeze(-2)
    return torch.relu(difference).pow(degree)


def build_design_matrix(
    params: torch.Tensor,
    internal_knots: torch.Tensor,
    activity: torch.Tensor,
    degree: int,
    eps: float = 1e-6,
    gate_transform: str = "sqrt",
) -> dict[str, torch.Tensor]:
    """Build a gated truncated-power design matrix.

    ``gate_transform='sqrt'`` preserves the legacy ``sqrt(a + eps)``
    parameterization. ``gate_transform='direct'`` treats ``activity`` as the
    already sampled gate and multiplies the increment columns by it directly.
    """
    if activity.shape != internal_knots.shape:
        raise ValueError("activity and internal_knots must have identical shape")
    if gate_transform not in {"sqrt", "direct"}:
        raise ValueError("gate_transform must be 'sqrt' or 'direct'")

    polynomial_basis = build_polynomial_basis(params, degree)
    increment_basis = build_truncated_power_basis(params, internal_knots, degree)
    if gate_transform == "sqrt":
        activity_gate = torch.sqrt(activity.clamp_min(0.0) + eps)
    else:
        activity_gate = activity
    weighted_increment_basis = increment_basis * activity_gate.unsqueeze(-2)
    design_matrix = torch.cat([polynomial_basis, weighted_increment_basis], dim=-1)

    return {
        "polynomial_basis": polynomial_basis,
        "increment_basis": increment_basis,
        "activity_gate": activity_gate,
        "weighted_increment_basis": weighted_increment_basis,
        "design_matrix": design_matrix,
    }
