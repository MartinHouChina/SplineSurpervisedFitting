from __future__ import annotations

import torch

from .truncated_power_basis import build_design_matrix


def reconstruct_from_design(
    design_matrix: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    return design_matrix @ coefficients


def evaluate_curve(
    params: torch.Tensor,
    internal_knots: torch.Tensor,
    activity: torch.Tensor,
    coefficients: torch.Tensor,
    degree: int,
    eps: float = 1e-6,
    gate_transform: str = "sqrt",
) -> torch.Tensor:
    design = build_design_matrix(
        params,
        internal_knots,
        activity,
        degree,
        eps,
        gate_transform=gate_transform,
    )["design_matrix"]
    return reconstruct_from_design(design, coefficients)


def sample_curve(
    internal_knots: torch.Tensor,
    activity: torch.Tensor,
    coefficients: torch.Tensor,
    degree: int,
    num_samples: int = 256,
    eps: float = 1e-6,
    gate_transform: str = "sqrt",
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = internal_knots.shape[0]
    params = torch.linspace(
        0.0,
        1.0,
        num_samples,
        device=internal_knots.device,
        dtype=internal_knots.dtype,
    ).unsqueeze(0).expand(batch, -1)
    curve = evaluate_curve(
        params,
        internal_knots,
        activity,
        coefficients,
        degree,
        eps,
        gate_transform=gate_transform,
    )
    return params, curve
