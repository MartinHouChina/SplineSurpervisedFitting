from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ParameterHead(nn.Module):
    """Predict strictly increasing point parameters through positive gaps."""

    def __init__(
        self,
        hidden_dim: int = 128,
        min_gap: float = 1e-4,
        gap_parameterization: str = "strict",
        gap_reference: str = "learned",
        residual_logit_limit: float = 0.5,
    ) -> None:
        super().__init__()
        if gap_parameterization not in {"strict", "legacy"}:
            raise ValueError("gap_parameterization must be 'strict' or 'legacy'")
        if min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        if gap_reference not in {
            "learned",
            "chord_residual",
            "uniform_residual",
        }:
            raise ValueError(
                "gap_reference must be 'learned', 'chord_residual', or "
                "'uniform_residual'"
            )
        if (
            gap_reference in {"chord_residual", "uniform_residual"}
            and gap_parameterization != "strict"
        ):
            raise ValueError(
                f"{gap_reference} requires strict gap parameterization"
            )
        if not math.isfinite(residual_logit_limit) or residual_logit_limit <= 0.0:
            raise ValueError("residual_logit_limit must be finite and positive")
        self.min_gap = min_gap
        self.gap_parameterization = gap_parameterization
        self.gap_reference = str(gap_reference)
        self.residual_logit_limit = float(residual_logit_limit)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        if self.gap_reference in {"chord_residual", "uniform_residual"}:
            # Residual parameterizations start exactly at their chord/uniform
            # reference instead of a random parameter distribution. This is
            # constructor-only behavior: historical ``learned`` checkpoints
            # retain their state layout and load bit-for-bit as before.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        *,
        reference_params: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, num_points, _ = local_features.shape
        global_expanded = global_features.unsqueeze(1).expand(-1, num_points, -1)
        fused = torch.cat([local_features, global_expanded], dim=-1)

        raw_gaps = self.mlp(fused[:, :-1]).squeeze(-1)
        if self.gap_reference == "chord_residual":
            if reference_params is None:
                raise ValueError(
                    "reference_params are required for chord_residual gaps"
                )
            if reference_params.shape != (batch, num_points):
                raise ValueError("reference_params must have shape [B,M]")
            if reference_params.device != local_features.device:
                raise ValueError("reference_params must share the feature device")
            if reference_params.dtype != local_features.dtype:
                raise ValueError("reference_params must share the feature dtype")
            if torch.any(reference_params[:, 1:] < reference_params[:, :-1]):
                raise ValueError("reference_params must be non-decreasing")

            num_gaps = num_points - 1
            if self.min_gap * num_gaps >= 1.0:
                raise ValueError(
                    "min_gap is too large for the number of sampled point intervals"
                )
            reference_gaps = reference_params[:, 1:] - reference_params[:, :-1]
            # Work in the free interval budget so a zero network residual is
            # exactly the chord-length parameterization whenever its gaps
            # already satisfy ``min_gap``.  Degenerate/repeated point spans are
            # projected to the same strict feasible simplex used elsewhere.
            free_budget = 1.0 - self.min_gap * num_gaps
            reference_free = (reference_gaps - self.min_gap).clamp_min(0.0)
            empty_reference = reference_free.sum(dim=-1, keepdim=True) <= 1e-12
            reference_free = torch.where(
                empty_reference,
                torch.ones_like(reference_free),
                reference_free,
            )
            centered_residual_logits = raw_gaps - raw_gaps.mean(
                dim=-1,
                keepdim=True,
            )
            bounded_residual_logits = self.residual_logit_limit * torch.tanh(
                centered_residual_logits / self.residual_logit_limit
            )
            residual_weights = reference_free * torch.exp(bounded_residual_logits)
            normalized_gaps = self.min_gap + free_budget * residual_weights / (
                residual_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            )
        elif self.gap_reference == "uniform_residual":
            num_gaps = num_points - 1
            if self.min_gap * num_gaps >= 1.0:
                raise ValueError(
                    "min_gap is too large for the number of sampled point intervals"
                )
            centered_residual_logits = raw_gaps - raw_gaps.mean(
                dim=-1,
                keepdim=True,
            )
            bounded_residual_logits = self.residual_logit_limit * torch.tanh(
                centered_residual_logits / self.residual_logit_limit
            )
            free_budget = 1.0 - self.min_gap * num_gaps
            normalized_gaps = self.min_gap + free_budget * F.softmax(
                bounded_residual_logits,
                dim=-1,
            )
        elif self.gap_parameterization == "legacy":
            positive_gaps = F.softplus(raw_gaps) + self.min_gap
            normalized_gaps = positive_gaps / positive_gaps.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        else:
            num_gaps = num_points - 1
            if self.min_gap * num_gaps >= 1.0:
                raise ValueError(
                    "min_gap is too large for the number of sampled point intervals"
                )
            gap_weights = F.softmax(raw_gaps, dim=-1)
            free_budget = 1.0 - self.min_gap * num_gaps
            normalized_gaps = self.min_gap + free_budget * gap_weights

        params = torch.cat(
            [
                torch.zeros(batch, 1, device=local_features.device, dtype=local_features.dtype),
                torch.cumsum(normalized_gaps, dim=-1),
            ],
            dim=-1,
        )
        params[:, -1] = 1.0
        return {
            "params": params,
            "parameter_gaps": normalized_gaps,
            "raw_parameter_gaps": raw_gaps,
            "parameter_reference_params": (
                reference_params
                if reference_params is not None
                else params.new_empty(0)
            ),
        }
