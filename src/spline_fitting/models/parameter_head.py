from __future__ import annotations

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
    ) -> None:
        super().__init__()
        if gap_parameterization not in {"strict", "legacy"}:
            raise ValueError("gap_parameterization must be 'strict' or 'legacy'")
        if min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        self.min_gap = min_gap
        self.gap_parameterization = gap_parameterization
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, num_points, _ = local_features.shape
        global_expanded = global_features.unsqueeze(1).expand(-1, num_points, -1)
        fused = torch.cat([local_features, global_expanded], dim=-1)

        raw_gaps = self.mlp(fused[:, :-1]).squeeze(-1)
        if self.gap_parameterization == "legacy":
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
        }
