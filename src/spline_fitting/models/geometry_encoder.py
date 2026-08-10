from __future__ import annotations

import torch
from torch import nn


class GeometryEncoder(nn.Module):
    """Shared 1D convolutional encoder for ordered curve samples."""

    def __init__(
        self,
        point_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        feature_mode: str = "chord_derivatives",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if feature_mode not in {"raw_differences", "chord_derivatives"}:
            raise ValueError("unsupported geometry feature mode")
        self.feature_mode = feature_mode

        layers: list[nn.Module] = []
        in_channels = point_dim * 3
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
                    nn.GELU(),
                ]
            )
            in_channels = hidden_dim
        self.backbone = nn.Sequential(*layers)

    def _geometric_features(self, points: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "chord_derivatives":
            segments = points[:, 1:] - points[:, :-1]
            lengths = segments.norm(dim=-1).clamp_min(1e-6)
            gaps = lengths / lengths.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            segment_derivative = segments / gaps.unsqueeze(-1)
            first = torch.zeros_like(points)
            first[:, 0] = segment_derivative[:, 0]
            first[:, 1:] = segment_derivative

            second = torch.zeros_like(points)
            if points.shape[1] > 2:
                midpoint_gap = 0.5 * (gaps[:, 1:] + gaps[:, :-1])
                second[:, 1:-1] = (
                    segment_derivative[:, 1:] - segment_derivative[:, :-1]
                ) / midpoint_gap.unsqueeze(-1).clamp_min(1e-6)
            return torch.cat([points, first, second], dim=-1)

        first = torch.zeros_like(points)
        first[:, 1:] = points[:, 1:] - points[:, :-1]

        second = torch.zeros_like(points)
        if points.shape[1] > 2:
            second[:, 1:-1] = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]

        return torch.cat([points, first, second], dim=-1)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3:
            raise ValueError("points must have shape [B, M, D]")
        features = self._geometric_features(points).transpose(1, 2)
        encoded = self.backbone(features)
        local_features = encoded.transpose(1, 2)
        global_features = encoded.amax(dim=-1)
        return local_features, global_features
