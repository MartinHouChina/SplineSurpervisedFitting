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
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

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

    @staticmethod
    def _geometric_features(points: torch.Tensor) -> torch.Tensor:
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
