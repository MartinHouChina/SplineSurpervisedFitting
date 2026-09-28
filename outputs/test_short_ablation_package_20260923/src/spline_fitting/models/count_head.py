from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CountHead(nn.Module):
    """Predict knot count with a historical or local ordinal head."""

    def __init__(
        self,
        hidden_dim: int = 128,
        max_internal_knots: int = 8,
        *,
        mode: str = "ordinal_local_attention",
        attention_heads: int = 4,
        query_count: int = 4,
    ) -> None:
        super().__init__()
        if max_internal_knots < 0:
            raise ValueError("max_internal_knots must be non-negative")
        if mode not in {"categorical_global", "ordinal_local_attention"}:
            raise ValueError("unsupported count-head mode")
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if query_count <= 0:
            raise ValueError("query_count must be positive")
        self.max_internal_knots = int(max_internal_knots)
        self.mode = mode
        self.register_buffer(
            "count_values",
            torch.arange(self.max_internal_knots + 1, dtype=torch.float32),
        )

        if mode == "categorical_global":
            # Exact v4 module layout for strict checkpoint restoration.
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self.max_internal_knots + 1),
            )
            return

        self.count_queries = nn.Parameter(torch.empty(query_count, hidden_dim))
        nn.init.trunc_normal_(self.count_queries, std=0.02)
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim * (query_count + 1), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        if self.max_internal_knots:
            survival = torch.arange(
                self.max_internal_knots, 0, -1, dtype=torch.float32
            ) / (self.max_internal_knots + 1)
            initial_thresholds = -torch.logit(survival)
            self.threshold_start = nn.Parameter(initial_thresholds[:1].clone())
            if self.max_internal_knots > 1:
                gaps = initial_thresholds[1:] - initial_thresholds[:-1]
                raw_gaps = torch.log(torch.expm1(gaps).clamp_min(1e-6))
            else:
                raw_gaps = torch.empty(0)
            self.threshold_gap_raw = nn.Parameter(raw_gaps)

    @staticmethod
    def _position_encoding(positions: torch.Tensor, dimension: int) -> torch.Tensor:
        half = dimension // 2
        if half == 0:
            return positions.unsqueeze(-1)
        frequencies = torch.exp(
            torch.arange(half, device=positions.device, dtype=positions.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = positions.unsqueeze(-1) * frequencies * (2.0 * math.pi)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if encoding.shape[-1] < dimension:
            encoding = F.pad(encoding, (0, dimension - encoding.shape[-1]))
        return encoding

    def _ordinal_thresholds(self) -> torch.Tensor:
        if self.max_internal_knots == 0:
            return self.count_values.new_empty(0)
        if self.max_internal_knots == 1:
            return self.threshold_start
        gaps = F.softplus(self.threshold_gap_raw) + 1e-4
        return torch.cat(
            [self.threshold_start, self.threshold_start + torch.cumsum(gaps, dim=0)]
        )

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if global_features.ndim != 2:
            raise ValueError("global_features must have shape [B, H]")
        if self.mode == "categorical_global":
            logits = self.mlp(global_features)
            probabilities = torch.softmax(logits, dim=-1)
            return {
                "count_logits": logits,
                "count_probabilities": probabilities,
                "predicted_knot_count": logits.argmax(dim=-1),
                "expected_knot_count": (
                    probabilities * self.count_values.to(probabilities)
                ).sum(dim=-1),
            }

        if local_features is None or positions is None:
            raise ValueError("ordinal local CountHead requires local features and positions")
        if local_features.shape[:2] != positions.shape:
            raise ValueError("local_features and positions must share [B, M]")
        memory = local_features + self._position_encoding(
            positions, local_features.shape[-1]
        )
        queries = self.count_queries.unsqueeze(0).expand(
            global_features.shape[0], -1, -1
        )
        queries = queries + self.global_projection(global_features).unsqueeze(1)
        attended, _ = self.cross_attention(
            queries, memory, memory, need_weights=False
        )
        tokens = self.attention_norm(queries + attended)
        tokens = self.output_norm(tokens + self.feed_forward(tokens))
        evidence_input = torch.cat(
            [global_features, tokens.flatten(start_dim=1)], dim=-1
        )
        evidence = self.evidence_head(evidence_input).squeeze(-1)

        if self.max_internal_knots == 0:
            ordinal_logits = global_features.new_empty(global_features.shape[0], 0)
            probabilities = global_features.new_ones(global_features.shape[0], 1)
        else:
            ordinal_logits = evidence.unsqueeze(-1) - self._ordinal_thresholds()
            survival = torch.sigmoid(ordinal_logits)
            boundaries = torch.cat(
                [
                    torch.ones_like(survival[:, :1]),
                    survival,
                    torch.zeros_like(survival[:, :1]),
                ],
                dim=-1,
            )
            probabilities = (boundaries[:, :-1] - boundaries[:, 1:]).clamp_min(1e-8)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        count_logits = probabilities.log()
        return {
            "count_logits": count_logits,
            "count_ordinal_logits": ordinal_logits,
            "count_probabilities": probabilities,
            "predicted_knot_count": probabilities.argmax(dim=-1),
            "expected_knot_count": (
                probabilities * self.count_values.to(probabilities)
            ).sum(dim=-1),
            "count_evidence": evidence,
        }
