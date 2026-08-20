from __future__ import annotations

import torch
from torch import nn

from .knot_head import KnotHead


class InteractiveStructureHead(nn.Module):
    """Infer count from knot-local queries with cross- and self-attention."""

    def __init__(
        self,
        hidden_dim: int = 128,
        max_internal_knots: int = 6,
        *,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if max_internal_knots < 0:
            raise ValueError("max_internal_knots must be non-negative")
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        self.max_internal_knots = int(max_internal_knots)
        self.structure_queries = nn.Parameter(
            torch.empty(self.max_internal_knots, hidden_dim)
        )
        nn.init.trunc_normal_(self.structure_queries, std=0.02)
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.stop_head = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.normal_(self.stop_head.weight, std=0.01)
        if self.max_internal_knots:
            conditional_stop = 1.0 / torch.arange(
                self.max_internal_knots + 1,
                1,
                -1,
                dtype=torch.float32,
            )
            initial_stop_bias = torch.logit(conditional_stop)
        else:
            initial_stop_bias = torch.empty(0)
        self.stop_bias = nn.Parameter(initial_stop_bias)
        self.register_buffer(
            "count_values",
            torch.arange(self.max_internal_knots + 1, dtype=torch.float32),
        )

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if global_features.ndim != 2:
            raise ValueError("global_features must have shape [B,H]")
        if local_features.ndim != 3 or positions.ndim != 2:
            raise ValueError("local_features and positions must be [B,M,H] and [B,M]")
        if local_features.shape[:2] != positions.shape:
            raise ValueError("local_features and positions must share [B,M]")

        batch = global_features.shape[0]
        if self.max_internal_knots == 0:
            tokens = local_features.new_empty(batch, 0, local_features.shape[-1])
            stop_logits = local_features.new_empty(batch, 0)
            survival = local_features.new_empty(batch, 0)
            probabilities = local_features.new_ones(batch, 1)
        else:
            memory = local_features + KnotHead._sinusoidal_position_encoding(
                positions, local_features.shape[-1]
            )
            queries = self.structure_queries.unsqueeze(0).expand(batch, -1, -1)
            queries = queries + self.global_projection(global_features).unsqueeze(1)
            attended, _ = self.cross_attention(
                queries, memory, memory, need_weights=False
            )
            tokens = self.cross_norm(queries + attended)
            interacted, _ = self.self_attention(
                tokens, tokens, tokens, need_weights=False
            )
            tokens = self.self_norm(tokens + interacted)
            tokens = self.output_norm(tokens + self.feed_forward(tokens))

            stop_logits = self.stop_head(tokens).squeeze(-1) + self.stop_bias
            stop_probabilities = torch.sigmoid(stop_logits)
            survival = torch.cumprod(1.0 - stop_probabilities, dim=-1)
            probability_parts = [stop_probabilities[:, :1]]
            if self.max_internal_knots > 1:
                probability_parts.append(
                    survival[:, :-1] * stop_probabilities[:, 1:]
                )
            probability_parts.append(survival[:, -1:])
            probabilities = torch.cat(probability_parts, dim=-1)
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

        survival_logits = torch.logit(survival.clamp(1e-6, 1.0 - 1e-6))
        count_logits = probabilities.clamp_min(1e-8).log()
        return {
            "structure_query_features": tokens,
            "structure_stop_logits": stop_logits,
            "structure_survival_probabilities": survival,
            "structure_survival_logits": survival_logits,
            "count_logits": count_logits,
            "count_probabilities": probabilities,
            "predicted_knot_count": probabilities.argmax(dim=-1),
            "expected_knot_count": (
                probabilities * self.count_values.to(probabilities)
            ).sum(dim=-1),
        }
