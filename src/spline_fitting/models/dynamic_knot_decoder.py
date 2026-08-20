from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .knot_head import KnotHead


class DynamicKnotDecoder(nn.Module):
    """Decode selected ``K+1`` intervals, skipping position work when K is zero."""

    def __init__(
        self,
        hidden_dim: int = 128,
        max_internal_knots: int = 6,
        *,
        min_gap: float = 1e-3,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if max_internal_knots < 0:
            raise ValueError("max_internal_knots must be non-negative")
        if min_gap < 0.0 or min_gap * (max_internal_knots + 1) >= 1.0:
            raise ValueError("min_gap must leave positive interval budget")
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        self.max_internal_knots = int(max_internal_knots)
        self.min_gap = float(min_gap)
        self.interval_queries = nn.Parameter(
            torch.empty(self.max_internal_knots + 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.interval_queries, std=0.02)
        self.count_embedding = nn.Embedding(
            self.max_internal_knots + 1, hidden_dim
        )
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        self.structure_projection = nn.Linear(hidden_dim, hidden_dim)
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
        self.interval_score = nn.Linear(hidden_dim, 1)

    def _decode_group(
        self,
        count: int,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        positions: torch.Tensor,
        structure_features: torch.Tensor,
    ) -> torch.Tensor:
        group_size = global_features.shape[0]
        local_memory = local_features + KnotHead._sinusoidal_position_encoding(
            positions, local_features.shape[-1]
        )
        memory = torch.cat([local_memory, structure_features], dim=1)
        structure_context = (
            structure_features.mean(dim=1)
            if structure_features.shape[1]
            else torch.zeros_like(global_features)
        )
        count_index = torch.full(
            (group_size,), count, device=global_features.device, dtype=torch.long
        )
        queries = self.interval_queries[: count + 1].unsqueeze(0).expand(
            group_size, -1, -1
        )
        queries = (
            queries
            + self.global_projection(global_features).unsqueeze(1)
            + self.structure_projection(structure_context).unsqueeze(1)
            + self.count_embedding(count_index).unsqueeze(1)
        )
        attended, _ = self.cross_attention(
            queries, memory, memory, need_weights=False
        )
        tokens = self.attention_norm(queries + attended)
        tokens = self.output_norm(tokens + self.feed_forward(tokens))
        raw_intervals = self.interval_score(tokens).squeeze(-1)
        free_budget = 1.0 - self.min_gap * (count + 1)
        intervals = self.min_gap + free_budget * F.softmax(raw_intervals, dim=-1)
        return torch.cumsum(intervals, dim=-1)[:, :-1]

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        positions: torch.Tensor,
        structure_features: torch.Tensor,
        selected_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = global_features.shape[0]
        if selected_count.shape != (batch,):
            raise ValueError("selected_count must have shape [B]")
        if selected_count.is_floating_point():
            raise ValueError("selected_count must use an integer dtype")
        if torch.any((selected_count < 0) | (selected_count > self.max_internal_knots)):
            raise ValueError("selected_count lies outside the decoder range")

        internal_knots = local_features.new_zeros(
            batch, self.max_internal_knots
        )
        for count_value in torch.unique(selected_count).detach().cpu().tolist():
            count = int(count_value)
            if count == 0:
                continue
            indices = torch.nonzero(selected_count == count, as_tuple=False).squeeze(-1)
            knots = self._decode_group(
                count,
                global_features.index_select(0, indices),
                local_features.index_select(0, indices),
                positions.index_select(0, indices),
                structure_features.index_select(0, indices),
            )
            internal_knots[indices, :count] = knots

        slots = torch.arange(self.max_internal_knots, device=selected_count.device)
        knot_mask = slots.unsqueeze(0) < selected_count.unsqueeze(1)
        return {
            "internal_knots": internal_knots,
            "knot_mask": knot_mask,
            "count_used_for_knots": selected_count,
            "decoded_interval_query_count": torch.where(
                selected_count > 0,
                selected_count + 1,
                torch.zeros_like(selected_count),
            ),
        }
