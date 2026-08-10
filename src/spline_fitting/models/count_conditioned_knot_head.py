from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .knot_head import KnotHead


class CountConditionedKnotHead(nn.Module):
    """Predict exactly ``K`` ordered knots from the decoder branch for ``K``.

    Branch ``K`` owns ``K + 1`` interval queries. Their positive normalized
    intervals partition ``[0, 1]`` and therefore yield exactly ``K`` strictly
    ordered internal knots without candidate deletion or a probability threshold.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_internal_knots: int = 8,
        min_gap: float = 1e-3,
        attention_heads: int = 4,
        mode: str = "shared_count_embedding",
    ) -> None:
        super().__init__()
        if max_internal_knots < 0:
            raise ValueError("max_internal_knots must be non-negative")
        if min_gap < 0.0 or min_gap * (max_internal_knots + 1) >= 1.0:
            raise ValueError("min_gap must leave positive interval budget")
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if mode not in {"independent_branches", "shared_count_embedding"}:
            raise ValueError("unsupported count-conditioned decoder mode")
        self.max_internal_knots = int(max_internal_knots)
        self.min_gap = float(min_gap)
        self.mode = mode
        if mode == "independent_branches":
            # Exact v4 layout for strict checkpoint restoration.
            self.interval_queries = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(count + 1, hidden_dim))
                    for count in range(1, self.max_internal_knots + 1)
                ]
            )
            for queries in self.interval_queries:
                nn.init.trunc_normal_(queries, std=0.02)
        else:
            self.interval_queries = nn.Parameter(
                torch.empty(self.max_internal_knots + 1, hidden_dim)
            )
            nn.init.trunc_normal_(self.interval_queries, std=0.02)
            self.count_embedding = nn.Embedding(
                self.max_internal_knots + 1, hidden_dim
            )
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.interval_score = nn.Linear(hidden_dim, 1)

    def _decode_branch(
        self,
        count: int,
        memory: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "independent_branches":
            base_queries = self.interval_queries[count - 1]
            count_condition = 0.0
        else:
            base_queries = self.interval_queries[: count + 1]
            count_index = torch.full(
                (global_features.shape[0],),
                count,
                device=global_features.device,
                dtype=torch.long,
            )
            count_condition = self.count_embedding(count_index).unsqueeze(1)
        queries = base_queries.unsqueeze(0).expand(
            global_features.shape[0], -1, -1
        )
        queries = (
            queries
            + self.global_projection(global_features).unsqueeze(1)
            + count_condition
        )
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            need_weights=False,
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
        selected_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if local_features.ndim != 3 or positions.ndim != 2:
            raise ValueError(
                "local_features and positions must have shapes [B, M, H] and [B, M]"
            )
        if local_features.shape[:2] != positions.shape:
            raise ValueError("local_features and positions must share [B, M]")
        if global_features.shape != (
            local_features.shape[0],
            local_features.shape[-1],
        ):
            raise ValueError("global_features must have shape [B, H]")
        if selected_count.shape != (global_features.shape[0],):
            raise ValueError("selected_count must have shape [B]")
        if selected_count.is_floating_point():
            raise ValueError("selected_count must use an integer dtype")
        if torch.any((selected_count < 0) | (selected_count > self.max_internal_knots)):
            raise ValueError("selected_count lies outside the decoder range")

        memory = local_features + KnotHead._sinusoidal_position_encoding(
            positions,
            local_features.shape[-1],
        )
        batch = global_features.shape[0]
        zero_branch = local_features.new_zeros(batch, self.max_internal_knots)
        padded_branches = [zero_branch]
        for count in range(1, self.max_internal_knots + 1):
            knots = self._decode_branch(count, memory, global_features)
            padded_branches.append(F.pad(knots, (0, self.max_internal_knots - count)))
        branch_internal_knots = torch.stack(padded_branches, dim=1)

        branch_index = selected_count.to(torch.long).view(batch, 1, 1).expand(
            -1, 1, self.max_internal_knots
        )
        internal_knots = torch.gather(
            branch_internal_knots,
            dim=1,
            index=branch_index,
        ).squeeze(1)
        slots = torch.arange(self.max_internal_knots, device=selected_count.device)
        knot_mask = slots.unsqueeze(0) < selected_count.unsqueeze(1)
        return {
            "internal_knots": internal_knots,
            "knot_mask": knot_mask,
            "count_used_for_knots": selected_count,
            "branch_internal_knots": branch_internal_knots,
        }
