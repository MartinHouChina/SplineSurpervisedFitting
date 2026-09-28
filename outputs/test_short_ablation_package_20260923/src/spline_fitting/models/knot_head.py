from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class KnotHead(nn.Module):
    """Predict candidate interior knots from positional local features.

    ``independent_queries`` gives every candidate its own position regressor.
    Sorting is applied only after regression so removing one candidate cannot
    redistribute the remaining positions. ``interval`` retains the historical
    coupled positive-interval parameterization for checkpoint compatibility.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_internal_knots: int = 8,
        min_gap: float = 1e-3,
        gap_parameterization: str = "strict",
        use_local_cross_attention: bool = True,
        attention_heads: int = 4,
        knot_parameterization: str = "independent_queries",
    ) -> None:
        super().__init__()
        if gap_parameterization not in {"strict", "legacy"}:
            raise ValueError("gap_parameterization must be 'strict' or 'legacy'")
        if max_internal_knots < 0:
            raise ValueError("max_internal_knots must be non-negative")
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if knot_parameterization not in {"independent_queries", "interval"}:
            raise ValueError(
                "knot_parameterization must be 'independent_queries' or 'interval'"
            )
        num_intervals = max_internal_knots + 1
        if min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        if knot_parameterization == "interval" and min_gap * num_intervals >= 1.0:
            raise ValueError(
                "min_gap must be non-negative and leave positive interval budget"
            )
        if knot_parameterization == "independent_queries" and min_gap >= 0.5:
            raise ValueError("min_gap must leave a non-empty interior domain")
        self.max_internal_knots = max_internal_knots
        self.min_gap = min_gap
        self.gap_parameterization = gap_parameterization
        self.knot_parameterization = knot_parameterization
        self.use_local_cross_attention = bool(use_local_cross_attention)
        if self.use_local_cross_attention:
            num_queries = (
                max_internal_knots
                if knot_parameterization == "independent_queries"
                else num_intervals
            )
            query_name = (
                "node_queries"
                if knot_parameterization == "independent_queries"
                else "interval_queries"
            )
            setattr(
                self, query_name, nn.Parameter(torch.empty(num_queries, hidden_dim))
            )
            queries = getattr(self, query_name)
            nn.init.trunc_normal_(queries, std=0.02)
            self.global_projection = nn.Linear(hidden_dim, hidden_dim)
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim,
                attention_heads,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            if knot_parameterization == "independent_queries":
                self.position_score = nn.Linear(hidden_dim, 1)
            else:
                self.interval_score = nn.Linear(hidden_dim, 1)
        else:
            # Historical global-only path retained for checkpoint compatibility.
            if knot_parameterization != "interval":
                raise ValueError("independent_queries requires local cross-attention")
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_intervals),
            )

    @staticmethod
    def _sinusoidal_position_encoding(
        positions: torch.Tensor,
        hidden_dim: int,
    ) -> torch.Tensor:
        frequencies = torch.exp(
            torch.arange(
                0,
                hidden_dim,
                2,
                device=positions.device,
                dtype=positions.dtype,
            )
            * (-math.log(10000.0) / hidden_dim)
        )
        scaled_positions = positions * max(positions.shape[-1] - 1, 1)
        angles = scaled_positions.unsqueeze(-1) * frequencies
        encoding = torch.zeros(
            *positions.shape,
            hidden_dim,
            device=positions.device,
            dtype=positions.dtype,
        )
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles[..., : hidden_dim // 2])
        return encoding

    def forward(
        self,
        global_features: torch.Tensor,
        *,
        local_features: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        attention_weights: torch.Tensor | None = None
        query_tokens: torch.Tensor | None = None
        if self.use_local_cross_attention:
            if local_features is None or positions is None:
                raise ValueError(
                    "local_features and positions are required for local cross-attention"
                )
            if local_features.ndim != 3 or positions.ndim != 2:
                raise ValueError(
                    "local_features and positions must have shapes [B, M, H] and [B, M]"
                )
            if local_features.shape[:2] != positions.shape:
                raise ValueError("local_features and positions must share [B, M]")
            if local_features.shape[0] != global_features.shape[0]:
                raise ValueError("local and global features must share the batch size")
            if local_features.shape[-1] != global_features.shape[-1]:
                raise ValueError("local and global feature dimensions must match")

            memory = local_features + self._sinusoidal_position_encoding(
                positions,
                local_features.shape[-1],
            )
            learned_queries = (
                self.node_queries
                if self.knot_parameterization == "independent_queries"
                else self.interval_queries
            )
            queries = learned_queries.unsqueeze(0).expand(
                global_features.shape[0], -1, -1
            )
            queries = queries + self.global_projection(global_features).unsqueeze(1)
            attended, attention_weights = self.cross_attention(
                queries,
                memory,
                memory,
                need_weights=True,
                average_attn_weights=True,
            )
            tokens = self.attention_norm(queries + attended)
            tokens = self.output_norm(tokens + self.feed_forward(tokens))
            query_tokens = tokens
            if self.knot_parameterization == "independent_queries":
                raw_positions = self.position_score(tokens).squeeze(-1)
            else:
                raw_intervals = self.interval_score(tokens).squeeze(-1)
        else:
            raw_intervals = self.mlp(global_features)

        if self.knot_parameterization == "independent_queries":
            free_domain = 1.0 - 2.0 * self.min_gap
            unsorted_knots = self.min_gap + free_domain * torch.sigmoid(raw_positions)
            internal_knots, sort_indices = torch.sort(unsorted_knots, dim=-1)
            assert query_tokens is not None
            feature_indices = sort_indices.unsqueeze(-1).expand(
                -1, -1, query_tokens.shape[-1]
            )
            sorted_tokens = torch.gather(query_tokens, 1, feature_indices)
            output = {
                "internal_knots": internal_knots,
                "raw_knot_positions": torch.gather(raw_positions, 1, sort_indices),
                "knot_query_features": sorted_tokens,
                "knot_sort_indices": sort_indices,
            }
            if attention_weights is not None:
                attention_indices = sort_indices.unsqueeze(-1).expand(
                    -1, -1, attention_weights.shape[-1]
                )
                output["knot_attention_weights"] = torch.gather(
                    attention_weights, 1, attention_indices
                )
            return output

        if self.gap_parameterization == "legacy":
            positive_intervals = F.softplus(raw_intervals) + self.min_gap
            normalized_intervals = positive_intervals / positive_intervals.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        else:
            interval_weights = F.softmax(raw_intervals, dim=-1)
            free_budget = 1.0 - self.min_gap * (self.max_internal_knots + 1)
            normalized_intervals = self.min_gap + free_budget * interval_weights
        internal_knots = torch.cumsum(normalized_intervals, dim=-1)[:, :-1]
        output = {
            "internal_knots": internal_knots,
            "knot_intervals": normalized_intervals,
            "raw_knot_intervals": raw_intervals,
        }
        if attention_weights is not None:
            output["knot_attention_weights"] = attention_weights
        return output
