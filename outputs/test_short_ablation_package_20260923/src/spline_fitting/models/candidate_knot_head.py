from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .knot_head import KnotHead


class CandidateKnotHead(nn.Module):
    """Generate a fixed, high-recall set of ordered candidate knots.

    ``interval_softmax`` predicts ``num_candidates + 1`` positive intervals
    that partition the normalized parameter domain.  The opt-in
    ``bounded_anchor_residual`` mode instead moves every uniform candidate by
    less than half a cell, preventing proposal collapse while retaining local
    adaptivity.  Both parameterizations are strictly ordered by construction;
    no post-hoc sort is needed and tokens keep a stable left-to-right identity.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_candidates: int = 24,
        *,
        min_gap: float = 1e-3,
        attention_heads: int = 4,
        local_attention_bandwidth: float = 0.0,
        interval_logit_limit: float = 0.0,
        position_parameterization: str = "interval_softmax",
    ) -> None:
        super().__init__()
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if min_gap <= 0.0 or min_gap * (num_candidates + 1) >= 1.0:
            raise ValueError(
                "min_gap must be positive and leave positive interval budget"
            )
        if (
            not math.isfinite(local_attention_bandwidth)
            or local_attention_bandwidth < 0.0
        ):
            raise ValueError(
                "local_attention_bandwidth must be finite and non-negative"
            )
        if not math.isfinite(interval_logit_limit) or interval_logit_limit < 0.0:
            raise ValueError("interval_logit_limit must be finite and non-negative")
        if position_parameterization not in {
            "interval_softmax",
            "bounded_anchor_residual",
        }:
            raise ValueError(
                "position_parameterization must be 'interval_softmax' or "
                "'bounded_anchor_residual'"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_candidates = int(num_candidates)
        self.min_gap = float(min_gap)
        self.attention_heads = int(attention_heads)
        self.local_attention_bandwidth = float(local_attention_bandwidth)
        self.interval_logit_limit = float(interval_logit_limit)
        self.position_parameterization = str(position_parameterization)

        # Interval queries have a fixed left-to-right identity.  Positional
        # anchors make the initial attention cover the complete domain, while
        # the learned interval scores can subsequently concentrate candidates
        # around geometrically difficult regions.
        self.interval_queries = nn.Parameter(
            torch.empty(self.num_candidates + 1, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.interval_queries, std=0.02)
        self.register_buffer(
            "interval_query_anchors",
            (torch.arange(self.num_candidates + 1, dtype=torch.float32) + 0.5)
            / (self.num_candidates + 1),
        )
        self.register_buffer(
            "candidate_position_anchors",
            torch.arange(1, self.num_candidates + 1, dtype=torch.float32)
            / (self.num_candidates + 1),
            persistent=False,
        )

        self.global_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.interval_score = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.interval_score.weight, std=0.01)
        nn.init.zeros_(self.interval_score.bias)

        # Candidate j is the boundary shared by interval j and j+1.  Fusing
        # both adjacent tokens gives the pruning stage evidence from both sides
        # of the candidate knot.
        self.candidate_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.candidate_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return ordered candidates and one feature token per candidate.

        Args:
            global_features: Curve-level features with shape ``[B, H]``.
            local_features: Ordered point features with shape ``[B, M, H]``.
            positions: Point parameters in ``[0, 1]`` with shape ``[B, M]``.
        """

        if global_features.ndim != 2:
            raise ValueError("global_features must have shape [B,H]")
        if local_features.ndim != 3 or positions.ndim != 2:
            raise ValueError(
                "local_features and positions must have shapes [B,M,H] and [B,M]"
            )
        if local_features.shape[:2] != positions.shape:
            raise ValueError("local_features and positions must share [B,M]")
        if local_features.shape[0] != global_features.shape[0]:
            raise ValueError("local and global features must share the batch size")
        if global_features.shape[-1] != self.hidden_dim:
            raise ValueError("global feature dimension does not match hidden_dim")
        if local_features.shape[-1] != self.hidden_dim:
            raise ValueError("local feature dimension does not match hidden_dim")
        if local_features.shape[1] == 0:
            raise ValueError("local_features must contain at least one point")

        batch = global_features.shape[0]
        memory = local_features + KnotHead._sinusoidal_position_encoding(
            positions,
            self.hidden_dim,
        )
        anchors = (
            self.interval_query_anchors.to(
                device=local_features.device,
                dtype=local_features.dtype,
            )
            .unsqueeze(0)
            .expand(batch, -1)
        )
        anchor_encoding = KnotHead._sinusoidal_position_encoding(
            anchors,
            self.hidden_dim,
        )
        queries = (
            self.interval_queries.to(local_features.dtype)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        queries = (
            queries
            + anchor_encoding
            + self.global_projection(global_features).unsqueeze(1)
        )
        attention_bias = None
        local_attention_bias = None
        if self.local_attention_bandwidth > 0.0:
            # Each interval query receives an anchor-centred Gaussian bias in
            # parameter space.  Unlike the former positional encoding alone,
            # this makes the cross-attention itself local while retaining a
            # smooth tail that can still collect wider geometric context.
            distance = positions.unsqueeze(1) - anchors.unsqueeze(-1)
            local_attention_bias = (
                -0.5 * (distance / self.local_attention_bandwidth).square()
            )
            local_attention_bias = local_attention_bias.clamp_min(-30.0)
            attention_bias = (
                local_attention_bias.unsqueeze(1)
                .expand(-1, self.attention_heads, -1, -1)
                .reshape(
                    batch * self.attention_heads,
                    self.num_candidates + 1,
                    positions.shape[1],
                )
            )
        attended, attention_weights = self.cross_attention(
            queries,
            memory,
            memory,
            attn_mask=attention_bias,
            need_weights=True,
            average_attn_weights=True,
        )
        interval_tokens = self.cross_norm(queries + attended)
        interval_tokens = self.output_norm(
            interval_tokens + self.feed_forward(interval_tokens)
        )

        raw_interval_logits = self.interval_score(interval_tokens).squeeze(-1)
        if self.interval_logit_limit > 0.0:
            # Softmax is invariant to a shared shift, so remove it before the
            # bounded transform.  The resulting logit range is at most
            # ``2 * limit``: no single interval can absorb essentially the
            # whole parameter domain and strand the remaining candidates at
            # ``min_gap``.  ``limit=0`` retains historical checkpoint output
            # exactly; the bounded path is opt-in for new training runs.
            centered_interval_logits = raw_interval_logits - raw_interval_logits.mean(
                dim=-1,
                keepdim=True,
            )
            effective_interval_logits = self.interval_logit_limit * torch.tanh(
                centered_interval_logits / self.interval_logit_limit
            )
        else:
            effective_interval_logits = raw_interval_logits
        if self.position_parameterization == "interval_softmax":
            free_budget = 1.0 - self.min_gap * (self.num_candidates + 1)
            candidate_intervals = self.min_gap + free_budget * F.softmax(
                effective_interval_logits,
                dim=-1,
            )
            candidate_positions = torch.cumsum(candidate_intervals, dim=-1)[:, :-1]
            candidate_position_residual = candidate_positions - (
                self.candidate_position_anchors.to(
                    device=local_features.device,
                    dtype=local_features.dtype,
                )
                .unsqueeze(0)
                .expand(batch, -1)
            )
        else:
            # A dense proposal is useful only if it covers the entire domain.
            # Parameterize every candidate as a bounded residual around its
            # uniform anchor instead of allowing a global interval softmax to
            # move nearly all probability mass into one span.  Adjacent slots
            # can each move by less than half their cell width; choosing the
            # exact feasible radius below guarantees their remaining gap is at
            # least ``min_gap`` without sorting or a projection.
            anchor_positions = self.candidate_position_anchors.to(
                device=local_features.device,
                dtype=local_features.dtype,
            ).unsqueeze(0).expand(batch, -1)
            uniform_interval = 1.0 / (self.num_candidates + 1)
            residual_radius = 0.5 * (uniform_interval - self.min_gap)
            boundary_signal = (
                effective_interval_logits[:, :-1]
                - effective_interval_logits[:, 1:]
            )
            candidate_position_residual = residual_radius * torch.tanh(
                boundary_signal
            )
            candidate_positions = anchor_positions + candidate_position_residual
            boundaries = torch.cat(
                [
                    candidate_positions.new_zeros(batch, 1),
                    candidate_positions,
                    candidate_positions.new_ones(batch, 1),
                ],
                dim=-1,
            )
            candidate_intervals = boundaries[:, 1:] - boundaries[:, :-1]

        adjacent_tokens = torch.cat(
            [interval_tokens[:, :-1], interval_tokens[:, 1:]],
            dim=-1,
        )
        position_encoding = KnotHead._sinusoidal_position_encoding(
            candidate_positions,
            self.hidden_dim,
        )
        candidate_tokens = self.candidate_norm(
            self.candidate_fusion(adjacent_tokens) + position_encoding
        )

        return {
            "candidate_positions": candidate_positions,
            # A knot-named alias makes the hand-off to spline solvers explicit.
            "candidate_knots": candidate_positions,
            "candidate_tokens": candidate_tokens,
            "candidate_intervals": candidate_intervals,
            "raw_candidate_interval_logits": raw_interval_logits,
            "effective_candidate_interval_logits": effective_interval_logits,
            "candidate_position_anchors": self.candidate_position_anchors.to(
                device=local_features.device,
                dtype=local_features.dtype,
            ).unsqueeze(0).expand(batch, -1),
            "candidate_position_residual": candidate_position_residual,
            "candidate_attention_weights": attention_weights,
            "candidate_local_attention_bias": (
                local_attention_bias
                if local_attention_bias is not None
                else candidate_positions.new_empty(0)
            ),
        }
