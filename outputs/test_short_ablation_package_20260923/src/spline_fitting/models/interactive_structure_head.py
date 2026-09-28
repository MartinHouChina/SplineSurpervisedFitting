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
        count_distribution_mode: str = "categorical",
        min_internal_knots: int = 0,
    ) -> None:
        super().__init__()
        if max_internal_knots < 0:
            raise ValueError("max_internal_knots must be non-negative")
        if count_distribution_mode not in {"hazard", "categorical"}:
            raise ValueError(
                "count_distribution_mode must be 'hazard' or 'categorical'"
            )
        if not 0 <= min_internal_knots <= max_internal_knots:
            raise ValueError(
                "min_internal_knots must lie between zero and max_internal_knots"
            )
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        self.max_internal_knots = int(max_internal_knots)
        self.min_internal_knots = int(min_internal_knots)
        self.count_distribution_mode = count_distribution_mode
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
        if self.count_distribution_mode == "hazard":
            # Historical v6 path. Keep the module names and tensor layout exact
            # so checkpoints written before categorical count prediction remain
            # strictly loadable.
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
        else:
            self.count_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.max_internal_knots + 1),
            )
            nn.init.normal_(self.count_classifier[-1].weight, std=0.01)
            nn.init.zeros_(self.count_classifier[-1].bias)
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

        if self.count_distribution_mode == "hazard":
            if self.max_internal_knots == 0:
                stop_logits = local_features.new_empty(batch, 0)
                survival = local_features.new_empty(batch, 0)
                probabilities = local_features.new_ones(batch, 1)
            else:
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

            if self.min_internal_knots:
                probabilities = probabilities.clone()
                probabilities[:, : self.min_internal_knots] = 0.0
                probabilities = probabilities / probabilities.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)

            survival_logits = torch.logit(survival.clamp(1e-6, 1.0 - 1e-6))
            count_logits = probabilities.clamp_min(1e-8).log()
            structure_output = {
                "structure_stop_logits": stop_logits,
                "structure_survival_probabilities": survival,
                "structure_survival_logits": survival_logits,
            }
        else:
            count_context = (
                tokens.mean(dim=1)
                if self.max_internal_knots
                else self.global_projection(global_features)
            )
            unmasked_logits = self.count_classifier(count_context)
            count_values = torch.arange(
                self.max_internal_knots + 1,
                device=unmasked_logits.device,
            )
            legal_count_mask = count_values >= self.min_internal_knots
            count_logits = unmasked_logits.masked_fill(
                ~legal_count_mask.unsqueeze(0),
                torch.finfo(unmasked_logits.dtype).min,
            )
            probabilities = torch.softmax(count_logits, dim=-1)
            structure_output = {
                "unmasked_count_logits": unmasked_logits,
                "legal_count_mask": legal_count_mask,
            }
        count_mode = probabilities.argmax(dim=-1)
        posterior_median = (
            probabilities.cumsum(dim=-1) < 0.5
        ).to(torch.long).sum(dim=-1)
        posterior_median = posterior_median.clamp(
            min=self.min_internal_knots,
            max=self.max_internal_knots,
        )
        return {
            "structure_query_features": tokens,
            **structure_output,
            "count_logits": count_logits,
            "count_probabilities": probabilities,
            "count_mode_knot_count": count_mode,
            # The posterior median is the Bayes estimator for absolute count
            # error. It is substantially more stable than argmax when a broad
            # count distribution has tiny endpoint-vs-interior differences.
            "predicted_knot_count": posterior_median,
            "expected_knot_count": (
                probabilities * self.count_values.to(probabilities)
            ).sum(dim=-1),
        }
