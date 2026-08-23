from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .knot_head import KnotHead


class InteractivePruningHead(nn.Module):
    """Rank, retain, and refine candidate knots using analytic fit evidence.

    Each candidate receives its truncated-power coefficient energy, the
    measured/approximated objective increase after deletion, two-sided spacing,
    and local residual features.  Self-attention lets candidates reason about
    redundancy jointly instead of making independent threshold decisions.

    Position updates are bounded to less than half of the available neighboring
    slack.  Consequently ordered inputs remain strictly ordered without sorting,
    preserving the correspondence between candidates and output scores.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        residual_feature_dim: int = 1,
        attention_heads: int = 4,
        min_gap: float = 1e-3,
        max_position_fraction: float = 0.45,
        initial_keep_probability: float = 0.9,
    ) -> None:
        super().__init__()
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if residual_feature_dim <= 0:
            raise ValueError("residual_feature_dim must be positive")
        if min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        if not 0.0 < max_position_fraction < 0.5:
            raise ValueError("max_position_fraction must lie strictly between 0 and 0.5")
        if not 0.0 < initial_keep_probability < 1.0:
            raise ValueError("initial_keep_probability must lie in (0,1)")

        self.hidden_dim = int(hidden_dim)
        self.residual_feature_dim = int(residual_feature_dim)
        self.min_gap = float(min_gap)
        self.max_position_fraction = float(max_position_fraction)

        # position + coefficient energy + deletion delta + left/right spacing
        # + the configurable local residual descriptor.
        analytic_dim = 5 + self.residual_feature_dim
        self.analytic_projection = nn.Sequential(
            nn.Linear(analytic_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            self.hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(self.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        self.keep_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.keep_head.weight, std=0.01)
        nn.init.constant_(
            self.keep_head.bias,
            math.log(initial_keep_probability / (1.0 - initial_keep_probability)),
        )
        self.deletion_cost_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.deletion_cost_head.weight, std=0.01)
        nn.init.constant_(self.deletion_cost_head.bias, -4.0)
        self.remove_action_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.remove_action_head.weight, std=0.01)
        nn.init.zeros_(self.remove_action_head.bias)
        self.stop_action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.stop_action_head[-1].weight)
        nn.init.constant_(self.stop_action_head[-1].bias, -2.0)
        self.position_residual_head = nn.Linear(self.hidden_dim, 1)
        # Begin from the high-recall candidates; position refinement is learned
        # only when its supervision supplies evidence for a move.
        nn.init.zeros_(self.position_residual_head.weight)
        nn.init.zeros_(self.position_residual_head.bias)

    @staticmethod
    def _stable_signed_log1p(values: torch.Tensor) -> torch.Tensor:
        values = torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        values = values.clamp(min=-1e6, max=1e6)
        return values.sign() * torch.log1p(values.abs())

    @staticmethod
    def _two_sided_spacing(candidate_positions: torch.Tensor) -> torch.Tensor:
        batch = candidate_positions.shape[0]
        boundaries = torch.cat(
            [
                candidate_positions.new_zeros(batch, 1),
                candidate_positions,
                candidate_positions.new_ones(batch, 1),
            ],
            dim=-1,
        )
        intervals = boundaries[:, 1:] - boundaries[:, :-1]
        return torch.stack([intervals[:, :-1], intervals[:, 1:]], dim=-1)

    def _validate_inputs(
        self,
        candidate_tokens: torch.Tensor,
        candidate_positions: torch.Tensor,
        coefficient_energy: torch.Tensor,
        deletion_delta: torch.Tensor,
        spacing: torch.Tensor,
        residual_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> None:
        if candidate_tokens.ndim != 3 or candidate_positions.ndim != 2:
            raise ValueError(
                "candidate_tokens and candidate_positions must be [B,K,H] and [B,K]"
            )
        if candidate_tokens.shape[:2] != candidate_positions.shape:
            raise ValueError("candidate tokens and positions must share [B,K]")
        if candidate_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("candidate token dimension does not match hidden_dim")
        expected_scalar_shape = candidate_positions.shape
        if coefficient_energy.shape != expected_scalar_shape:
            raise ValueError("coefficient_energy must have shape [B,K]")
        if deletion_delta.shape != expected_scalar_shape:
            raise ValueError("deletion_delta must have shape [B,K]")
        if spacing.shape != (*expected_scalar_shape, 2):
            raise ValueError("spacing must have shape [B,K,2]")
        if residual_features.shape != (
            *expected_scalar_shape,
            self.residual_feature_dim,
        ):
            raise ValueError(
                "residual_features must have shape [B,K,residual_feature_dim]"
            )
        if candidate_mask.shape != expected_scalar_shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean with shape [B,K]")
        if candidate_positions.shape[1] == 0:
            raise ValueError("at least one candidate is required")
        if torch.any(~candidate_mask.any(dim=-1)):
            raise ValueError("every sample must contain at least one active candidate")

        # The bounded refinement guarantee assumes a valid ordered input.  The
        # small tolerance avoids false failures from floating-point round-off.
        full_spacing = self._two_sided_spacing(candidate_positions)
        if torch.any(full_spacing < self.min_gap - 1e-6):
            raise ValueError(
                "candidate_positions must be ordered and respect min_gap"
            )

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        candidate_positions: torch.Tensor,
        *,
        coefficient_energy: torch.Tensor,
        deletion_delta: torch.Tensor,
        residual_features: torch.Tensor,
        spacing: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate and refine a fixed candidate set.

        ``spacing`` contains left/right gaps and defaults to the exact gaps
        derived from ``candidate_positions``.  A scalar residual descriptor can
        be passed as ``[B,K]`` when ``residual_feature_dim == 1``.
        """

        if candidate_positions.ndim != 2:
            raise ValueError("candidate_positions must have shape [B,K]")
        if spacing is None:
            spacing = self._two_sided_spacing(candidate_positions)
        if residual_features.ndim == 2 and self.residual_feature_dim == 1:
            residual_features = residual_features.unsqueeze(-1)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(candidate_positions, dtype=torch.bool)

        self._validate_inputs(
            candidate_tokens,
            candidate_positions,
            coefficient_energy,
            deletion_delta,
            spacing,
            residual_features,
            candidate_mask,
        )

        coefficient_feature = torch.log1p(
            torch.nan_to_num(
                coefficient_energy,
                nan=0.0,
                posinf=1e6,
                neginf=0.0,
            ).clamp(min=0.0, max=1e6)
        )
        deletion_feature = self._stable_signed_log1p(deletion_delta)
        spacing_feature = torch.nan_to_num(
            spacing,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(min=0.0, max=1.0)
        residual_feature = self._stable_signed_log1p(residual_features)
        analytic_features = torch.cat(
            [
                candidate_positions.unsqueeze(-1),
                coefficient_feature.unsqueeze(-1),
                deletion_feature.unsqueeze(-1),
                spacing_feature,
                residual_feature,
            ],
            dim=-1,
        )

        position_encoding = KnotHead._sinusoidal_position_encoding(
            candidate_positions,
            self.hidden_dim,
        )
        tokens = self.input_norm(
            candidate_tokens
            + position_encoding
            + self.analytic_projection(analytic_features)
        )
        interacted, attention_weights = self.self_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~candidate_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        tokens = self.self_norm(tokens + interacted)
        tokens = self.output_norm(tokens + self.feed_forward(tokens))

        keep_logits = self.keep_head(tokens).squeeze(-1)
        keep_logits = keep_logits.masked_fill(
            ~candidate_mask,
            torch.finfo(keep_logits.dtype).min,
        )
        keep_probabilities = torch.sigmoid(keep_logits)

        predicted_deletion_cost = F.softplus(
            self.deletion_cost_head(tokens).squeeze(-1)
        )
        predicted_deletion_cost = predicted_deletion_cost.masked_fill(
            ~candidate_mask,
            0.0,
        )

        # Sequential pruning uses one action per state.  The last category is
        # an explicit STOP action, so K=0 and "no safe deletion" are first-
        # class outcomes rather than a special count-head convention.
        remove_logits = self.remove_action_head(tokens).squeeze(-1)
        remove_logits = remove_logits.masked_fill(
            ~candidate_mask,
            torch.finfo(remove_logits.dtype).min,
        )
        valid_weight = candidate_mask.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * valid_weight).sum(dim=1) / valid_weight.sum(
            dim=1
        ).clamp_min(1.0)
        stop_logit = self.stop_action_head(pooled).squeeze(-1)
        remove_stop_logits = torch.cat(
            [remove_logits, stop_logit.unsqueeze(-1)], dim=-1
        )
        predicted_log_deletion_cost = torch.log(
            predicted_deletion_cost.clamp_min(
                torch.finfo(predicted_deletion_cost.dtype).tiny
            )
        )

        two_sided_spacing = self._two_sided_spacing(candidate_positions)
        left_slack = (two_sided_spacing[..., 0] - self.min_gap).clamp_min(0.0)
        right_slack = (two_sided_spacing[..., 1] - self.min_gap).clamp_min(0.0)
        residual_signal = torch.tanh(
            self.position_residual_head(tokens).squeeze(-1)
        )
        position_residual = self.max_position_fraction * torch.where(
            residual_signal >= 0.0,
            residual_signal * right_slack,
            residual_signal * left_slack,
        )
        position_residual = position_residual.masked_fill(~candidate_mask, 0.0)
        refined_candidates = candidate_positions + position_residual

        return {
            "pruning_tokens": tokens,
            "analytic_contribution_features": analytic_features,
            "pruning_attention_weights": attention_weights,
            "keep_logits": keep_logits,
            "keep_probabilities": keep_probabilities,
            "keep_probability": keep_probabilities,
            "predicted_deletion_cost": predicted_deletion_cost,
            "remove_logits": remove_logits,
            "stop_logit": stop_logit,
            "remove_stop_logits": remove_stop_logits,
            "remove_stop_probability": F.softmax(remove_stop_logits, dim=-1),
            "predicted_log_deletion_cost": predicted_log_deletion_cost,
            "raw_position_residual": residual_signal,
            "position_residual": position_residual,
            "refined_candidate_positions": refined_candidates,
            "refined_candidate_knots": refined_candidates,
            "candidate_mask": candidate_mask,
        }
