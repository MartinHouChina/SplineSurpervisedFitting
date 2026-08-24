from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .knot_head import KnotHead


class InteractivePruningHead(nn.Module):
    """Select and refine candidate knots using a curve-adaptive threshold.

    Each candidate receives its truncated-power coefficient energy, the
    measured/approximated objective increase after deletion, two-sided spacing,
    and local residual features.  Self-attention lets candidates reason about
    redundancy jointly instead of making independent threshold decisions.

    A single global threshold is predicted from the pooled candidate state.  A
    knot is kept according to ``sigmoid(raw_importance - threshold)`` rather
    than an unrelated fixed 0.5 classifier threshold.  The resulting soft keep
    state is pooled back into every candidate before position refinement, which
    couples selection and location in one head.

    Position updates remain bounded to less than half of the available
    neighboring slack.  Consequently ordered inputs remain strictly ordered
    without sorting, preserving candidate/output correspondence.
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
        one_shot_adaptive: bool = False,
    ) -> None:
        super().__init__()
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if residual_feature_dim <= 0:
            raise ValueError("residual_feature_dim must be positive")
        if min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        if not 0.0 < max_position_fraction < 0.5:
            raise ValueError(
                "max_position_fraction must lie strictly between 0 and 0.5"
            )
        if not 0.0 < initial_keep_probability < 1.0:
            raise ValueError("initial_keep_probability must lie in (0,1)")

        self.hidden_dim = int(hidden_dim)
        self.residual_feature_dim = int(residual_feature_dim)
        self.min_gap = float(min_gap)
        self.max_position_fraction = float(max_position_fraction)
        self.one_shot_adaptive = bool(one_shot_adaptive)

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
        if self.one_shot_adaptive:
            self.adaptive_threshold_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.normal_(self.adaptive_threshold_head[-1].weight, std=0.01)
            nn.init.zeros_(self.adaptive_threshold_head[-1].bias)
            self.keep_context_projection = nn.Sequential(
                nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.keep_context_norm = nn.LayerNorm(self.hidden_dim)
            # A separately named position -> selection path lets staged
            # training unfreeze the bidirectional feedback without also
            # changing the candidate encoder or the position decoder.  It is
            # intentionally created only for v8 one-shot heads: legacy v7
            # state_dict layouts remain byte-for-byte unchanged.
            self.position_to_keep_feedback = nn.Sequential(
                nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            nn.init.normal_(self.position_to_keep_feedback[-1].weight, std=0.01)
            nn.init.zeros_(self.position_to_keep_feedback[-1].bias)
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

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Early one-shot v8 checkpoints predate the explicit position->keep
        # feedback block.  Supplying this block's constructor initialization
        # keeps strict loading usable while still loading every historical
        # tensor exactly.  Legacy v7 heads never construct the block, so their
        # 30-key layout and strict-loading behavior are untouched.
        if self.one_shot_adaptive:
            feedback_prefix = prefix + "position_to_keep_feedback."
            for name, value in self.position_to_keep_feedback.state_dict().items():
                key = feedback_prefix + name
                if key not in state_dict:
                    state_dict[key] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        if (
            candidate_mask.shape != expected_scalar_shape
            or candidate_mask.dtype != torch.bool
        ):
            raise ValueError("candidate_mask must be boolean with shape [B,K]")
        if candidate_positions.shape[1] == 0:
            raise ValueError("at least one candidate is required")
        if torch.any(~candidate_mask.any(dim=-1)):
            raise ValueError("every sample must contain at least one active candidate")

        # The bounded refinement guarantee assumes a valid ordered input.  The
        # small tolerance avoids false failures from floating-point round-off.
        full_spacing = self._two_sided_spacing(candidate_positions)
        if torch.any(full_spacing < self.min_gap - 1e-6):
            raise ValueError("candidate_positions must be ordered and respect min_gap")

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
        global_features: torch.Tensor | None = None,
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
        if global_features is not None and global_features.shape != (
            candidate_positions.shape[0],
            self.hidden_dim,
        ):
            raise ValueError("global_features must have shape [B,H]")

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

        valid_weight = candidate_mask.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * valid_weight).sum(dim=1) / valid_weight.sum(dim=1).clamp_min(
            1.0
        )
        if self.one_shot_adaptive:
            # Fixed-pass bidirectional interaction (there is deliberately no
            # data-dependent loop):
            #   keep_0 -> provisional position -> keep_1 -> final position.
            preliminary_raw_importance = self.keep_head(tokens).squeeze(-1)
            preliminary_raw_importance = preliminary_raw_importance.masked_fill(
                ~candidate_mask,
                torch.finfo(preliminary_raw_importance.dtype).min,
            )
            preliminary_threshold_context = pooled
            normalized_global = None
            if global_features is not None:
                normalized_global = F.layer_norm(
                    torch.nan_to_num(global_features),
                    (self.hidden_dim,),
                )
                preliminary_threshold_context = 0.5 * (pooled + normalized_global)
            preliminary_adaptive_keep_threshold = self.adaptive_threshold_head(
                preliminary_threshold_context
            ).squeeze(-1)
            preliminary_keep_logits = (
                preliminary_raw_importance
                - preliminary_adaptive_keep_threshold.unsqueeze(-1)
            )
            preliminary_keep_logits = preliminary_keep_logits.masked_fill(
                ~candidate_mask,
                torch.finfo(preliminary_keep_logits.dtype).min,
            )
            preliminary_keep_probabilities = torch.sigmoid(preliminary_keep_logits)

            two_sided_spacing = self._two_sided_spacing(candidate_positions)
            left_slack = (two_sided_spacing[..., 0] - self.min_gap).clamp_min(0.0)
            right_slack = (two_sided_spacing[..., 1] - self.min_gap).clamp_min(0.0)

            preliminary_soft_keep_weight = (
                preliminary_keep_probabilities * candidate_mask.to(tokens.dtype)
            )
            preliminary_soft_keep_context = (
                tokens * preliminary_soft_keep_weight.unsqueeze(-1)
            ).sum(dim=1) / preliminary_soft_keep_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            provisional_context = preliminary_soft_keep_context.unsqueeze(1).expand_as(
                tokens
            )
            provisional_keep_interaction = torch.cat(
                [
                    tokens,
                    provisional_context,
                    tokens * torch.tanh(provisional_context),
                    preliminary_keep_probabilities.unsqueeze(-1),
                ],
                dim=-1,
            )
            provisional_position_refinement_tokens = self.keep_context_norm(
                tokens + self.keep_context_projection(provisional_keep_interaction)
            )
            provisional_raw_position_residual = torch.tanh(
                self.position_residual_head(
                    provisional_position_refinement_tokens
                ).squeeze(-1)
            )
            provisional_keep_conditioned_position_signal = (
                provisional_raw_position_residual * preliminary_keep_probabilities
            )
            provisional_position_residual = self.max_position_fraction * torch.where(
                provisional_keep_conditioned_position_signal >= 0.0,
                provisional_keep_conditioned_position_signal * right_slack,
                provisional_keep_conditioned_position_signal * left_slack,
            )
            provisional_position_residual = provisional_position_residual.masked_fill(
                ~candidate_mask,
                0.0,
            )
            provisional_candidates = candidate_positions + provisional_position_residual

            # Feed the *actual provisional locations* back into the keep
            # decision.  Subtracting the original encoding isolates the
            # proposed motion while preserving the local provisional token.
            provisional_position_encoding = KnotHead._sinusoidal_position_encoding(
                provisional_candidates,
                self.hidden_dim,
            )
            position_feedback_encoding = (
                provisional_position_encoding - position_encoding
            )
            position_feedback_state = (
                provisional_position_refinement_tokens + position_feedback_encoding
            )
            position_feedback_interaction = torch.cat(
                [
                    tokens,
                    position_feedback_state,
                    tokens * torch.tanh(position_feedback_state),
                    preliminary_keep_probabilities.unsqueeze(-1),
                ],
                dim=-1,
            )
            final_decision_tokens = self.keep_context_norm(
                tokens + self.position_to_keep_feedback(position_feedback_interaction)
            )
            final_pooled = (final_decision_tokens * valid_weight).sum(
                dim=1
            ) / valid_weight.sum(dim=1).clamp_min(1.0)
            raw_importance = self.keep_head(final_decision_tokens).squeeze(-1)
            raw_importance = raw_importance.masked_fill(
                ~candidate_mask,
                torch.finfo(raw_importance.dtype).min,
            )
            threshold_context = final_pooled
            if normalized_global is not None:
                threshold_context = 0.5 * (final_pooled + normalized_global)
            adaptive_keep_threshold = self.adaptive_threshold_head(
                threshold_context
            ).squeeze(-1)
            keep_logits = raw_importance - adaptive_keep_threshold.unsqueeze(-1)
        else:
            # Exact v7 behavior: keep_head directly emits the Bernoulli logits.
            raw_importance = self.keep_head(tokens).squeeze(-1)
            raw_importance = raw_importance.masked_fill(
                ~candidate_mask,
                torch.finfo(raw_importance.dtype).min,
            )
            adaptive_keep_threshold = raw_importance.new_zeros(raw_importance.shape[0])
            keep_logits = raw_importance
            preliminary_raw_importance = raw_importance
            preliminary_adaptive_keep_threshold = adaptive_keep_threshold
            preliminary_keep_logits = keep_logits
            final_decision_tokens = tokens
        keep_logits = keep_logits.masked_fill(
            ~candidate_mask,
            torch.finfo(keep_logits.dtype).min,
        )
        keep_probabilities = torch.sigmoid(keep_logits)
        if not self.one_shot_adaptive:
            preliminary_keep_probabilities = keep_probabilities

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
        stop_logit = self.stop_action_head(pooled).squeeze(-1)
        remove_stop_logits = torch.cat(
            [remove_logits, stop_logit.unsqueeze(-1)], dim=-1
        )
        predicted_log_deletion_cost = torch.log(
            predicted_deletion_cost.clamp_min(
                torch.finfo(predicted_deletion_cost.dtype).tiny
            )
        )

        if self.one_shot_adaptive:
            # ``soft_keep_context`` remains a useful diagnostic of the final
            # probabilities.  The actual final location path below does not
            # use it: its forward value contains hard-kept candidates only.
            final_soft_keep_weight = keep_probabilities * candidate_mask.to(
                tokens.dtype
            )
            soft_keep_context = (
                final_decision_tokens * final_soft_keep_weight.unsqueeze(-1)
            ).sum(dim=1) / final_soft_keep_weight.sum(dim=1, keepdim=True).clamp_min(
                1e-6
            )

            final_hard_keep_mask = (keep_probabilities >= 0.5) & candidate_mask
            hard_keep_weight = final_hard_keep_mask.to(tokens.dtype)
            # Straight-through KeepMask: exactly hard in the forward pass,
            # with the sigmoid probability supplying the backward derivative.
            final_hard_st_keep_gate = (
                hard_keep_weight + keep_probabilities - keep_probabilities.detach()
            ) * candidate_mask.to(tokens.dtype)
            hard_count = hard_keep_weight.sum(dim=1, keepdim=True)
            st_count = final_hard_st_keep_gate.sum(dim=1, keepdim=True)
            safe_context_denominator = (
                hard_count.clamp_min(1.0) + st_count - st_count.detach()
            )
            final_hard_st_keep_context = (
                final_decision_tokens * final_hard_st_keep_gate.unsqueeze(-1)
            ).sum(dim=1) / safe_context_denominator
            final_context = final_hard_st_keep_context.unsqueeze(1).expand_as(
                final_decision_tokens
            )
            final_keep_interaction = torch.cat(
                [
                    final_decision_tokens,
                    final_context,
                    final_decision_tokens * torch.tanh(final_context),
                    final_hard_st_keep_gate.unsqueeze(-1),
                ],
                dim=-1,
            )
            position_refinement_tokens = self.keep_context_norm(
                final_decision_tokens
                + self.keep_context_projection(final_keep_interaction)
            )
            residual_signal = torch.tanh(
                self.position_residual_head(position_refinement_tokens).squeeze(-1)
            )
            keep_conditioned_signal = residual_signal * final_hard_st_keep_gate
        else:
            # Keep the old position path bit-for-bit: no selection context was
            # used by v7 checkpoints.
            two_sided_spacing = self._two_sided_spacing(candidate_positions)
            left_slack = (two_sided_spacing[..., 0] - self.min_gap).clamp_min(0.0)
            right_slack = (two_sided_spacing[..., 1] - self.min_gap).clamp_min(0.0)
            soft_keep_weight = keep_probabilities * candidate_mask.to(tokens.dtype)
            soft_keep_context = (tokens * soft_keep_weight.unsqueeze(-1)).sum(
                dim=1
            ) / soft_keep_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
            position_refinement_tokens = tokens
            residual_signal = torch.tanh(
                self.position_residual_head(position_refinement_tokens).squeeze(-1)
            )
            keep_conditioned_signal = residual_signal
        position_residual = self.max_position_fraction * torch.where(
            keep_conditioned_signal >= 0.0,
            keep_conditioned_signal * right_slack,
            keep_conditioned_signal * left_slack,
        )
        position_residual = position_residual.masked_fill(~candidate_mask, 0.0)
        refined_candidates = candidate_positions + position_residual

        if not self.one_shot_adaptive:
            # Diagnostic aliases only.  They add no parameters and leave all
            # legacy output tensors and computations unchanged.
            preliminary_soft_keep_context = soft_keep_context
            provisional_position_refinement_tokens = position_refinement_tokens
            provisional_raw_position_residual = residual_signal
            provisional_keep_conditioned_position_signal = keep_conditioned_signal
            provisional_position_residual = position_residual
            provisional_candidates = refined_candidates
            position_feedback_encoding = torch.zeros_like(tokens)
            position_feedback_state = tokens
            final_hard_keep_mask = (keep_probabilities >= 0.5) & candidate_mask
            final_hard_st_keep_gate = (
                final_hard_keep_mask.to(tokens.dtype)
                + keep_probabilities
                - keep_probabilities.detach()
            ) * candidate_mask.to(tokens.dtype)
            hard_count = final_hard_keep_mask.to(tokens.dtype).sum(dim=1, keepdim=True)
            st_count = final_hard_st_keep_gate.sum(dim=1, keepdim=True)
            safe_context_denominator = (
                hard_count.clamp_min(1.0) + st_count - st_count.detach()
            )
            final_hard_st_keep_context = (
                tokens * final_hard_st_keep_gate.unsqueeze(-1)
            ).sum(dim=1) / safe_context_denominator

        return {
            "pruning_tokens": tokens,
            "analytic_contribution_features": analytic_features,
            "pruning_attention_weights": attention_weights,
            "preliminary_raw_importance": preliminary_raw_importance,
            "preliminary_adaptive_keep_threshold": (
                preliminary_adaptive_keep_threshold
            ),
            "preliminary_keep_logits": preliminary_keep_logits,
            "preliminary_keep_probabilities": preliminary_keep_probabilities,
            "preliminary_keep_probability": preliminary_keep_probabilities,
            "preliminary_soft_keep_context": preliminary_soft_keep_context,
            "provisional_position_refinement_tokens": (
                provisional_position_refinement_tokens
            ),
            "provisional_raw_position_residual": (provisional_raw_position_residual),
            "provisional_keep_conditioned_position_signal": (
                provisional_keep_conditioned_position_signal
            ),
            "provisional_position_residual": provisional_position_residual,
            "provisional_candidate_positions": provisional_candidates,
            "provisional_candidate_knots": provisional_candidates,
            "position_feedback_encoding": position_feedback_encoding,
            "position_feedback_tokens": position_feedback_state,
            "final_decision_tokens": final_decision_tokens,
            "raw_importance": raw_importance,
            "raw_keep_importance": raw_importance,
            "final_raw_importance": raw_importance,
            "adaptive_keep_threshold": adaptive_keep_threshold,
            "adaptive_keep_logit_threshold": adaptive_keep_threshold,
            "adaptive_raw_importance_probability_threshold": torch.sigmoid(
                adaptive_keep_threshold
            ),
            "keep_threshold": adaptive_keep_threshold,
            "keep_logits": keep_logits,
            "keep_probabilities": keep_probabilities,
            "keep_probability": keep_probabilities,
            "final_keep_logits": keep_logits,
            "final_keep_probabilities": keep_probabilities,
            "final_keep_probability": keep_probabilities,
            "final_hard_keep_mask": final_hard_keep_mask,
            "final_hard_st_keep_gate": final_hard_st_keep_gate,
            "final_hard_st_keep_context": final_hard_st_keep_context,
            "predicted_deletion_cost": predicted_deletion_cost,
            "remove_logits": remove_logits,
            "stop_logit": stop_logit,
            "remove_stop_logits": remove_stop_logits,
            "remove_stop_probability": F.softmax(remove_stop_logits, dim=-1),
            "predicted_log_deletion_cost": predicted_log_deletion_cost,
            "soft_keep_context": soft_keep_context,
            "position_refinement_tokens": position_refinement_tokens,
            "raw_position_residual": residual_signal,
            "keep_conditioned_position_signal": keep_conditioned_signal,
            "position_residual": position_residual,
            "refined_candidate_positions": refined_candidates,
            "refined_candidate_knots": refined_candidates,
            "candidate_mask": candidate_mask,
        }
