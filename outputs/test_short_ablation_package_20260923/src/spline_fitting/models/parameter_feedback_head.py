from __future__ import annotations

import math

import torch
from torch import nn

from .knot_head import KnotHead


class ParameterFeedbackHead(nn.Module):
    """Correct point-parameter gaps from one-shot structural evidence.

    The head runs after candidate selection and survivor relocation, but before
    the final spline refit.  It reuses the already available all-candidate
    pilot residual and deletion-risk descriptors, so enabling it adds no spline
    solve and does not turn deployment into an iterative network procedure.

    Corrections act on the free part of the strictly positive parameter gaps.
    The terminal projection is initialized to zero, making a newly enabled head
    an identity map for an existing proposal checkpoint.
    """

    _INTERVAL_FEATURE_DIM = 4

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        min_gap: float = 1e-4,
        attention_heads: int = 4,
        risk_bandwidth: float = 0.05,
        max_logit_shift: float = 0.5,
        fusion_mode: str = "cross_attention",
    ) -> None:
        super().__init__()
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        if not math.isfinite(risk_bandwidth) or risk_bandwidth <= 0.0:
            raise ValueError("risk_bandwidth must be finite and positive")
        if not math.isfinite(max_logit_shift) or max_logit_shift <= 0.0:
            raise ValueError("max_logit_shift must be finite and positive")
        if fusion_mode not in {
            "fast_global",
            "fast_structural",
            "gaussian_pool",
            "cross_attention",
        }:
            raise ValueError("unsupported parameter-feedback fusion mode")

        self.hidden_dim = int(hidden_dim)
        self.min_gap = float(min_gap)
        self.risk_bandwidth = float(risk_bandwidth)
        self.max_logit_shift = float(max_logit_shift)
        self.fusion_mode = str(fusion_mode)

        # This scalar exposes the empirically strong chord-length prior
        # directly instead of asking the terminal MLP to rediscover a global
        # convex blend from scratch.  It starts at zero so attaching the head
        # to an old checkpoint is still an exact identity operation.  The
        # feedback calibration stage may seed it (0.6 by default) and then
        # learn it jointly with the structure-conditioned residual correction.
        self.chord_blend_weight = nn.Parameter(torch.zeros(()))

        self.interval_projection = nn.Sequential(
            nn.Linear(self._INTERVAL_FEATURE_DIM, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.interval_input_norm = nn.LayerNorm(self.hidden_dim)
        self.survivor_input_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(self.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.gap_logit_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.gap_logit_head.weight)
        nn.init.zeros_(self.gap_logit_head.bias)
        self.fast_input_projection = nn.Linear(
            self._INTERVAL_FEATURE_DIM + 2,
            32,
        )
        self.fast_gap_logit_head = nn.Linear(32, 1)
        self.fast_chord_adjust_head = nn.Linear(32, 1)
        nn.init.zeros_(self.fast_gap_logit_head.weight)
        nn.init.zeros_(self.fast_gap_logit_head.bias)
        nn.init.zeros_(self.fast_chord_adjust_head.weight)
        nn.init.zeros_(self.fast_chord_adjust_head.bias)
        self.global_input_projection = nn.Linear(5, 16)
        self.global_chord_adjust_head = nn.Linear(16, 1)
        nn.init.zeros_(self.global_chord_adjust_head.weight)
        nn.init.zeros_(self.global_chord_adjust_head.bias)

    @staticmethod
    def _normalized_chord_gaps(points: torch.Tensor) -> torch.Tensor:
        segments = points[:, 1:] - points[:, :-1]
        lengths = segments.norm(dim=-1)
        total = lengths.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(lengths, 1.0 / max(lengths.shape[-1], 1))
        return torch.where(
            total > torch.finfo(lengths.dtype).eps,
            lengths / total.clamp_min(torch.finfo(lengths.dtype).eps),
            uniform,
        )

    @staticmethod
    def _validate_inputs(
        local_features: torch.Tensor,
        points: torch.Tensor,
        proposal_params: torch.Tensor,
        proposal_parameter_gaps: torch.Tensor,
        pilot_reconstructed_points: torch.Tensor,
        risk_candidate_positions: torch.Tensor,
        deletion_delta: torch.Tensor,
        survivor_tokens: torch.Tensor,
        survivor_positions: torch.Tensor,
        survivor_gate: torch.Tensor,
        hard_survivor_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        if local_features.ndim != 3 or points.ndim != 3:
            raise ValueError("local_features and points must have shape [B,M,*]")
        if local_features.shape[:2] != points.shape[:2]:
            raise ValueError("local_features and points must share [B,M]")
        if local_features.shape[-1] != hidden_dim:
            raise ValueError("local feature dimension does not match hidden_dim")
        if proposal_params.shape != points.shape[:2]:
            raise ValueError("proposal_params must have shape [B,M]")
        if proposal_parameter_gaps.shape != (
            points.shape[0],
            points.shape[1] - 1,
        ):
            raise ValueError("proposal_parameter_gaps must have shape [B,M-1]")
        if pilot_reconstructed_points.shape != points.shape:
            raise ValueError("pilot reconstruction and points must share shape")
        if survivor_tokens.ndim != 3:
            raise ValueError("survivor_tokens must have shape [B,K,H]")
        if survivor_tokens.shape[-1] != hidden_dim:
            raise ValueError("survivor token dimension does not match hidden_dim")
        candidate_shape = survivor_tokens.shape[:2]
        for name, value in (
            ("risk_candidate_positions", risk_candidate_positions),
            ("deletion_delta", deletion_delta),
            ("survivor_positions", survivor_positions),
            ("survivor_gate", survivor_gate),
            ("hard_survivor_mask", hard_survivor_mask),
            ("candidate_mask", candidate_mask),
        ):
            if value.shape != candidate_shape:
                raise ValueError(f"{name} must have shape [B,K]")
        if candidate_shape[0] != points.shape[0]:
            raise ValueError("point and candidate tensors must share batch size")
        if candidate_shape[1] == 0:
            raise ValueError("at least one candidate slot is required")

    def _fast_global_feedback(
        self,
        proposal_params: torch.Tensor,
        proposal_parameter_gaps: torch.Tensor,
        chord_gaps: torch.Tensor,
        point_residual: torch.Tensor,
        normalized_delta: torch.Tensor,
        survivor_gate: torch.Tensor,
        hard_survivor_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Cheap per-curve structural feedback for latency-sensitive deployment."""

        batch, gap_count = proposal_parameter_gaps.shape
        candidate_float = candidate_mask.to(survivor_gate.dtype)
        active = hard_survivor_mask.to(torch.bool) & candidate_mask.to(torch.bool)
        active_strength = active.to(survivor_gate.dtype) * survivor_gate.clamp(0.0, 1.0)
        removed_strength = (1.0 - survivor_gate.clamp(0.0, 1.0)) * candidate_float
        kept_risk = (normalized_delta * active_strength).sum(
            dim=-1
        ) / active_strength.sum(dim=-1).clamp_min(1.0)
        removed_risk = (normalized_delta * removed_strength).sum(
            dim=-1
        ) / removed_strength.sum(dim=-1).clamp_min(1.0)
        keep_fraction = active.to(proposal_params.dtype).sum(
            dim=-1
        ) / candidate_float.sum(dim=-1).clamp_min(1.0)
        summary = torch.stack(
            [
                torch.log1p(100.0 * point_residual.mean(dim=-1)),
                torch.log1p(100.0 * point_residual.amax(dim=-1)),
                keep_fraction,
                kept_risk,
                removed_risk,
            ],
            dim=-1,
        )
        global_token = torch.nn.functional.gelu(self.global_input_projection(summary))
        chord_adjustment = 0.15 * torch.tanh(
            self.global_chord_adjust_head(global_token)
        )
        chord_blend_weight = (self.chord_blend_weight + chord_adjustment).clamp(
            0.0, 1.0
        )
        free_budget = 1.0 - self.min_gap * gap_count
        projected_chord_gaps = self.min_gap + free_budget * chord_gaps
        corrected_gaps = (
            1.0 - chord_blend_weight
        ) * proposal_parameter_gaps + chord_blend_weight * projected_chord_gaps
        corrected_gap_logits = corrected_gaps.log()
        raw_logit_signal = torch.zeros_like(corrected_gaps)
        gap_logit_delta = torch.zeros_like(corrected_gaps)
        corrected_params = torch.cat(
            [
                proposal_params.new_zeros(batch, 1),
                torch.cumsum(corrected_gaps, dim=-1),
            ],
            dim=-1,
        )
        corrected_params[:, -1] = 1.0

        residual_scale = point_residual.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized_residual = torch.log1p(point_residual / residual_scale)
        interval_residual = 0.5 * (
            normalized_residual[:, :-1] + normalized_residual[:, 1:]
        )
        empty = ~active.any(dim=-1)
        fallback_index = candidate_mask.to(torch.long).argmax(dim=-1, keepdim=True)
        fallback_mask = torch.zeros_like(active).scatter(1, fallback_index, True)
        safe_memory_mask = active | (fallback_mask & empty.unsqueeze(-1))
        return {
            "feedback_params": corrected_params,
            "feedback_parameter_gaps": corrected_gaps,
            "feedback_raw_parameter_gaps": corrected_gap_logits,
            "parameter_feedback_gap_logit_delta": gap_logit_delta,
            "parameter_feedback_raw_logit_signal": raw_logit_signal,
            "parameter_feedback_interval_tokens": global_token.unsqueeze(1).expand(
                -1, gap_count, -1
            ),
            "parameter_feedback_chord_gaps": chord_gaps,
            "parameter_feedback_chord_blend_weight": chord_blend_weight[:, 0],
            "parameter_feedback_interval_residual": interval_residual,
            "parameter_feedback_removal_risk": removed_risk.unsqueeze(-1).expand(
                -1, gap_count
            ),
            "parameter_feedback_safe_memory_mask": safe_memory_mask,
        }

    def forward(
        self,
        local_features: torch.Tensor,
        points: torch.Tensor,
        proposal_params: torch.Tensor,
        proposal_parameter_gaps: torch.Tensor,
        pilot_reconstructed_points: torch.Tensor,
        *,
        risk_candidate_positions: torch.Tensor,
        deletion_delta: torch.Tensor,
        survivor_tokens: torch.Tensor,
        survivor_positions: torch.Tensor,
        survivor_gate: torch.Tensor,
        hard_survivor_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return a strictly increasing corrected parameterization.

        Candidate tensors use the fixed proposal slot layout ``[B,K]`` even
        when no slot survives.  In that case a zero-valued fallback memory key
        keeps multi-head attention finite while local residual/chord evidence
        remains available to every interval query.
        """

        self._validate_inputs(
            local_features,
            points,
            proposal_params,
            proposal_parameter_gaps,
            pilot_reconstructed_points,
            risk_candidate_positions,
            deletion_delta,
            survivor_tokens,
            survivor_positions,
            survivor_gate,
            hard_survivor_mask,
            candidate_mask,
            self.hidden_dim,
        )

        batch, num_points, _ = points.shape
        num_gaps = num_points - 1
        if num_gaps < 1:
            raise ValueError("parameter feedback requires at least two points")
        free_budget = 1.0 - self.min_gap * num_gaps
        if free_budget <= 0.0:
            raise ValueError("min_gap leaves no free parameter budget")

        interval_midpoints = 0.5 * (proposal_params[:, :-1] + proposal_params[:, 1:])
        interval_local = 0.5 * (local_features[:, :-1] + local_features[:, 1:])
        chord_gaps = self._normalized_chord_gaps(points)

        point_residual = (pilot_reconstructed_points - points).norm(dim=-1)
        residual_scale = point_residual.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized_point_residual = torch.log1p(point_residual / residual_scale)
        interval_residual = 0.5 * (
            normalized_point_residual[:, :-1] + normalized_point_residual[:, 1:]
        )

        finite_delta = torch.nan_to_num(
            deletion_delta,
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        ).clamp(min=0.0, max=1e6)
        log_delta = torch.log1p(finite_delta)
        delta_scale = log_delta.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized_delta = log_delta / delta_scale
        if self.fusion_mode == "fast_global":
            return self._fast_global_feedback(
                proposal_params,
                proposal_parameter_gaps,
                chord_gaps,
                point_residual,
                normalized_delta,
                survivor_gate,
                hard_survivor_mask,
                candidate_mask,
            )
        removed_weight = (1.0 - survivor_gate.clamp(0.0, 1.0)) * candidate_mask.to(
            survivor_gate.dtype
        )
        risk_distance = interval_midpoints.unsqueeze(
            -1
        ) - risk_candidate_positions.unsqueeze(1)
        risk_kernel = torch.exp(
            -0.5 * (risk_distance / self.risk_bandwidth).square()
        ) * candidate_mask.to(risk_distance.dtype).unsqueeze(1)
        removal_risk = (
            risk_kernel * removed_weight.unsqueeze(1) * normalized_delta.unsqueeze(1)
        ).sum(dim=-1) / risk_kernel.sum(dim=-1).clamp_min(1e-8)

        safe_proposal_gap = proposal_parameter_gaps.clamp_min(1e-8)
        safe_chord_gap = chord_gaps.clamp_min(1e-8)
        interval_features = torch.stack(
            [
                safe_proposal_gap.log(),
                (safe_chord_gap / safe_proposal_gap).log().clamp(-8.0, 8.0),
                interval_residual,
                removal_risk,
            ],
            dim=-1,
        )
        active = hard_survivor_mask.to(torch.bool) & candidate_mask.to(torch.bool)
        empty = ~active.any(dim=-1)
        fallback_index = candidate_mask.to(torch.long).argmax(dim=-1, keepdim=True)
        fallback_mask = torch.zeros_like(active).scatter(1, fallback_index, True)
        safe_memory_mask = active | (fallback_mask & empty.unsqueeze(-1))
        attention_weights: torch.Tensor | None
        if self.fusion_mode == "fast_structural":
            survivor_distance = (
                interval_midpoints.unsqueeze(-1) - survivor_positions.unsqueeze(1)
            ).abs()
            active_strength = active.to(survivor_gate.dtype) * survivor_gate.clamp(
                0.0, 1.0
            )
            survivor_kernel = torch.exp(
                -0.5 * (survivor_distance / self.risk_bandwidth).square()
            ) * active_strength.unsqueeze(1)
            survivor_density = survivor_kernel.sum(dim=-1) / active_strength.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            nearest_survivor = torch.where(
                active.unsqueeze(1),
                survivor_distance,
                torch.ones_like(survivor_distance),
            ).amin(dim=-1)
            nearest_survivor = torch.where(
                empty.unsqueeze(-1),
                torch.ones_like(nearest_survivor),
                nearest_survivor,
            )
            fast_features = torch.cat(
                [
                    interval_features,
                    survivor_density.unsqueeze(-1),
                    (nearest_survivor / self.risk_bandwidth)
                    .clamp(0.0, 8.0)
                    .unsqueeze(-1),
                ],
                dim=-1,
            )
            feedback_tokens = torch.nn.functional.gelu(
                self.fast_input_projection(fast_features)
            )
            raw_logit_signal = torch.tanh(
                self.fast_gap_logit_head(feedback_tokens).squeeze(-1)
            )
            chord_adjustment = 0.15 * torch.tanh(
                self.fast_chord_adjust_head(feedback_tokens.mean(dim=1))
            )
            chord_blend_weight = (self.chord_blend_weight + chord_adjustment).clamp(
                0.0, 1.0
            )
            attention_weights = None
        else:
            interval_position_encoding = KnotHead._sinusoidal_position_encoding(
                interval_midpoints,
                self.hidden_dim,
            )
            interval_queries = self.interval_input_norm(
                interval_local
                + interval_position_encoding
                + self.interval_projection(interval_features)
            )
            survivor_position_encoding = KnotHead._sinusoidal_position_encoding(
                survivor_positions,
                self.hidden_dim,
            )
            survivor_memory = self.survivor_input_norm(
                survivor_tokens + survivor_position_encoding
            )
            survivor_values = survivor_memory * survivor_gate.unsqueeze(-1)

        if self.fusion_mode == "gaussian_pool":
            survivor_distance = interval_midpoints.unsqueeze(
                -1
            ) - survivor_positions.unsqueeze(1)
            local_weights = torch.exp(
                -0.5 * (survivor_distance / self.risk_bandwidth).square()
            )
            local_weights = local_weights * active.to(local_weights.dtype).unsqueeze(1)
            normalized_local_weights = local_weights / local_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            attended = normalized_local_weights @ survivor_values
            attention_weights = (
                normalized_local_weights if return_attention_weights else None
            )
        elif self.fusion_mode == "cross_attention":
            attended, attention_weights = self.cross_attention(
                interval_queries,
                survivor_values,
                survivor_values,
                key_padding_mask=~safe_memory_mask,
                need_weights=return_attention_weights,
                average_attn_weights=True,
            )
        if self.fusion_mode != "fast_structural":
            feedback_tokens = self.cross_attention_norm(interval_queries + attended)
            feedback_tokens = self.output_norm(
                feedback_tokens + self.feed_forward(feedback_tokens)
            )
            raw_logit_signal = torch.tanh(
                self.gap_logit_head(feedback_tokens).squeeze(-1)
            )
            chord_blend_weight = self.chord_blend_weight.clamp(0.0, 1.0)
        gap_logit_delta = self.max_logit_shift * raw_logit_signal

        base_free_gap = (proposal_parameter_gaps - self.min_gap).clamp_min(
            torch.finfo(proposal_parameter_gaps.dtype).tiny
        )
        proposal_free_weight = base_free_gap / base_free_gap.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(proposal_parameter_gaps.dtype).tiny)
        chord_free_weight = chord_gaps / chord_gaps.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(chord_gaps.dtype).tiny
        )
        base_free_weight = (
            1.0 - chord_blend_weight
        ) * proposal_free_weight + chord_blend_weight * chord_free_weight
        base_free_weight = base_free_weight.clamp_min(
            torch.finfo(base_free_weight.dtype).tiny
        )
        base_free_weight = base_free_weight / base_free_weight.sum(dim=-1, keepdim=True)
        corrected_gap_logits = base_free_weight.log() + gap_logit_delta
        corrected_free_weight = torch.softmax(corrected_gap_logits, dim=-1)
        corrected_gaps = self.min_gap + free_budget * corrected_free_weight
        corrected_params = torch.cat(
            [
                proposal_params.new_zeros(batch, 1),
                torch.cumsum(corrected_gaps, dim=-1),
            ],
            dim=-1,
        )
        corrected_params[:, -1] = 1.0

        result = {
            "feedback_params": corrected_params,
            "feedback_parameter_gaps": corrected_gaps,
            "feedback_raw_parameter_gaps": corrected_gap_logits,
            "parameter_feedback_gap_logit_delta": gap_logit_delta,
            "parameter_feedback_raw_logit_signal": raw_logit_signal,
            "parameter_feedback_interval_tokens": feedback_tokens,
            "parameter_feedback_chord_gaps": chord_gaps,
            "parameter_feedback_chord_blend_weight": (
                chord_blend_weight.expand(batch)
                if chord_blend_weight.ndim == 0
                else chord_blend_weight.reshape(batch)
            ),
            "parameter_feedback_interval_residual": interval_residual,
            "parameter_feedback_removal_risk": removal_risk,
            "parameter_feedback_safe_memory_mask": safe_memory_mask,
        }
        if attention_weights is not None:
            result["parameter_feedback_attention_weights"] = (
                attention_weights * active.to(attention_weights.dtype).unsqueeze(1)
            )
        return result
