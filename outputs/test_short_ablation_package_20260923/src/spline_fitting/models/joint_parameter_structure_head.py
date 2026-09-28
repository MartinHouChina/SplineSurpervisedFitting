from __future__ import annotations

import math

import torch
from torch import nn

from .knot_head import KnotHead


class JointParameterStructureHead(nn.Module):
    """Revisit candidate selection after parameter feedback.

    The first pruning pass supplies a structural hypothesis to the parameter
    feedback head.  This module closes the loop: the same candidate tokens are
    queried against geometry encoded at the corrected parameters, then produce
    bounded residual corrections for Keep logits and candidate locations.

    All three terminal projections are zero initialised.  Attaching the module
    to an existing checkpoint is therefore neutral until the joint calibration
    stage is trained.
    """

    _SCALAR_FEATURE_DIM = 8

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        attention_heads: int = 4,
        local_bandwidth: float = 0.08,
        max_keep_logit_shift: float = 2.0,
    ) -> None:
        super().__init__()
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if not math.isfinite(local_bandwidth) or local_bandwidth <= 0.0:
            raise ValueError("local_bandwidth must be finite and positive")
        if not math.isfinite(max_keep_logit_shift) or max_keep_logit_shift <= 0.0:
            raise ValueError("max_keep_logit_shift must be finite and positive")

        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.local_bandwidth = float(local_bandwidth)
        self.max_keep_logit_shift = float(max_keep_logit_shift)

        self.scalar_projection = nn.Sequential(
            nn.Linear(self._SCALAR_FEATURE_DIM, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.local_cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.attention_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        self.token_delta_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.keep_delta_head = nn.Linear(self.hidden_dim, 1)
        self.position_delta_head = nn.Linear(self.hidden_dim, 1)
        self.relocation_scale = nn.Parameter(torch.zeros(()))
        for module in (
            self.token_delta_head,
            self.keep_delta_head,
            self.position_delta_head,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _normalise_nonnegative(values: torch.Tensor) -> torch.Tensor:
        finite = torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=0.0)
        transformed = torch.log1p(finite.clamp(min=0.0, max=1e6))
        scale = transformed.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return (transformed / scale).clamp(0.0, 8.0)

    @staticmethod
    def _validate_inputs(
        local_features: torch.Tensor,
        corrected_params: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_positions: torch.Tensor,
        preliminary_keep_logits: torch.Tensor,
        deletion_delta: torch.Tensor,
        candidate_local_residual: torch.Tensor,
        preliminary_position_residual: torch.Tensor,
        preliminary_keep_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        if local_features.ndim != 3:
            raise ValueError("local_features must have shape [B,M,H]")
        if local_features.shape[-1] != hidden_dim:
            raise ValueError("local feature dimension does not match hidden_dim")
        if corrected_params.shape != local_features.shape[:2]:
            raise ValueError("corrected_params must have shape [B,M]")
        if candidate_tokens.ndim != 3 or candidate_tokens.shape[-1] != hidden_dim:
            raise ValueError("candidate_tokens must have shape [B,K,H]")
        candidate_shape = candidate_tokens.shape[:2]
        if candidate_shape[0] != local_features.shape[0]:
            raise ValueError("point and candidate tensors must share batch size")
        for name, value in (
            ("candidate_positions", candidate_positions),
            ("preliminary_keep_logits", preliminary_keep_logits),
            ("deletion_delta", deletion_delta),
            ("candidate_local_residual", candidate_local_residual),
            ("preliminary_position_residual", preliminary_position_residual),
            ("preliminary_keep_mask", preliminary_keep_mask),
            ("candidate_mask", candidate_mask),
        ):
            if value.shape != candidate_shape:
                raise ValueError(f"{name} must have shape [B,K]")
        if (
            preliminary_keep_mask.dtype != torch.bool
            or candidate_mask.dtype != torch.bool
        ):
            raise ValueError("candidate masks must be boolean")

    def forward(
        self,
        local_features: torch.Tensor,
        corrected_params: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_positions: torch.Tensor,
        *,
        preliminary_keep_logits: torch.Tensor,
        deletion_delta: torch.Tensor,
        candidate_local_residual: torch.Tensor,
        preliminary_position_residual: torch.Tensor,
        preliminary_keep_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return residual joint updates in the corrected parameter domain."""

        self._validate_inputs(
            local_features,
            corrected_params,
            candidate_tokens,
            candidate_positions,
            preliminary_keep_logits,
            deletion_delta,
            candidate_local_residual,
            preliminary_position_residual,
            preliminary_keep_mask,
            candidate_mask,
            self.hidden_dim,
        )

        corrected_point_encoding = KnotHead._sinusoidal_position_encoding(
            corrected_params,
            self.hidden_dim,
        )
        candidate_encoding = KnotHead._sinusoidal_position_encoding(
            candidate_positions,
            self.hidden_dim,
        )
        keep_probability = torch.sigmoid(preliminary_keep_logits)
        candidate_spacing = torch.cat(
            [
                candidate_positions[:, :1],
                candidate_positions[:, 1:] - candidate_positions[:, :-1],
                1.0 - candidate_positions[:, -1:],
            ],
            dim=-1,
        )
        left_spacing = candidate_spacing[:, :-1]
        right_spacing = candidate_spacing[:, 1:]
        scalar_features = torch.stack(
            [
                candidate_positions,
                keep_probability,
                preliminary_keep_logits.clamp(-8.0, 8.0) / 8.0,
                self._normalise_nonnegative(deletion_delta),
                self._normalise_nonnegative(candidate_local_residual),
                preliminary_position_residual.clamp(-0.5, 0.5) * 2.0,
                left_spacing,
                right_spacing,
            ],
            dim=-1,
        )
        scalar_features = scalar_features * candidate_mask.to(
            scalar_features.dtype
        ).unsqueeze(-1)
        query = self.query_norm(
            candidate_tokens
            + candidate_encoding
            + self.scalar_projection(scalar_features)
        )
        memory = self.memory_norm(local_features + corrected_point_encoding)

        # A Gaussian log prior makes attention local without hard window edges.
        # MultiheadAttention accepts a different additive mask for every
        # batch/head pair as [B*heads,K,M].
        distance = candidate_positions.unsqueeze(-1) - corrected_params.unsqueeze(1)
        gaussian_log_prior = -0.5 * (distance / self.local_bandwidth).square()
        gaussian_log_prior = gaussian_log_prior.clamp(min=-30.0, max=0.0)
        attention_mask = (
            gaussian_log_prior.unsqueeze(1)
            .expand(-1, self.attention_heads, -1, -1)
            .reshape(
                candidate_positions.shape[0] * self.attention_heads,
                candidate_positions.shape[1],
                corrected_params.shape[1],
            )
        )
        attended, attention_weights = self.local_cross_attention(
            query,
            memory,
            memory,
            attn_mask=attention_mask,
            need_weights=return_attention_weights,
            average_attn_weights=True,
        )
        interacted = self.attention_norm(query + attended)
        interacted = self.output_norm(interacted + self.feed_forward(interacted))
        valid = candidate_mask.to(interacted.dtype).unsqueeze(-1)
        interacted = interacted * valid

        token_delta = self.token_delta_head(interacted) * valid
        joint_tokens = candidate_tokens + token_delta
        keep_logit_delta = self.max_keep_logit_shift * torch.tanh(
            self.keep_delta_head(interacted).squeeze(-1)
        )
        keep_logit_delta = keep_logit_delta.masked_fill(~candidate_mask, 0.0)
        raw_position_signal = torch.tanh(
            self.position_delta_head(interacted).squeeze(-1)
        ).masked_fill(~candidate_mask, 0.0)
        final_keep_logits = preliminary_keep_logits + keep_logit_delta
        final_keep_probability = torch.sigmoid(final_keep_logits) * candidate_mask.to(
            final_keep_logits.dtype
        )

        result = {
            "joint_structure_tokens": joint_tokens,
            "joint_structure_token_delta": token_delta,
            "joint_keep_logit_delta": keep_logit_delta,
            "joint_keep_logits": final_keep_logits,
            "joint_keep_probability": final_keep_probability,
            "joint_raw_position_signal": raw_position_signal,
            "joint_relocation_scale": torch.tanh(self.relocation_scale),
            "joint_local_attention_log_prior": gaussian_log_prior,
        }
        if return_attention_weights:
            result["joint_local_attention_weights"] = attention_weights
        return result
