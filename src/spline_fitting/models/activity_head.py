from __future__ import annotations

import torch
from torch import nn

from .hard_concrete import HardConcreteGate


class ActivityHead(nn.Module):
    """Predict a keep probability and an optimization gate for each knot."""

    def __init__(
        self,
        hidden_dim: int = 128,
        initial_bias: float = -2.0,
        gate_mode: str = "hard_concrete",
        gate_temperature: float = 2.0 / 3.0,
        gate_stretch_low: float = -0.1,
        gate_stretch_high: float = 1.1,
        activity_threshold: float = 0.5,
        gate_eps: float = 1e-6,
        use_local_context: bool = False,
        context_bandwidth: float = 0.08,
        use_query_features: bool = False,
        use_pilot_importance: bool = False,
        pilot_importance_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"hard_concrete", "legacy_soft"}:
            raise ValueError("gate_mode must be 'hard_concrete' or 'legacy_soft'")
        if gate_eps < 0.0:
            raise ValueError("gate_eps must be non-negative")
        if context_bandwidth <= 0.0:
            raise ValueError("context_bandwidth must be positive")
        if pilot_importance_gain < 0.0:
            raise ValueError("pilot_importance_gain must be non-negative")

        self.gate_mode = gate_mode
        self.gate_eps = float(gate_eps)
        self.use_local_context = bool(use_local_context)
        self.context_bandwidth = float(context_bandwidth)
        self.use_query_features = bool(use_query_features)
        self.use_pilot_importance = bool(use_pilot_importance)
        self.pilot_importance_gain = float(pilot_importance_gain)
        activity_input_dim = hidden_dim + 1
        if self.use_query_features:
            activity_input_dim += hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(activity_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, initial_bias)
        self.hard_concrete = HardConcreteGate(
            temperature=gate_temperature,
            stretch_low=gate_stretch_low,
            stretch_high=gate_stretch_high,
            threshold=activity_threshold,
        )

    def set_gate_temperature(self, value: float) -> None:
        self.hard_concrete.set_temperature(value)

    def set_force_open_gates(self, enabled: bool) -> None:
        self.hard_concrete.set_force_open_gates(enabled)

    def set_activity_threshold(self, value: float) -> None:
        self.hard_concrete.set_threshold(value)

    def forward(
        self,
        global_features: torch.Tensor,
        internal_knots: torch.Tensor,
        local_features: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        query_features: torch.Tensor | None = None,
        normalized_knot_importance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, num_knots = internal_knots.shape
        global_expanded = global_features.unsqueeze(1).expand(-1, num_knots, -1)
        if self.use_local_context:
            if local_features is None or params is None:
                raise ValueError(
                    "local_features and params are required when local context is enabled"
                )
            if local_features.ndim != 3 or params.ndim != 2:
                raise ValueError(
                    "local_features and params must have shapes [B, M, H] and [B, M]"
                )
            if local_features.shape[:2] != params.shape:
                raise ValueError("local_features and params must share [B, M]")
            if local_features.shape[0] != batch:
                raise ValueError("activity inputs must have the same batch size")
            if local_features.shape[-1] != global_features.shape[-1]:
                raise ValueError("local and global feature dimensions must match")

            scaled_distance = (
                params.unsqueeze(1) - internal_knots.unsqueeze(-1)
            ) / self.context_bandwidth
            attention = torch.softmax(-0.5 * scaled_distance.square(), dim=-1)
            local_context = attention @ local_features
            activity_features = global_expanded + local_context
        else:
            activity_features = global_expanded

        features = [activity_features]
        if self.use_query_features:
            if query_features is None:
                raise ValueError(
                    "query_features is required when query conditioning is enabled"
                )
            if query_features.shape != (batch, num_knots, global_features.shape[-1]):
                raise ValueError("query_features must have shape [B, K, H]")
            features.append(query_features)
        features.append(internal_knots.unsqueeze(-1))
        logits = self.mlp(torch.cat(features, dim=-1)).squeeze(-1)
        if self.use_pilot_importance:
            if normalized_knot_importance is None:
                raise ValueError(
                    "normalized_knot_importance is required when pilot importance "
                    "is enabled"
                )
            if normalized_knot_importance.shape != internal_knots.shape:
                raise ValueError("normalized_knot_importance must have shape [B, K]")
            logits = logits + self.pilot_importance_gain * normalized_knot_importance

        if self.gate_mode == "hard_concrete":
            gate_output = self.hard_concrete(logits)
        else:
            probability = torch.sigmoid(logits)
            if self.hard_concrete.force_open_gates:
                gate = torch.ones_like(probability)
            else:
                gate = torch.sqrt(probability.clamp_min(0.0) + self.gate_eps)
            gate_output = {
                "activity": probability,
                "l0_probability": probability,
                "expected_l0": probability.sum(dim=-1),
                "activity_gate": gate,
            }

        return {
            "activity_logits": logits,
            **gate_output,
        }
