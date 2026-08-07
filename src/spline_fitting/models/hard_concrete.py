from __future__ import annotations

import math

import torch
from torch import nn


class HardConcreteGate(nn.Module):
    """Sample differentiable :math:`L_0` gates from Hard-Concrete variables.

    ``log_alpha`` is the location predicted for each candidate knot. During
    training, the returned gate is a reparameterized Hard-Concrete sample in
    ``[0, 1]``. During evaluation, it is the hard decision
    ``P(z > 0) >= threshold`` and is therefore exactly zero or one.

    Temperature is stored in the state dict so the gate semantics at the best
    epoch survive checkpoint loading. A compatibility loader supplies the
    constructor temperature for checkpoints created before this module existed.
    """

    def __init__(
        self,
        temperature: float = 2.0 / 3.0,
        stretch_low: float = -0.1,
        stretch_high: float = 1.1,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._validate_stretch(stretch_low, stretch_high)
        self._validate_threshold(threshold)
        self._validate_eps(eps)

        self.stretch_low = float(stretch_low)
        self.stretch_high = float(stretch_high)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.register_buffer(
            "_temperature_buffer", torch.tensor(float(temperature)), persistent=True
        )
        self._force_open_gates = False
        self.set_temperature(temperature)

    @staticmethod
    def _validate_stretch(stretch_low: float, stretch_high: float) -> None:
        if not stretch_low < 0.0:
            raise ValueError("stretch_low must be negative")
        if not stretch_high > 1.0:
            raise ValueError("stretch_high must be greater than one")
        if not stretch_low < stretch_high:
            raise ValueError("stretch_low must be less than stretch_high")

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")

    @staticmethod
    def _validate_eps(eps: float) -> None:
        if not 0.0 < eps < 0.5:
            raise ValueError("eps must lie in (0, 0.5)")

    @property
    def temperature(self) -> float:
        return float(self._temperature_buffer.item())

    @property
    def force_open_gates(self) -> bool:
        return self._force_open_gates

    def set_temperature(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self._temperature_buffer.fill_(value)

    def set_threshold(self, value: float) -> None:
        """Set the deterministic deployment threshold used by ``eval()``."""
        self._validate_threshold(value)
        self.threshold = float(value)

    def set_force_open_gates(self, enabled: bool) -> None:
        self._force_open_gates = bool(enabled)

    def l0_probability(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Return the closed-form probability ``P(z > 0)`` per gate."""
        return torch.sigmoid(self.l0_probability_logits(log_alpha))

    def l0_probability_logits(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Return logits of the closed-form non-zero probability."""
        boundary = math.log(-self.stretch_low / self.stretch_high)
        temperature = self._temperature_buffer.to(dtype=log_alpha.dtype)
        return log_alpha - temperature * boundary

    def _sample_relaxed_gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(log_alpha).clamp(self.eps, 1.0 - self.eps)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        temperature = self._temperature_buffer.to(dtype=log_alpha.dtype)
        concrete = torch.sigmoid((log_alpha + logistic_noise) / temperature)
        stretched = concrete * (self.stretch_high - self.stretch_low) + self.stretch_low
        return stretched.clamp(0.0, 1.0)

    def forward(self, log_alpha: torch.Tensor) -> dict[str, torch.Tensor]:
        probability_logits = self.l0_probability_logits(log_alpha)
        probability = torch.sigmoid(probability_logits)

        if self.force_open_gates:
            gate = torch.ones_like(log_alpha)
        elif self.training:
            gate = self._sample_relaxed_gate(log_alpha)
        else:
            gate = (probability >= self.threshold).to(dtype=log_alpha.dtype)

        return {
            "activity": probability,
            "activity_probability_logits": probability_logits,
            "l0_probability": probability,
            "expected_l0": probability.sum(dim=-1),
            "activity_gate": gate,
        }

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
        # Checkpoints created before Hard-Concrete was introduced have no
        # temperature entry. Supplying the constructor value keeps strict
        # loading compatible without changing the legacy MLP state keys.
        temperature_key = prefix + "_temperature_buffer"
        if temperature_key not in state_dict:
            state_dict[temperature_key] = self._temperature_buffer.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature:g}, "
            f"stretch=({self.stretch_low:g}, {self.stretch_high:g}), "
            f"threshold={self.threshold:g}"
        )
