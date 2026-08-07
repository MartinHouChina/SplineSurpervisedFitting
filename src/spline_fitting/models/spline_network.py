from __future__ import annotations

import torch
from torch import nn

from ..spline.curve_evaluation import reconstruct_from_design
from ..spline.derivatives import build_derivative_design_matrix
from ..spline.differentiable_solver import (
    coefficient_drop_objective_delta,
    solve_coefficients,
)
from ..spline.truncated_power_basis import build_design_matrix
from .activity_head import ActivityHead
from .geometry_encoder import GeometryEncoder
from .knot_head import KnotHead
from .parameter_head import ParameterHead


class SplineFittingNetwork(nn.Module):
    """Jointly predict t, candidate U and sparse Hard-Concrete knot gates."""

    def __init__(
        self,
        point_dim: int = 2,
        degree: int = 3,
        hidden_dim: int = 128,
        encoder_layers: int = 4,
        max_internal_knots: int = 8,
        min_parameter_gap: float = 1e-4,
        min_knot_gap: float = 1e-3,
        gap_parameterization: str = "strict",
        lambda_poly: float = 1e-6,
        lambda_knot: float = 1e-4,
        gate_eps: float = 1e-6,
        gate_mode: str = "hard_concrete",
        gate_temperature: float = 2.0 / 3.0,
        gate_stretch_low: float = -0.1,
        gate_stretch_high: float = 1.1,
        hard_concrete_gamma: float | None = None,
        hard_concrete_zeta: float | None = None,
        activity_threshold: float = 0.5,
        activity_initial_bias: float = -2.0,
        activity_use_local_context: bool = True,
        activity_context_bandwidth: float = 0.08,
        activity_use_query_features: bool = True,
        activity_use_pilot_importance: bool = False,
        activity_pilot_importance_gain: float = 1.0,
        knot_use_local_cross_attention: bool = True,
        knot_attention_heads: int = 4,
        knot_parameterization: str = "independent_queries",
        compute_first_derivative: bool = False,
    ) -> None:
        super().__init__()
        self.degree = degree
        self.lambda_poly = lambda_poly
        self.lambda_knot = lambda_knot
        self.gate_eps = gate_eps
        self.gate_mode = gate_mode
        # The current objective does not need curve tangents.  This switch is
        # retained only so historical checkpoints with the projection-
        # orthogonality loss can reproduce their original forward pass.
        self.compute_first_derivative = compute_first_derivative
        if hard_concrete_gamma is not None:
            gate_stretch_low = hard_concrete_gamma
        if hard_concrete_zeta is not None:
            gate_stretch_high = hard_concrete_zeta

        self.encoder = GeometryEncoder(point_dim, hidden_dim, encoder_layers)
        self.parameter_head = ParameterHead(
            hidden_dim, min_parameter_gap, gap_parameterization=gap_parameterization
        )
        use_knot_cross_attention = (
            knot_use_local_cross_attention and gate_mode != "legacy_soft"
        )
        if gate_mode == "legacy_soft":
            knot_parameterization = "interval"
        self.knot_head = KnotHead(
            hidden_dim,
            max_internal_knots,
            min_knot_gap,
            gap_parameterization=gap_parameterization,
            use_local_cross_attention=use_knot_cross_attention,
            attention_heads=knot_attention_heads,
            knot_parameterization=knot_parameterization,
        )
        # Legacy checkpoints did not use knot-local context. Keeping it disabled
        # in legacy mode preserves their forward semantics and MLP dimensions.
        use_local_context = activity_use_local_context and gate_mode != "legacy_soft"
        use_pilot_importance = (
            activity_use_pilot_importance and gate_mode != "legacy_soft"
        )
        use_query_features = (
            activity_use_query_features
            and gate_mode != "legacy_soft"
            and knot_parameterization == "independent_queries"
        )
        self.activity_head = ActivityHead(
            hidden_dim,
            initial_bias=activity_initial_bias,
            gate_mode=gate_mode,
            gate_temperature=gate_temperature,
            gate_stretch_low=gate_stretch_low,
            gate_stretch_high=gate_stretch_high,
            activity_threshold=activity_threshold,
            gate_eps=gate_eps,
            use_local_context=use_local_context,
            context_bandwidth=activity_context_bandwidth,
            use_query_features=use_query_features,
            use_pilot_importance=use_pilot_importance,
            pilot_importance_gain=activity_pilot_importance_gain,
        )

    def _pilot_knot_importance(
        self,
        params: torch.Tensor,
        internal_knots: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate per-knot drop cost using an all-candidate pilot fit."""
        if internal_knots.shape[-1] == 0:
            empty = internal_knots.detach().clone()
            return empty, empty

        # Importance is a structural feature, not a second optimization path
        # for t/U.  Detaching avoids costly higher-order gradients through an
        # inverse normal matrix while gradients still train the activity MLP.
        with torch.no_grad():
            detached_params = params.detach()
            detached_knots = internal_knots.detach()
            detached_points = points.detach()
            pilot_basis = build_design_matrix(
                params=detached_params,
                internal_knots=detached_knots,
                activity=torch.ones_like(detached_knots),
                degree=self.degree,
                eps=0.0,
                gate_transform="direct",
            )
            pilot_solver = solve_coefficients(
                design_matrix=pilot_basis["design_matrix"],
                points=detached_points,
                degree=self.degree,
                lambda_poly=self.lambda_poly,
                lambda_knot=self.lambda_knot,
            )
            delta = coefficient_drop_objective_delta(
                pilot_solver["coefficients"],
                pilot_solver["normal_matrix"],
                first_column=self.degree + 1,
            )
            log_delta = torch.log(delta.clamp_min(torch.finfo(delta.dtype).tiny))
            centered = log_delta - log_delta.mean(dim=-1, keepdim=True)
            scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            normalized = (centered / scale).clamp(-3.0, 3.0)
        return delta, normalized

    def set_gate_temperature(self, value: float) -> None:
        """Update the Hard-Concrete temperature used on subsequent forwards."""
        self.activity_head.set_gate_temperature(value)

    def set_force_open_gates(self, enabled: bool) -> None:
        """Force actual design-matrix gates open, e.g. during warm-up."""
        self.activity_head.set_force_open_gates(enabled)

    def set_activity_threshold(self, value: float) -> None:
        """Set the fixed probability threshold used by deterministic gates."""
        self.activity_head.set_activity_threshold(value)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        local_features, global_features = self.encoder(points)
        parameter_output = self.parameter_head(local_features, global_features)
        knot_output = self.knot_head(
            global_features,
            local_features=local_features,
            positions=parameter_output["params"],
        )
        if self.activity_head.use_pilot_importance:
            pilot_delta, normalized_importance = self._pilot_knot_importance(
                parameter_output["params"],
                knot_output["internal_knots"],
                points,
            )
        else:
            pilot_delta = torch.zeros_like(knot_output["internal_knots"])
            normalized_importance = torch.zeros_like(knot_output["internal_knots"])
        activity_output = self.activity_head(
            global_features,
            knot_output["internal_knots"],
            local_features=local_features,
            params=parameter_output["params"],
            query_features=knot_output.get("knot_query_features"),
            normalized_knot_importance=normalized_importance,
        )

        basis_output = build_design_matrix(
            params=parameter_output["params"],
            internal_knots=knot_output["internal_knots"],
            activity=activity_output["activity_gate"],
            degree=self.degree,
            eps=0.0,
            gate_transform="direct",
        )
        solver_output = solve_coefficients(
            design_matrix=basis_output["design_matrix"],
            points=points,
            degree=self.degree,
            lambda_poly=self.lambda_poly,
            lambda_knot=self.lambda_knot,
        )
        reconstructed = reconstruct_from_design(
            basis_output["design_matrix"], solver_output["coefficients"]
        )
        output = {
            "local_features": local_features,
            "global_features": global_features,
            **parameter_output,
            **knot_output,
            **activity_output,
            "pilot_drop_objective_delta": pilot_delta,
            "normalized_knot_importance": normalized_importance,
            **basis_output,
            **solver_output,
            "reconstructed_points": reconstructed,
        }
        if self.compute_first_derivative:
            first_derivative_design = build_derivative_design_matrix(
                params=parameter_output["params"],
                internal_knots=knot_output["internal_knots"],
                activity=activity_output["activity_gate"],
                degree=self.degree,
                order=1,
                eps=0.0,
                gate_transform="direct",
            )
            output["first_derivative"] = (
                first_derivative_design @ solver_output["coefficients"]
            )
        return output
