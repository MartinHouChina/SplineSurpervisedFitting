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
from .candidate_knot_head import CandidateKnotHead
from .count_conditioned_knot_head import CountConditionedKnotHead
from .count_head import CountHead
from .dynamic_knot_decoder import DynamicKnotDecoder
from .geometry_encoder import GeometryEncoder
from .interactive_structure_head import InteractiveStructureHead
from .interactive_pruning_head import InteractivePruningHead
from .knot_head import KnotHead
from .parameter_head import ParameterHead


class SplineFittingNetwork(nn.Module):
    """Predict parameters, spline structure, and ordered internal knots."""

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
        activity_use_candidate_self_attention: bool = True,
        activity_candidate_attention_heads: int = 4,
        activity_use_pilot_importance: bool = False,
        activity_pilot_importance_gain: float = 1.0,
        detach_activity_gate_for_fit: bool = True,
        knot_use_local_cross_attention: bool = True,
        knot_attention_heads: int = 4,
        knot_parameterization: str = "independent_queries",
        compute_first_derivative: bool = False,
        structure_mode: str = "hard_concrete",
        count_attention_heads: int = 4,
        count_head_mode: str = "ordinal_local_attention",
        count_query_count: int = 4,
        count_decoder_mode: str = "shared_count_embedding",
        geometry_feature_mode: str | None = None,
        structure_attention_heads: int = 4,
        structure_count_mode: str = "categorical",
        min_internal_knots: int = 0,
        pruning_residual_bandwidth: float = 0.05,
        pruning_initial_keep_probability: float = 0.9,
    ) -> None:
        super().__init__()
        if structure_mode not in {
            "hard_concrete",
            "count_conditioned",
            "interactive_dynamic",
            "candidate_pruning",
            "candidate_pruning_one_shot",
        }:
            raise ValueError("unsupported structure_mode")
        self.structure_mode = structure_mode
        self._force_open_candidate_gate = False
        self.degree = degree
        self.lambda_poly = lambda_poly
        self.lambda_knot = lambda_knot
        self.gate_eps = gate_eps
        self.gate_mode = gate_mode
        if pruning_residual_bandwidth <= 0.0:
            raise ValueError("pruning_residual_bandwidth must be positive")
        self.pruning_residual_bandwidth = float(pruning_residual_bandwidth)
        # In the supervised-existence objective, curve fitting must not teach
        # the structural classifier to open every useful basis column. The
        # sampled Hard-Concrete value is still used numerically by the solver;
        # only its fit-gradient is stopped. Legacy models retain the old path.
        self.detach_activity_gate_for_fit = bool(
            detach_activity_gate_for_fit and gate_mode != "legacy_soft"
        )
        # The current objective does not need curve tangents.  This switch is
        # retained only so historical checkpoints with the projection-
        # orthogonality loss can reproduce their original forward pass.
        self.compute_first_derivative = compute_first_derivative
        if hard_concrete_gamma is not None:
            gate_stretch_low = hard_concrete_gamma
        if hard_concrete_zeta is not None:
            gate_stretch_high = hard_concrete_zeta

        if geometry_feature_mode is None:
            geometry_feature_mode = (
                "chord_derivatives"
                if self.structure_mode
                in {
                    "count_conditioned",
                    "interactive_dynamic",
                    "candidate_pruning",
                    "candidate_pruning_one_shot",
                }
                else "raw_differences"
            )
        self.encoder = GeometryEncoder(
            point_dim,
            hidden_dim,
            encoder_layers,
            feature_mode=geometry_feature_mode,
        )
        self.parameter_head = ParameterHead(
            hidden_dim, min_parameter_gap, gap_parameterization=gap_parameterization
        )
        if self.structure_mode in {
            "candidate_pruning",
            "candidate_pruning_one_shot",
        }:
            self.candidate_head = CandidateKnotHead(
                hidden_dim,
                max_internal_knots,
                min_gap=min_knot_gap,
                attention_heads=structure_attention_heads,
            )
            self.pruning_head = InteractivePruningHead(
                hidden_dim,
                attention_heads=structure_attention_heads,
                min_gap=min_knot_gap,
                initial_keep_probability=pruning_initial_keep_probability,
                one_shot_adaptive=(self.structure_mode == "candidate_pruning_one_shot"),
            )
        elif self.structure_mode == "interactive_dynamic":
            self.structure_head = InteractiveStructureHead(
                hidden_dim,
                max_internal_knots,
                attention_heads=structure_attention_heads,
                count_distribution_mode=structure_count_mode,
                min_internal_knots=min_internal_knots,
            )
            self.knot_head = DynamicKnotDecoder(
                hidden_dim,
                max_internal_knots,
                min_gap=min_knot_gap,
                attention_heads=structure_attention_heads,
            )
        elif self.structure_mode == "count_conditioned":
            self.count_head = CountHead(
                hidden_dim,
                max_internal_knots,
                mode=count_head_mode,
                attention_heads=count_attention_heads,
                query_count=count_query_count,
            )
            self.knot_head = CountConditionedKnotHead(
                hidden_dim,
                max_internal_knots,
                min_gap=min_knot_gap,
                attention_heads=count_attention_heads,
                mode=count_decoder_mode,
            )
        else:
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
        if self.structure_mode == "hard_concrete":
            use_local_context = (
                activity_use_local_context and gate_mode != "legacy_soft"
            )
            use_pilot_importance = (
                activity_use_pilot_importance and gate_mode != "legacy_soft"
            )
            use_query_features = (
                activity_use_query_features
                and gate_mode != "legacy_soft"
                and knot_parameterization == "independent_queries"
            )
            use_candidate_self_attention = (
                activity_use_candidate_self_attention and use_query_features
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
                use_candidate_self_attention=use_candidate_self_attention,
                candidate_attention_heads=activity_candidate_attention_heads,
                use_pilot_importance=use_pilot_importance,
                pilot_importance_gain=activity_pilot_importance_gain,
            )

    def _candidate_local_residual(
        self,
        params: torch.Tensor,
        candidate_knots: torch.Tensor,
        point_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Pool geometric residual around every candidate parameter."""
        distance = (params.unsqueeze(-1) - candidate_knots.unsqueeze(1)).abs()
        weights = torch.exp(
            -0.5 * (distance / self.pruning_residual_bandwidth).square()
        )
        return (weights * point_residual.unsqueeze(-1)).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1e-8)

    def _forward_candidate_pruning(
        self,
        points: torch.Tensor,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        parameter_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Propose all knots, measure their fit contribution, then interact."""
        params = parameter_output["params"]
        candidate_output = self.candidate_head(
            global_features,
            local_features,
            params,
        )
        candidates = candidate_output["candidate_knots"]
        all_open = torch.ones_like(candidates)

        pilot_basis = build_design_matrix(
            params=params,
            internal_knots=candidates,
            activity=all_open,
            degree=self.degree,
            eps=0.0,
            gate_transform="direct",
        )
        pilot_solver = solve_coefficients(
            design_matrix=pilot_basis["design_matrix"],
            points=points,
            degree=self.degree,
            lambda_poly=self.lambda_poly,
            lambda_knot=self.lambda_knot,
        )
        pilot_reconstruction = reconstruct_from_design(
            pilot_basis["design_matrix"],
            pilot_solver["coefficients"],
        )

        # These are measured structural descriptors.  Stopping their gradient
        # avoids second-order inverse-system derivatives while the proposal
        # positions still receive direct coverage, position and fit gradients.
        with torch.no_grad():
            analytic_delta = coefficient_drop_objective_delta(
                pilot_solver["coefficients"].detach(),
                pilot_solver["normal_matrix"].detach(),
                first_column=self.degree + 1,
            )
            knot_coefficients = pilot_solver["coefficients"][
                :, self.degree + 1 :
            ].detach()
            coefficient_energy = knot_coefficients.square().sum(dim=-1)
            point_residual = (pilot_reconstruction.detach() - points.detach()).norm(
                dim=-1
            )
            local_residual = self._candidate_local_residual(
                params.detach(),
                candidates.detach(),
                point_residual,
            )

        pruning_output = self.pruning_head(
            candidate_output["candidate_tokens"],
            candidates,
            coefficient_energy=coefficient_energy,
            deletion_delta=analytic_delta,
            residual_features=local_residual,
            global_features=global_features,
        )
        refined_knots = pruning_output["refined_candidate_knots"]
        keep_probability = pruning_output["keep_probability"]
        learned_keep_mask = pruning_output.get(
            "final_hard_keep_mask", keep_probability >= 0.5
        )
        if self._force_open_candidate_gate:
            fit_activity_gate = torch.ones_like(refined_knots)
        elif self.structure_mode == "candidate_pruning_one_shot":
            # A hard forward mask with a straight-through probability gradient
            # makes every truncated-power knot contribution independently
            # switchable while deployment remains a truly discrete subset.
            fit_activity_gate = pruning_output.get(
                "final_hard_st_keep_gate",
                learned_keep_mask.to(keep_probability.dtype)
                + keep_probability
                - keep_probability.detach(),
            )
        else:
            # v7 keeps the learned mask diagnostic-only and fits all proposals.
            fit_activity_gate = torch.ones_like(refined_knots)
        basis_output = build_design_matrix(
            params=params,
            internal_knots=refined_knots,
            activity=fit_activity_gate,
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
        knot_mask = learned_keep_mask
        return {
            "local_features": local_features,
            "global_features": global_features,
            **parameter_output,
            **candidate_output,
            **pruning_output,
            # Compatibility aliases used by Trainer and diagnostics.  The
            # mask is only a learned diagnostic; hard deployment pruning does
            # not trust this 0.5 threshold.
            "internal_knots": refined_knots,
            "activity": keep_probability,
            "activity_probability_logits": pruning_output["keep_logits"],
            "activity_gate": knot_mask.to(points.dtype),
            "knot_mask": knot_mask,
            "learned_keep_mask": knot_mask,
            "fit_activity_gate": fit_activity_gate,
            "expected_knot_count": keep_probability.sum(dim=-1),
            "predicted_knot_count": knot_mask.sum(dim=-1),
            "analytic_drop_objective_delta": analytic_delta,
            "coefficient_energy": coefficient_energy,
            "candidate_local_residual": local_residual,
            "pilot_reconstructed_points": pilot_reconstruction,
            **basis_output,
            **solver_output,
            "reconstructed_points": reconstructed,
        }

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
        if hasattr(self, "activity_head"):
            self.activity_head.set_gate_temperature(value)

    def set_force_open_gates(self, enabled: bool) -> None:
        """Force actual design-matrix gates open, e.g. during warm-up."""
        if hasattr(self, "activity_head"):
            self.activity_head.set_force_open_gates(enabled)
        if self.structure_mode == "candidate_pruning_one_shot":
            self._force_open_candidate_gate = bool(enabled)

    def set_activity_threshold(self, value: float) -> None:
        """Set the fixed probability threshold used by deterministic gates."""
        if hasattr(self, "activity_head"):
            self.activity_head.set_activity_threshold(value)

    @staticmethod
    def _select_knot_count(
        predicted_count: torch.Tensor,
        teacher_count: torch.Tensor | None,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must lie in [0, 1]")
        if teacher_count is None or teacher_forcing_ratio == 0.0:
            return predicted_count
        teacher_count = teacher_count.to(
            device=predicted_count.device,
            dtype=torch.long,
        )
        if teacher_count.shape != predicted_count.shape:
            raise ValueError("teacher and predicted knot counts must share shape [B]")
        if teacher_forcing_ratio == 1.0:
            return teacher_count
        use_teacher = (
            torch.rand(
                predicted_count.shape,
                device=predicted_count.device,
            )
            < teacher_forcing_ratio
        )
        return torch.where(use_teacher, teacher_count, predicted_count)

    def forward(
        self,
        points: torch.Tensor,
        true_internal_knot_count: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        local_features, global_features = self.encoder(points)
        parameter_output = self.parameter_head(local_features, global_features)
        if self.structure_mode in {
            "candidate_pruning",
            "candidate_pruning_one_shot",
        }:
            return self._forward_candidate_pruning(
                points,
                local_features,
                global_features,
                parameter_output,
            )
        if self.structure_mode == "interactive_dynamic":
            count_output = self.structure_head(
                global_features,
                local_features,
                parameter_output["params"],
            )
            selected_count = self._select_knot_count(
                count_output["predicted_knot_count"],
                true_internal_knot_count,
                teacher_forcing_ratio,
            )
            knot_output = self.knot_head(
                global_features,
                local_features,
                parameter_output["params"],
                count_output["structure_query_features"],
                selected_count,
            )
            fit_activity_gate = knot_output["knot_mask"].to(points.dtype)
            pilot_delta = torch.zeros_like(knot_output["internal_knots"])
            normalized_importance = torch.zeros_like(knot_output["internal_knots"])
            activity_output: dict[str, torch.Tensor] = {}
        elif self.structure_mode == "count_conditioned":
            count_output = self.count_head(
                global_features,
                local_features,
                parameter_output["params"],
            )
            selected_count = self._select_knot_count(
                count_output["predicted_knot_count"],
                true_internal_knot_count,
                teacher_forcing_ratio,
            )
            knot_output = self.knot_head(
                global_features,
                local_features,
                parameter_output["params"],
                selected_count,
            )
            fit_activity_gate = knot_output["knot_mask"].to(points.dtype)
            pilot_delta = torch.zeros_like(knot_output["internal_knots"])
            normalized_importance = torch.zeros_like(knot_output["internal_knots"])
            activity_output: dict[str, torch.Tensor] = {}
        else:
            count_output = {}
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
            sampled_activity_gate = activity_output["activity_gate"]
            fit_activity_gate = (
                sampled_activity_gate.detach()
                if self.detach_activity_gate_for_fit
                else sampled_activity_gate
            )

        basis_output = build_design_matrix(
            params=parameter_output["params"],
            internal_knots=knot_output["internal_knots"],
            activity=fit_activity_gate,
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
            **count_output,
            **knot_output,
            **activity_output,
            "fit_activity_gate": fit_activity_gate,
            "pilot_drop_objective_delta": pilot_delta,
            "normalized_knot_importance": normalized_importance,
            **basis_output,
            **solver_output,
            "reconstructed_points": reconstructed,
        }
        if self.structure_mode == "hard_concrete":
            output["sampled_activity_gate"] = sampled_activity_gate
        if self.compute_first_derivative:
            first_derivative_design = build_derivative_design_matrix(
                params=parameter_output["params"],
                internal_knots=knot_output["internal_knots"],
                activity=fit_activity_gate,
                degree=self.degree,
                order=1,
                eps=0.0,
                gate_transform="direct",
            )
            output["first_derivative"] = (
                first_derivative_design @ solver_output["coefficients"]
            )
        return output
