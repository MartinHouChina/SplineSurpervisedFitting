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
from .joint_parameter_structure_head import JointParameterStructureHead
from .knot_head import KnotHead
from .parameter_feedback_head import ParameterFeedbackHead
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
        parameter_gap_reference: str = "learned",
        parameter_residual_logit_limit: float = 0.5,
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
        one_shot_fixed_proposal_geometry: bool = False,
        one_shot_selection_policy: str = "threshold",
        one_shot_safety_sigma: float = 0.0,
        one_shot_selector_layers: int = 1,
        one_shot_coverage_bins: int = 0,
        candidate_local_attention_bandwidth: float = 0.0,
        candidate_interval_logit_limit: float = 0.0,
        candidate_position_parameterization: str = "interval_softmax",
        one_shot_joint_position_refinement: bool = False,
        one_shot_survivor_relocation: bool = False,
        one_shot_max_position_shift: float = 0.05,
        stable_pilot_descriptors: bool = False,
        parameter_feedback_fusion: bool = False,
        parameter_feedback_attention_heads: int | None = None,
        parameter_feedback_max_logit_shift: float = 0.5,
        parameter_feedback_fusion_mode: str = "fast_global",
        joint_parameter_structure_feedback: bool = False,
        joint_parameter_structure_local_bandwidth: float = 0.08,
        joint_parameter_structure_max_keep_logit_shift: float = 2.0,
        enforce_ordered_joint_candidates: bool = False,
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
        if parameter_feedback_fusion and (
            structure_mode != "candidate_pruning_one_shot"
        ):
            raise ValueError(
                "parameter feedback fusion requires candidate_pruning_one_shot mode"
            )
        if parameter_feedback_fusion and gap_parameterization != "strict":
            raise ValueError(
                "parameter feedback fusion requires strict gap parameterization"
            )
        if joint_parameter_structure_feedback and not parameter_feedback_fusion:
            raise ValueError(
                "joint parameter/structure feedback requires parameter feedback"
            )
        if joint_parameter_structure_feedback and not one_shot_survivor_relocation:
            raise ValueError(
                "joint parameter/structure feedback requires survivor relocation"
            )
        self.parameter_feedback_fusion = bool(parameter_feedback_fusion)
        self.joint_parameter_structure_feedback = bool(
            joint_parameter_structure_feedback
        )
        self.enforce_ordered_joint_candidates = bool(
            enforce_ordered_joint_candidates
        )
        if self.enforce_ordered_joint_candidates and not (
            self.joint_parameter_structure_feedback
        ):
            raise ValueError(
                "ordered joint candidates require joint parameter/structure feedback"
            )
        self._force_open_candidate_gate = False
        self.degree = degree
        self.lambda_poly = lambda_poly
        self.lambda_knot = lambda_knot
        # v13 computes the analytic proposal descriptors from a detached
        # float64 pilot solve.  The legacy float32 path is retained by default
        # so v12 and earlier checkpoints reproduce their original semantics.
        self.stable_pilot_descriptors = bool(stable_pilot_descriptors)
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
            hidden_dim,
            min_parameter_gap,
            gap_parameterization=gap_parameterization,
            gap_reference=parameter_gap_reference,
            residual_logit_limit=parameter_residual_logit_limit,
        )
        if self.parameter_feedback_fusion:
            feedback_heads = (
                structure_attention_heads
                if parameter_feedback_attention_heads is None
                else parameter_feedback_attention_heads
            )
            self.parameter_feedback_head = ParameterFeedbackHead(
                hidden_dim,
                min_gap=min_parameter_gap,
                attention_heads=feedback_heads,
                risk_bandwidth=pruning_residual_bandwidth,
                max_logit_shift=parameter_feedback_max_logit_shift,
                fusion_mode=parameter_feedback_fusion_mode,
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
                local_attention_bandwidth=candidate_local_attention_bandwidth,
                interval_logit_limit=candidate_interval_logit_limit,
                position_parameterization=candidate_position_parameterization,
            )
            self.pruning_head = InteractivePruningHead(
                hidden_dim,
                attention_heads=structure_attention_heads,
                min_gap=min_knot_gap,
                initial_keep_probability=pruning_initial_keep_probability,
                one_shot_adaptive=(self.structure_mode == "candidate_pruning_one_shot"),
                one_shot_fixed_proposal_geometry=(
                    one_shot_fixed_proposal_geometry
                    and self.structure_mode == "candidate_pruning_one_shot"
                ),
                one_shot_selection_policy=one_shot_selection_policy,
                one_shot_safety_sigma=one_shot_safety_sigma,
                one_shot_selector_layers=one_shot_selector_layers,
                one_shot_coverage_bins=one_shot_coverage_bins,
                one_shot_joint_position_refinement=(
                    one_shot_joint_position_refinement
                    and self.structure_mode == "candidate_pruning_one_shot"
                ),
                one_shot_survivor_relocation=(
                    one_shot_survivor_relocation
                    and self.structure_mode == "candidate_pruning_one_shot"
                ),
                one_shot_max_position_shift=one_shot_max_position_shift,
            )
            if self.joint_parameter_structure_feedback:
                self.joint_parameter_structure_head = JointParameterStructureHead(
                    hidden_dim,
                    attention_heads=structure_attention_heads,
                    local_bandwidth=joint_parameter_structure_local_bandwidth,
                    max_keep_logit_shift=(
                        joint_parameter_structure_max_keep_logit_shift
                    ),
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

    @staticmethod
    def _chord_length_parameters(points: torch.Tensor) -> torch.Tensor:
        """Return normalized chord-length parameters for an ordered point batch."""

        if points.ndim != 3 or points.shape[1] < 2:
            raise ValueError("points must have shape [B,M,D] with M >= 2")
        segment_lengths = torch.nan_to_num(
            (points[:, 1:] - points[:, :-1]).norm(dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        total_length = segment_lengths.sum(dim=-1, keepdim=True)
        uniform_gaps = torch.full_like(
            segment_lengths,
            1.0 / segment_lengths.shape[-1],
        )
        chord_gaps = torch.where(
            total_length > torch.finfo(points.dtype).eps,
            segment_lengths / total_length.clamp_min(torch.finfo(points.dtype).eps),
            uniform_gaps,
        )
        chord_params = torch.cat(
            [
                points.new_zeros(points.shape[0], 1),
                torch.cumsum(chord_gaps, dim=-1),
            ],
            dim=-1,
        )
        chord_params[:, -1] = 1.0
        return chord_params

    def _parameter_head_forward(
        self,
        points: torch.Tensor,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reference_params = None
        if self.parameter_head.gap_reference == "chord_residual":
            reference_params = self._chord_length_parameters(points)
        return self.parameter_head(
            local_features,
            global_features,
            reference_params=reference_params,
        )

    @staticmethod
    def _warp_parameter_coordinates(
        values: torch.Tensor,
        source_params: torch.Tensor,
        target_params: torch.Tensor,
    ) -> torch.Tensor:
        """Transport ``t0`` coordinates through the monotone ``t0 -> t1`` map."""

        if source_params.shape != target_params.shape:
            raise ValueError("source and target parameters must share shape [B,M]")
        if values.ndim != 2 or values.shape[0] != source_params.shape[0]:
            raise ValueError("values must have shape [B,K]")
        right = torch.searchsorted(
            source_params.detach().contiguous(),
            values.detach().contiguous(),
            right=True,
        ).clamp(1, source_params.shape[-1] - 1)
        left = right - 1
        source_left = torch.gather(source_params, 1, left)
        source_right = torch.gather(source_params, 1, right)
        target_left = torch.gather(target_params, 1, left)
        target_right = torch.gather(target_params, 1, right)
        fraction = (values - source_left) / (source_right - source_left).clamp_min(
            torch.finfo(source_params.dtype).eps
        )
        return target_left + fraction * (target_right - target_left)

    def _forward_candidate_pruning(
        self,
        points: torch.Tensor,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        parameter_output: dict[str, torch.Tensor],
        *,
        compute_fit_surrogate: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Propose all knots, measure their fit contribution, then interact."""
        params = parameter_output["params"]
        candidate_output = self.candidate_head(
            global_features,
            local_features,
            params,
        )
        candidates = candidate_output["candidate_knots"]
        if self.stable_pilot_descriptors:
            # The truncated-power normal matrix is commonly ill-conditioned
            # (about 1e8 for the Kc=28 training setup).  CUDA's float32 batched
            # solve may therefore turn harmless 1e-7 upstream round-off into
            # order-one residual/importance changes, making KeepMask depend on
            # unrelated batch companions.  Pilot quantities are detached
            # structural features, so solve them in float64 and cast only the
            # resulting descriptors back to the network dtype.
            with torch.no_grad():
                stable_params = params.detach().to(torch.float64)
                stable_candidates = candidates.detach().to(torch.float64)
                stable_points = points.detach().to(torch.float64)
                stable_basis = build_design_matrix(
                    params=stable_params,
                    internal_knots=stable_candidates,
                    activity=torch.ones_like(stable_candidates),
                    degree=self.degree,
                    eps=0.0,
                    gate_transform="direct",
                )
                pilot_solver = solve_coefficients(
                    design_matrix=stable_basis["design_matrix"],
                    points=stable_points,
                    degree=self.degree,
                    lambda_poly=self.lambda_poly,
                    lambda_knot=self.lambda_knot,
                )
                stable_reconstruction = reconstruct_from_design(
                    stable_basis["design_matrix"],
                    pilot_solver["coefficients"],
                )
                analytic_delta = coefficient_drop_objective_delta(
                    pilot_solver["coefficients"],
                    pilot_solver["normal_matrix"],
                    first_column=self.degree + 1,
                ).to(points.dtype)
                knot_coefficients = pilot_solver["coefficients"][:, self.degree + 1 :]
                coefficient_energy = (
                    knot_coefficients.square().sum(dim=-1).to(points.dtype)
                )
                pilot_reconstruction = stable_reconstruction.to(points.dtype)
                point_residual = (pilot_reconstruction - points.detach()).norm(dim=-1)
                local_residual = self._candidate_local_residual(
                    params.detach(),
                    candidates.detach(),
                    point_residual,
                )
        else:
            pilot_basis = build_design_matrix(
                params=params,
                internal_knots=candidates,
                activity=torch.ones_like(candidates),
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

            # These are measured structural descriptors. Stopping their gradient
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
        proposal_knots = pruning_output.get(
            "proposal_candidate_knots", pruning_output["refined_candidate_knots"]
        )
        refined_knots = pruning_output.get(
            "deployment_candidate_knots", pruning_output["refined_candidate_knots"]
        )
        keep_probability = pruning_output["keep_probability"]
        learned_keep_mask = pruning_output.get(
            "final_hard_keep_mask", keep_probability >= 0.5
        )

        final_parameter_output = dict(parameter_output)
        parameter_feedback_output: dict[str, torch.Tensor] = {}
        joint_structure_output: dict[str, torch.Tensor] = {}
        deployment_knots = refined_knots
        final_keep_logits = pruning_output["keep_logits"]
        if self.parameter_feedback_fusion:
            survivor_gate = pruning_output.get(
                "final_hard_st_keep_gate",
                learned_keep_mask.to(keep_probability.dtype)
                + keep_probability
                - keep_probability.detach(),
            )
            if self.detach_activity_gate_for_fit:
                survivor_gate = survivor_gate.detach()
            candidate_mask = pruning_output.get(
                "candidate_mask",
                torch.ones_like(learned_keep_mask, dtype=torch.bool),
            )
            survivor_tokens = pruning_output.get(
                "relocation_tokens",
                pruning_output.get(
                    "position_refinement_tokens",
                    pruning_output["final_decision_tokens"],
                ),
            )
            parameter_feedback_output = self.parameter_feedback_head(
                local_features,
                points,
                params,
                parameter_output["parameter_gaps"],
                pilot_reconstruction.detach(),
                risk_candidate_positions=candidates,
                deletion_delta=analytic_delta.detach(),
                survivor_tokens=survivor_tokens,
                survivor_positions=refined_knots,
                survivor_gate=survivor_gate,
                hard_survivor_mask=learned_keep_mask,
                candidate_mask=candidate_mask,
            )
            final_parameter_output = {
                "params": parameter_feedback_output["feedback_params"],
                "parameter_gaps": parameter_feedback_output["feedback_parameter_gaps"],
                "raw_parameter_gaps": parameter_feedback_output[
                    "feedback_raw_parameter_gaps"
                ],
            }
            deployment_knots = self._warp_parameter_coordinates(
                refined_knots,
                params,
                final_parameter_output["params"],
            )
            parameter_feedback_output.update(
                {
                    "parameter_feedback_source_internal_knots": refined_knots,
                    "feedback_internal_knots": deployment_knots,
                }
            )
            if self.joint_parameter_structure_feedback:
                corrected_params = final_parameter_output["params"]
                corrected_candidates = self._warp_parameter_coordinates(
                    candidates,
                    params,
                    corrected_params,
                )
                transported_preliminary_knots = deployment_knots
                preliminary_position_residual = (
                    transported_preliminary_knots - corrected_candidates
                )
                corrected_local_residual = self._candidate_local_residual(
                    corrected_params.detach(),
                    corrected_candidates.detach(),
                    point_residual,
                )
                joint_head_output = self.joint_parameter_structure_head(
                    local_features,
                    corrected_params,
                    pruning_output["final_decision_tokens"],
                    corrected_candidates,
                    preliminary_keep_logits=pruning_output["keep_logits"],
                    deletion_delta=analytic_delta.detach(),
                    candidate_local_residual=corrected_local_residual.detach(),
                    preliminary_position_residual=preliminary_position_residual,
                    preliminary_keep_mask=learned_keep_mask,
                    candidate_mask=candidate_mask,
                )
                raw_joint_position_residual = (
                    self.pruning_head._bounded_all_candidate_residual(
                        transported_preliminary_knots,
                        joint_head_output["joint_raw_position_signal"],
                        candidate_mask,
                    )
                )
                unprojected_joint_candidates = (
                    transported_preliminary_knots + raw_joint_position_residual
                )
                if self.enforce_ordered_joint_candidates:
                    # The preliminary survivor-only relocation is ordered only
                    # on its selected subsequence: dormant slots may have been
                    # crossed.  The joint pass reconsiders every candidate, so
                    # restore a legal full Kc geometry before that re-selection.
                    joint_pre_relocation_knots = (
                        self.pruning_head._project_ordered_positions(
                            unprojected_joint_candidates,
                            candidate_mask,
                            reference_positions=corrected_candidates,
                        )
                    )
                else:
                    joint_pre_relocation_knots = unprojected_joint_candidates
                joint_position_residual = (
                    joint_pre_relocation_knots - transported_preliminary_knots
                )
                joint_order_projection_delta = (
                    joint_pre_relocation_knots - unprojected_joint_candidates
                )
                final_keep_logits = joint_head_output["joint_keep_logits"]
                keep_probability = joint_head_output["joint_keep_probability"]
                (
                    learned_keep_mask,
                    joint_selected_count,
                    joint_selection_uncertainty,
                ) = self.pruning_head._select_hard_keep_mask(
                    keep_probability,
                    candidate_mask,
                    joint_pre_relocation_knots,
                )
                joint_st_gate = self.pruning_head._straight_through_keep_gate(
                    keep_probability,
                    learned_keep_mask,
                    candidate_mask,
                )
                relocation_output = self.pruning_head._relocate_survivors(
                    joint_head_output["joint_structure_tokens"],
                    joint_pre_relocation_knots,
                    learned_keep_mask,
                    candidate_mask,
                    joint_st_gate,
                )
                joint_relocation_residual = (
                    self.pruning_head._bounded_selected_residual(
                        joint_pre_relocation_knots,
                        relocation_output["raw_signal"]
                        * joint_head_output["joint_relocation_scale"],
                        learned_keep_mask,
                        candidate_mask,
                        reference_positions=corrected_candidates,
                    )
                )
                unprojected_deployment_knots = (
                    joint_pre_relocation_knots + joint_relocation_residual
                )
                if self.enforce_ordered_joint_candidates:
                    # A second feasibility guard handles the accumulated
                    # parameter warp + joint move + selected-only relocation.
                    # Only retained slots constrain this terminal projection,
                    # so survivors may still use space vacated by deleted
                    # proposals while the deployed knot vector stays legal.
                    projected_survivor_knots = (
                        self.pruning_head._project_ordered_positions(
                            unprojected_deployment_knots,
                            learned_keep_mask & candidate_mask,
                            reference_positions=corrected_candidates,
                        )
                    )
                else:
                    projected_survivor_knots = unprojected_deployment_knots
                joint_survivor_order_projection_delta = (
                    projected_survivor_knots - unprojected_deployment_knots
                )
                joint_relocation_residual = (
                    projected_survivor_knots - joint_pre_relocation_knots
                )
                if self.enforce_ordered_joint_candidates:
                    deployment_knots = (
                        self.pruning_head._fill_inactive_ordered_slots(
                            projected_survivor_knots,
                            learned_keep_mask,
                            candidate_mask,
                        )
                    )
                else:
                    deployment_knots = projected_survivor_knots
                joint_inactive_order_fill_delta = (
                    deployment_knots - projected_survivor_knots
                )
                joint_probability_mass = (
                    keep_probability * candidate_mask.to(keep_probability.dtype)
                ).sum(dim=-1)
                joint_requested_count_score = (
                    joint_probability_mass
                    + self.pruning_head.one_shot_safety_sigma
                    * joint_selection_uncertainty
                )
                joint_context = self.pruning_head._hard_st_context(
                    joint_head_output["joint_structure_tokens"],
                    learned_keep_mask,
                    joint_st_gate,
                )
                joint_structure_output = {
                    **joint_head_output,
                    "joint_preliminary_keep_logits": pruning_output["keep_logits"],
                    "joint_preliminary_keep_probability": pruning_output[
                        "keep_probability"
                    ],
                    "joint_preliminary_hard_keep_mask": pruning_output[
                        "final_hard_keep_mask"
                    ],
                    "joint_preliminary_deployment_knots": (
                        transported_preliminary_knots
                    ),
                    "joint_corrected_candidate_knots": corrected_candidates,
                    "joint_candidate_local_residual": corrected_local_residual,
                    "joint_raw_position_residual": raw_joint_position_residual,
                    "joint_position_residual": joint_position_residual,
                    "joint_unprojected_candidate_knots": (
                        unprojected_joint_candidates
                    ),
                    "joint_order_projection_delta": joint_order_projection_delta,
                    "joint_pre_relocation_candidate_positions": (
                        joint_pre_relocation_knots
                    ),
                    "joint_pre_relocation_candidate_knots": (
                        joint_pre_relocation_knots
                    ),
                    "joint_relocation_relative_features": relocation_output[
                        "relative_features"
                    ],
                    "joint_relocation_previous_survivor_position": (
                        relocation_output["previous_position"]
                    ),
                    "joint_relocation_next_survivor_position": relocation_output[
                        "next_position"
                    ],
                    "joint_relocation_input_tokens": relocation_output["input_tokens"],
                    "joint_relocation_tokens": relocation_output["tokens"],
                    "joint_relocation_attention_weights": relocation_output[
                        "attention_weights"
                    ],
                    "joint_relocation_raw_position_signal": relocation_output[
                        "raw_signal"
                    ],
                    "joint_relocation_position_residual": (joint_relocation_residual),
                    "joint_unprojected_deployment_knots": (
                        unprojected_deployment_knots
                    ),
                    "joint_survivor_order_projection_delta": (
                        joint_survivor_order_projection_delta
                    ),
                    "joint_inactive_order_fill_delta": (
                        joint_inactive_order_fill_delta
                    ),
                    "joint_final_deployment_knots": deployment_knots,
                    # Final compatibility names intentionally override the
                    # preliminary pruning pass in the merged output below.
                    "keep_logits": final_keep_logits,
                    "keep_probabilities": keep_probability,
                    "keep_probability": keep_probability,
                    "final_keep_logits": final_keep_logits,
                    "final_keep_probabilities": keep_probability,
                    "final_keep_probability": keep_probability,
                    "final_hard_keep_mask": learned_keep_mask,
                    "one_shot_selected_count": joint_selected_count,
                    "one_shot_selection_uncertainty": (joint_selection_uncertainty),
                    "one_shot_probability_mass": joint_probability_mass,
                    "one_shot_requested_count_score": (joint_requested_count_score),
                    "final_hard_st_keep_gate": joint_st_gate,
                    "final_hard_st_keep_context": joint_context,
                    "pre_relocation_candidate_positions": (joint_pre_relocation_knots),
                    "pre_relocation_candidate_knots": joint_pre_relocation_knots,
                    "relocation_relative_features": relocation_output[
                        "relative_features"
                    ],
                    "relocation_previous_survivor_position": relocation_output[
                        "previous_position"
                    ],
                    "relocation_next_survivor_position": relocation_output[
                        "next_position"
                    ],
                    "relocation_input_tokens": relocation_output["input_tokens"],
                    "relocation_tokens": relocation_output["tokens"],
                    "relocation_attention_weights": relocation_output[
                        "attention_weights"
                    ],
                    "relocation_raw_position_signal": relocation_output["raw_signal"],
                    "relocation_position_residual": joint_relocation_residual,
                    "relocation_selected_mask": learned_keep_mask,
                    "position_residual": deployment_knots - corrected_candidates,
                    "refined_candidate_positions": deployment_knots,
                    "refined_candidate_knots": deployment_knots,
                    "deployment_candidate_positions": deployment_knots,
                    "deployment_candidate_knots": deployment_knots,
                }
                parameter_feedback_output.update(
                    {
                        "parameter_feedback_transport_internal_knots": (
                            transported_preliminary_knots
                        ),
                        "feedback_internal_knots": deployment_knots,
                    }
                )
        final_params = final_parameter_output["params"]

        if self._force_open_candidate_gate:
            fit_activity_gate = torch.ones_like(deployment_knots)
        elif self.structure_mode == "candidate_pruning_one_shot":
            fit_activity_gate = joint_structure_output.get(
                "final_hard_st_keep_gate",
                pruning_output.get(
                    "final_hard_st_keep_gate",
                    learned_keep_mask.to(keep_probability.dtype)
                    + keep_probability
                    - keep_probability.detach(),
                ),
            )
            if self.detach_activity_gate_for_fit:
                fit_activity_gate = fit_activity_gate.detach()
        else:
            fit_activity_gate = torch.ones_like(deployment_knots)

        fit_surrogate_output: dict[str, torch.Tensor] = {}
        if compute_fit_surrogate:
            basis_output = build_design_matrix(
                params=final_params,
                internal_knots=deployment_knots,
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
            fit_surrogate_output = {
                **basis_output,
                **solver_output,
                "reconstructed_points": reconstructed,
            }
        knot_mask = learned_keep_mask
        return {
            "local_features": local_features,
            "global_features": global_features,
            "proposal_params": parameter_output["params"],
            "proposal_parameter_gaps": parameter_output["parameter_gaps"],
            "proposal_raw_parameter_gaps": parameter_output["raw_parameter_gaps"],
            **final_parameter_output,
            **parameter_feedback_output,
            **candidate_output,
            **pruning_output,
            **joint_structure_output,
            # Compatibility aliases used by Trainer and diagnostics.  The
            # historical internal_knots name remains the actual deployment
            # geometry.  v11 additionally preserves the immutable proposal
            # geometry used by the offline Hard-RMS teacher.
            "proposal_internal_knots": proposal_knots,
            "deployment_internal_knots": deployment_knots,
            "internal_knots": deployment_knots,
            "activity": keep_probability,
            "activity_probability_logits": final_keep_logits,
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
            **fit_surrogate_output,
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

    def forward_deployment(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return one-shot deployment structure without the training surrogate fit.

        Candidate proposal, pilot descriptors, KeepMask and survivor relocation
        are identical to :meth:`forward`.  Only the final differentiable
        truncated-power solve is omitted because production immediately performs
        a standard B-spline control-point refit from the selected knots.
        """

        if self.structure_mode != "candidate_pruning_one_shot":
            raise RuntimeError(
                "forward_deployment requires candidate_pruning_one_shot mode"
            )
        local_features, global_features = self.encoder(points)
        parameter_output = self._parameter_head_forward(
            points,
            local_features,
            global_features,
        )
        return self._forward_candidate_pruning(
            points,
            local_features,
            global_features,
            parameter_output,
            compute_fit_surrogate=False,
        )

    def forward(
        self,
        points: torch.Tensor,
        true_internal_knot_count: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        local_features, global_features = self.encoder(points)
        parameter_output = self._parameter_head_forward(
            points,
            local_features,
            global_features,
        )
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
