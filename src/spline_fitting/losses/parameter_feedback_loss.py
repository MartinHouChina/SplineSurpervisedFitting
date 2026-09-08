from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .deployment_bspline_loss import differentiable_hard_gated_bspline_fit


@dataclass(frozen=True)
class ParameterFeedbackLossWeights:
    """Weights for the post-selection parameter-feedback calibration stage."""

    fit: float = 0.10
    threshold_violation: float = 0.50
    true_parameter: float = 1.00
    chord_prior: float = 0.05
    identity: float = 0.01
    deployment_fit: float = 0.00
    deployment_threshold_violation: float = 0.00
    joint_keep: float = 0.00
    joint_critical_recall: float = 0.00
    joint_position: float = 0.00
    joint_set_position: float = 0.00
    joint_spacing: float = 0.00
    joint_count: float = 0.00


class ParameterFeedbackLoss(nn.Module):
    """Calibrate parameter feedback and the optional final structure loop.

    Parameter errors are divided by ``parameter_error_scale**2``.  In v15 the
    optional deployment terms differentiate through the hard-mask standard
    B-spline refit, while Keep logits remain supervised by the offline teacher
    and the exact ``mass_topk`` requested-count score.  Zero-valued new weights
    preserve the historical one-way v14 loss exactly.
    """

    def __init__(
        self,
        weights: ParameterFeedbackLossWeights | None = None,
        *,
        fit_tolerance: float = 5e-3,
        parameter_error_scale: float = 0.02,
        knot_position_beta: float = 0.01,
        positive_keep_weight: float = 1.5,
        degree: int = 3,
        deployment_smoothness_weight: float = 1e-6,
        deployment_control_ridge: float = 0.0,
        deployment_solver_jitter: float = 1e-8,
    ) -> None:
        super().__init__()
        self.weights = weights or ParameterFeedbackLossWeights()
        if fit_tolerance <= 0.0:
            raise ValueError("fit_tolerance must be positive")
        if parameter_error_scale <= 0.0:
            raise ValueError("parameter_error_scale must be positive")
        if knot_position_beta <= 0.0:
            raise ValueError("knot_position_beta must be positive")
        if positive_keep_weight <= 0.0:
            raise ValueError("positive_keep_weight must be positive")
        if degree < 1:
            raise ValueError("degree must be positive")
        if (
            deployment_smoothness_weight < 0.0
            or deployment_control_ridge < 0.0
            or deployment_solver_jitter < 0.0
        ):
            raise ValueError("deployment refit regularization must be non-negative")
        if any(value < 0.0 for value in vars(self.weights).values()):
            raise ValueError("parameter-feedback loss weights must be non-negative")
        self.fit_tolerance = float(fit_tolerance)
        self.parameter_error_scale = float(parameter_error_scale)
        self.knot_position_beta = float(knot_position_beta)
        self.positive_keep_weight = float(positive_keep_weight)
        self.degree = int(degree)
        self.deletion_smoothness_weight = float(deployment_smoothness_weight)
        self.deletion_control_ridge = float(deployment_control_ridge)
        self.deployment_solver_jitter = float(deployment_solver_jitter)

    @staticmethod
    def _warp_parameter_coordinates(
        values: torch.Tensor,
        source_params: torch.Tensor,
        target_params: torch.Tensor,
    ) -> torch.Tensor:
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

    @staticmethod
    def _ordered_pair_indices(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> list[tuple[int, int]]:
        """Minimum-L1 ordered pairs, matching every item of the smaller set."""

        if first.numel() == 0 or second.numel() == 0:
            return []
        if first.numel() < second.numel():
            reversed_pairs = ParameterFeedbackLoss._ordered_pair_indices(second, first)
            return [(right, left) for left, right in reversed_pairs]
        first_values = first.detach().cpu().tolist()
        second_values = second.detach().cpu().tolist()
        first_count = len(first_values)
        second_count = len(second_values)
        costs = [[float("inf")] * (second_count + 1) for _ in range(first_count + 1)]
        take = [[False] * (second_count + 1) for _ in range(first_count + 1)]
        for first_index in range(first_count + 1):
            costs[first_index][0] = 0.0
        for first_index in range(1, first_count + 1):
            for second_index in range(1, min(first_index, second_count) + 1):
                skip_cost = costs[first_index - 1][second_index]
                match_cost = costs[first_index - 1][second_index - 1] + abs(
                    first_values[first_index - 1] - second_values[second_index - 1]
                )
                if match_cost <= skip_cost:
                    costs[first_index][second_index] = match_cost
                    take[first_index][second_index] = True
                else:
                    costs[first_index][second_index] = skip_cost
        first_index = first_count
        second_index = second_count
        pairs: list[tuple[int, int]] = []
        while second_index > 0:
            if take[first_index][second_index]:
                pairs.append((first_index - 1, second_index - 1))
                first_index -= 1
                second_index -= 1
            else:
                first_index -= 1
        pairs.reverse()
        return pairs

    def forward(
        self,
        output: dict[str, torch.Tensor],
        points: torch.Tensor,
        chord_params: torch.Tensor | None = None,
        true_params: torch.Tensor | None = None,
        teacher_retained_mask: torch.Tensor | None = None,
        teacher_internal_knots: torch.Tensor | None = None,
        teacher_internal_knot_mask: torch.Tensor | None = None,
        teacher_count: torch.Tensor | None = None,
        teacher_soft_keep_risk: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        required = {"params", "reconstructed_points"}
        missing = sorted(required.difference(output))
        if missing:
            raise KeyError(
                "parameter-feedback output is missing: " + ", ".join(missing)
            )
        params = output["params"]
        reconstructed = output["reconstructed_points"]
        if params.shape != points.shape[:2]:
            raise ValueError("params must share the point batch/sequence shape")
        if reconstructed.shape != points.shape:
            raise ValueError("reconstructed_points must have the same shape as points")

        per_sample_fit_mse = (reconstructed - points).square().sum(dim=-1).mean(dim=-1)
        fit_loss = per_sample_fit_mse.mean()
        fit_scale = self.fit_tolerance**2
        normalized_fit_loss = fit_loss / fit_scale
        per_sample_fit_rms = per_sample_fit_mse.clamp_min(0.0).sqrt()
        threshold_violation_loss = (
            torch.relu(per_sample_fit_rms / self.fit_tolerance - 1.0).square().mean()
        )

        # v15 aligns calibration with the actual deployed curve.  Unlike the
        # historical truncated-power proxy above, this is the same standard
        # open-clamped B-spline basis and endpoint-constrained refit used at
        # evaluation.  It is opt-in so old v14 checkpoints retain their exact
        # loss semantics.
        zero = points.new_zeros(())
        deployment_fit_loss = zero
        normalized_deployment_fit_loss = zero
        deployment_threshold_violation_loss = zero
        deployment_threshold_satisfied_rate = zero
        if (
            self.weights.deployment_fit > 0.0
            or self.weights.deployment_threshold_violation > 0.0
        ):
            deployment_knots = output.get("deployment_internal_knots")
            final_mask = output.get("learned_keep_mask")
            if deployment_knots is None or final_mask is None:
                raise KeyError(
                    "deployment-aligned feedback requires deployed knots and final mask"
                )
            exact_fit = differentiable_hard_gated_bspline_fit(
                params,
                points,
                deployment_knots,
                final_mask,
                degree=self.degree,
                smoothness_weight=self.deletion_smoothness_weight,
                control_ridge=self.deletion_control_ridge,
                solver_jitter=self.deployment_solver_jitter,
            )
            deployment_fit_loss = exact_fit["fit_mse"]
            normalized_deployment_fit_loss = deployment_fit_loss / fit_scale
            deployment_rms = exact_fit["per_sample_rms"]
            deployment_threshold_violation_loss = torch.relu(
                deployment_rms / self.fit_tolerance - 1.0
            ).square().mean()
            deployment_threshold_satisfied_rate = (
                (deployment_rms <= self.fit_tolerance).to(points.dtype).mean()
            )

        parameter_scale = self.parameter_error_scale**2
        if true_params is None:
            true_parameter_loss = zero
            normalized_true_parameter_loss = zero
            true_parameter_supervision_fraction = zero
        else:
            if true_params.shape != params.shape:
                raise ValueError("true_params and params must share shape [B,M]")
            true_parameter_loss = (params - true_params).square().mean()
            normalized_true_parameter_loss = true_parameter_loss / parameter_scale
            true_parameter_supervision_fraction = points.new_ones(())

        if chord_params is None:
            chord_prior_loss = zero
            normalized_chord_prior_loss = zero
        else:
            if chord_params.shape != params.shape:
                raise ValueError("chord_params and params must share shape [B,M]")
            chord_prior_loss = (params - chord_params).square().mean()
            normalized_chord_prior_loss = chord_prior_loss / parameter_scale

        proposal_params = output.get("proposal_params")
        if proposal_params is None:
            identity_loss = zero
            normalized_identity_loss = zero
        else:
            if proposal_params.shape != params.shape:
                raise ValueError("proposal_params and params must share shape [B,M]")
            identity_loss = (params - proposal_params.detach()).square().mean()
            normalized_identity_loss = identity_loss / parameter_scale

        joint_keep_loss = zero
        joint_critical_recall_loss = zero
        joint_position_loss = zero
        joint_set_position_loss = zero
        joint_spacing_loss = zero
        joint_count_loss = zero
        joint_requested_count_mean = zero
        joint_target_count_mean = zero
        joint_target_policy_score_mean = zero
        joint_supervision_fraction = zero
        keep_logits = output.get("keep_logits")
        keep_probability = output.get("keep_probability")
        candidate_mask = output.get("candidate_mask")
        if teacher_retained_mask is not None:
            if keep_logits is None or keep_probability is None:
                raise KeyError(
                    "joint feedback requires final Keep logits/probabilities"
                )
            target_keep = teacher_retained_mask.to(
                device=keep_logits.device,
                dtype=keep_logits.dtype,
            )
            if target_keep.shape != keep_logits.shape:
                raise ValueError("teacher_retained_mask must share candidate shape")
            valid_mask = (
                torch.ones_like(target_keep, dtype=torch.bool)
                if candidate_mask is None
                else candidate_mask.to(torch.bool)
            )
            positive_weight = keep_logits.new_tensor(self.positive_keep_weight)
            elementwise_keep = F.binary_cross_entropy_with_logits(
                keep_logits,
                target_keep,
                pos_weight=positive_weight,
                reduction="none",
            )
            keep_bce = elementwise_keep[valid_mask].mean()
            masked_probability = keep_probability * valid_mask.to(
                keep_probability.dtype
            )
            intersection = (masked_probability * target_keep).sum(dim=-1)
            dice = 1.0 - (2.0 * intersection + 1.0) / (
                masked_probability.sum(dim=-1) + target_keep.sum(dim=-1) + 1.0
            )
            joint_keep_loss = keep_bce + dice.mean()
            if teacher_soft_keep_risk is not None:
                risk = teacher_soft_keep_risk.to(
                    device=keep_logits.device,
                    dtype=keep_logits.dtype,
                )
                if risk.shape != keep_logits.shape:
                    raise ValueError(
                        "teacher_soft_keep_risk must share candidate shape"
                    )
                if torch.any((risk < 0.0) | (risk > 1.0)):
                    raise ValueError("teacher_soft_keep_risk must lie in [0,1]")
                critical_weight = risk * target_keep * valid_mask.to(risk.dtype)
                joint_critical_recall_loss = (
                    critical_weight * F.softplus(-keep_logits)
                ).sum() / critical_weight.sum().clamp_min(1.0)
            if teacher_count is None:
                target_count = target_keep.sum(dim=-1)
            else:
                target_count = teacher_count.to(
                    device=keep_probability.device,
                    dtype=keep_probability.dtype,
                ).reshape(-1)
            predicted_count = masked_probability.sum(dim=-1)
            requested_count_score = output.get(
                "one_shot_requested_count_score", predicted_count
            )
            if requested_count_score.shape != predicted_count.shape:
                raise ValueError(
                    "one_shot_requested_count_score must have shape [B]"
                )
            # Deployment uses score<0.5 for zero and ceil(score) otherwise.
            # Supervising plain sum(p) at integer K ignored the positive
            # uncertainty reserve and systematically deployed K+1 nodes.
            target_policy_score = torch.where(
                target_count > 0,
                target_count - 0.25,
                torch.full_like(target_count, 0.25),
            )
            count_scale = (
                valid_mask.sum(dim=-1).to(predicted_count.dtype).clamp_min(1.0)
            )
            joint_count_loss = F.smooth_l1_loss(
                requested_count_score / count_scale,
                target_policy_score / count_scale,
                beta=0.05,
            )
            joint_requested_count_mean = requested_count_score.mean()
            joint_target_count_mean = target_count.mean()
            joint_target_policy_score_mean = target_policy_score.mean()
            joint_supervision_fraction = points.new_ones(())

        if teacher_internal_knots is not None or teacher_internal_knot_mask is not None:
            if teacher_internal_knots is None or teacher_internal_knot_mask is None:
                raise ValueError(
                    "teacher_internal_knots and teacher_internal_knot_mask "
                    "must be supplied together"
                )
            deployment_knots = output.get("deployment_internal_knots")
            final_mask = output.get("learned_keep_mask")
            if deployment_knots is None or final_mask is None:
                raise KeyError("joint feedback requires deployed knots and final mask")
            if teacher_internal_knots.shape != deployment_knots.shape:
                raise ValueError("teacher internal knots must share candidate shape")
            if teacher_internal_knot_mask.shape != deployment_knots.shape:
                raise ValueError("teacher knot mask must share candidate shape")
            source_params = output.get("proposal_params")
            if source_params is None:
                raise KeyError(
                    "joint feedback requires proposal_params for t0->t1 warp"
                )
            warped_targets = self._warp_parameter_coordinates(
                teacher_internal_knots.to(params.dtype),
                source_params,
                params,
            )
            position_terms: list[torch.Tensor] = []
            set_position_terms: list[torch.Tensor] = []
            spacing_terms: list[torch.Tensor] = []
            for batch_index in range(points.shape[0]):
                predicted = torch.sort(
                    deployment_knots[
                        batch_index, final_mask[batch_index].to(torch.bool)
                    ]
                ).values
                target = torch.sort(
                    warped_targets[
                        batch_index,
                        teacher_internal_knot_mask[batch_index].to(torch.bool),
                    ]
                ).values
                pairs = self._ordered_pair_indices(predicted, target)
                if not pairs:
                    continue
                predicted_indices = torch.tensor(
                    [left for left, _ in pairs],
                    dtype=torch.long,
                    device=points.device,
                )
                target_indices = torch.tensor(
                    [right for _, right in pairs],
                    dtype=torch.long,
                    device=points.device,
                )
                position_terms.append(
                    F.smooth_l1_loss(
                        predicted[predicted_indices],
                        target[target_indices],
                        beta=self.knot_position_beta,
                    )
                )
                pairwise_distance = (predicted[:, None] - target[None, :]).abs()
                nearest_target = pairwise_distance.argmin(dim=1)
                nearest_prediction = pairwise_distance.argmin(dim=0)
                set_position_terms.append(
                    0.5
                    * (
                        F.smooth_l1_loss(
                            predicted,
                            target[nearest_target],
                            beta=self.knot_position_beta,
                        )
                        + F.smooth_l1_loss(
                            target,
                            predicted[nearest_prediction],
                            beta=self.knot_position_beta,
                        )
                    )
                )
                if predicted.numel() == target.numel():
                    predicted_gaps = torch.cat(
                        [predicted.new_zeros(1), predicted, predicted.new_ones(1)]
                    ).diff()
                    target_gaps = torch.cat(
                        [target.new_zeros(1), target, target.new_ones(1)]
                    ).diff()
                    spacing_terms.append(
                        F.smooth_l1_loss(
                            predicted_gaps,
                            target_gaps,
                            beta=self.knot_position_beta,
                        )
                    )
            if position_terms:
                joint_position_loss = torch.stack(position_terms).mean()
            if set_position_terms:
                joint_set_position_loss = torch.stack(set_position_terms).mean()
            if spacing_terms:
                joint_spacing_loss = torch.stack(spacing_terms).mean()

        total = (
            self.weights.fit * normalized_fit_loss
            + self.weights.threshold_violation * threshold_violation_loss
            + self.weights.true_parameter * normalized_true_parameter_loss
            + self.weights.chord_prior * normalized_chord_prior_loss
            + self.weights.identity * normalized_identity_loss
            + self.weights.deployment_fit * normalized_deployment_fit_loss
            + self.weights.deployment_threshold_violation
            * deployment_threshold_violation_loss
            + self.weights.joint_keep * joint_keep_loss
            + self.weights.joint_critical_recall * joint_critical_recall_loss
            + self.weights.joint_position * joint_position_loss
            + self.weights.joint_set_position * joint_set_position_loss
            + self.weights.joint_spacing * joint_spacing_loss
            + self.weights.joint_count * joint_count_loss
        )

        feedback_shift = output.get("parameter_feedback_gap_logit_delta")
        if feedback_shift is None:
            # Compatibility with early experimental feedback-head checkpoints.
            feedback_shift = output.get("parameter_feedback_gap_logit_shift")
        feedback_shift_mean_abs = (
            feedback_shift.abs().mean() if feedback_shift is not None else zero
        )
        chord_blend = output.get("parameter_feedback_chord_blend_weight")
        chord_blend_mean = chord_blend.mean() if chord_blend is not None else zero
        return {
            "loss": total,
            "fit_loss": fit_loss,
            "normalized_fit_loss": normalized_fit_loss,
            "threshold_violation_loss": threshold_violation_loss,
            "threshold_satisfied_rate": (
                (per_sample_fit_rms <= self.fit_tolerance).to(points.dtype).mean()
            ),
            "deployment_fit_loss": deployment_fit_loss,
            "normalized_deployment_fit_loss": normalized_deployment_fit_loss,
            "deployment_training_threshold_violation_loss": (
                deployment_threshold_violation_loss
            ),
            "deployment_training_threshold_satisfied_rate": (
                deployment_threshold_satisfied_rate
            ),
            "true_parameter_loss": true_parameter_loss,
            "normalized_true_parameter_loss": normalized_true_parameter_loss,
            "true_parameter_supervision_fraction": (
                true_parameter_supervision_fraction
            ),
            "parameter_prior_loss": chord_prior_loss,
            "normalized_parameter_prior_loss": normalized_chord_prior_loss,
            "parameter_feedback_identity_loss": identity_loss,
            "normalized_parameter_feedback_identity_loss": normalized_identity_loss,
            "parameter_feedback_gap_logit_shift_mean_abs": feedback_shift_mean_abs,
            "parameter_feedback_chord_blend_weight": chord_blend_mean,
            "joint_keep_loss": joint_keep_loss,
            "joint_critical_recall_loss": joint_critical_recall_loss,
            "joint_knot_position_loss": joint_position_loss,
            "joint_set_position_loss": joint_set_position_loss,
            "joint_spacing_loss": joint_spacing_loss,
            "joint_count_loss": joint_count_loss,
            "joint_requested_count_mean": joint_requested_count_mean,
            "joint_target_count_mean": joint_target_count_mean,
            "joint_target_policy_score_mean": joint_target_policy_score_mean,
            "joint_structure_supervision_fraction": joint_supervision_fraction,
        }
