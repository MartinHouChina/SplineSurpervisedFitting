from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..spline.bspline_deletion_teacher import single_knot_deletion_rmse_batch
from .total_loss import SplineFittingLoss


@dataclass
class CandidatePruningLossWeights:
    """Weights for high-recall proposal and supervised pruning training."""

    fit: float = 0.25
    threshold_violation: float = 5.0
    true_parameter: float = 5e-2
    candidate_coverage: float = 5.0
    candidate_repulsion: float = 5e-2
    keep: float = 2.5e-1
    remove_action: float = 1.0
    knot_position: float = 2.0
    count_consistency: float = 0.0
    deletion_cost: float = 5e-2
    teacher_risk: float = 0.0
    teacher_ranking: float = 0.0
    teacher_distribution: float = 0.0
    teacher_critical_recall: float = 0.0
    teacher_false_positive: float = 0.0
    teacher_count: float = 0.0
    policy_count: float = 0.0
    canonical_selection: float = 0.0
    complexity: float = 0.0


class CandidatePruningLoss(nn.Module):
    """Train proposals for recall and pruning tokens for safe deletion.

    In v8 the final keep mask is distilled from an offline hard-RMS teacher and
    deployment performs one standard B-spline refit.  Historical v7 callers
    can still enable sequential deletion supervision through the same loss.
    """

    def __init__(
        self,
        weights: CandidatePruningLossWeights | None = None,
        *,
        knot_position_beta: float = 0.01,
        candidate_match_tolerance: float = 0.02,
        fit_tolerance: float = 5e-3,
        repulsion_distance: float | None = None,
        positive_keep_weight: float = 2.0,
        exact_deletion_supervision: bool = True,
        deletion_smoothness_weight: float = 1e-6,
        deletion_control_ridge: float = 0.0,
        teacher_ranking_margin: float = 1.0,
        candidate_coverage_tolerances: tuple[float, ...] = (),
        position_aware_distribution: bool = False,
        joint_position_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.weights = weights or CandidatePruningLossWeights()
        if knot_position_beta <= 0.0:
            raise ValueError("knot_position_beta must be positive")
        if candidate_match_tolerance < 0.0:
            raise ValueError("candidate_match_tolerance must be non-negative")
        if fit_tolerance <= 0.0:
            raise ValueError("fit_tolerance must be positive")
        if repulsion_distance is not None and repulsion_distance < 0.0:
            raise ValueError("repulsion_distance must be non-negative or None")
        if positive_keep_weight <= 0.0:
            raise ValueError("positive_keep_weight must be positive")
        if deletion_smoothness_weight < 0.0 or deletion_control_ridge < 0.0:
            raise ValueError("deletion solver weights must be non-negative")
        if teacher_ranking_margin < 0.0:
            raise ValueError("teacher_ranking_margin must be non-negative")
        if any(
            (not isinstance(value, (int, float))) or value <= 0.0
            for value in candidate_coverage_tolerances
        ):
            raise ValueError("candidate coverage tolerances must be positive")
        self.knot_position_beta = float(knot_position_beta)
        self.candidate_match_tolerance = float(candidate_match_tolerance)
        self.fit_tolerance = float(fit_tolerance)
        self.repulsion_distance = repulsion_distance
        self.positive_keep_weight = float(positive_keep_weight)
        self.exact_deletion_supervision = bool(exact_deletion_supervision)
        self.deletion_smoothness_weight = float(deletion_smoothness_weight)
        self.deletion_control_ridge = float(deletion_control_ridge)
        self.teacher_ranking_margin = float(teacher_ranking_margin)
        self.candidate_coverage_tolerances = tuple(
            float(value) for value in candidate_coverage_tolerances
        )
        self.position_aware_distribution = bool(position_aware_distribution)
        self.joint_position_supervision = bool(joint_position_supervision)

    @staticmethod
    def _fit_loss(reconstructed: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return (reconstructed - points).pow(2).sum(dim=-1).mean()

    @staticmethod
    def _ordered_pair_indices(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> list[tuple[int, int]]:
        """Minimum-L1 ordered pairs, matching every item of the smaller set."""

        if first.numel() == 0 or second.numel() == 0:
            return []
        if first.numel() < second.numel():
            reversed_pairs = CandidatePruningLoss._ordered_pair_indices(second, first)
            return [(right, left) for left, right in reversed_pairs]

        candidates = first.detach().cpu().tolist()
        targets = second.detach().cpu().tolist()
        candidate_count = len(candidates)
        target_count = len(targets)
        infinity = float("inf")
        costs = [[infinity] * (target_count + 1) for _ in range(candidate_count + 1)]
        take = [[False] * (target_count + 1) for _ in range(candidate_count + 1)]
        for candidate_index in range(candidate_count + 1):
            costs[candidate_index][0] = 0.0
        for candidate_index in range(1, candidate_count + 1):
            for target_index in range(1, min(candidate_index, target_count) + 1):
                skip_cost = costs[candidate_index - 1][target_index]
                match_cost = costs[candidate_index - 1][target_index - 1] + abs(
                    candidates[candidate_index - 1] - targets[target_index - 1]
                )
                if match_cost <= skip_cost:
                    costs[candidate_index][target_index] = match_cost
                    take[candidate_index][target_index] = True
                else:
                    costs[candidate_index][target_index] = skip_cost
        candidate_index = candidate_count
        target_index = target_count
        pairs: list[tuple[int, int]] = []
        while target_index > 0:
            if take[candidate_index][target_index]:
                pairs.append((candidate_index - 1, target_index - 1))
                candidate_index -= 1
                target_index -= 1
            else:
                candidate_index -= 1
        pairs.reverse()
        return pairs

    def forward(
        self,
        output: dict[str, torch.Tensor],
        points: torch.Tensor,
        chord_params: torch.Tensor | None = None,
        true_params: torch.Tensor | None = None,
        true_internal_knots: torch.Tensor | None = None,
        true_internal_knot_mask: torch.Tensor | None = None,
        teacher_retained_mask: torch.Tensor | None = None,
        teacher_soft_keep_risk: torch.Tensor | None = None,
        teacher_internal_knots: torch.Tensor | None = None,
        teacher_internal_knot_mask: torch.Tensor | None = None,
        teacher_count: torch.Tensor | None = None,
        teacher_fit_rms: torch.Tensor | None = None,
        teacher_threshold_satisfied: torch.Tensor | None = None,
        teacher_single_deletion_rms: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        required = {
            "candidate_knots",
            "internal_knots",
            "keep_logits",
            "keep_probability",
            "reconstructed_points",
            "params",
        }
        missing = sorted(required.difference(output))
        if missing:
            raise KeyError("candidate-pruning output is missing: " + ", ".join(missing))
        if true_internal_knots is None or true_internal_knot_mask is None:
            raise ValueError("candidate-pruning training requires true internal knots")

        candidates = output["candidate_knots"]
        refined = output["internal_knots"]
        proposal_refined = output.get("proposal_internal_knots", refined)
        keep_logits = output["keep_logits"]
        keep_probability = output["keep_probability"]
        if not (
            candidates.shape
            == refined.shape
            == proposal_refined.shape
            == keep_logits.shape
            == keep_probability.shape
        ):
            raise ValueError("all candidate tensors must share shape [B,Kc]")

        true_mask = true_internal_knot_mask.to(torch.bool)
        true_count = true_mask.sum(dim=-1).to(torch.long)
        existence_targets, position_targets, matched_mask = (
            SplineFittingLoss._ordered_knot_assignment(
                proposal_refined,
                true_internal_knots,
                true_mask,
            )
        )

        coverage_terms: list[torch.Tensor] = []
        multiscale_coverage_terms: list[torch.Tensor] = []
        multiscale_recall_counts = {
            tolerance: points.new_zeros(())
            for tolerance in self.candidate_coverage_tolerances
        }
        recall_count = points.new_zeros(())
        initial_recall_count = points.new_zeros(())
        target_count = points.new_zeros(())
        nearest_error_sum = points.new_zeros(())
        for batch_index in range(points.shape[0]):
            targets = true_internal_knots[batch_index, true_mask[batch_index]]
            if targets.numel() == 0:
                continue
            distances = (
                targets.unsqueeze(-1) - candidates[batch_index].unsqueeze(0)
            ).abs()
            initial_nearest = distances.amin(dim=-1)
            refined_nearest = (
                (targets.unsqueeze(-1) - proposal_refined[batch_index].unsqueeze(0))
                .abs()
                .amin(dim=-1)
            )
            coverage_scale = max(self.candidate_match_tolerance, 1e-3)
            coverage_terms.append(
                0.5 * (initial_nearest.mean() + refined_nearest.mean()) / coverage_scale
            )
            for tolerance in self.candidate_coverage_tolerances:
                # Mean nearest distance alone can sacrifice a few difficult
                # knots.  Multi-scale hinge barriers directly train recall at
                # the tolerances used by evaluation and saturate once covered.
                for nearest in (initial_nearest, refined_nearest):
                    multiscale_coverage_terms.append(
                        torch.relu(nearest / tolerance - 1.0).square().mean()
                    )
                multiscale_recall_counts[tolerance] = (
                    multiscale_recall_counts[tolerance]
                    + (refined_nearest <= tolerance).to(points.dtype).sum()
                )
            recall_count = (
                recall_count
                + (refined_nearest <= self.candidate_match_tolerance)
                .to(points.dtype)
                .sum()
            )
            initial_recall_count = (
                initial_recall_count
                + (initial_nearest <= self.candidate_match_tolerance)
                .to(points.dtype)
                .sum()
            )
            target_count = target_count + targets.numel()
            nearest_error_sum = nearest_error_sum + refined_nearest.sum()
        candidate_coverage_loss = (
            torch.stack(coverage_terms).mean()
            if coverage_terms
            else points.new_zeros(())
        )
        if multiscale_coverage_terms:
            candidate_coverage_loss = (
                candidate_coverage_loss + torch.stack(multiscale_coverage_terms).mean()
            )

        candidate_count = candidates.shape[-1]
        desired_gap = (
            self.repulsion_distance
            if self.repulsion_distance is not None
            else 0.5 / max(candidate_count + 1, 1)
        )
        if candidate_count > 1:
            adjacent_gaps = candidates[:, 1:] - candidates[:, :-1]
            candidate_repulsion_loss = (
                (torch.relu(desired_gap - adjacent_gaps) / max(desired_gap, 1e-6))
                .pow(2)
                .mean()
            )
        else:
            candidate_repulsion_loss = points.new_zeros(())

        if teacher_retained_mask is not None:
            if teacher_retained_mask.shape != candidates.shape:
                raise ValueError("teacher_retained_mask must have shape [B,Kc]")
            keep_targets = teacher_retained_mask.to(points.dtype)
        else:
            keep_targets = existence_targets
        positive_weight = points.new_tensor(self.positive_keep_weight)
        keep_bce_loss = F.binary_cross_entropy_with_logits(
            keep_logits,
            keep_targets,
            pos_weight=positive_weight,
        )
        if teacher_retained_mask is not None:
            # Slot-wise BCE calibrates probabilities, while the soft Dice term
            # prevents the heavily imbalanced candidate set from finding an
            # easy all-remove or all-keep solution.  This term is v8-only;
            # historical v7 supervision keeps its exact loss semantics.
            dice_numerator = 2.0 * (keep_probability * keep_targets).sum(dim=-1)
            dice_denominator = keep_probability.sum(dim=-1) + keep_targets.sum(dim=-1)
            keep_dice_loss = (
                1.0 - (dice_numerator + 1.0) / (dice_denominator + 1.0)
            ).mean()
        else:
            keep_dice_loss = points.new_zeros(())
        keep_loss = keep_bce_loss + keep_dice_loss
        canonical_selection_loss = F.binary_cross_entropy_with_logits(
            keep_logits,
            existence_targets,
        )
        if teacher_retained_mask is not None:
            # A canonical-positive slot may be an interchangeable alternative
            # to the particular subset chosen by greedy Hard-RMS.  Do not
            # teach it as a strong false positive merely because the teacher
            # retained a neighbouring equivalent slot.
            negative_weight = (1.0 - keep_targets) * (1.0 - existence_targets)
            teacher_false_positive_loss = (
                F.softplus(keep_logits) * negative_weight
            ).sum() / negative_weight.sum().clamp_min(1.0)
        else:
            teacher_false_positive_loss = points.new_zeros(())
        if teacher_retained_mask is not None:
            ranking_terms: list[torch.Tensor] = []
            retained_mask = teacher_retained_mask.to(torch.bool)
            for batch_index in range(points.shape[0]):
                retained_logits = keep_logits[batch_index, retained_mask[batch_index]]
                removed_logits = keep_logits[batch_index, ~retained_mask[batch_index]]
                if retained_logits.numel() and removed_logits.numel():
                    pairwise_difference = retained_logits.unsqueeze(
                        -1
                    ) - removed_logits.unsqueeze(0)
                    ranking_terms.append(
                        F.softplus(
                            self.teacher_ranking_margin - pairwise_difference
                        ).mean()
                    )
            teacher_ranking_loss = (
                torch.stack(ranking_terms).mean()
                if ranking_terms
                else points.new_zeros(())
            )
        else:
            teacher_ranking_loss = points.new_zeros(())
        if teacher_single_deletion_rms is not None:
            if teacher_single_deletion_rms.shape != candidates.shape:
                raise ValueError("teacher_single_deletion_rms must have shape [B,Kc]")
            deletion_rms = teacher_single_deletion_rms.to(points.dtype)
            safe_delete_mask = deletion_rms <= self.fit_tolerance
        elif self.exact_deletion_supervision:
            deletion_rms = single_knot_deletion_rmse_batch(
                output["params"].detach(),
                points.detach(),
                refined.detach(),
                degree=3,
                smoothness_weight=self.deletion_smoothness_weight,
                control_ridge=self.deletion_control_ridge,
                interpolate_endpoints=True,
            )
            safe_delete_mask = deletion_rms <= self.fit_tolerance
        else:
            deletion_rms = None
            # Cheap fallback used in candidate-only pretraining and evaluation
            # loss reporting. It must never be confused with an RMS guarantee.
            safe_delete_mask = ~keep_targets.to(torch.bool)

        if teacher_soft_keep_risk is not None:
            if teacher_soft_keep_risk.shape != candidates.shape:
                raise ValueError("teacher_soft_keep_risk must have shape [B,Kc]")
            teacher_risk_loss = F.binary_cross_entropy(
                keep_probability.clamp(1e-6, 1.0 - 1e-6),
                teacher_soft_keep_risk.to(points.dtype).clamp(0.0, 1.0),
            )
        else:
            teacher_risk_loss = points.new_zeros(())

        # A slot-wise classifier can obtain a reasonable average F1 while
        # placing most selected knots in the same parameter-domain region.
        # Comparing normalized cumulative mass is the 1-D Wasserstein/CDF
        # distance between the predicted and teacher knot sets.  It explicitly
        # trains global spatial allocation without requiring a second spline
        # solve or an arbitrary candidate-to-candidate matching operation.
        if teacher_retained_mask is not None and self.position_aware_distribution:
            grid = torch.linspace(
                0.0,
                1.0,
                65,
                device=points.device,
                dtype=points.dtype,
            )
            temperature = max(self.candidate_match_tolerance * 0.5, 2.5e-3)
            predicted_steps = torch.sigmoid(
                (grid.view(1, 1, -1) - refined.unsqueeze(-1)) / temperature
            )
            predicted_mass = keep_probability.sum(dim=-1, keepdim=True)
            predicted_cdf = (predicted_steps * keep_probability.unsqueeze(-1)).sum(
                dim=1
            ) / predicted_mass.clamp_min(1e-6)
            true_steps = torch.sigmoid(
                (
                    grid.view(1, 1, -1)
                    - true_internal_knots.to(points.dtype).unsqueeze(-1)
                )
                / temperature
            )
            true_mass = true_mask.to(points.dtype).sum(dim=-1, keepdim=True)
            true_cdf = (true_steps * true_mask.to(points.dtype).unsqueeze(-1)).sum(
                dim=1
            ) / true_mass.clamp_min(1.0)
            valid_distribution = true_mass.squeeze(-1) > 0
            per_sample_distribution = (predicted_cdf - true_cdf).abs().mean(dim=-1)
            teacher_distribution_loss = (
                per_sample_distribution[valid_distribution].mean()
                if valid_distribution.any()
                else points.new_zeros(())
            )
        elif teacher_retained_mask is not None:
            teacher_mass = keep_targets.sum(dim=-1, keepdim=True)
            predicted_mass = keep_probability.sum(dim=-1, keepdim=True)
            valid_distribution = teacher_mass.squeeze(-1) > 0
            predicted_density = keep_probability / predicted_mass.clamp_min(1e-6)
            teacher_density = keep_targets / teacher_mass.clamp_min(1.0)
            per_sample_distribution = (
                (predicted_density.cumsum(dim=-1) - teacher_density.cumsum(dim=-1))
                .abs()
                .mean(dim=-1)
            )
            teacher_distribution_loss = (
                per_sample_distribution[valid_distribution].mean()
                if valid_distribution.any()
                else points.new_zeros(())
            )
        else:
            teacher_distribution_loss = points.new_zeros(())

        # False negatives at the teacher's final stopping state are much more
        # damaging than harmless extra knots: deleting any such critical slot
        # violates epsilon.  The positive-only softplus term therefore gives
        # retained high-risk slots an explicit recall gradient, while count and
        # complexity terms continue to penalize unnecessary additions.
        if teacher_retained_mask is not None:
            critical_weight = keep_targets
            if teacher_soft_keep_risk is not None:
                critical_weight = critical_weight * (
                    1.0 + teacher_soft_keep_risk.to(points.dtype)
                )
            teacher_critical_recall_loss = (
                F.softplus(-keep_logits) * critical_weight
            ).sum() / critical_weight.sum().clamp_min(1.0)
        else:
            teacher_critical_recall_loss = points.new_zeros(())

        remove_stop_logits = output.get("remove_stop_logits")
        if remove_stop_logits is not None:
            if remove_stop_logits.shape != (
                candidates.shape[0],
                candidates.shape[1] + 1,
            ):
                raise ValueError("remove_stop_logits must have shape [B,Kc+1]")
            log_probability = F.log_softmax(remove_stop_logits, dim=-1)
            action_terms: list[torch.Tensor] = []
            for batch_index in range(points.shape[0]):
                if safe_delete_mask[batch_index].any():
                    # Several deletions can satisfy the same geometric bound.
                    # Multi-positive likelihood avoids inventing an arbitrary
                    # action order when the standard B-spline teacher says
                    # that several next steps are valid.
                    action_terms.append(
                        -torch.logsumexp(
                            log_probability[batch_index, :-1][
                                safe_delete_mask[batch_index]
                            ],
                            dim=0,
                        )
                    )
                else:
                    action_terms.append(-log_probability[batch_index, -1])
            remove_action_loss = torch.stack(action_terms).mean()
        else:
            remove_action_loss = points.new_zeros(())
        teacher_anchor_position_loss = points.new_zeros(())
        if teacher_internal_knots is not None or teacher_internal_knot_mask is not None:
            if teacher_internal_knots is None or teacher_internal_knot_mask is None:
                raise ValueError(
                    "teacher_internal_knots and teacher_internal_knot_mask "
                    "must be supplied together"
                )
            if (
                teacher_internal_knots.shape != candidates.shape
                or teacher_internal_knot_mask.shape != candidates.shape
            ):
                raise ValueError("teacher knot tensors must have shape [B,Kc]")
            if teacher_retained_mask is None:
                raise ValueError(
                    "teacher knot position supervision requires teacher_retained_mask"
                )
            teacher_position_terms: list[torch.Tensor] = []
            teacher_knot_mask = teacher_internal_knot_mask.to(torch.bool)
            retained_teacher_mask = teacher_retained_mask.to(torch.bool)
            for batch_index in range(points.shape[0]):
                slot_targets = teacher_internal_knots[
                    batch_index, teacher_knot_mask[batch_index]
                ].to(points.dtype)
                selected_positions = refined[
                    batch_index, retained_teacher_mask[batch_index]
                ]
                if slot_targets.numel() != selected_positions.numel():
                    raise ValueError(
                        "teacher packed knots disagree with teacher_retained_mask"
                    )
                if slot_targets.numel():
                    teacher_position_terms.append(
                        F.smooth_l1_loss(
                            selected_positions,
                            slot_targets,
                            beta=self.knot_position_beta,
                        )
                    )
            teacher_anchor_position_loss = (
                torch.stack(teacher_position_terms).mean()
                if teacher_position_terms
                else points.new_zeros(())
            )
        canonical_position_loss = (
            F.smooth_l1_loss(
                proposal_refined[matched_mask],
                position_targets[matched_mask],
                beta=self.knot_position_beta,
            )
            if matched_mask.any()
            else points.new_zeros(())
        )
        joint_deployment_position_terms: list[torch.Tensor] = []
        if self.joint_position_supervision and teacher_retained_mask is not None:
            retained_teacher_mask = teacher_retained_mask.to(torch.bool)
            for batch_index in range(points.shape[0]):
                retained_slots = torch.nonzero(
                    retained_teacher_mask[batch_index], as_tuple=False
                ).squeeze(-1)
                target_slots = torch.nonzero(
                    true_mask[batch_index], as_tuple=False
                ).squeeze(-1)
                if retained_slots.numel() == 0 or target_slots.numel() == 0:
                    continue
                retained_positions, retained_order = torch.sort(
                    refined[batch_index, retained_slots]
                )
                target_positions, target_order = torch.sort(
                    true_internal_knots[batch_index, target_slots].to(points.dtype)
                )
                pairs = self._ordered_pair_indices(
                    retained_positions,
                    target_positions,
                )
                predicted_indices = retained_slots.new_tensor(
                    [
                        int(retained_slots[retained_order[left]].item())
                        for left, _ in pairs
                    ]
                )
                target_indices = target_slots.new_tensor(
                    [
                        int(target_slots[target_order[right]].item())
                        for _, right in pairs
                    ]
                )
                joint_deployment_position_terms.append(
                    F.smooth_l1_loss(
                        refined[batch_index, predicted_indices],
                        true_internal_knots[batch_index, target_indices].to(
                            points.dtype
                        ),
                        beta=self.knot_position_beta,
                    )
                )
        joint_deployment_position_loss = (
            torch.stack(joint_deployment_position_terms).mean()
            if joint_deployment_position_terms
            else points.new_zeros(())
        )
        # Proposal pretraining learns canonical localization.  Once an offline
        # Hard-RMS teacher exists, its mask is bound to that exact proposal
        # geometry: pulling the same slots toward a second canonical target
        # makes the cached deletion labels stale.  v9 therefore uses the
        # teacher anchor exclusively during distillation.
        if self.joint_position_supervision and teacher_retained_mask is not None:
            knot_position_loss = joint_deployment_position_loss
        elif teacher_internal_knots is not None:
            knot_position_loss = teacher_anchor_position_loss
        else:
            knot_position_loss = canonical_position_loss

        expected_count = keep_probability.sum(dim=-1)
        if teacher_retained_mask is not None:
            hard_keep_for_count = keep_probability >= 0.5
            hard_st_gate = output.get("final_hard_st_keep_gate")
            if hard_st_gate is None:
                hard_st_gate = (
                    hard_keep_for_count.to(points.dtype)
                    + keep_probability
                    - keep_probability.detach()
                )
            if hard_st_gate.shape != candidates.shape:
                raise ValueError("final_hard_st_keep_gate must have shape [B,Kc]")
            # Forward value is the deployed integer K; backward derivative is
            # that of sum(p).  This closes the former loophole where every
            # probability could stay below 0.5 while sum(p) matched the teacher.
            structure_count = hard_st_gate.sum(dim=-1)
        else:
            structure_count = expected_count
        if teacher_count is not None and teacher_count.shape != (candidates.shape[0],):
            raise ValueError("teacher_count must have shape [B]")
        target_count_for_structure = (
            teacher_count.to(points.dtype)
            if teacher_count is not None
            else true_count.to(points.dtype)
        )
        if teacher_threshold_satisfied is not None:
            if teacher_threshold_satisfied.shape != (candidates.shape[0],):
                raise ValueError("teacher_threshold_satisfied must have shape [B]")
            feasible_teacher = teacher_threshold_satisfied.to(torch.bool)
        else:
            feasible_teacher = torch.ones(
                candidates.shape[0], dtype=torch.bool, device=points.device
            )
        if teacher_fit_rms is not None:
            if teacher_fit_rms.shape != (candidates.shape[0],):
                raise ValueError("teacher_fit_rms must have shape [B]")
            if not torch.isfinite(teacher_fit_rms).all():
                raise ValueError("teacher_fit_rms must be finite")

        per_sample_count_loss = F.smooth_l1_loss(
            structure_count / max(candidate_count, 1),
            target_count_for_structure / max(candidate_count, 1),
            beta=0.1,
            reduction="none",
        )
        if teacher_count is not None:
            count_consistency_loss = (
                per_sample_count_loss[feasible_teacher].mean()
                if feasible_teacher.any()
                else points.new_zeros(())
            )
        else:
            count_consistency_loss = per_sample_count_loss.mean()
        teacher_count_loss = (
            count_consistency_loss
            if teacher_count is not None
            else points.new_zeros(())
        )
        if teacher_count is not None:
            requested_count_score = output.get(
                "one_shot_requested_count_score", expected_count
            )
            if requested_count_score.shape != (candidates.shape[0],):
                raise ValueError("one_shot_requested_count_score must have shape [B]")
            # score<0.5 is the stable zero-knot region; positive K uses ceil.
            # Place the continuous target safely inside the corresponding
            # deployment interval instead of forcing probabilities to underflow.
            target_policy_score = torch.where(
                teacher_count > 0,
                teacher_count.to(points.dtype) - 0.25,
                torch.full_like(teacher_count, 0.25, dtype=points.dtype),
            )
            per_sample_policy_count = F.smooth_l1_loss(
                requested_count_score / max(candidate_count, 1),
                target_policy_score / max(candidate_count, 1),
                beta=0.05,
                reduction="none",
            )
            policy_count_loss = (
                per_sample_policy_count[feasible_teacher].mean()
                if feasible_teacher.any()
                else points.new_zeros(())
            )
        else:
            policy_count_loss = points.new_zeros(())
        per_sample_complexity = structure_count / max(candidate_count, 1)
        complexity_loss = (
            per_sample_complexity[feasible_teacher].mean()
            if feasible_teacher.any()
            else points.new_zeros(())
        )

        predicted_log_cost = output.get("predicted_log_deletion_cost")
        analytic_delta = output.get("analytic_drop_objective_delta")
        if predicted_log_cost is not None and deletion_rms is not None:
            tiny = torch.finfo(points.dtype).eps
            target_log_cost = torch.log(
                (deletion_rms + tiny) / (self.fit_tolerance + tiny)
            ).clamp(-20.0, 20.0)
            deletion_cost_loss = F.smooth_l1_loss(
                predicted_log_cost,
                target_log_cost,
                beta=1.0,
            )
            deletion_log_ratio_mae = (predicted_log_cost - target_log_cost).abs().mean()
        elif predicted_log_cost is not None and analytic_delta is not None:
            target_log_cost = torch.log(
                analytic_delta.detach().clamp_min(torch.finfo(points.dtype).tiny)
            ).clamp(-30.0, 20.0)
            deletion_cost_loss = F.smooth_l1_loss(
                predicted_log_cost,
                target_log_cost,
                beta=1.0,
            )
            deletion_log_ratio_mae = points.new_zeros(())
        else:
            deletion_cost_loss = points.new_zeros(())
            deletion_log_ratio_mae = points.new_zeros(())

        fit_loss = self._fit_loss(output["reconstructed_points"], points)
        normalized_fit_loss = fit_loss / (self.fit_tolerance**2)
        per_sample_fit_mse = (
            (output["reconstructed_points"] - points).pow(2).sum(dim=-1).mean(dim=-1)
        )
        per_sample_fit_rms = per_sample_fit_mse.clamp_min(0.0).sqrt()
        threshold_violation_loss = (
            torch.relu(per_sample_fit_rms / self.fit_tolerance - 1.0).square().mean()
        )
        threshold_satisfied_rate = (
            (per_sample_fit_rms <= self.fit_tolerance).to(points.dtype).mean()
        )
        if true_params is None:
            true_parameter_loss = points.new_zeros(())
        else:
            true_parameter_loss = (output["params"] - true_params).pow(2).mean()
        parameter_prior_loss = (
            (output["params"] - chord_params).pow(2).mean()
            if chord_params is not None
            else points.new_zeros(())
        )

        total = (
            self.weights.fit * normalized_fit_loss
            + self.weights.threshold_violation * threshold_violation_loss
            + self.weights.true_parameter * true_parameter_loss
            + self.weights.candidate_coverage * candidate_coverage_loss
            + self.weights.candidate_repulsion * candidate_repulsion_loss
            + self.weights.keep * keep_loss
            + self.weights.remove_action * remove_action_loss
            + self.weights.knot_position * knot_position_loss
            + self.weights.count_consistency * count_consistency_loss
            + self.weights.deletion_cost * deletion_cost_loss
            + self.weights.teacher_risk * teacher_risk_loss
            + self.weights.teacher_ranking * teacher_ranking_loss
            + self.weights.teacher_distribution * teacher_distribution_loss
            + self.weights.teacher_critical_recall * teacher_critical_recall_loss
            + self.weights.teacher_false_positive * teacher_false_positive_loss
            + self.weights.teacher_count * teacher_count_loss
            + self.weights.policy_count * policy_count_loss
            + self.weights.canonical_selection * canonical_selection_loss
            + self.weights.complexity * complexity_loss
        )

        threshold_hard_keep = keep_probability >= 0.5
        hard_keep = output.get("final_hard_keep_mask", threshold_hard_keep)
        if hard_keep.shape != candidates.shape:
            raise ValueError("final_hard_keep_mask must have shape [B,Kc]")
        hard_keep = hard_keep.to(torch.bool)
        predicted_count = hard_keep.sum(dim=-1)
        metric_targets = keep_targets.to(torch.bool)
        true_positive = (hard_keep & metric_targets).to(points.dtype).sum(dim=-1).mean()
        predicted_positive = hard_keep.to(points.dtype).sum(dim=-1).mean()
        target_positive = metric_targets.to(points.dtype).sum(dim=-1).mean()
        target_count_metric = target_count_for_structure.to(torch.long)
        candidate_recall = recall_count / target_count.clamp_min(1.0)
        candidate_nearest_mae = nearest_error_sum / target_count.clamp_min(1.0)
        batch_size = max(int(points.shape[0]), 1)
        if remove_stop_logits is not None:
            predicted_action = remove_stop_logits.argmax(dim=-1)
            stop_action = predicted_action == candidate_count
            selected_index = predicted_action.clamp(max=candidate_count - 1)
            selected_safe = safe_delete_mask.gather(
                1, selected_index.unsqueeze(-1)
            ).squeeze(-1)
            safe_action_top1 = ((~stop_action) & selected_safe).to(points.dtype).mean()
            false_stop_rate = (
                (stop_action & safe_delete_mask.any(dim=-1)).to(points.dtype).mean()
            )
            unsafe_delete_rate = (
                ((~stop_action) & (~selected_safe)).to(points.dtype).mean()
            )
        else:
            safe_action_top1 = points.new_zeros(())
            false_stop_rate = points.new_zeros(())
            unsafe_delete_rate = points.new_zeros(())

        zero = points.new_zeros(())
        result = {
            "loss": total,
            "fit_loss": fit_loss,
            "normalized_fit_loss": normalized_fit_loss,
            "threshold_violation_loss": threshold_violation_loss,
            # This is the differentiable truncated-power training surrogate.
            # Trainer adds the exact one-shot standard-B-spline deployment
            # metric during validation and uses that value for checkpointing.
            "threshold_satisfied_rate": threshold_satisfied_rate,
            "surrogate_threshold_satisfied_rate": threshold_satisfied_rate,
            "fit_rms": per_sample_fit_rms.mean(),
            "candidate_coverage_loss": candidate_coverage_loss,
            "candidate_repulsion_loss": candidate_repulsion_loss,
            "keep_loss": keep_loss,
            "keep_bce_loss": keep_bce_loss,
            "keep_dice_loss": keep_dice_loss,
            "remove_action_loss": remove_action_loss,
            "deletion_cost_loss": deletion_cost_loss,
            "teacher_risk_loss": teacher_risk_loss,
            "teacher_ranking_loss": teacher_ranking_loss,
            "teacher_distribution_loss": teacher_distribution_loss,
            "teacher_critical_recall_loss": teacher_critical_recall_loss,
            "teacher_false_positive_loss": teacher_false_positive_loss,
            "teacher_count_loss": teacher_count_loss,
            "policy_count_loss": policy_count_loss,
            "canonical_selection_loss": canonical_selection_loss,
            "complexity_loss": complexity_loss,
            "deletion_log_ratio_mae": deletion_log_ratio_mae,
            "safe_action_top1": safe_action_top1,
            "false_stop_rate": false_stop_rate,
            "unsafe_delete_rate": unsafe_delete_rate,
            "safe_delete_fraction": safe_delete_mask.to(points.dtype).mean(),
            "true_parameter_loss": true_parameter_loss,
            "parameter_prior_loss": parameter_prior_loss,
            "knot_position_loss": knot_position_loss,
            "teacher_anchor_position_loss": teacher_anchor_position_loss,
            "joint_deployment_position_loss": joint_deployment_position_loss,
            "canonical_position_loss": canonical_position_loss,
            "count_loss": count_consistency_loss,
            "count_consistency_loss": count_consistency_loss,
            "candidate_recall": candidate_recall,
            "candidate_nearest_mae": candidate_nearest_mae,
            # Trainer accumulates per-sample means and reconstructs the global
            # ratio after the epoch, so expose count sums normalized by B.
            "candidate_match_count": recall_count / batch_size,
            "candidate_initial_match_count": initial_recall_count / batch_size,
            "candidate_target_count": target_count / batch_size,
            "candidate_nearest_error_sum": nearest_error_sum / batch_size,
            "existence_loss": keep_loss,
            "existence_true_positive_count": true_positive,
            "existence_predicted_count": predicted_positive,
            "existence_target_count": target_positive,
            "count_accuracy": (predicted_count == target_count_metric)
            .to(points.dtype)
            .mean(),
            "count_absolute_error": (predicted_count - target_count_metric)
            .abs()
            .to(points.dtype)
            .mean(),
            "teacher_mask_accuracy": (
                (hard_keep == metric_targets).to(points.dtype).mean()
                if teacher_retained_mask is not None
                else zero
            ),
            "threshold_mask_accuracy": (
                (threshold_hard_keep == metric_targets).to(points.dtype).mean()
                if teacher_retained_mask is not None
                else zero
            ),
            "threshold_active_count": threshold_hard_keep.to(points.dtype)
            .sum(dim=-1)
            .mean(),
            "teacher_feasible_rate": feasible_teacher.to(points.dtype).mean(),
            "adaptive_keep_threshold_mean": output.get(
                "adaptive_keep_threshold", points.new_tensor(0.0)
            )
            .to(points.dtype)
            .mean(),
            "expected_active_count": expected_count.mean(),
            "structured_active_count": structure_count.mean(),
            "l0_loss": zero,
            "activity_loss": zero,
            "binary_loss": zero,
            "gap_loss": zero,
            "over_count_loss": zero,
        }
        for tolerance, count in multiscale_recall_counts.items():
            suffix = f"{tolerance:.6f}".rstrip("0").rstrip(".").replace(".", "p")
            result[f"candidate_match_count_at_{suffix}"] = count / batch_size
        return result
