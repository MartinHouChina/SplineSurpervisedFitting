"""Online subset learning from the deployed, endpoint-constrained B-spline fit.

Only independent Bernoulli draws enter the score-function estimator. Discrete
counterfactuals supply online distillation targets, never fake policy samples.
All extra fits are training work; deployment still selects and fits one subset.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .deployment_bspline_loss import differentiable_hard_gated_bspline_fit
from ..spline.bspline_deletion_teacher import single_knot_deletion_mse_batch


def subset_cost(
    mse: torch.Tensor, counts: torch.Tensor, tolerance: torch.Tensor, capacity: int,
) -> torch.Tensor:
    """Feasible costs are below one; every infeasible cost is at least two.

Among feasible subsets, removing one knot always outweighs the entire MSE
tie-break range. Infeasible subsets receive no reward for having fewer knots.
"""
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if not torch.isfinite(mse).all() or torch.any(mse < 0):
        raise ValueError("subset MSE must be finite and non-negative")
    if not torch.isfinite(tolerance).all() or torch.any(tolerance <= 0):
        raise ValueError("tolerance must be finite and positive")
    if not torch.isfinite(counts).all() or torch.any((counts < 0) | (counts > capacity)):
        raise ValueError("subset counts must lie within the candidate capacity")
    ratio_log = torch.log(mse + tolerance * 1e-12) - torch.log(tolerance)
    feasible_cost = (counts + 0.25 * (mse / tolerance).clamp(max=1.0)) / (capacity + 1)
    return torch.where(mse <= tolerance, feasible_cost, 2.0 + ratio_log.clamp_min(0.0))


def select_best_subset(
    masks: torch.Tensor, mse: torch.Tensor, tolerance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose feasible minimum K, then MSE; if none is feasible, minimum MSE."""
    if masks.ndim != 3 or masks.dtype != torch.bool or mse.shape != masks.shape[:2]:
        raise ValueError("masks/MSE must have shapes [J,B,K]/[J,B], with boolean masks")
    if masks.shape[0] < 1 or tolerance.shape != mse.shape[1:]:
        raise ValueError("at least one subset and a tolerance per curve are required")
    if not torch.isfinite(mse).all() or torch.any(mse < 0):
        raise ValueError("subset MSE must be finite and non-negative")
    if not torch.isfinite(tolerance).all() or torch.any(tolerance <= 0):
        raise ValueError("tolerance must be finite and positive")
    counts = masks.sum(-1)
    feasible = mse <= tolerance.unsqueeze(0)
    best = torch.zeros(mse.shape[1], dtype=torch.long, device=mse.device)
    columns = torch.arange(mse.shape[1], device=mse.device)
    for index in range(1, masks.shape[0]):
        old_feasible = feasible[best, columns]
        both_feasible = feasible[index] & old_feasible
        better_feasible = (counts[index] < counts[best, columns]) | (
            (counts[index] == counts[best, columns]) & (mse[index] < mse[best, columns])
        )
        better = (feasible[index] & ~old_feasible) | (both_feasible & better_feasible)
        better |= (~feasible[index] & ~old_feasible) & (mse[index] < mse[best, columns])
        best = torch.where(better, index, best)
    return masks[best, columns], best


def bernoulli_subset_policy_loss(
    logits: torch.Tensor, sampled_masks: torch.Tensor, sampled_costs: torch.Tensor,
) -> torch.Tensor:
    """Score-function loss with an independent leave-one-out cost baseline.

    ``sampled_masks`` must be IID Bernoulli draws from ``sigmoid(logits)``.
    Deterministic masks and counterfactual edits must never enter this estimator.
    Costs are detached: geometry is trained separately by the selected refits.
    The scalar loss may be negative; its gradient estimates expected-cost descent.
    """
    if logits.ndim != 2 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("policy logits must be finite floating-point [B,K]")
    if sampled_masks.ndim != 3 or sampled_masks.dtype != torch.bool:
        raise ValueError("policy samples must be boolean [S,B,K]")
    if sampled_masks.shape[0] < 2 or sampled_masks.shape[1:] != logits.shape:
        raise ValueError("policy requires at least two samples matching logits")
    if sampled_costs.shape != sampled_masks.shape[:2] or not torch.isfinite(sampled_costs).all():
        raise ValueError("sampled costs must be finite with shape [S,B]")
    if sampled_masks.device != logits.device or sampled_costs.device != logits.device:
        raise ValueError("policy tensors must share a device")
    costs = sampled_costs.detach()
    baseline = (costs.sum(0, keepdim=True) - costs) / (costs.shape[0] - 1)
    log_probability = -F.binary_cross_entropy_with_logits(
        logits.expand_as(sampled_masks), sampled_masks.to(logits.dtype), reduction="none",
    ).sum(-1)
    return ((costs - baseline) * log_probability).mean()


class V16SubsetLoss(nn.Module):
    """Joint parameter, survivor-position and subset learning without cached labels."""

    def __init__(
        self, mse_tolerance: float = 2.5e-5, policy_samples: int = 4,
        counterfactual_edits: int = 4, fit_weight: float = 1.0,
        policy_weight: float = 0.25, distillation_weight: float = 2.0,
        count_weight: float = 2.0, ranking_weight: float = 1.0,
        supervised_count_weight: float = 1.0,
        supervised_over_count_weight: float = 0.0,
        true_parameter_weight: float = 0.1,
        proposal_knot_coverage_weight: float = 1.0,
        selected_knot_position_weight: float = 1.0,
        false_remove_weight: float = 5.0, ranking_margin: float = 1.0,
        dense_weight: float = 0.25, entropy_weight: float = 0.0,
        complexity_weight: float = 0.05, complexity_activation_ratio: float = 0.8,
        tail_weight: float = 0.5, tail_fraction: float = 0.2,
        solver_jitter: float = 1e-10, ranked_prefix_teacher: bool = True,
        teacher_prefix_search_steps: int = 7, knot_position_beta: float = 0.01,
        teacher_refinement_steps: int = 0, teacher_refinement_candidates: int = 2,
        boundary_ranking_weight: float = 0.0, boundary_ranking_candidates: int = 4,
        parameter_counterfactual_weight: float = 0.0,
        teacher_geometry_candidates: int = 0, local_fit_weight: float = 0.0,
        proposal_ordered_weight: float = 0.0,
        feasible_objective: bool = False, feasible_fit_margin: float = 0.8,
        feasible_fit_weight: float = 0.02, teacher_greedy_steps: int = 0,
        teacher_greedy_max_curves: int = 2,
        teacher_geometry_distillation_weight: float = 0.0,
        count_reserve_alignment: bool = False,
        teacher_greedy_trajectory_checks: int = 0,
        teacher_geometry_trajectory_targets: int = 0,
        teacher_greedy_priority: str = "sequential",
        teacher_compact_mask_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(mse_tolerance) or mse_tolerance <= 0:
            raise ValueError("mse_tolerance must be finite and positive")
        for name, value, minimum in (
            ("policy_samples", policy_samples, 2),
            ("counterfactual_edits", counterfactual_edits, 0),
            ("teacher_refinement_steps", teacher_refinement_steps, 0),
            ("teacher_refinement_candidates", teacher_refinement_candidates, 1),
            ("boundary_ranking_candidates", boundary_ranking_candidates, 1),
            ("teacher_geometry_candidates", teacher_geometry_candidates, 0),
            ("teacher_greedy_steps", teacher_greedy_steps, 0),
            ("teacher_greedy_max_curves", teacher_greedy_max_curves, 1),
            ("teacher_greedy_trajectory_checks", teacher_greedy_trajectory_checks, 0),
            ("teacher_geometry_trajectory_targets", teacher_geometry_trajectory_targets, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("fit_weight", fit_weight), ("policy_weight", policy_weight),
            ("distillation_weight", distillation_weight), ("dense_weight", dense_weight),
            ("count_weight", count_weight), ("ranking_weight", ranking_weight),
            ("boundary_ranking_weight", boundary_ranking_weight),
            ("parameter_counterfactual_weight", parameter_counterfactual_weight),
            ("local_fit_weight", local_fit_weight),
            ("proposal_ordered_weight", proposal_ordered_weight),
            ("teacher_geometry_distillation_weight", teacher_geometry_distillation_weight),
            ("teacher_compact_mask_weight", teacher_compact_mask_weight),
            ("supervised_count_weight", supervised_count_weight),
            ("supervised_over_count_weight", supervised_over_count_weight),
            ("true_parameter_weight", true_parameter_weight),
            ("proposal_knot_coverage_weight", proposal_knot_coverage_weight),
            ("selected_knot_position_weight", selected_knot_position_weight),
            ("false_remove_weight", false_remove_weight),
            ("ranking_margin", ranking_margin),
            ("entropy_weight", entropy_weight), ("complexity_weight", complexity_weight),
            ("complexity_activation_ratio", complexity_activation_ratio),
            ("tail_weight", tail_weight), ("tail_fraction", tail_fraction),
            ("solver_jitter", solver_jitter),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, float(value))
        if false_remove_weight < 1.0:
            raise ValueError("false_remove_weight must be at least one")
        if not 0 < complexity_activation_ratio <= 1:
            raise ValueError("complexity_activation_ratio must lie in (0,1]")
        if not 0 < tail_fraction <= 1:
            raise ValueError("tail_fraction must lie in (0,1]")
        if not math.isfinite(knot_position_beta) or knot_position_beta <= 0:
            raise ValueError("knot_position_beta must be finite and positive")
        self.mse_tolerance = float(mse_tolerance)
        self.policy_samples = policy_samples
        self.counterfactual_edits = counterfactual_edits
        if not isinstance(ranked_prefix_teacher, bool):
            raise ValueError("ranked_prefix_teacher must be boolean")
        if (
            isinstance(teacher_prefix_search_steps, bool)
            or not isinstance(teacher_prefix_search_steps, int)
            or teacher_prefix_search_steps < 1
        ):
            raise ValueError("teacher_prefix_search_steps must be a positive integer")
        self.ranked_prefix_teacher = ranked_prefix_teacher
        self.teacher_prefix_search_steps = teacher_prefix_search_steps
        self.knot_position_beta = float(knot_position_beta)
        self.teacher_refinement_steps = teacher_refinement_steps
        self.teacher_refinement_candidates = teacher_refinement_candidates
        self.boundary_ranking_candidates = boundary_ranking_candidates
        self.teacher_geometry_candidates = teacher_geometry_candidates
        for name, value in (("feasible_objective", feasible_objective),
                            ("count_reserve_alignment", count_reserve_alignment)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, value)
        if not math.isfinite(feasible_fit_margin) or not 0 < feasible_fit_margin <= 1:
            raise ValueError("feasible_fit_margin must lie in (0,1]")
        if not math.isfinite(feasible_fit_weight) or not 0 <= feasible_fit_weight <= 1:
            raise ValueError("feasible_fit_weight must lie in [0,1]")
        self.feasible_fit_margin = float(feasible_fit_margin)
        self.feasible_fit_weight = float(feasible_fit_weight)
        self.teacher_greedy_steps = teacher_greedy_steps
        self.teacher_greedy_max_curves = teacher_greedy_max_curves
        if teacher_geometry_trajectory_targets and not teacher_greedy_trajectory_checks:
            raise ValueError("trajectory geometry targets require trajectory decode checks")
        self.teacher_greedy_trajectory_checks = teacher_greedy_trajectory_checks
        self.teacher_geometry_trajectory_targets = teacher_geometry_trajectory_targets
        if teacher_greedy_priority not in ("sequential", "compact"):
            raise ValueError("teacher_greedy_priority must be 'sequential' or 'compact'")
        self.teacher_greedy_priority = teacher_greedy_priority

    @staticmethod
    def _validate_points(points: torch.Tensor, degree: int) -> None:
        if not isinstance(points, torch.Tensor) or points.ndim != 3:
            raise ValueError("points must have shape [B,M,D]")
        if points.shape[0] < 1 or points.shape[1] < degree + 1 or points.shape[2] < 1:
            raise ValueError("points require a nonempty batch and enough samples for the degree")
        if not points.is_floating_point() or not torch.isfinite(points).all():
            raise ValueError("points must be finite floating-point values")

    @staticmethod
    def _validate_geometry(parameters, knots, mask, points) -> None:
        if not all(isinstance(value, torch.Tensor) for value in (parameters, knots, mask)):
            raise ValueError("parameters, knots and mask must be tensors")
        if parameters.shape != points.shape[:2] or knots.ndim != 2:
            raise ValueError("parameters/knots must have shapes [B,M]/[B,K]")
        if knots.shape[0] != points.shape[0] or mask.shape != knots.shape or mask.dtype != torch.bool:
            raise ValueError("knots and boolean masks must share [B,K]")
        if any(value.device != points.device for value in (parameters, knots, mask)):
            raise ValueError("all geometry tensors must share the points device")
        if any(not value.is_floating_point() or not torch.isfinite(value).all()
               for value in (parameters, knots)):
            raise ValueError("parameters and knots must be finite floating-point values")
        if torch.any(parameters[:, 1:] <= parameters[:, :-1]):
            raise ValueError("parameters must be strictly increasing")
        if torch.any(parameters < 0) or torch.any(parameters > 1):
            raise ValueError("parameters must lie in [0,1]")
        if not torch.allclose(parameters[:, 0], torch.zeros_like(parameters[:, 0]), atol=1e-7):
            raise ValueError("parameters must begin at zero")
        if not torch.allclose(parameters[:, -1], torch.ones_like(parameters[:, -1]), atol=1e-7):
            raise ValueError("parameters must end at one")
        if torch.any((knots[mask] <= 0) | (knots[mask] >= 1)):
            raise ValueError("selected knots must lie strictly inside (0,1)")

    def _fit_details(self, parameters, knots, mask, points, degree):
        self._validate_geometry(parameters, knots, mask, points)
        result = differentiable_hard_gated_bspline_fit(
            parameters.double(), points.double(), knots.double(), mask,
            degree=degree, smoothness_weight=0.0, control_ridge=0.0,
            solver_jitter=self.solver_jitter,
        )
        if not torch.isfinite(result["per_sample_mse"]).all():
            raise RuntimeError("non-finite standard B-spline subset refit")
        return result

    def _fit(self, parameters, knots, mask, points, degree):
        return self._fit_details(parameters, knots, mask, points, degree)["per_sample_mse"]

    @staticmethod
    def _decode_output(model, context, mask):
        output = model.decode_subset(context, mask)
        if not isinstance(output, Mapping):
            raise ValueError("decode_subset must return a mapping")
        returned_mask = output.get("learned_keep_mask")
        if not isinstance(returned_mask, torch.Tensor) or not torch.equal(returned_mask, mask):
            raise ValueError("decode_subset must preserve the requested discrete mask")
        return output

    def _decode_mse(self, model, context, mask, points, degree):
        output = self._decode_output(model, context, mask)
        return self._fit(output["params"], output["internal_knots"], mask, points, degree)

    def _parameter_counterfactual(self, parameters, knots, mask, chord, mse,
                                  points, degree, tolerance):
        """Same selected knots/K in chord coordinates, as a detached target.

        Relocated survivor knots are warped together with the parameters.  A
        reference constructed from unchanged knot coordinates would compare
        different geometric locations and supply a misleading trust target.
        No decoder or deployment-time alternative search is involved.
        """
        if not isinstance(chord, torch.Tensor) or chord.shape != parameters.shape:
            raise ValueError("parameter counterfactual requires chord_params [B,M]")
        with torch.no_grad():
            reference_knots = self._warp_knots_to_target_parameterization(
                knots.detach(), parameters.detach(), chord.detach(),
            )
            reference_mse = self._fit(
                chord.detach(), reference_knots, mask, points, degree,
            ).detach()
        floor = tolerance * 1e-12
        penalty = F.relu(torch.log(mse + floor) - torch.log(reference_mse + floor))
        return self._tail_aware_mean(penalty), reference_mse

    def _local_fit_penalty(self, squared_error, parameters, tolerance):
        """Normalized interval/end-neighborhood loss; global MSE stays intact.

        Four equal parameter intervals and two endpoint neighborhoods each
        contribute one mean, regardless of sample count. Empty intervals are
        excluded rather than treated as zero-error evidence. The endpoints
        themselves interpolate exactly; their nearby samples carry the signal.
        """
        if squared_error.shape != parameters.shape or squared_error.ndim != 2:
            raise ValueError("local residuals and parameters must share [B,M]")
        if not torch.isfinite(squared_error).all() or bool((squared_error < 0).any()):
            raise ValueError("local squared residuals must be finite and non-negative")
        bins = (parameters.detach() * 4).long().clamp(0, 3)
        regions = [bins == index for index in range(4)]
        sample_ids = torch.arange(parameters.shape[1], device=parameters.device)
        endpoint_count = max(2, math.ceil(parameters.shape[1] * 0.1))
        regions.extend([
            (sample_ids < endpoint_count).expand_as(parameters),
            (sample_ids >= parameters.shape[1] - endpoint_count).expand_as(parameters),
        ])
        masks = torch.stack(regions, dim=1)
        counts = masks.sum(-1)
        region_mse = (squared_error[:, None, :] * masks).sum(-1) / counts.clamp_min(1)
        penalties = self._objective_fit_penalty(region_mse, tolerance[:, None])
        per_curve = (penalties * (counts > 0)).sum(-1) / (counts > 0).sum(-1)
        return self._tail_aware_mean(per_curve), region_mse.amax(-1).mean()

    @staticmethod
    def _trust_statistics(value, *, prefix, zero):
        """Scalar gate diagnostics; absent legacy/stage-specific fields are zero."""
        if value is None:
            return {f"{prefix}_{name}": zero for name in ("mean", "min", "max")}
        if (not isinstance(value, torch.Tensor) or not value.numel()
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"{prefix} must be a nonempty finite floating-point tensor")
        return {
            f"{prefix}_mean": value.mean(),
            f"{prefix}_min": value.amin(),
            f"{prefix}_max": value.amax(),
        }

    @staticmethod
    def _fit_penalty(mse, tolerance):
        # Log space avoids the huge MSE/tolerance penalties of earlier versions.
        log_mse = torch.log(mse + tolerance * 1e-12)
        log_tolerance = torch.log(tolerance)
        return torch.logaddexp(log_mse, log_tolerance) - log_tolerance + F.relu(log_mse - log_tolerance)

    def _objective_fit_penalty(self, mse, tolerance):
        original = self._fit_penalty(mse, tolerance)
        if not self.feasible_objective:
            return original
        # Continuous at the safety margin. Above it the historical fit
        # gradient is unchanged; safely feasible rows receive only a weak
        # accuracy tie-break instead of paying many knots for unnecessary MSE.
        boundary = math.log1p(self.feasible_fit_margin + 1e-12)
        return torch.where(
            mse <= tolerance * self.feasible_fit_margin,
            self.feasible_fit_weight * original,
            original - (1 - self.feasible_fit_weight) * boundary,
        )

    def _tail_aware_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Mean plus a CVaR-style upper-tail term for hard-curve recall."""
        if values.ndim != 1 or values.numel() < 1:
            raise ValueError("tail-aware values must be a nonempty vector")
        tail_count = max(1, math.ceil(values.numel() * self.tail_fraction))
        return values.mean() + self.tail_weight * values.topk(tail_count).values.mean()

    @staticmethod
    def _topk_mask(probabilities: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=order.device).expand_as(order),
        )
        return rank < counts.unsqueeze(-1)

    def _respect_minimum_count(
        self, mask: torch.Tensor, probabilities: torch.Tensor, minimum: int,
    ) -> torch.Tensor:
        if minimum <= 0:
            return mask
        counts = mask.sum(-1)
        if bool((counts >= minimum).all()):
            return mask
        add_order = torch.argsort(
            probabilities.masked_fill(mask, float("-inf")),
            dim=-1,
            descending=True,
            stable=True,
        )
        add_rank = torch.empty_like(add_order)
        add_rank.scatter_(
            1,
            add_order,
            torch.arange(add_order.shape[1], device=mask.device).expand_as(add_order),
        )
        needed = (minimum - counts).clamp_min(0)
        return mask | (add_rank < needed.unsqueeze(-1))

    def _counterfactual_masks(
        self, mask, probabilities, minimum=0, *, ranked_mask=None,
        constrain=None,
    ):
        """Safe add/remove edits and score-ranked budgets; never policy draws."""
        batch, capacity = mask.shape
        columns = torch.arange(batch, device=mask.device)
        remove_order = probabilities.masked_fill(~mask, float("inf")).argsort(-1)
        add_order = probabilities.masked_fill(mask, -float("inf")).argsort(-1, descending=True)
        for edit in range(self.counterfactual_edits):
            rank, kind = divmod(edit, 3)
            rank %= capacity
            remove = remove_order[:, rank]
            add = add_order[:, rank]
            trial = mask.clone()
            if kind in (0, 2):
                valid = mask[columns, remove] & (mask.sum(-1) > minimum)
                trial[columns[valid], remove[valid]] = False
            if kind in (1, 2):
                valid = ~mask[columns, add]
                trial[columns[valid], add[valid]] = True
            trial = self._respect_minimum_count(trial, probabilities, minimum)
            yield constrain(trial) if constrain is not None else trial
        yield torch.ones_like(mask)
        # Unlike the removed equal-index half-capacity shortcut, every global
        # budget respects the selector ranking and therefore teaches a subset
        # that the deployed Top-K policy can actually reproduce.
        if self.counterfactual_edits:
            lower_budget = max(minimum, 1)
            for step in range(1, self.counterfactual_edits + 1):
                # Log-spaced budgets cover both high-recall and compact sets.
                # The removed Kc/2 floor meant Kc=96 could never teach K<48.
                ratio = (lower_budget / capacity) ** (
                    step / self.counterfactual_edits
                )
                count = max(minimum, math.ceil(capacity * ratio))
                budgets = mask.new_full((batch,), count, dtype=torch.long)
                yield (
                    ranked_mask(budgets)
                    if ranked_mask is not None
                    else self._topk_mask(probabilities, budgets)
                )

    def _minimum_feasible_ranked_prefix(
        self, model, context, probabilities, points, degree, tolerance, minimum,
        *, guide_counts=None, guide_valid=None,
    ):
        """Find a fine-grained feasible deployed Top-K prefix in training only.

        Ranked masks are nested, but the subset decoder also relocates knots and
        updates parameters, so decoded MSE need not be perfectly monotone in K.
        We therefore combine a fast batched boundary search with an exact local
        sweep around its boundary, and choose by feasibility/count over *all*
        evaluated prefixes.  Full-prefix-infeasible rows receive broad count
        probes; if no evaluated prefix passes, the full prefix is the safe
        distillation fallback.
        """
        batch, capacity = probabilities.shape
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise ValueError("minimum selected count must be an integer")
        if not 0 <= minimum <= capacity:
            raise ValueError("minimum selected count must lie within capacity")

        selector = getattr(model, "select_mask_at_count", None)

        def ranked_mask(counts):
            trial = (
                selector(context, counts)
                if callable(selector)
                else self._topk_mask(probabilities, counts)
            )
            if (
                not isinstance(trial, torch.Tensor)
                or trial.shape != probabilities.shape
                or trial.dtype != torch.bool
                or trial.device != probabilities.device
            ):
                raise ValueError(
                    "ranked prefix selector must return boolean candidate masks"
                )
            if not torch.equal(trial.sum(-1), counts):
                raise ValueError("ranked prefix selector must preserve requested counts")
            return trial

        # First inspect a low-count-dense quadratic grid for *every* curve.
        # This does not assume that the relocated decoder's MSE is monotone in
        # K, and it includes both deployment extrema.  Seven intervals at
        # Kc=96 evaluate approximately 4, 6, 12, 21, 34, 51, 72 and 96.
        grid_denominator = self.teacher_prefix_search_steps
        coarse_counts = sorted({
            minimum + round(
                (capacity - minimum) * (step / grid_denominator) ** 2
            )
            for step in range(grid_denominator + 1)
        } | {minimum, capacity})
        evaluated_masks = []
        evaluated_mse = []
        for count_value in coarse_counts:
            counts = torch.full(
                (batch,), count_value, dtype=torch.long,
                device=probabilities.device,
            )
            trial_mask = ranked_mask(counts)
            evaluated_masks.append(trial_mask)
            evaluated_mse.append(
                self._decode_mse(model, context, trial_mask, points, degree)
            )

        full_index = coarse_counts.index(capacity)
        full_mask = evaluated_masks[full_index]
        full_mse = evaluated_mse[full_index]

        if guide_counts is not None:
            if (
                not isinstance(guide_counts, torch.Tensor)
                or guide_counts.shape != (batch,)
            ):
                raise ValueError("guide counts must have shape [B]")
            if (
                guide_valid is None
                or not isinstance(guide_valid, torch.Tensor)
                or guide_valid.shape != (batch,)
                or guide_valid.dtype != torch.bool
            ):
                raise ValueError("guide validity must be boolean [B]")
            guided = guide_counts.to(
                device=probabilities.device, dtype=torch.long,
            ).clamp(min=minimum, max=capacity)
            guided = torch.where(
                guide_valid.to(probabilities.device), guided,
                torch.full_like(guided, capacity),
            )
            guided_mask = ranked_mask(guided)
            evaluated_masks.append(guided_mask)
            evaluated_mse.append(
                self._decode_mse(
                    model, context, guided_mask, points, degree,
                )
            )

        coarse_mse = torch.stack(evaluated_mse[:len(coarse_counts)])
        coarse_feasible = coarse_mse <= tolerance.unsqueeze(0)
        grid_values = torch.tensor(
            coarse_counts, dtype=torch.long, device=probabilities.device,
        )
        grid_indices = torch.arange(
            len(coarse_counts), device=probabilities.device,
        ).unsqueeze(-1).expand_as(coarse_feasible)
        sentinel = torch.full_like(grid_indices, len(coarse_counts))
        first_feasible = torch.where(
            coarse_feasible, grid_indices, sentinel,
        ).amin(dim=0)
        coarse_found = first_feasible < len(coarse_counts)
        safe_index = first_feasible.clamp(max=len(coarse_counts) - 1)
        high = grid_values[safe_index]
        previous_index = (safe_index - 1).clamp_min(0)
        low = grid_values[previous_index]
        low = torch.where(safe_index == 0, high, low)
        full_counts = torch.full_like(high, capacity)
        low = torch.where(coarse_found, low, full_counts)
        high = torch.where(coarse_found, high, full_counts)

        # Refine only inside the first coarse infeasible->feasible interval.
        # The subsequent exact neighborhood handles modest local non-monotonicity.
        for _ in range(self.teacher_prefix_search_steps):
            active = low < high
            if not bool(active.any()):
                break
            midpoint = torch.div(low + high, 2, rounding_mode="floor")
            trial_counts = torch.where(active, midpoint, high)
            trial_mask = ranked_mask(trial_counts)
            trial_mse = self._decode_mse(
                model, context, trial_mask, points, degree,
            )
            evaluated_masks.append(trial_mask)
            evaluated_mse.append(trial_mse)
            feasible = trial_mse <= tolerance
            high = torch.where(active & feasible, midpoint, high)
            low = torch.where(active & ~feasible, midpoint + 1, low)

        preliminary_masks = torch.stack(evaluated_masks)
        preliminary_mse = torch.stack(evaluated_mse)
        preliminary, _ = select_best_subset(
            preliminary_masks, preliminary_mse, tolerance,
        )
        columns = torch.arange(batch, device=probabilities.device)
        preliminary_any_feasible = (
            preliminary_mse <= tolerance.unsqueeze(0)
        ).any(0)
        base_counts = torch.where(
            preliminary_any_feasible,
            preliminary.sum(-1),
            full_counts,
        )

        # Decode a small exact neighborhood around the best evaluated prefix.
        for offset in range(-2, 3):
            local_counts = (base_counts + offset).clamp(
                min=minimum, max=capacity,
            )
            local_mask = ranked_mask(local_counts)
            evaluated_masks.append(local_mask)
            evaluated_mse.append(
                self._decode_mse(
                    model, context, local_mask, points, degree,
                )
            )

        masks = torch.stack(evaluated_masks)
        mse = torch.stack(evaluated_mse)
        selected, indices = select_best_subset(masks, mse, tolerance)
        columns = torch.arange(batch, device=probabilities.device)
        selected_mse = mse[indices, columns]
        any_feasible = (mse <= tolerance.unsqueeze(0)).any(0)
        best_mask = torch.where(any_feasible.unsqueeze(-1), selected, full_mask)
        best_mse = torch.where(any_feasible, selected_mse, full_mse)
        return best_mask, best_mse, any_feasible, len(evaluated_masks)

    def _teacher_neighbor_masks(self, model, context, mask, probabilities, minimum):
        """Bounded local edits of one frozen teacher, using actual constrained masks.

        Each candidate rank supplies a removal, an addition and a genuine
        equal-count swap. Unlike a removal, a swap is legal at minimum K.
        This separate opt-in path leaves the historical counterfactual pool intact.
        """
        batch, capacity = mask.shape
        columns = torch.arange(batch, device=mask.device)
        counts = mask.sum(-1)
        remove_order = probabilities.masked_fill(~mask, float("inf")).argsort(
            dim=-1, stable=True,
        )
        add_order = probabilities.masked_fill(mask, -float("inf")).argsort(
            dim=-1, descending=True, stable=True,
        )
        constrain = getattr(model, "constrain_selection_mask", None)
        for rank in range(min(self.teacher_refinement_candidates, capacity)):
            remove = remove_order[:, rank]
            add = add_order[:, rank]
            can_remove = mask[columns, remove]
            can_add = ~mask[columns, add]
            for kind in ("remove", "add", "swap"):
                trial = mask.clone()
                if kind == "remove":
                    valid = can_remove & (counts > minimum)
                    trial[columns[valid], remove[valid]] = False
                elif kind == "add":
                    trial[columns[can_add], add[can_add]] = True
                else:
                    valid = can_remove & can_add
                    trial[columns[valid], remove[valid]] = False
                    trial[columns[valid], add[valid]] = True
                if callable(constrain):
                    trial = constrain(context, trial)
                if (
                    not isinstance(trial, torch.Tensor)
                    or trial.shape != mask.shape
                    or trial.dtype != torch.bool
                    or trial.device != mask.device
                    or bool((trial.sum(-1) < minimum).any())
                ):
                    raise ValueError(
                        "teacher refinement constraints must return boolean [B,K] "
                        "masks respecting the deployment minimum"
                    )
                yield trial

    @torch.no_grad()
    def _refine_teacher(
        self, model, context, best_mask, best_mse, probabilities,
        points, degree, tolerance, minimum,
    ):
        """Refine a teacher with real subset decodes/refits, never policy draws.

        A round compares all neighbors of its frozen incumbent, then accepts
        only a feasible lexicographic (K, MSE) improvement. Infeasible fallback
        rows retain their old teacher until a feasible alternative is found.
        """
        initial_mask = best_mask.clone()
        initial_mse = best_mse.detach().clone()
        best_mask = best_mask.detach()
        best_mse = best_mse.detach()
        evaluations = steps_used = 0
        columns = torch.arange(best_mask.shape[0], device=best_mask.device)
        for _ in range(self.teacher_refinement_steps):
            compared_masks = [best_mask]
            compared_mse = [best_mse]
            for trial in self._teacher_neighbor_masks(
                model, context, best_mask, probabilities.detach(), minimum,
            ):
                if any(torch.equal(trial, previous) for previous in compared_masks):
                    continue
                compared_masks.append(trial)
                compared_mse.append(
                    self._decode_mse(model, context, trial, points, degree)
                )
                evaluations += 1
            if len(compared_masks) == 1:
                break
            steps_used += 1
            masks = torch.stack(compared_masks)
            mse = torch.stack(compared_mse)
            selected, indices = select_best_subset(masks, mse, tolerance)
            selected_mse = mse[indices, columns]
            accept = (indices != 0) & (selected_mse <= tolerance)
            if not bool(accept.any()):
                break
            best_mask = torch.where(accept.unsqueeze(-1), selected, best_mask)
            best_mse = torch.where(accept, selected_mse, best_mse)
        diagnostics = {
            "teacher_refinement_evaluations": initial_mse.new_tensor(float(evaluations)),
            "teacher_refinement_steps_used": initial_mse.new_tensor(float(steps_used)),
            "teacher_refinement_improved_fraction": (
                (best_mask != initial_mask).any(-1).double().mean()
            ),
            "teacher_refinement_count_reduction": (
                initial_mask.sum(-1) - best_mask.sum(-1)
            ).double().mean(),
            "teacher_refinement_mse_gain": (initial_mse - best_mse).mean(),
            "teacher_refinement_feasible_fraction": (best_mse <= tolerance).double().mean(),
        }
        return best_mask, best_mse, diagnostics

    def _boundary_ranking_loss(self, logits, teacher_mask, teacher_feasible):
        """Focus on the weakest kept and strongest rejected feasible-teacher slots.

        This is ordinary differentiable ranking supervision of a real-refit
        teacher, not a surrogate fit or a policy-gradient sample. The existing
        all-pair objective remains unchanged. Infeasible full-set fallbacks
        provide no reliable keep/reject boundary and are excluded.
        """
        candidates = min(self.boundary_ranking_candidates, logits.shape[1])
        positive_indices = logits.detach().masked_fill(
            ~teacher_mask, float("inf"),
        ).argsort(dim=-1, stable=True)[:, :candidates]
        negative_indices = logits.detach().masked_fill(
            teacher_mask, -float("inf"),
        ).argsort(dim=-1, descending=True, stable=True)[:, :candidates]
        positive_logits = logits.gather(1, positive_indices)
        negative_logits = logits.gather(1, negative_indices)
        valid_pairs = (
            teacher_mask.gather(1, positive_indices).unsqueeze(-1)
            & ~teacher_mask.gather(1, negative_indices).unsqueeze(-2)
            & teacher_feasible[:, None, None]
        )
        if not bool(valid_pairs.any()):
            return logits.sum() * 0.0
        penalties = F.softplus(
            self.ranking_margin - positive_logits.unsqueeze(-1)
            + negative_logits.unsqueeze(-2)
        )
        # Average per eligible curve so changing its teacher K does not change
        # that curve's influence merely by changing the number of valid pairs.
        pair_counts = valid_pairs.sum(dim=(-1, -2))
        per_curve = (penalties * valid_pairs).sum(dim=(-1, -2)) / pair_counts.clamp_min(1)
        return per_curve[pair_counts > 0].mean()

    def _geometry_teacher_masks(self, model, context, mask, points, minimum):
        """Bounded curvature/coverage and spatially dispersed probes, no logits.

        Alternating removals and equal-count swaps explore survivors excluded
        by student probability ranking. Geometry ranks only propose masks;
        acceptance always uses the actual decoded endpoint-constrained refit.
        """
        parameters = context["proposal_params"].detach()
        chord = context.get("chord_params", parameters).detach()
        knots = self._warp_knots_to_target_parameterization(
            context["proposal_internal_knots"].detach(), parameters, chord,
        )
        edges = points.detach().diff(dim=1)
        directions = F.normalize(edges, dim=-1)
        turns = (directions[:, 1:] - directions[:, :-1]).norm(dim=-1)
        turns = F.pad(turns, (1, 1))
        sample_ids = torch.searchsorted(chord.contiguous(), knots.contiguous()).clamp(
            0, chord.shape[1] - 1,
        )
        salience = turns.gather(1, sample_ids) + 0.05
        remove_orders = []
        add_orders = []
        selected_ids = []
        for row in range(mask.shape[0]):
            kept = mask[row].nonzero(as_tuple=False).flatten()
            rejected = (~mask[row]).nonzero(as_tuple=False).flatten()
            selected_ids.append(kept)
            locations = knots[row, kept]
            boundaries = torch.cat([locations.new_zeros(1), locations, locations.new_ones(1)])
            if kept.numel():
                spacing = torch.minimum(boundaries.diff()[:-1], boundaries.diff()[1:])
                importance = spacing * salience[row, kept]
                remove_orders.append(kept[importance.argsort(stable=True)])
            else:
                remove_orders.append(kept)
            if rejected.numel():
                distance = (knots[row, rejected, None] - boundaries[None, :]).abs().amin(-1)
                coverage = distance * salience[row, rejected]
                add_orders.append(rejected[coverage.argsort(descending=True, stable=True)])
            else:
                add_orders.append(rejected)
        constrain = getattr(model, "constrain_selection_mask", None)
        for index in range(self.teacher_geometry_candidates):
            trial = mask.clone()
            for row, kept in enumerate(selected_ids):
                if not kept.numel():
                    continue
                round_index = index // 4
                if index % 4 < 2:
                    remove = remove_orders[row][round_index % kept.numel()]
                else:
                    # Golden-ratio spatial traversal spreads probes over the
                    # whole survivor list instead of revisiting low-score slots.
                    fraction = ((round_index + 1) * 0.6180339887498949) % 1.0
                    remove = kept[min(int(fraction * kept.numel()), kept.numel() - 1)]
                if index % 2 == 0:
                    if kept.numel() > minimum:
                        trial[row, remove] = False
                elif add_orders[row].numel():
                    add = add_orders[row][(index // 2) % add_orders[row].numel()]
                    trial[row, remove] = False
                    trial[row, add] = True
            if callable(constrain):
                trial = constrain(context, trial)
            if (not isinstance(trial, torch.Tensor) or trial.shape != mask.shape
                    or trial.dtype != torch.bool or trial.device != mask.device
                    or bool((trial.sum(-1) < minimum).any())):
                raise ValueError("geometry teacher masks must respect shape and minimum K")
            yield trial

    @torch.no_grad()
    def _geometry_teacher(self, model, context, mask, mse, points, degree,
                          tolerance, minimum):
        masks, errors = [mask], [mse.detach()]
        for trial in self._geometry_teacher_masks(model, context, mask, points, minimum):
            if any(torch.equal(trial, previous) for previous in masks):
                continue
            masks.append(trial)
            errors.append(self._decode_mse(model, context, trial, points, degree))
        selected, indices = select_best_subset(torch.stack(masks), torch.stack(errors), tolerance)
        columns = torch.arange(mask.shape[0], device=mask.device)
        selected_mse = torch.stack(errors)[indices, columns]
        # Never exchange an infeasible fallback for another infeasible mask.
        # Feasible improvements are lexicographic (minimum K, then MSE), so a
        # lower-K fit may use more of the allowed error without harming feasibility.
        accept = (indices != 0) & (selected_mse <= tolerance)
        best_mask = torch.where(accept[:, None], selected, mask)
        best_mse = torch.where(accept, selected_mse, mse)
        return best_mask, best_mse, {
            "teacher_geometry_evaluations": mse.new_tensor(float(len(masks) - 1)),
            "teacher_geometry_improved_fraction": accept.double().mean(),
            "teacher_geometry_count_reduction": (mask.sum(-1) - best_mask.sum(-1)).double().mean(),
            "teacher_geometry_mse_gain": (mse - best_mse).mean(),
            "teacher_geometry_feasible_fraction": (best_mse <= tolerance).double().mean(),
        }

    @staticmethod
    def _greedy_metrics(zero):
        return {name: zero for name in (
            "teacher_greedy_curve_fraction", "teacher_greedy_fixed_count",
            "teacher_greedy_fixed_pass_rate", "teacher_greedy_decoded_pass_rate",
            "teacher_greedy_fixed_delta_k", "teacher_greedy_accepted_delta_k",
            "teacher_greedy_accepted_fraction", "teacher_greedy_candidate_refits",
            "teacher_greedy_verification_refits", "teacher_greedy_decode_refits",
            "teacher_greedy_refits", "teacher_geometry_distillation_loss",
            "teacher_geometry_fit_violation", "teacher_geometry_aux_refits",
            "teacher_greedy_trajectory_checks", "teacher_greedy_trajectory_pass_rate",
            "teacher_greedy_trajectory_accepted_delta_k", "teacher_geometry_target_count",
            "teacher_geometry_target_k_mean", "teacher_geometry_target_pass_rate",
            "teacher_compact_mask_loss", "teacher_compact_bce_loss",
            "teacher_compact_count_loss", "teacher_compact_ranking_loss",
            "teacher_compact_target_count", "teacher_compact_target_k_mean",
        )}

    @staticmethod
    def _row_context(context, row, batch):
        return {
            key: (value[row:row + 1] if isinstance(value, torch.Tensor)
                  and value.ndim and value.shape[0] == batch else value)
            for key, value in context.items()
        }

    @torch.no_grad()
    def _greedy_priority_rows(self, best_mask, best_mse, deployment_mask, tolerance,
                              minimum, *, deployment_mse=None, supervised_counts=None,
                              supervised_valid=None):
        """Allocate a bounded search budget, without changing the random stream.

        Compact mode exploits feasible curves with error slack and redundant
        knots. One quarter of the budget (at least one slot when possible) is
        reserved for the hardest remaining curve(s); priority must not starve
        difficult curves. True counts affect search allocation only, never
        certify a feasible teacher subset. Loader shuffling supplies diversity.
        """
        batch, capacity = best_mask.shape
        budget = min(batch, self.teacher_greedy_max_curves)
        if self.teacher_greedy_priority == "sequential":
            return list(range(budget))
        ratio = best_mse.detach() / tolerance.detach()
        if deployment_mse is not None:
            ratio = torch.minimum(ratio, deployment_mse.detach() / tolerance.detach())
        ratio = torch.nan_to_num(ratio, nan=float("inf"), posinf=float("inf"))
        current_k = torch.maximum(best_mask.sum(-1), deployment_mask.sum(-1))
        reference_k = best_mask.sum(-1).to(ratio.dtype)
        if supervised_counts is not None:
            valid = (torch.ones_like(reference_k, dtype=torch.bool) if supervised_valid is None
                     else supervised_valid.detach())
            reference_k = torch.where(
                valid, torch.minimum(reference_k, supervised_counts.detach().to(ratio.dtype)),
                reference_k,
            )
        removable = (current_k - minimum).clamp_min(0).to(ratio.dtype) / capacity
        excess = (current_k - reference_k).clamp_min(0) / capacity
        slack = (1 - ratio).clamp(0, 1)
        feasible = torch.isfinite(ratio) & (ratio <= 1)
        score = slack * removable + excess
        # One small host transfer per field, not scalar GPU synchronizations
        # inside Python's sorting comparisons.
        feasible_values, score_values = feasible.cpu().tolist(), score.cpu().tolist()
        ratio_values, count_values = ratio.cpu().tolist(), current_k.cpu().tolist()
        # Feasibility is lexicographically prior to estimated compactness.
        exploitation = sorted(range(batch), key=lambda row: (
            -int(feasible_values[row]), -score_values[row], row,
        ))
        explore = max(1, budget // 4) if budget >= 2 else 0
        selected = exploitation[:budget - explore]
        remaining = [row for row in range(batch) if row not in selected]
        remaining.sort(key=lambda row: (-ratio_values[row], -count_values[row], row))
        return selected + remaining[:explore]

    @torch.no_grad()
    def _fixed_geometry_greedy(self, model, context, parameters, knots, mask,
                               points, degree, tolerance, minimum, *, trajectory=None):
        """Full single-deletion scan, bounded by accepted steps, on one curve.

        The batched solver examines EVERY selected location in a round. The
        counted budget is actual least-squares systems, including candidates
        later rejected by coverage constraints, not Python call count. A second
        fit with the training solver verifies each accepted deletion.
        """
        parameters, knots, mask = parameters.detach(), knots.detach(), mask.detach().clone()
        current_mse = self._fit(parameters, knots, mask, points, degree)
        scans, checks = 0, 1
        if bool((current_mse > tolerance).any()):
            return mask, current_mse, scans, checks
        constrain = getattr(model, "constrain_selection_mask", None)
        for _ in range(self.teacher_greedy_steps):
            ids = mask[0].nonzero(as_tuple=False).flatten()
            if ids.numel() <= minimum:
                break
            trials, permitted = [], []
            for index in ids:
                trial = mask.clone()
                trial[0, index] = False
                repaired = constrain(context, trial) if callable(constrain) else trial
                valid = (isinstance(repaired, torch.Tensor)
                         and repaired.dtype == torch.bool and repaired.shape == trial.shape
                         and repaired.device == trial.device and torch.equal(repaired, trial))
                trials.append(trial)
                permitted.append(valid)
            if not any(permitted):
                break
            deleted_mse = single_knot_deletion_mse_batch(
                parameters.double(), points.double(), knots[:, ids].double(),
                degree=degree, smoothness_weight=0.0,
                # Same interior ridge rows as differentiable solver_jitter.
                control_ridge=self.solver_jitter, interpolate_endpoints=True,
            )[0]
            scans += ids.numel()
            allowed = torch.tensor(permitted, dtype=torch.bool, device=mask.device)
            allowed &= torch.isfinite(deleted_mse) & (deleted_mse <= tolerance[0])
            order = deleted_mse.masked_fill(~allowed, float("inf")).argsort(stable=True)
            accepted = False
            for candidate in order:
                if not bool(allowed[candidate]):
                    break
                trial = trials[int(candidate)]
                verified = self._fit(parameters, knots, trial, points, degree)
                checks += 1
                if bool((verified <= tolerance).all()):
                    mask, current_mse = trial, verified
                    if trajectory is not None:
                        trajectory.append((mask.detach().clone(), current_mse.detach().clone()))
                    accepted = True
                    break
            if not accepted:
                break
        return mask, current_mse, scans, checks

    @staticmethod
    def _trajectory_check_indices(length, budget):
        """Bounded samples including one deletion and the final deletion.

        We do not assume decoded feasibility is monotonic along fixed-geometry
        deletions. Equally spaced probes cover the path without an extra search
        over all its states; a one-check budget intentionally means endpoint.
        """
        count = min(length, budget)
        if count < 1:
            return []
        if count == 1:
            return [length - 1]
        return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})

    def _trajectory_geometry_targets(self, candidates):
        """Teach the reachable frontier and its nearest harder checked state.

        A long-jump, decoder-infeasible endpoint is not automatically a useful
        first learning target. If none of the checked deletions is reachable,
        start with the one-deletion state instead. Every target still has an
        independently verified feasible *fixed* geometry and is detached.
        """
        unique = []
        for candidate in sorted(candidates, key=lambda item: item["rank"]):
            if not any(torch.equal(candidate["mask"], previous["mask"]) for previous in unique):
                unique.append(candidate)
        if not unique:
            return []
        reachable = [item for item in unique if item["decoded_feasible"]]
        frontier = (min(reachable, key=lambda item: item["rank"]) if reachable else
                    min(unique, key=lambda item: (item["depth"], item["rank"])))
        selected = [frontier]
        # A harder state is useful only when it is not already realizable.
        # Prefer the nearest lower cardinality, not the farthest endpoint.
        harder = [item for item in unique if item is not frontier
                  and item["rank"][0] < frontier["rank"][0]]
        others = [item for item in unique if item is not frontier
                  and item["rank"][0] >= frontier["rank"][0]]
        harder.sort(key=lambda item: (frontier["rank"][0] - item["rank"][0], item["rank"][1]))
        others.sort(key=lambda item: (abs(item["rank"][0] - frontier["rank"][0]), item["rank"][1]))
        selected.extend((harder + others)[:max(self.teacher_geometry_trajectory_targets - 1, 0)])
        return selected

    @torch.no_grad()
    def _trajectory_teacher(self, model, context, best_mask, best_mse, deployment_mask,
                            deployment_output, points, degree, tolerance, minimum,
                            *, row_indices=None):
        """Training-only fixed-geometry paths with actual decoded acceptance.

        At most trajectory_checks decoded refits are performed per distinct
        seed and bounded row. Fixed geometry and masks do not bypass the actual
        decoder. Failed endpoints remain diagnostics, not feasible labels.
        """
        zero = best_mse.new_zeros(())
        metrics = self._greedy_metrics(zero)
        batch = points.shape[0]
        original_mask = best_mask.clone()
        best_mask, best_mse = best_mask.clone(), best_mse.clone()
        targets, terminals = [], []
        scans = checks = decoded_refits = decoded_passes = 0
        rows = (list(range(min(batch, self.teacher_greedy_max_curves)))
                if row_indices is None else row_indices)
        searched = len(rows)
        for row in rows:
            row_context = self._row_context(context, row, batch)
            row_points, row_tolerance = points[row:row + 1], tolerance[row:row + 1]
            seeds = [original_mask[row:row + 1]]
            if not torch.equal(seeds[0], deployment_mask[row:row + 1]):
                seeds.append(deployment_mask[row:row + 1])
            row_candidates, row_terminals = [], []
            for seed in seeds:
                if (deployment_output is not None
                        and torch.equal(seed, deployment_mask[row:row + 1])):
                    output = {key: value[row:row + 1] for key, value in deployment_output.items()
                              if isinstance(value, torch.Tensor) and value.ndim
                              and value.shape[0] == batch}
                else:
                    output = self._decode_output(model, row_context, seed)
                trajectory = []
                fixed_mask, fixed_mse, evaluated, verified = self._fixed_geometry_greedy(
                    model, row_context, output["params"], output["internal_knots"], seed,
                    row_points, degree, row_tolerance, minimum, trajectory=trajectory,
                )
                scans += evaluated
                checks += verified
                if not bool((fixed_mse <= row_tolerance).all()):
                    continue
                if not trajectory:
                    trajectory = [(fixed_mask, fixed_mse)]
                for index in self._trajectory_check_indices(
                    len(trajectory), self.teacher_greedy_trajectory_checks,
                ):
                    trial_mask, trial_fixed_mse = trajectory[index]
                    decoded_mse = self._decode_mse(model, row_context, trial_mask, row_points, degree)
                    decoded_refits += 1
                    feasible = bool((decoded_mse <= row_tolerance).all())
                    decoded_passes += int(feasible)
                    current_k, old_k = int(trial_mask.sum()), int(best_mask[row].sum())
                    improve = (not bool(best_mse[row] <= tolerance[row])
                               or current_k < old_k
                               or (current_k == old_k and bool(decoded_mse[0] < best_mse[row])))
                    if feasible and improve:
                        best_mask[row] = trial_mask[0]
                        best_mse[row] = decoded_mse[0]
                    candidate = dict(
                        row=row, mask=trial_mask.detach().clone(),
                        params=output["params"].detach().clone(),
                        knots=output["internal_knots"].detach().clone(),
                        rank=(current_k, float(trial_fixed_mse[0])),
                        decoded_feasible=feasible, depth=int(seed.sum()) - current_k,
                    )
                    row_candidates.append(candidate)
                    if index == len(trajectory) - 1:
                        row_terminals.append(candidate)
            if row_terminals:
                terminal = min(row_terminals, key=lambda item: item["rank"])
                terminals.append(terminal)
                if self.teacher_geometry_trajectory_targets:
                    targets.extend(self._trajectory_geometry_targets(row_candidates))
                else:
                    targets.append(terminal)
        changed = (best_mask != original_mask).any(-1)
        accepted_delta = (original_mask.sum(-1) - best_mask.sum(-1)).double().mean()
        metrics.update(
            teacher_greedy_curve_fraction=zero.new_tensor(searched / batch),
            teacher_greedy_fixed_count=zero.new_tensor(
                sum(item["rank"][0] for item in terminals) / max(len(terminals), 1)),
            teacher_greedy_fixed_pass_rate=zero.new_tensor(len(terminals) / searched),
            teacher_greedy_decoded_pass_rate=zero.new_tensor(
                sum(item["decoded_feasible"] for item in terminals) / max(len(terminals), 1)),
            teacher_greedy_fixed_delta_k=zero.new_tensor(sum(
                int(original_mask[item["row"]].sum()) - item["rank"][0]
                for item in terminals) / max(len(terminals), 1)),
            teacher_greedy_accepted_delta_k=accepted_delta,
            teacher_greedy_accepted_fraction=changed.double().mean(),
            teacher_greedy_candidate_refits=zero.new_tensor(float(scans)),
            teacher_greedy_verification_refits=zero.new_tensor(float(checks)),
            teacher_greedy_decode_refits=zero.new_tensor(float(decoded_refits)),
            teacher_greedy_refits=zero.new_tensor(float(scans + checks + decoded_refits)),
            teacher_greedy_trajectory_checks=zero.new_tensor(float(decoded_refits)),
            teacher_greedy_trajectory_pass_rate=zero.new_tensor(decoded_passes / max(decoded_refits, 1)),
            teacher_greedy_trajectory_accepted_delta_k=accepted_delta,
            teacher_geometry_target_count=zero.new_tensor(float(len(targets))),
            # These statistics cover ALL selected auxiliary targets, including
            # the harder, not-yet-reachable target; they are not frontier-only.
            teacher_geometry_target_k_mean=zero.new_tensor(
                sum(item["rank"][0] for item in targets) / max(len(targets), 1)),
            teacher_geometry_target_pass_rate=zero.new_tensor(
                sum(item["decoded_feasible"] for item in targets) / max(len(targets), 1)),
        )
        return best_mask, best_mse, targets, metrics

    @torch.no_grad()
    def _greedy_teacher(self, model, context, best_mask, best_mse, deployment_mask,
                        deployment_output, points, degree, tolerance, minimum,
                        *, deployment_mse=None, supervised_counts=None, supervised_valid=None):
        """Discover compact fixed geometry, then admit ONLY decoded-feasible labels.

        Up to max_curves rows are searched using the configured priority.
        Each distinct seed, incumbent teacher and deployed mask, receives the
        configured number of accepted deletion steps. A fixed-geometry target
        that the current decoder cannot realize is auxiliary geometry training
        data, NEVER a feasible mask/count label.
        """
        rows = self._greedy_priority_rows(
            best_mask, best_mse, deployment_mask, tolerance, minimum,
            deployment_mse=deployment_mse, supervised_counts=supervised_counts,
            supervised_valid=supervised_valid,
        )
        if self.teacher_greedy_trajectory_checks:
            return self._trajectory_teacher(
                model, context, best_mask, best_mse, deployment_mask,
                deployment_output, points, degree, tolerance, minimum, row_indices=rows,
            )
        zero = best_mse.new_zeros(())
        metrics = self._greedy_metrics(zero)
        batch = points.shape[0]
        original_mask = best_mask.clone()
        best_mask, best_mse = best_mask.clone(), best_mse.clone()
        targets = []
        fixed_counts, fixed_pass, decoded_pass, fixed_delta = [], [], [], []
        scans = checks = decoded_refits = 0
        for row in rows:
            row_context = self._row_context(context, row, batch)
            row_points, row_tolerance = points[row:row + 1], tolerance[row:row + 1]
            seeds = [original_mask[row:row + 1]]
            if not torch.equal(seeds[0], deployment_mask[row:row + 1]):
                seeds.append(deployment_mask[row:row + 1])
            target = None
            for seed in seeds:
                if (deployment_output is not None
                        and torch.equal(seed, deployment_mask[row:row + 1])):
                    output = {key: value[row:row + 1] for key, value in deployment_output.items()
                              if isinstance(value, torch.Tensor) and value.ndim
                              and value.shape[0] == batch}
                else:
                    output = self._decode_output(model, row_context, seed)
                fixed_mask, fixed_mse, evaluated, verified = self._fixed_geometry_greedy(
                    model, row_context, output["params"], output["internal_knots"], seed,
                    row_points, degree, row_tolerance, minimum,
                )
                scans += evaluated
                checks += verified
                if not bool((fixed_mse <= row_tolerance).all()):
                    continue
                decoded_mse = self._decode_mse(model, row_context, fixed_mask, row_points, degree)
                decoded_refits += 1
                current_k, old_k = int(fixed_mask.sum()), int(best_mask[row].sum())
                feasible = bool((decoded_mse <= row_tolerance).all())
                improve = (not bool(best_mse[row] <= tolerance[row])
                           or current_k < old_k
                           or (current_k == old_k and bool(decoded_mse[0] < best_mse[row])))
                if feasible and improve:
                    best_mask[row] = fixed_mask[0]
                    best_mse[row] = decoded_mse[0]
                if target is None or (current_k, float(fixed_mse[0])) < target["rank"]:
                    target = dict(
                        row=row, mask=fixed_mask.detach().clone(),
                        params=output["params"].detach().clone(),
                        knots=output["internal_knots"].detach().clone(),
                        rank=(current_k, float(fixed_mse[0])),
                        decoded_feasible=feasible,
                    )
            if target is not None:
                targets.append(target)
                fixed_counts.append(target["rank"][0])
                fixed_pass.append(1.0)
                decoded_pass.append(float(target["decoded_feasible"]))
                fixed_delta.append(int(original_mask[row].sum()) - target["rank"][0])
        searched = len(rows)
        changed = (best_mask != original_mask).any(-1)
        metrics.update(
            teacher_greedy_curve_fraction=zero.new_tensor(searched / batch),
            teacher_greedy_fixed_count=zero.new_tensor(sum(fixed_counts) / max(len(fixed_counts), 1)),
            teacher_greedy_fixed_pass_rate=zero.new_tensor(sum(fixed_pass) / searched),
            teacher_greedy_decoded_pass_rate=zero.new_tensor(sum(decoded_pass) / max(len(decoded_pass), 1)),
            teacher_greedy_fixed_delta_k=zero.new_tensor(sum(fixed_delta) / max(len(fixed_delta), 1)),
            teacher_greedy_accepted_delta_k=(original_mask.sum(-1) - best_mask.sum(-1)).double().mean(),
            teacher_greedy_accepted_fraction=changed.double().mean(),
            teacher_greedy_candidate_refits=zero.new_tensor(float(scans)),
            teacher_greedy_verification_refits=zero.new_tensor(float(checks)),
            teacher_greedy_decode_refits=zero.new_tensor(float(decoded_refits)),
            teacher_greedy_refits=zero.new_tensor(float(scans + checks + decoded_refits)),
        )
        return best_mask, best_mse, targets, metrics

    def _greedy_geometry_distillation(self, model, context, targets, points, degree, tolerance):
        """Detached compact target plus a differentiable REAL refit violation."""
        losses, violations = [], []
        beta = self.knot_position_beta
        for target in targets:
            row, mask = target["row"], target["mask"].detach()
            row_context = self._row_context(context, row, points.shape[0])
            output = self._decode_output(model, row_context, mask)
            parameter_loss = F.smooth_l1_loss(
                output["params"], target["params"].detach(), beta=beta,
            ) / beta
            position_loss = output["params"].new_zeros(())
            if bool(mask.any()):
                position_loss = F.smooth_l1_loss(
                    output["internal_knots"][mask], target["knots"].detach()[mask], beta=beta,
                ) / beta
            mse = self._fit(output["params"], output["internal_knots"], mask,
                            points[row:row + 1], degree)
            threshold = tolerance[row:row + 1]
            violation = F.relu(torch.log(mse + threshold * 1e-12) - torch.log(threshold)).mean()
            violations.append(violation)
            losses.append(parameter_loss + position_loss + violation)
        zero = points.new_zeros(())
        if losses and self.teacher_geometry_trajectory_targets:
            # A curve with two frontier targets must not receive twice the
            # weight of a curve with only one reachable compact state.
            rows = sorted({target["row"] for target in targets})
            mean_loss = torch.stack([
                torch.stack([loss for loss, target in zip(losses, targets)
                             if target["row"] == row]).mean() for row in rows
            ]).mean()
            mean_violation = torch.stack([
                torch.stack([value for value, target in zip(violations, targets)
                             if target["row"] == row]).mean() for row in rows
            ]).mean()
            return mean_loss, mean_violation, len(losses)
        return (torch.stack(losses).mean() if losses else zero,
                torch.stack(violations).mean() if violations else zero, len(losses))

    def _compact_mask_distillation(self, model, context, targets, deployment_mask):
        """Emphasize already reachable compact sets, not impossible deletions.

        Geometry-only targets can be decoder-infeasible and MUST be excluded.
        For each searched curve choose just its smallest decoded-feasible state
        that improves on deployment. Unweighted, mass-aligned BCE avoids adding
        another false-remove class weight to the probability-based count head.
        Averaging across selected curves keeps scarce deep-search labels from
        being diluted by all the unsearched curves in the batch.
        """
        logits = context["keep_logits"]
        zero = logits.new_zeros(())
        metrics = {name: zero for name in (
            "teacher_compact_mask_loss", "teacher_compact_bce_loss",
            "teacher_compact_count_loss", "teacher_compact_ranking_loss",
            "teacher_compact_target_count", "teacher_compact_target_k_mean",
        )}
        chosen = {}
        for target in targets:
            row = target["row"]
            if not target["decoded_feasible"]:
                continue
            mask = target["mask"].detach()
            count = int(mask.sum())
            if count >= int(deployment_mask[row].sum()):
                continue
            rank = (count, target["rank"][1])
            if row not in chosen or rank < chosen[row][0]:
                chosen[row] = (rank, mask)
        if not chosen:
            return zero, metrics
        if getattr(model, "one_shot_selection_policy", None) != "mass_topk":
            raise ValueError("compact mask distillation requires mass_topk deployment")
        rows = sorted(chosen)
        masks = torch.cat([chosen[row][1] for row in rows], dim=0)
        labels, reachable_count = self._count_aligned_targets(
            masks, sigma=float(getattr(model, "one_shot_safety_sigma", 0)),
            reserve=float(getattr(model, "one_shot_safety_knots", 0)), dtype=logits.dtype,
        )
        # A configured reserve can make a small target count unattainable.
        # Do not create conflicting supervision for such rows.
        if not bool(reachable_count.any()):
            return zero, metrics
        row_ids = torch.tensor(rows, device=logits.device, dtype=torch.long)[reachable_count]
        masks, labels = masks[reachable_count], labels[reachable_count].detach()
        selected_logits = logits[row_ids]
        bce = F.binary_cross_entropy_with_logits(selected_logits, labels)
        requested = context.get("one_shot_requested_count_score", logits.sigmoid().sum(-1))
        target_counts = (masks.sum(-1).to(logits.dtype) - .25).clamp_min(0).detach()
        count_loss = F.smooth_l1_loss(
            torch.log1p(requested[row_ids].clamp_min(0)), torch.log1p(target_counts),
        )
        pair_mask = masks.unsqueeze(-1) & (~masks).unsqueeze(-2)
        pair_penalty = F.softplus(
            self.ranking_margin - selected_logits.unsqueeze(-1) + selected_logits.unsqueeze(-2),
        )
        # Give each curve equal weight even when its cardinality differs.
        ranking_by_row = []
        for pair, penalty in zip(pair_mask, pair_penalty):
            if bool(pair.any()):
                ranking_by_row.append(penalty[pair].mean())
            else:
                ranking_by_row.append(zero)
        ranking = torch.stack(ranking_by_row).mean()
        loss = bce + count_loss + ranking
        metrics.update(
            teacher_compact_mask_loss=loss, teacher_compact_bce_loss=bce,
            teacher_compact_count_loss=count_loss, teacher_compact_ranking_loss=ranking,
            teacher_compact_target_count=zero.new_tensor(float(len(row_ids))),
            teacher_compact_target_k_mean=masks.sum(-1).to(logits.dtype).mean(),
        )
        return loss, metrics

    @staticmethod
    def _count_aligned_targets(mask, *, sigma, reserve, dtype):
        """Soft BCE targets whose MASS + safety maps to the teacher cardinality.

        Only positive slots receive probability; ordering labels stay boolean.
        This is not an extra count network. Impossible targets below a fixed
        reserve are reported explicitly, rather than silently called aligned.
        """
        if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(reserve) or reserve < 0:
            raise ValueError("selection safety must be finite and non-negative")
        counts = mask.sum(-1).to(dtype)
        desired = (counts - 0.25).clamp_min(0)
        feasible = desired >= reserve
        low, high = torch.zeros_like(counts), torch.ones_like(counts)
        for _ in range(40):
            midpoint = (low + high) * 0.5
            score = counts * midpoint + sigma * (counts * midpoint * (1 - midpoint)).clamp_min(0).sqrt() + reserve
            low = torch.where(score < desired, midpoint, low)
            high = torch.where(score >= desired, midpoint, high)
        # A reserve larger than the desired score makes exact calibration
        # impossible. Preserve the hard positive labels in that case; turning
        # them all into negatives would destroy the feasible mask teacher.
        probability = torch.where(
            feasible & (counts > 0), (low + high) * .5,
            (counts > 0).to(dtype),
        )
        target = mask.to(dtype) * probability.unsqueeze(-1)
        return target, feasible

    @staticmethod
    def _teacher_match_metrics(mask, teacher_mask, teacher_feasible):
        """Micro keep/reject agreement against feasible online teachers only."""
        valid = teacher_feasible.unsqueeze(-1)
        true_positive = (mask & teacher_mask & valid).sum().double()
        predicted = (mask & valid).sum().double()
        target = (teacher_mask & valid).sum().double()
        one = true_positive.new_ones(())
        zero = true_positive.new_zeros(())
        eligible = teacher_feasible.any()
        precision = torch.where(predicted > 0, true_positive / predicted.clamp_min(1), one)
        recall = torch.where(target > 0, true_positive / target.clamp_min(1), one)
        f1 = torch.where(
            predicted + target > 0,
            2 * true_positive / (predicted + target).clamp_min(1), one,
        )
        return {
            "teacher_keep_precision": torch.where(eligible, precision, zero),
            "teacher_keep_recall": torch.where(eligible, recall, zero),
            "teacher_keep_f1": torch.where(eligible, f1, zero),
            "teacher_false_remove_rate": torch.where(eligible, 1 - recall, zero),
            "teacher_match_valid_fraction": teacher_feasible.double().mean(),
        }

    @staticmethod
    def _supervised_counts(
        target_count, target_valid, *, batch, capacity, device,
    ):
        """Validate optional certified synthetic counts without touching real rows."""
        if target_count is None:
            if target_valid is not None:
                raise ValueError(
                    "synthetic_target_valid requires synthetic_target_count"
                )
            return None, torch.zeros(batch, dtype=torch.bool, device=device)
        counts = torch.as_tensor(target_count, dtype=torch.float64, device=device)
        if counts.shape != (batch,):
            raise ValueError("synthetic_target_count must have shape [B]")
        if target_valid is None:
            valid = torch.ones(batch, dtype=torch.bool, device=device)
        else:
            if not isinstance(target_valid, torch.Tensor) or target_valid.dtype != torch.bool:
                raise ValueError("synthetic_target_valid must be a boolean tensor")
            if target_valid.shape != (batch,):
                raise ValueError("synthetic_target_valid must have shape [B]")
            valid = target_valid.to(device=device)
        certified = counts[valid]
        if (
            not torch.isfinite(certified).all()
            or torch.any(certified < 0)
            or torch.any(certified > capacity)
            or not torch.equal(certified, certified.round())
        ):
            raise ValueError(
                "valid synthetic target counts must be integers within capacity"
            )
        # Invalid rows may use NaN/-1 sentinels; replacing them prevents those
        # placeholders from contaminating masked arithmetic below.
        return torch.where(valid, counts, torch.zeros_like(counts)), valid

    @staticmethod
    def _geometry_targets(
        target_params, target_knots, target_mask, target_valid, *, points,
    ):
        supplied = tuple(
            value is not None
            for value in (target_params, target_knots, target_mask, target_valid)
        )
        if not any(supplied):
            return None, None, None, torch.zeros(
                points.shape[0], dtype=torch.bool, device=points.device,
            )
        if not all(supplied):
            raise ValueError(
                "target_params, target_internal_knots, "
                "target_internal_knot_mask and target_geometry_valid "
                "must be supplied together"
            )
        params = torch.as_tensor(
            target_params, dtype=points.dtype, device=points.device,
        ).detach()
        knots = torch.as_tensor(
            target_knots, dtype=points.dtype, device=points.device,
        ).detach()
        if not isinstance(target_mask, torch.Tensor) or target_mask.dtype != torch.bool:
            raise ValueError("target_internal_knot_mask must be a boolean tensor")
        if not isinstance(target_valid, torch.Tensor) or target_valid.dtype != torch.bool:
            raise ValueError("target_geometry_valid must be a boolean tensor")
        mask = target_mask.to(device=points.device).detach()
        valid = target_valid.to(device=points.device).detach()
        if params.shape != points.shape[:2]:
            raise ValueError("target_params must have shape [B,M]")
        if knots.ndim != 2 or knots.shape[0] != points.shape[0]:
            raise ValueError("target_internal_knots must have shape [B,T]")
        if mask.shape != knots.shape:
            raise ValueError(
                "target_internal_knot_mask must match target_internal_knots"
            )
        if valid.shape != points.shape[:1]:
            raise ValueError("target_geometry_valid must have shape [B]")
        supervised_params = params[valid]
        supervised_knots = knots[mask & valid.unsqueeze(-1)]
        if (
            not torch.isfinite(supervised_params).all()
            or torch.any(supervised_params < 0)
            or torch.any(supervised_params > 1)
        ):
            raise ValueError("valid target parameters must be finite in [0,1]")
        if supervised_params.numel() and (
            torch.any(supervised_params[:, 1:] <= supervised_params[:, :-1])
            or not torch.allclose(
                supervised_params[:, 0],
                torch.zeros_like(supervised_params[:, 0]), atol=1e-7,
            )
            or not torch.allclose(
                supervised_params[:, -1],
                torch.ones_like(supervised_params[:, -1]), atol=1e-7,
            )
        ):
            raise ValueError(
                "valid target parameters must increase strictly from zero to one"
            )
        if (
            not torch.isfinite(supervised_knots).all()
            or torch.any(supervised_knots <= 0)
            or torch.any(supervised_knots >= 1)
        ):
            raise ValueError("valid target knots must be finite inside (0,1)")
        mask = mask & valid.unsqueeze(-1)
        params = torch.where(valid.unsqueeze(-1), params, torch.zeros_like(params))
        knots = torch.where(mask, knots, torch.zeros_like(knots))
        return params, knots, mask, valid

    @staticmethod
    def _warp_knots_to_target_parameterization(
        knots: torch.Tensor,
        source_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Piecewise-linearly express batched knots in the target domain.

        Search intervals are discrete, but interpolation within the selected
        interval remains differentiable with respect to the predicted knot and
        source parameters.  This mirrors validation's parameterization warp
        and prevents position supervision from comparing unlike domains.
        """
        if knots.ndim != 2 or source_parameters.ndim != 2:
            raise ValueError("knots and source parameters must be batched matrices")
        if target_parameters.shape != source_parameters.shape:
            raise ValueError("source and target parameters must share shape [B,M]")
        if knots.shape[0] != source_parameters.shape[0]:
            raise ValueError("knots and parameters must share the batch dimension")
        right = torch.searchsorted(
            source_parameters.contiguous(), knots.contiguous(), right=True,
        ).clamp(1, source_parameters.shape[1] - 1)
        left = right - 1
        source_left = source_parameters.gather(1, left)
        source_right = source_parameters.gather(1, right)
        target_left = target_parameters.gather(1, left)
        target_right = target_parameters.gather(1, right)
        denominator = (source_right - source_left).clamp_min(
            torch.finfo(source_parameters.dtype).eps
        )
        fraction = (knots - source_left) / denominator
        warped = target_left + fraction * (target_right - target_left)
        return warped.clamp(
            min=target_parameters[:, :1], max=target_parameters[:, -1:]
        )

    def _directed_knot_loss(
        self, predicted, predicted_mask, target, target_mask, valid,
    ):
        """Recall-direction position loss: every true knot needs a prediction."""
        zero = predicted.new_zeros(())
        active_targets = target_mask & valid.unsqueeze(-1)
        if not bool(active_targets.any()):
            return zero, zero
        distances = (predicted.unsqueeze(-1) - target.unsqueeze(1)).abs()
        if predicted_mask is not None:
            if predicted_mask.shape != predicted.shape or predicted_mask.dtype != torch.bool:
                raise ValueError("predicted knot mask must be boolean [B,K]")
            distances = distances.masked_fill(
                ~predicted_mask.unsqueeze(-1), 1.0,
            )
        nearest = distances.amin(dim=1)[active_targets]
        loss = F.smooth_l1_loss(
            nearest, torch.zeros_like(nearest), beta=self.knot_position_beta,
        )
        return loss, nearest.mean()

    def _selected_knot_loss(
        self, predicted, predicted_mask, target, target_mask, valid,
    ):
        """Monotone one-to-one loss plus nearest-neighbour diagnostics.

        The assignment is computed from detached one-dimensional coordinates,
        like a Hungarian target assignment, while SmoothL1 gradients still
        flow through the selected predicted coordinates.  Every element of the
        shorter set is matched exactly once; the certified log-count loss is
        responsible for eliminating unmatched survivors or adding missing
        ones.  This avoids the many-to-one collapse permitted by Chamfer loss.
        """
        zero = predicted.new_zeros(())
        if predicted_mask.shape != predicted.shape or predicted_mask.dtype != torch.bool:
            raise ValueError("predicted knot mask must be boolean [B,K]")

        matched_predictions = []
        matched_targets = []
        target_nearest = []
        prediction_nearest = []
        for row in valid.nonzero(as_tuple=False).flatten().tolist():
            row_predictions = predicted[row][predicted_mask[row]].sort().values
            row_targets = target[row][target_mask[row]].sort().values
            if row_predictions.numel() and row_targets.numel():
                distances = (
                    row_predictions.unsqueeze(-1) - row_targets.unsqueeze(0)
                ).abs()
                target_nearest.append(distances.amin(dim=0))
                prediction_nearest.append(distances.amin(dim=1))
                prediction_ids, target_ids = self._ordered_pair_indices(
                    row_predictions.detach(), row_targets.detach()
                )
                matched_predictions.append(row_predictions[prediction_ids])
                matched_targets.append(row_targets[target_ids])
            elif row_targets.numel():
                target_nearest.append(row_targets.new_ones(row_targets.shape))
            elif row_predictions.numel():
                prediction_nearest.append(
                    row_predictions.new_ones(row_predictions.shape)
                )

        if matched_predictions:
            paired_predictions = torch.cat(matched_predictions)
            paired_targets = torch.cat(matched_targets)
            position_loss = F.smooth_l1_loss(
                paired_predictions,
                paired_targets,
                beta=self.knot_position_beta,
            )
        else:
            position_loss = zero
        return (
            position_loss,
            torch.cat(target_nearest).mean() if target_nearest else zero,
            torch.cat(prediction_nearest).mean() if prediction_nearest else zero,
        )

    @staticmethod
    def _ordered_pair_indices(
        predicted: torch.Tensor, target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimum-L1 monotone maximum-cardinality assignment in 1D."""
        if predicted.ndim != 1 or target.ndim != 1:
            raise ValueError("ordered assignment inputs must be vectors")
        if not predicted.numel() or not target.numel():
            empty = torch.empty(0, dtype=torch.long, device=predicted.device)
            return empty, empty

        predicted_is_long = predicted.numel() >= target.numel()
        long_values = predicted if predicted_is_long else target
        short_values = target if predicted_is_long else predicted
        long_cpu = long_values.detach().double().cpu()
        short_cpu = short_values.detach().double().cpu()
        long_count, short_count = long_cpu.numel(), short_cpu.numel()
        infinity = float("inf")
        cost = [[infinity] * (short_count + 1) for _ in range(long_count + 1)]
        take = [[False] * (short_count + 1) for _ in range(long_count + 1)]
        for index in range(long_count + 1):
            cost[index][0] = 0.0
        for i in range(1, long_count + 1):
            for j in range(1, min(i, short_count) + 1):
                skip_cost = cost[i - 1][j]
                match_cost = cost[i - 1][j - 1] + abs(
                    float(long_cpu[i - 1] - short_cpu[j - 1])
                )
                if match_cost <= skip_cost:
                    cost[i][j] = match_cost
                    take[i][j] = True
                else:
                    cost[i][j] = skip_cost

        long_ids = []
        short_ids = []
        i, j = long_count, short_count
        while j:
            if take[i][j]:
                long_ids.append(i - 1)
                short_ids.append(j - 1)
                i -= 1
                j -= 1
            else:
                i -= 1
        long_ids.reverse()
        short_ids.reverse()
        if predicted_is_long:
            predicted_ids, target_ids = long_ids, short_ids
        else:
            predicted_ids, target_ids = short_ids, long_ids
        return (
            torch.tensor(predicted_ids, dtype=torch.long, device=predicted.device),
            torch.tensor(target_ids, dtype=torch.long, device=predicted.device),
        )

    def forward(
        self, model, points, *, stage="joint", mse_tolerance=None,
        complexity_scale: float = 1.0, synthetic_target_count=None,
        synthetic_target_valid=None, target_params=None,
        target_internal_knots=None, target_internal_knot_mask=None,
        target_geometry_valid=None,
    ):
        if stage not in ("proposal", "joint"):
            raise ValueError("stage must be 'proposal' or 'joint'")
        if not math.isfinite(complexity_scale) or complexity_scale < 0:
            raise ValueError("complexity_scale must be finite and non-negative")
        degree = getattr(model, "degree", 3)
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError("model.degree must be a positive integer")
        self._validate_points(points, degree)
        requested_tolerance = self.mse_tolerance if mse_tolerance is None else mse_tolerance
        tolerance = torch.as_tensor(requested_tolerance, dtype=torch.float64, device=points.device)
        if tolerance.ndim == 0:
            tolerance = tolerance.expand(points.shape[0])
        if tolerance.shape != points.shape[:1] or not torch.isfinite(tolerance).all() or torch.any(tolerance <= 0):
            raise ValueError("mse_tolerance must be positive finite scalar or shape [B]")
        context = model.encode_candidates(points, mse_tolerance=requested_tolerance)
        if not isinstance(context, Mapping):
            raise ValueError("encode_candidates must return a mapping")
        proposals = context["proposal_internal_knots"]
        if proposals.ndim != 2 or proposals.shape[1] < 1:
            raise ValueError("proposal_internal_knots must have shape [B,K] with K >= 1")
        supervised_counts, supervised_valid = self._supervised_counts(
            synthetic_target_count, synthetic_target_valid,
            batch=points.shape[0], capacity=proposals.shape[1], device=points.device,
        )
        (
            geometry_params,
            geometry_knots,
            geometry_mask,
            geometry_valid,
        ) = self._geometry_targets(
            target_params, target_internal_knots, target_internal_knot_mask,
            target_geometry_valid, points=points,
        )
        zero = points.new_zeros(())
        proposal_ordered_loss = zero
        if bool(geometry_valid.any()):
            proposal_true_parameter_loss = F.mse_loss(
                context["proposal_params"][geometry_valid],
                geometry_params[geometry_valid],
            )
            proposal_true_parameter_mae = (
                context["proposal_params"][geometry_valid]
                - geometry_params[geometry_valid]
            ).abs().mean()
            supervised_proposals = proposals.clone()
            supervised_proposals[geometry_valid] = (
                self._warp_knots_to_target_parameterization(
                    proposals[geometry_valid],
                    context["proposal_params"][geometry_valid],
                    geometry_params[geometry_valid],
                )
            )
            proposal_knot_coverage_loss, proposal_knot_nearest_mae = (
                self._directed_knot_loss(
                    supervised_proposals,
                    None,
                    geometry_knots,
                    geometry_mask,
                    geometry_valid,
                )
            )
            if self.proposal_ordered_weight:
                proposal_ordered_loss, _, _ = self._selected_knot_loss(
                    supervised_proposals,
                    torch.ones_like(proposals, dtype=torch.bool),
                    geometry_knots, geometry_mask, geometry_valid,
                )
        else:
            proposal_true_parameter_loss = proposal_true_parameter_mae = zero
            proposal_knot_coverage_loss = proposal_knot_nearest_mae = zero
        dense_mask = torch.ones_like(proposals, dtype=torch.bool)
        dense_details = None
        if self.local_fit_weight:
            dense_details = self._fit_details(
                context["proposal_params"], proposals, dense_mask, points, degree,
            )
            dense_mse = dense_details["per_sample_mse"]
        else:
            dense_mse = self._fit(context["proposal_params"], proposals, dense_mask, points, degree)
        dense_penalty = self._tail_aware_mean(self._objective_fit_penalty(dense_mse, tolerance))
        zero = dense_mse.new_zeros(())
        boundary_ranking_loss = zero
        proposal_counterfactual_loss = deployment_counterfactual_loss = zero
        proposal_chord_mse = deployment_chord_mse = zero
        proposal_counterfactual_win = deployment_counterfactual_win = zero
        local_fit_loss = local_max_mse = zero
        greedy_metrics = self._greedy_metrics(zero)
        greedy_targets = []
        geometry_distillation_loss = zero
        compact_mask_loss = zero
        complexity_active_fraction = zero
        aligned_target_mass = count_reserve_infeasible_fraction = zero
        counterfactual_refits = 0
        if self.parameter_counterfactual_weight:
            proposal_counterfactual_loss, reference_mse = self._parameter_counterfactual(
                context["proposal_params"], proposals, dense_mask,
                context.get("chord_params"), dense_mse, points, degree, tolerance,
            )
            proposal_chord_mse = reference_mse.mean()
            proposal_counterfactual_win = (dense_mse.detach() <= reference_mse).double().mean()
            counterfactual_refits += 1
        geometry_teacher_metrics = {
            name: zero for name in (
                "teacher_geometry_evaluations", "teacher_geometry_improved_fraction",
                "teacher_geometry_count_reduction", "teacher_geometry_mse_gain",
                "teacher_geometry_feasible_fraction",
            )
        }
        deployment_output = None
        refinement_metrics = {
            name: zero for name in (
                "teacher_refinement_evaluations", "teacher_refinement_steps_used",
                "teacher_refinement_improved_fraction",
                "teacher_refinement_count_reduction", "teacher_refinement_mse_gain",
                "teacher_refinement_feasible_fraction",
            )
        }
        teacher_match_metrics = {
            name: zero for name in (
                "teacher_keep_precision", "teacher_keep_recall", "teacher_keep_f1",
                "teacher_false_remove_rate", "teacher_match_valid_fraction",
            )
        }
        if stage == "proposal":
            loss = (
                self.fit_weight * dense_penalty
                + self.true_parameter_weight * proposal_true_parameter_loss
                + self.proposal_knot_coverage_weight
                * proposal_knot_coverage_loss
            )
            deployment_mse = best_mse = dense_mse
            count = dense_mse.new_full(dense_mse.shape, proposals.shape[1])
            best_count = count
            policy_loss = distillation_loss = count_loss = ranking_loss = zero
            entropy = complexity = zero
            supervised_count_loss = supervised_over_count_loss = zero
            supervised_count_mae = zero
            supervised_excess_count = zero
            prefix_feasible_fraction = zero
            prefix_fallback_fraction = zero
            prefix_search_evaluations = zero
            selected_knot_position_loss = selected_knot_nearest_mae = zero
            selected_to_true_knot_mae = zero
            deployment_true_parameter_loss = deployment_true_parameter_mae = zero
            true_parameter_loss = proposal_true_parameter_loss
            true_parameter_mae = proposal_true_parameter_mae
            if self.local_fit_weight:
                local_fit_loss, local_max_mse = self._local_fit_penalty(
                    dense_details["per_point_squared_error"], context["proposal_params"], tolerance,
                )
        else:
            logits = context["keep_logits"]
            if logits.shape != proposals.shape or logits.device != points.device:
                raise ValueError("keep_logits must share proposal [B,K] shape and device")
            if not logits.is_floating_point() or not torch.isfinite(logits).all():
                raise ValueError("keep_logits must be finite floating-point values")
            probabilities = logits.sigmoid()
            mask = model.select_mask(context)
            if not isinstance(mask, torch.Tensor) or mask.shape != logits.shape or mask.dtype != torch.bool:
                raise ValueError("select_mask must return a boolean [B,K] tensor")
            deployment_output = None
            if (self.parameter_counterfactual_weight or self.local_fit_weight
                    or getattr(model, "parameter_trust_enabled", False)
                    or self.teacher_greedy_steps):
                deployment_output = self._decode_output(model, context, mask)
                deployment_details = self._fit_details(
                    deployment_output["params"], deployment_output["internal_knots"],
                    mask, points, degree,
                )
                deployment_mse = deployment_details["per_sample_mse"]
                if self.local_fit_weight:
                    local_fit_loss, local_max_mse = self._local_fit_penalty(
                        deployment_details["per_point_squared_error"],
                        deployment_output["params"], tolerance,
                    )
                if self.parameter_counterfactual_weight:
                    deployment_counterfactual_loss, reference_mse = self._parameter_counterfactual(
                        deployment_output["params"], deployment_output["internal_knots"], mask,
                        context.get("chord_params"), deployment_mse, points, degree, tolerance,
                    )
                    deployment_chord_mse = reference_mse.mean()
                    deployment_counterfactual_win = (
                        deployment_mse.detach() <= reference_mse
                    ).double().mean()
                    counterfactual_refits += 1
            else:
                deployment_mse = self._decode_mse(model, context, mask, points, degree)
            count = mask.sum(-1).to(dense_mse.dtype)
            # Independent draws only: neither the deterministic deployment nor
            # any edited/optimized target is assigned a Bernoulli log-probability.
            random_masks = torch.bernoulli(
                probabilities.detach().expand(self.policy_samples, -1, -1)
            ).bool()
            minimum = int(getattr(model, "min_selected_knots", 0))
            # Keep these draws exactly IID Bernoulli.  Projecting them to the
            # deployment minimum would invalidate the score-function log
            # probability.  Deployment-constrained masks are supplied by the
            # deterministic counterfactual pool below instead.
            with torch.no_grad():
                sampled_mse = torch.stack([
                    self._decode_mse(model, context, draw, points, degree) for draw in random_masks
                ])
                costs = subset_cost(sampled_mse, random_masks.sum(-1), tolerance, proposals.shape[1])
            policy_loss = bernoulli_subset_policy_loss(logits, random_masks, costs)
            with torch.no_grad():
                # Random policy draws can violate the deployment minimum, so
                # they train the unbiased policy estimator but cannot become
                # structured deployment teachers.
                if self.ranked_prefix_teacher:
                    (
                        best_mask,
                        searched_best_mse,
                        prefix_teacher_feasible,
                        search_evaluations,
                    ) = self._minimum_feasible_ranked_prefix(
                        model, context, probabilities.detach(), points, degree,
                        tolerance, minimum,
                        guide_counts=supervised_counts,
                        guide_valid=supervised_valid,
                    )
                    prefix_feasible_fraction = prefix_teacher_feasible.double().mean()
                    prefix_fallback_fraction = (~prefix_teacher_feasible).double().mean()
                    prefix_search_evaluations = dense_mse.new_tensor(
                        float(search_evaluations)
                    )
                    # Retain a few add/remove/swap alternatives.  They expose
                    # interactions that a pure prefix search cannot see; BCE
                    # and ranking distillation can then reorder such a useful
                    # combination into the future deployed Top-K prefix.
                    compared_masks = [best_mask, mask]
                    compared_mse = [searched_best_mse, deployment_mse.detach()]
                    constrain_method = getattr(
                        model, "constrain_selection_mask", None
                    )
                    constrain = (
                        (lambda trial: constrain_method(context, trial))
                        if callable(constrain_method) else None
                    )
                    alternatives = self._counterfactual_masks(
                        mask, probabilities.detach(), minimum,
                        constrain=constrain,
                    )
                    for alternative_index, trial in enumerate(alternatives):
                        if alternative_index >= self.counterfactual_edits:
                            break
                        compared_masks.append(trial)
                        compared_mse.append(
                            self._decode_mse(model, context, trial, points, degree)
                        )
                    candidate_masks = torch.stack(compared_masks)
                    candidate_mse = torch.stack(compared_mse)
                    selected, best_index = select_best_subset(
                        candidate_masks, candidate_mse, tolerance,
                    )
                    columns = torch.arange(points.shape[0], device=points.device)
                    selected_mse = candidate_mse[best_index, columns]
                    any_feasible = (
                        candidate_mse <= tolerance.unsqueeze(0)
                    ).any(0)
                    # Prefix helper already supplies the conservative full-set
                    # fallback; never replace it with an infeasible local edit.
                    best_mask = torch.where(
                        any_feasible.unsqueeze(-1), selected, best_mask,
                    )
                    searched_best_mse = torch.where(
                        any_feasible, selected_mse, searched_best_mse,
                    )
                else:
                    compared_masks = [mask]
                    compared_mse = [deployment_mse.detach()]
                    use_deployment_family = (
                        getattr(model, "one_shot_selection_policy", None)
                        == "mass_topk"
                        and callable(getattr(model, "select_mask_at_count", None))
                        and callable(getattr(model, "constrain_selection_mask", None))
                    )
                    ranked_mask = (
                        (lambda counts: model.select_mask_at_count(context, counts))
                        if use_deployment_family else None
                    )
                    constrain = (
                        (lambda trial: model.constrain_selection_mask(context, trial))
                        if use_deployment_family else None
                    )
                    for trial in self._counterfactual_masks(
                        mask, probabilities.detach(), minimum,
                        ranked_mask=ranked_mask, constrain=constrain,
                    ):
                        compared_masks.append(trial)
                        compared_mse.append(
                            self._decode_mse(model, context, trial, points, degree)
                        )
                    best_mask, best_index = select_best_subset(
                        torch.stack(compared_masks), torch.stack(compared_mse), tolerance,
                    )
                    columns = torch.arange(points.shape[0], device=points.device)
                    searched_best_mse = torch.stack(compared_mse)[best_index, columns]
                    prefix_feasible_fraction = (
                        searched_best_mse <= tolerance
                    ).double().mean()
                    prefix_fallback_fraction = zero
                    prefix_search_evaluations = dense_mse.new_tensor(
                        float(len(compared_masks))
                    )
                if self.teacher_refinement_steps:
                    best_mask, searched_best_mse, refinement_metrics = self._refine_teacher(
                        model, context, best_mask, searched_best_mse,
                        probabilities.detach(), points, degree, tolerance, minimum,
                    )
                if self.teacher_geometry_candidates:
                    best_mask, searched_best_mse, geometry_teacher_metrics = self._geometry_teacher(
                        model, context, best_mask, searched_best_mse,
                        points, degree, tolerance, minimum,
                    )
                if self.teacher_greedy_steps:
                    best_mask, searched_best_mse, greedy_targets, greedy_metrics = self._greedy_teacher(
                        model, context, best_mask, searched_best_mse, mask,
                        deployment_output, points, degree, tolerance, minimum,
                        deployment_mse=deployment_mse, supervised_counts=supervised_counts,
                        supervised_valid=supervised_valid,
                    )
                teacher_feasible = searched_best_mse <= tolerance
                teacher_match_metrics = self._teacher_match_metrics(
                    mask, best_mask, teacher_feasible,
                )
            if self.teacher_geometry_distillation_weight and greedy_targets:
                geometry_distillation_loss, violation, auxiliary_refits = self._greedy_geometry_distillation(
                    model, context, greedy_targets, points, degree, tolerance,
                )
                greedy_metrics.update(
                    teacher_geometry_distillation_loss=geometry_distillation_loss,
                    teacher_geometry_fit_violation=violation,
                    teacher_geometry_aux_refits=zero.new_tensor(float(auxiliary_refits)),
                )
                greedy_metrics["teacher_greedy_refits"] = (
                    greedy_metrics["teacher_greedy_refits"] + auxiliary_refits
                )
            if self.teacher_compact_mask_weight and greedy_targets:
                compact_mask_loss, compact_metrics = self._compact_mask_distillation(
                    model, context, greedy_targets, mask,
                )
                greedy_metrics.update(compact_metrics)
            # Re-decode the actual chosen set with gradients. Targets are from
            # this model/context, so relocation is trained for the selected set.
            best_mse = self._decode_mse(model, context, best_mask, points, degree)
            best_count = best_mask.sum(-1).to(dense_mse.dtype)
            target = best_mask.to(logits.dtype)
            element_weight = 1.0 + (self.false_remove_weight - 1.0) * target
            if self.count_reserve_alignment:
                target, alignment_feasible = self._count_aligned_targets(
                    best_mask, sigma=float(getattr(model, "one_shot_safety_sigma", 0)),
                    reserve=float(getattr(model, "one_shot_safety_knots", 0)), dtype=logits.dtype,
                )
                aligned_target_mass = target.sum(-1).mean()
                count_reserve_infeasible_fraction = (~alignment_feasible).double().mean()
            distillation_loss = F.binary_cross_entropy_with_logits(
                logits, target, weight=element_weight,
            )
            requested_score = context.get(
                "one_shot_requested_count_score", probabilities.sum(-1)
            )
            if (
                not isinstance(requested_score, torch.Tensor)
                or requested_score.shape != best_count.shape
                or not requested_score.is_floating_point()
                or not torch.isfinite(requested_score).all()
            ):
                raise ValueError(
                    "one_shot_requested_count_score must be finite floating-point [B]"
                )
            # K-0.25 lies safely inside mass-TopK's ceil interval (K-1,K].
            count_target = (best_count - 0.25).clamp_min(0.0)
            # Log-count supervision has comparable scale for Kc=20 and Kc=96;
            # the historical division by capacity made the selector count
            # gradient vanish as candidate capacity increased.
            count_loss = F.smooth_l1_loss(
                torch.log1p(requested_score.clamp_min(0.0)),
                torch.log1p(count_target),
            )
            structured_count_mae = (requested_score - best_count).abs().mean()
            positive = best_mask.unsqueeze(-1)
            negative = (~best_mask).unsqueeze(-2)
            pair_mask = positive & negative
            pair_penalty = F.softplus(
                self.ranking_margin - logits.unsqueeze(-1) + logits.unsqueeze(-2)
            )
            ranking_loss = (
                pair_penalty[pair_mask].mean() if bool(pair_mask.any()) else zero
            )
            if self.boundary_ranking_weight:
                boundary_ranking_loss = self._boundary_ranking_loss(
                    logits, best_mask, teacher_feasible,
                )
            entropy = -(probabilities * F.logsigmoid(logits) + (1 - probabilities) * F.logsigmoid(-logits)).mean()
            # Complexity is earned only with a strict feasibility margin.  The
            # trainer additionally ramps ``complexity_scale`` after validation
            # reaches its deployment gate.
            safe = deployment_mse.detach() <= tolerance * self.complexity_activation_ratio
            complexity_active_fraction = safe.double().mean()
            complexity = (safe * probabilities.mean(-1)).mean()
            if bool(geometry_valid.any()):
                if deployment_output is None:
                    deployment_output = model.decode_subset(context, mask)
                if not isinstance(deployment_output, Mapping):
                    raise ValueError("decode_subset must return a mapping")
                returned_mask = deployment_output.get("learned_keep_mask")
                if (
                    not isinstance(returned_mask, torch.Tensor)
                    or not torch.equal(returned_mask, mask)
                ):
                    raise ValueError(
                        "decode_subset must preserve the requested discrete mask"
                    )
                selected_knots = deployment_output.get("internal_knots")
                if (
                    not isinstance(selected_knots, torch.Tensor)
                    or selected_knots.shape != proposals.shape
                    or not torch.isfinite(selected_knots).all()
                ):
                    raise ValueError(
                        "decode_subset internal_knots must be finite [B,K]"
                    )
                deployment_params = deployment_output.get("params")
                if (
                    not isinstance(deployment_params, torch.Tensor)
                    or deployment_params.shape != points.shape[:2]
                    or not torch.isfinite(deployment_params).all()
                ):
                    raise ValueError("decode_subset params must be finite [B,M]")
                supervised_selected_knots = selected_knots.clone()
                supervised_selected_knots[geometry_valid] = (
                    self._warp_knots_to_target_parameterization(
                        selected_knots[geometry_valid],
                        deployment_params[geometry_valid],
                        geometry_params[geometry_valid],
                    )
                )
                (
                    selected_knot_position_loss,
                    selected_knot_nearest_mae,
                    selected_to_true_knot_mae,
                ) = self._selected_knot_loss(
                    supervised_selected_knots,
                    mask,
                    geometry_knots,
                    geometry_mask,
                    geometry_valid,
                )
                deployment_true_parameter_loss = F.mse_loss(
                    deployment_params[geometry_valid],
                    geometry_params[geometry_valid],
                )
                deployment_true_parameter_mae = (
                    deployment_params[geometry_valid]
                    - geometry_params[geometry_valid]
                ).abs().mean()
                # The final knot coordinates and their true labels now share a
                # directly supervised parameterization, while the proposal
                # ParameterHead remains anchored as well.
                true_parameter_loss = 0.5 * (
                    proposal_true_parameter_loss
                    + deployment_true_parameter_loss
                )
                true_parameter_mae = 0.5 * (
                    proposal_true_parameter_mae
                    + deployment_true_parameter_mae
                )
            else:
                selected_knot_position_loss = selected_knot_nearest_mae = zero
                selected_to_true_knot_mae = zero
                deployment_true_parameter_loss = deployment_true_parameter_mae = zero
                true_parameter_loss = proposal_true_parameter_loss
                true_parameter_mae = proposal_true_parameter_mae
            if supervised_counts is None or not bool(supervised_valid.any()):
                supervised_count_loss = supervised_over_count_loss = zero
                supervised_count_mae = zero
                supervised_excess_count = zero
            else:
                supervised_target = supervised_counts.to(requested_score.dtype)
                # K-0.25 maps to exactly K under the deployed ceil-based
                # mass-TopK cardinality rule.  This symmetric label term keeps
                # certified Synthetic predictions close to source K, while the
                # separate one-sided term below still discourages safe
                # over-retention more strongly.
                supervised_score_target = (
                    supervised_target[supervised_valid] - 0.25
                ).clamp_min(0.0)
                supervised_requested = requested_score[
                    supervised_valid
                ].clamp_min(0.0)
                # Before the deployed subset is feasible, the true-count label
                # may safely correct under-selection but must not demand still
                # more pruning.  Once feasible, use the full symmetric pull.
                count_label_active = (
                    (deployment_mse.detach()[supervised_valid] <= tolerance[supervised_valid])
                    | (supervised_requested < supervised_score_target)
                )
                if bool(count_label_active.any()):
                    supervised_count_loss = F.smooth_l1_loss(
                        torch.log1p(supervised_requested[count_label_active]),
                        torch.log1p(
                            supervised_score_target[count_label_active]
                        ),
                    )
                else:
                    supervised_count_loss = zero
                supervised_count_mae = (
                    count[supervised_valid] - supervised_target[supervised_valid]
                ).abs().mean()
                supervised_excess_count = F.relu(
                    count[supervised_valid] - supervised_target[supervised_valid]
                ).mean()
                eligible = supervised_valid & (deployment_mse.detach() <= tolerance)
                if bool(eligible.any()):
                    log_excess = F.relu(
                        torch.log1p(requested_score[eligible].clamp_min(0.0))
                        - torch.log1p(supervised_target[eligible])
                    )
                    supervised_over_count_loss = F.smooth_l1_loss(
                        log_excess, torch.zeros_like(log_excess),
                    )
                else:
                    supervised_over_count_loss = zero
            selected_fit = 0.5 * (
                self._tail_aware_mean(self._objective_fit_penalty(deployment_mse, tolerance))
                + self._tail_aware_mean(self._objective_fit_penalty(best_mse, tolerance))
            )
            loss = (self.fit_weight * selected_fit + self.dense_weight * dense_penalty
                    + self.policy_weight * policy_loss + self.distillation_weight * distillation_loss
                    + self.count_weight * count_loss + self.ranking_weight * ranking_loss
                    + self.entropy_weight * entropy
                    + complexity_scale * self.complexity_weight * complexity
                    + self.supervised_count_weight * supervised_count_loss
                    + self.supervised_over_count_weight
                    * supervised_over_count_loss
                    + self.true_parameter_weight * true_parameter_loss
                    + self.proposal_knot_coverage_weight
                    * proposal_knot_coverage_loss
                    + self.selected_knot_position_weight
                    * selected_knot_position_loss)
            if self.boundary_ranking_weight:
                loss = loss + self.boundary_ranking_weight * boundary_ranking_loss
        parameter_counterfactual_loss = (
            0.5 * (proposal_counterfactual_loss + deployment_counterfactual_loss)
            if stage == "joint" else proposal_counterfactual_loss
        )
        # Conditional additions preserve the historical objective bit-for-bit
        # with all four new mechanisms disabled.
        if self.parameter_counterfactual_weight:
            loss = loss + self.parameter_counterfactual_weight * parameter_counterfactual_loss
        if self.local_fit_weight:
            loss = loss + self.local_fit_weight * local_fit_loss
        if self.proposal_ordered_weight:
            loss = loss + self.proposal_ordered_weight * proposal_ordered_loss
        if self.teacher_geometry_distillation_weight:
            loss = loss + self.teacher_geometry_distillation_weight * geometry_distillation_loss
        if self.teacher_compact_mask_weight:
            loss = loss + self.teacher_compact_mask_weight * compact_mask_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite v16 subset objective")
        proposal_trust = context.get("proposal_parameter_trust")
        subset_trust = (
            deployment_output.get("subset_parameter_trust")
            if deployment_output is not None else None
        )
        if (stage == "joint" and subset_trust is None
                and isinstance(proposal_trust, torch.Tensor)
                and not getattr(model, "parameter_trust_enabled", False)):
            # Disabled native gates are identically one. Do not decode again
            # merely to observe that constant; legacy models without gate
            # fields retain the zero-safe missing-field diagnostic.
            subset_trust = torch.ones_like(proposal_trust)
        trust_metrics = {
            **self._trust_statistics(proposal_trust, prefix="proposal_parameter_trust", zero=zero),
            **self._trust_statistics(subset_trust, prefix="subset_parameter_trust", zero=zero),
        }
        metrics = {
            "loss": loss, "dense_mse": dense_mse.mean(),
            "dense_pass_rate": (dense_mse <= tolerance).double().mean(),
            "deployment_mse": deployment_mse.mean(),
            "deployment_pass_rate": (deployment_mse <= tolerance).double().mean(),
            "keep_count": count.mean(), "policy_loss": policy_loss,
            "mask_distillation_loss": distillation_loss,
            "structured_count_loss": count_loss,
            "structured_count_mae": (
                structured_count_mae if stage == "joint" else zero
            ),
            "supervised_count_loss": supervised_count_loss,
            "supervised_over_count_loss": supervised_over_count_loss,
            "supervised_count_mae": supervised_count_mae,
            "supervised_excess_count": supervised_excess_count,
            "true_parameter_loss": true_parameter_loss,
            "true_parameter_mae": true_parameter_mae,
            "proposal_true_parameter_loss": proposal_true_parameter_loss,
            "proposal_true_parameter_mae": proposal_true_parameter_mae,
            "deployment_true_parameter_loss": deployment_true_parameter_loss,
            "deployment_true_parameter_mae": deployment_true_parameter_mae,
            "proposal_knot_coverage_loss": proposal_knot_coverage_loss,
            "proposal_knot_nearest_mae": proposal_knot_nearest_mae,
            "proposal_ordered_loss": proposal_ordered_loss,
            "parameter_counterfactual_loss": parameter_counterfactual_loss,
            "proposal_parameter_counterfactual_loss": proposal_counterfactual_loss,
            "deployment_parameter_counterfactual_loss": deployment_counterfactual_loss,
            "proposal_chord_counterfactual_mse": proposal_chord_mse,
            "deployment_chord_counterfactual_mse": deployment_chord_mse,
            "proposal_parameter_counterfactual_win_fraction": proposal_counterfactual_win,
            "deployment_parameter_counterfactual_win_fraction": deployment_counterfactual_win,
            "parameter_counterfactual_refits": zero.new_tensor(float(counterfactual_refits)),
            "local_fit_loss": local_fit_loss,
            "local_max_region_mse": local_max_mse,
            "selected_knot_position_loss": selected_knot_position_loss,
            "selected_knot_nearest_mae": selected_knot_nearest_mae,
            "selected_to_true_knot_mae": selected_to_true_knot_mae,
            "teacher_ranking_loss": ranking_loss,
            "teacher_boundary_ranking_loss": boundary_ranking_loss,
            "proposal_feasible_fraction": (dense_mse <= tolerance).double().mean(),
            "subset_best_mse": best_mse.mean(), "subset_best_count": best_count.mean(),
            "subset_best_pass_rate": (best_mse <= tolerance).double().mean(),
            "prefix_teacher_feasible_fraction": prefix_feasible_fraction,
            "prefix_teacher_fallback_fraction": prefix_fallback_fraction,
            "prefix_teacher_search_evaluations": prefix_search_evaluations,
            "mask_entropy": entropy, "feasible_complexity_loss": complexity,
            "complexity_active_fraction": complexity_active_fraction,
            "aligned_teacher_probability_mass": aligned_target_mass,
            "count_reserve_infeasible_fraction": count_reserve_infeasible_fraction,
            **greedy_metrics,
            **refinement_metrics,
            **geometry_teacher_metrics,
            **trust_metrics,
            **teacher_match_metrics,
        }
        return loss, {name: value.detach() for name, value in metrics.items()}
