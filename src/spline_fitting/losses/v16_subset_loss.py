"""Supervised and legacy-online objectives for v16 candidate selection.

The formal path uses certified synthetic parameters and knots as direct labels:
an ordered one-to-one assignment supplies the KeepMask target, the labelled knot
count supervises adaptive mass-TopK, and the labelled subset supervises survivor
relocation.  A cached per-source-knot deletion MSE from the same minimality
certificate can additionally weight critical slots without any online solve.
The historical online counterfactual path remains available only for explicit
legacy ablations.
Deployment always selects once and performs one final standard B-spline refit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .deployment_bspline_loss import differentiable_hard_gated_bspline_fit


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
    """Joint parameter, survivor-position and subset learning.

    ``joint_supervision='synthetic_ground_truth'`` is the formal, fast path.
    ``'online_teacher'`` preserves the former counterfactual objective for
    controlled historical ablations only.
    """

    def __init__(
        self, mse_tolerance: float = 2.5e-5, policy_samples: int = 4,
        counterfactual_edits: int = 4, fit_weight: float = 1.0,
        policy_weight: float = 0.25, distillation_weight: float = 2.0,
        count_weight: float = 2.0, ranking_weight: float = 1.0,
        supervised_count_weight: float = 1.0,
        supervised_over_count_weight: float = 0.0,
        true_parameter_weight: float = 0.1,
        proposal_knot_coverage_weight: float = 1.0,
        proposal_knot_assignment_weight: float = 1.0,
        proposal_multiscale_recall_weight: float = 0.0,
        selected_knot_position_weight: float = 1.0,
        keep_dice_weight: float = 0.0,
        keep_cdf_weight: float = 0.0,
        parameter_gap_weight: float = 0.0,
        parameter_bias_weight: float = 0.0,
        fine_teacher_weight: float = 0.0,
        fine_teacher_ranking_weight: float = 0.0,
        fine_teacher_temperature: float = 0.5,
        keep_fuzzy_negative_radius: float = 0.01,
        keep_fuzzy_negative_floor: float = 0.1,
        proposal_parameter_warp_gradient_scale: float = 0.0,
        joint_parameter_warp_gradient_scale: float = 0.1,
        false_remove_weight: float = 5.0, ranking_margin: float = 1.0,
        dense_weight: float = 0.25, entropy_weight: float = 0.0,
        complexity_weight: float = 0.05, complexity_activation_ratio: float = 0.8,
        tail_weight: float = 0.5, tail_fraction: float = 0.2,
        solver_jitter: float = 1e-10, ranked_prefix_teacher: bool = True,
        teacher_prefix_search_steps: int = 7, teacher_low_count_sweep: int = 0,
        synthetic_count_role: str = "exact",
        synthetic_geometry_oracle_teacher: bool = False,
        oracle_teacher_extra_knots: int = 2,
        knot_position_beta: float = 0.01,
        joint_supervision: str = "online_teacher",
    ) -> None:
        super().__init__()
        if not math.isfinite(mse_tolerance) or mse_tolerance <= 0:
            raise ValueError("mse_tolerance must be finite and positive")
        for name, value, minimum in (
            ("policy_samples", policy_samples, 2),
            ("counterfactual_edits", counterfactual_edits, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("fit_weight", fit_weight), ("policy_weight", policy_weight),
            ("distillation_weight", distillation_weight), ("dense_weight", dense_weight),
            ("count_weight", count_weight), ("ranking_weight", ranking_weight),
            ("supervised_count_weight", supervised_count_weight),
            ("supervised_over_count_weight", supervised_over_count_weight),
            ("true_parameter_weight", true_parameter_weight),
            ("proposal_knot_coverage_weight", proposal_knot_coverage_weight),
            ("proposal_knot_assignment_weight", proposal_knot_assignment_weight),
            ("proposal_multiscale_recall_weight", proposal_multiscale_recall_weight),
            ("selected_knot_position_weight", selected_knot_position_weight),
            ("keep_dice_weight", keep_dice_weight),
            ("keep_cdf_weight", keep_cdf_weight),
            ("parameter_gap_weight", parameter_gap_weight),
            ("parameter_bias_weight", parameter_bias_weight),
            ("fine_teacher_weight", fine_teacher_weight),
            ("fine_teacher_ranking_weight", fine_teacher_ranking_weight),
            ("keep_fuzzy_negative_radius", keep_fuzzy_negative_radius),
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
        if (
            not math.isfinite(fine_teacher_temperature)
            or fine_teacher_temperature <= 0
        ):
            raise ValueError("fine_teacher_temperature must be finite and positive")
        self.fine_teacher_temperature = float(fine_teacher_temperature)
        if keep_fuzzy_negative_radius <= 0:
            raise ValueError("keep_fuzzy_negative_radius must be positive")
        if (
            not math.isfinite(keep_fuzzy_negative_floor)
            or not 0 <= keep_fuzzy_negative_floor <= 1
        ):
            raise ValueError("keep_fuzzy_negative_floor must lie in [0,1]")
        self.keep_fuzzy_negative_floor = float(keep_fuzzy_negative_floor)
        for name, value in (
            (
                "proposal_parameter_warp_gradient_scale",
                proposal_parameter_warp_gradient_scale,
            ),
            ("joint_parameter_warp_gradient_scale", joint_parameter_warp_gradient_scale),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
            setattr(self, name, float(value))
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
        if (
            isinstance(teacher_low_count_sweep, bool)
            or not isinstance(teacher_low_count_sweep, int)
            or teacher_low_count_sweep < 0
        ):
            raise ValueError("teacher_low_count_sweep must be a non-negative integer")
        if synthetic_count_role not in {"exact", "upper_bound"}:
            raise ValueError(
                "synthetic_count_role must be 'exact' or 'upper_bound'"
            )
        if joint_supervision not in {"synthetic_ground_truth", "online_teacher"}:
            raise ValueError(
                "joint_supervision must be 'synthetic_ground_truth' or "
                "'online_teacher'"
            )
        if not isinstance(synthetic_geometry_oracle_teacher, bool):
            raise ValueError("synthetic_geometry_oracle_teacher must be boolean")
        if (
            isinstance(oracle_teacher_extra_knots, bool)
            or not isinstance(oracle_teacher_extra_knots, int)
            or oracle_teacher_extra_knots < 0
        ):
            raise ValueError("oracle_teacher_extra_knots must be non-negative")
        self.ranked_prefix_teacher = ranked_prefix_teacher
        self.teacher_prefix_search_steps = teacher_prefix_search_steps
        self.teacher_low_count_sweep = teacher_low_count_sweep
        self.synthetic_count_role = synthetic_count_role
        self.synthetic_geometry_oracle_teacher = (
            synthetic_geometry_oracle_teacher
        )
        self.oracle_teacher_extra_knots = oracle_teacher_extra_knots
        self.knot_position_beta = float(knot_position_beta)
        self.joint_supervision = joint_supervision

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

    def _fit(self, parameters, knots, mask, points, degree):
        self._validate_geometry(parameters, knots, mask, points)
        result = differentiable_hard_gated_bspline_fit(
            parameters.double(), points.double(), knots.double(), mask,
            degree=degree, smoothness_weight=0.0, control_ridge=0.0,
            solver_jitter=self.solver_jitter,
        )["per_sample_mse"]
        if not torch.isfinite(result).all():
            raise RuntimeError("non-finite standard B-spline subset refit")
        return result

    def _decode_mse(self, model, context, mask, points, degree):
        output = model.decode_subset(context, mask)
        if not isinstance(output, Mapping):
            raise ValueError("decode_subset must return a mapping")
        returned_mask = output.get("learned_keep_mask")
        if not isinstance(returned_mask, torch.Tensor) or not torch.equal(returned_mask, mask):
            raise ValueError("decode_subset must preserve the requested discrete mask")
        return self._fit(output["params"], output["internal_knots"], mask, points, degree)

    @staticmethod
    def _fit_penalty(mse, tolerance):
        # Log space avoids the huge MSE/tolerance penalties of earlier versions.
        log_mse = torch.log(mse + tolerance * 1e-12)
        log_tolerance = torch.log(tolerance)
        return torch.logaddexp(log_mse, log_tolerance) - log_tolerance + F.relu(log_mse - log_tolerance)

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
        low_count_limit = min(capacity, self.teacher_low_count_sweep)
        low_count_counts = (
            set(range(minimum, low_count_limit + 1))
            if low_count_limit >= minimum else set()
        )
        coarse_counts = sorted({
            minimum + round(
                (capacity - minimum) * (step / grid_denominator) ** 2
            )
            for step in range(grid_denominator + 1)
        } | {minimum, capacity} | low_count_counts)
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

    @staticmethod
    def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
        """Keep the forward value while limiting a cross-task gradient path."""
        return value.detach() + float(scale) * (value - value.detach())

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

    @staticmethod
    def _parameter_gap_and_bias_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve local interval distortion and signed parameter drift.

        Pointwise parameter MSE alone is dominated by the many easy samples in
        smooth regions.  Log-gap supervision gives every sampling interval a
        relative error signal, while the per-curve signed mean explicitly
        exposes systematic left/right parameter bias.
        """
        if predicted.shape != target.shape or predicted.ndim != 2:
            raise ValueError("parameter supervision requires matching [B,M] tensors")
        if not predicted.is_floating_point() or not target.is_floating_point():
            raise ValueError("parameter supervision tensors must be floating point")
        epsilon = torch.finfo(predicted.dtype).eps
        predicted_gaps = predicted.diff(dim=-1).clamp_min(epsilon)
        target_gaps = target.diff(dim=-1).clamp_min(epsilon)
        log_gap_error = predicted_gaps.log() - target_gaps.log()
        gap_loss = F.smooth_l1_loss(
            log_gap_error,
            torch.zeros_like(log_gap_error),
            beta=0.25,
        )
        signed_bias = (predicted - target).mean(dim=-1)
        # Report in raw parameter units, optimize relative to the 0.01 knot
        # matching scale so this term cannot disappear next to fit penalties.
        bias_loss = (signed_bias / 0.01).square().mean()
        return gap_loss, bias_loss, signed_bias.abs().mean()

    @staticmethod
    def _multiscale_candidate_recall(
        predicted: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strict 0.005/0.010/0.020 recall surrogate and diagnostics.

        The logarithmic normalized distance prevents the loose 0.02 scale from
        hiding misses at publication-scale tolerances, without the exploding
        gradients of a raw ``distance / 0.005`` penalty.
        """
        active = target_mask & valid.unsqueeze(-1)
        zero = predicted.new_zeros(())
        if not bool(active.any()):
            return zero, zero, zero, zero
        nearest = (
            predicted.unsqueeze(-1) - target.unsqueeze(1)
        ).abs().amin(dim=1)[active]
        training_scales = nearest.new_tensor((0.0025, 0.005, 0.010))
        normalized = torch.log1p(
            (nearest.unsqueeze(-1) / training_scales).square()
        ).mean(-1)
        tail_count = max(1, math.ceil(normalized.numel() * 0.2))
        loss = normalized.mean() + 0.5 * normalized.topk(tail_count).values.mean()
        reporting_scales = nearest.new_tensor((0.005, 0.010, 0.020))
        recalls = tuple(
            (nearest <= scale).to(nearest.dtype).mean()
            for scale in reporting_scales
        )
        return loss, recalls[0], recalls[1], recalls[2]

    @staticmethod
    def _keep_distribution_losses(
        probabilities: torch.Tensor,
        target_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Set overlap and ordered spatial-mass losses for KeepMask learning."""
        if probabilities.shape != target_mask.shape or positions.shape != target_mask.shape:
            raise ValueError("Keep distribution tensors must share [B,K]")
        target = target_mask.to(probabilities.dtype)
        epsilon = torch.finfo(probabilities.dtype).eps
        overlap = 2.0 * (probabilities * target).sum(-1)
        dice = 1.0 - (
            (overlap + epsilon)
            / (probabilities.sum(-1) + target.sum(-1) + epsilon)
        )

        order = positions.detach().argsort(dim=-1, stable=True)
        ordered_probability = probabilities.gather(1, order)
        ordered_target = target.gather(1, order)
        probability_distribution = ordered_probability / ordered_probability.sum(
            -1, keepdim=True
        ).clamp_min(epsilon)
        target_distribution = ordered_target / ordered_target.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        cdf = F.mse_loss(
            probability_distribution.cumsum(-1),
            target_distribution.cumsum(-1),
            reduction="none",
        ).mean(-1)
        return dice.mean(), cdf.mean()

    @staticmethod
    def _weighted_pairwise_ranking_loss(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_confidence: torch.Tensor,
        *,
        margin: float,
        positive_margin: torch.Tensor | None = None,
        positive_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rank positives without treating plausible substitutes as hard negatives."""
        if (
            logits.ndim != 2
            or positive_mask.shape != logits.shape
            or positive_mask.dtype != torch.bool
            or negative_confidence.shape != logits.shape
        ):
            raise ValueError("ranking tensors must share [B,K] shape")
        if (
            not logits.is_floating_point()
            or not negative_confidence.is_floating_point()
            or not torch.isfinite(logits).all()
            or not torch.isfinite(negative_confidence).all()
            or torch.any((negative_confidence < 0) | (negative_confidence > 1))
        ):
            raise ValueError(
                "ranking logits/confidence must be finite with confidence in [0,1]"
            )
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("ranking margin must be finite and non-negative")
        if positive_margin is None:
            positive_margin = torch.zeros_like(logits)
        if positive_weight is None:
            positive_weight = torch.ones_like(logits)
        for name, value in (
            ("positive_margin", positive_margin),
            ("positive_weight", positive_weight),
        ):
            if (
                value.shape != logits.shape
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
                or torch.any(value < 0)
            ):
                raise ValueError(f"{name} must be finite non-negative [B,K]")

        pair_mask = positive_mask.unsqueeze(-1) & (~positive_mask).unsqueeze(-2)
        if not bool(pair_mask.any()):
            return logits.new_zeros(())
        penalty = F.softplus(
            margin
            + positive_margin.unsqueeze(-1)
            - logits.unsqueeze(-1)
            + logits.unsqueeze(-2)
        )
        pair_weight = (
            positive_weight.unsqueeze(-1)
            * negative_confidence.unsqueeze(-2)
        )
        active_weight = pair_weight[pair_mask]
        return (
            penalty[pair_mask] * active_weight
        ).sum() / active_weight.sum().clamp_min(torch.finfo(logits.dtype).eps)

    def _proposal_ordered_assignment_loss(
        self, predicted, target, target_mask, valid,
    ):
        """Supervise one distinct ordered proposal for every certified knot.

        Coverage remains a separate recall-direction objective.  This term
        additionally prevents several targets from sharing the same nearest
        proposal: when ``Kc > target K`` exactly ``target K`` proposals are
        selected by a minimum-L1 monotone assignment, while equal cardinality
        forces the unique rank-to-rank pairing.  Assignment indices are
        discrete and detached; gradients flow through the matched proposal
        coordinates only.
        """
        if predicted.ndim != 2 or target.ndim != 2:
            raise ValueError("proposal assignment inputs must be batched matrices")
        if predicted.shape[0] != target.shape[0] or target_mask.shape != target.shape:
            raise ValueError("proposal assignment inputs must share the batch")
        if target_mask.dtype != torch.bool or valid.dtype != torch.bool:
            raise ValueError("proposal assignment mask and validity must be boolean")
        if valid.shape != predicted.shape[:1]:
            raise ValueError("proposal assignment validity must have shape [B]")

        zero = predicted.new_zeros(())
        # One batch transfer avoids a CUDA synchronization for every labelled
        # curve while the small dynamic programs run on detached coordinates.
        sorted_predictions = predicted.sort(dim=-1).values
        predicted_cpu = sorted_predictions.detach().double().cpu()
        target_cpu = target.detach().double().cpu()
        target_mask_cpu = target_mask.detach().cpu()
        valid_rows = valid.detach().cpu().nonzero(as_tuple=False).flatten().tolist()
        matched_rows = []
        matched_prediction_ids = []
        matched_target_values = []
        matched_counts = []
        for row in valid_rows:
            row_targets_cpu = target_cpu[row][target_mask_cpu[row]].sort().values
            target_count = row_targets_cpu.numel()
            if target_count > predicted.shape[1]:
                raise ValueError("proposal assignment target count exceeds candidate capacity")
            matched_counts.append(target_count)
            if not target_count:
                continue
            prediction_ids, target_ids = self._ordered_pair_indices(
                predicted_cpu[row], row_targets_cpu,
            )
            matched_rows.extend([row] * target_count)
            matched_prediction_ids.extend(prediction_ids.tolist())
            matched_target_values.append(row_targets_cpu[target_ids])

        if not matched_rows:
            mean_count = (
                predicted.new_tensor(float(sum(matched_counts) / len(matched_counts)))
                if matched_counts else zero
            )
            return zero, zero, mean_count
        # Materialize all discrete indices/labels on the accelerator once,
        # rather than issuing one small CPU-to-GPU transfer per curve.
        paired_predictions = sorted_predictions[
            torch.tensor(matched_rows, dtype=torch.long, device=predicted.device),
            torch.tensor(
                matched_prediction_ids, dtype=torch.long, device=predicted.device,
            ),
        ]
        paired_targets = torch.cat(matched_target_values).to(
            device=predicted.device, dtype=predicted.dtype,
        )
        absolute_error = (paired_predictions - paired_targets).abs()
        return (
            F.smooth_l1_loss(
                paired_predictions,
                paired_targets,
                beta=self.knot_position_beta,
            ),
            absolute_error.mean(),
            predicted.new_tensor(float(sum(matched_counts) / len(matched_counts))),
        )

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

    def _ordered_ground_truth_candidate_mask(
        self, predicted, target, target_mask, valid,
    ) -> torch.Tensor:
        """Map certified knot labels to distinct ordered candidate slots.

        The mapping uses no Keep scores and no fitted-error search. ``predicted``
        and ``target`` must already share the certified target parameterization.
        """
        if predicted.ndim != 2 or target.ndim != 2:
            raise ValueError("oracle candidates and targets must be batched matrices")
        if predicted.shape[0] != target.shape[0] or target_mask.shape != target.shape:
            raise ValueError("oracle candidates and targets must share the batch")
        if target_mask.dtype != torch.bool or valid.dtype != torch.bool:
            raise ValueError("oracle target mask and validity must be boolean")
        if valid.shape != predicted.shape[:1]:
            raise ValueError("oracle validity must have shape [B]")
        # Transfer the small assignment batch once in each direction. Calling
        # ``.cpu()`` separately for every labelled row forces dozens of CUDA
        # synchronizations on mixed batches and can dominate a 3090 training
        # step even though the dynamic program itself is tiny (Kc <= 64 here).
        predicted_cpu = predicted.detach().double().cpu()
        target_cpu = target.detach().double().cpu()
        target_mask_cpu = target_mask.detach().cpu()
        valid_cpu = valid.detach().cpu()
        result_cpu = torch.zeros_like(predicted_cpu, dtype=torch.bool)
        for row in valid_cpu.nonzero(as_tuple=False).flatten().tolist():
            row_targets = target_cpu[row][target_mask_cpu[row]].sort().values
            if row_targets.numel() > predicted_cpu.shape[1]:
                raise ValueError("oracle target count exceeds candidate capacity")
            if not row_targets.numel():
                continue
            candidate_order = predicted_cpu[row].argsort(stable=True)
            candidate_ids, _ = self._ordered_pair_indices(
                predicted_cpu[row][candidate_order],
                row_targets,
            )
            result_cpu[row, candidate_order[candidate_ids]] = True
        return result_cpu.to(device=predicted.device)

    def _ordered_ground_truth_candidate_values(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        valid: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move per-true-knot teacher values onto their assigned candidate slots."""
        if target_values.shape != target.shape:
            raise ValueError("per-knot teacher values must match target knot shape")
        if not target_values.is_floating_point() or not torch.isfinite(target_values).all():
            raise ValueError("per-knot teacher values must be finite floating point")
        if torch.any(target_values[target_mask] < 0):
            raise ValueError("per-knot teacher values must be non-negative")
        predicted_cpu = predicted.detach().double().cpu()
        target_cpu = target.detach().double().cpu()
        target_mask_cpu = target_mask.detach().cpu()
        valid_cpu = valid.detach().cpu()
        value_cpu = target_values.detach().double().cpu()
        result_cpu = torch.zeros_like(predicted_cpu)
        result_valid_cpu = torch.zeros_like(predicted_cpu, dtype=torch.bool)
        for row in valid_cpu.nonzero(as_tuple=False).flatten().tolist():
            target_ids = target_mask_cpu[row].nonzero(as_tuple=False).flatten()
            target_order = target_cpu[row, target_ids].argsort(stable=True)
            sorted_target_ids = target_ids[target_order]
            row_targets = target_cpu[row, sorted_target_ids]
            if row_targets.numel() > predicted_cpu.shape[1]:
                raise ValueError("teacher knot count exceeds candidate capacity")
            if not row_targets.numel():
                continue
            candidate_order = predicted_cpu[row].argsort(stable=True)
            candidate_ids, matched_target_ids = self._ordered_pair_indices(
                predicted_cpu[row, candidate_order], row_targets,
            )
            assigned = candidate_order[candidate_ids]
            source_ids = sorted_target_ids[matched_target_ids.cpu()]
            result_cpu[row, assigned] = value_cpu[row, source_ids]
            result_valid_cpu[row, assigned] = True
        return (
            result_cpu.to(device=predicted.device, dtype=predicted.dtype),
            result_valid_cpu.to(device=predicted.device),
        )

    def _synthetic_oracle_candidate_mask(
        self, predicted, target, target_mask, valid,
    ) -> torch.Tensor:
        """Historical online-teacher alias for the ordered label mapping."""
        return self._ordered_ground_truth_candidate_mask(
            predicted, target, target_mask, valid,
        )

    @staticmethod
    def _expand_teacher_mask(
        mask: torch.Tensor,
        scores: torch.Tensor,
        extra_knots: int,
        eligible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add the highest-scored unused slots without changing existing slots."""
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError("teacher mask and scores must share boolean [B,K] shape")
        if eligible is not None and (
            eligible.dtype != torch.bool or eligible.shape != mask.shape[:1]
        ):
            raise ValueError("teacher expansion eligibility must be boolean [B]")
        if isinstance(extra_knots, bool) or not isinstance(extra_knots, int):
            raise ValueError("extra_knots must be an integer")
        if extra_knots <= 0:
            return mask.clone()
        available = (~mask).sum(-1)
        additions = torch.minimum(
            available, available.new_full(available.shape, extra_knots)
        )
        order = torch.argsort(
            scores.masked_fill(mask, float("-inf")),
            dim=-1, descending=True, stable=True,
        )
        rank = torch.empty_like(order)
        rank.scatter_(
            1, order,
            torch.arange(order.shape[1], device=order.device).expand_as(order),
        )
        expanded = mask | (rank < additions.unsqueeze(-1))
        if eligible is not None:
            expanded = torch.where(eligible.unsqueeze(-1), expanded, mask)
        return expanded

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

    def _supervised_joint_forward(
        self,
        model,
        context,
        points,
        degree,
        tolerance,
        proposals,
        supervised_proposals,
        supervised_counts,
        supervised_valid,
        geometry_params,
        geometry_knots,
        geometry_mask,
        geometry_valid,
        dense_mse,
        dense_penalty,
        proposal_true_parameter_loss,
        proposal_true_parameter_mae,
        proposal_knot_coverage_loss,
        proposal_knot_nearest_mae,
        proposal_knot_assignment_loss,
        proposal_knot_assignment_mae,
        proposal_knot_assignment_count,
        proposal_multiscale_recall_loss,
        proposal_recall_at_005,
        proposal_recall_at_010,
        proposal_recall_at_020,
        proposal_parameter_gap_loss,
        proposal_parameter_bias_loss,
        proposal_parameter_bias_mae,
        geometry_single_deletion_mse,
        fine_teacher_available,
    ):
        """Train Joint directly from certified synthetic knot labels.

        The discrete label is the minimum-cost monotone one-to-one assignment
        between the ordered proposal candidates and the certified source knots.
        No fitted subset is searched to create this target: Joint uses exactly
        one dense fit, one deployed-mask fit and one labelled-mask fit.
        """
        batch = points.shape[0]
        if supervised_counts is None:
            raise ValueError(
                "synthetic_ground_truth Joint requires labelled knot counts"
            )
        if not bool(supervised_valid.all()) or not bool(geometry_valid.all()):
            raise ValueError(
                "synthetic_ground_truth Joint accepts only fully labelled "
                "synthetic training rows"
            )
        labelled_counts = geometry_mask.sum(-1)
        if not torch.equal(
            supervised_counts.to(dtype=torch.long), labelled_counts.to(dtype=torch.long)
        ):
            raise ValueError("synthetic knot-count and knot-vector labels disagree")

        logits = context["keep_logits"]
        if logits.shape != proposals.shape or logits.device != points.device:
            raise ValueError("keep_logits must share proposal [B,K] shape and device")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("keep_logits must be finite floating-point values")
        probabilities = logits.sigmoid()
        target_mask = self._ordered_ground_truth_candidate_mask(
            supervised_proposals,
            geometry_knots,
            geometry_mask,
            geometry_valid,
        )
        if not torch.equal(target_mask.sum(-1), labelled_counts):
            raise RuntimeError("ordered ground-truth assignment lost labelled knots")

        fine_teacher_active = (
            self.fine_teacher_weight > 0
            or self.fine_teacher_ranking_weight > 0
        )
        if fine_teacher_active and not fine_teacher_available:
            raise ValueError(
                "fine-grained teacher weights require synthetic "
                "target_single_deletion_mse labels"
            )
        if fine_teacher_available:
            teacher_delete_mse, teacher_value_valid = (
                self._ordered_ground_truth_candidate_values(
                    supervised_proposals,
                    geometry_knots,
                    geometry_mask,
                    geometry_valid,
                    geometry_single_deletion_mse,
                )
            )
            if not torch.equal(teacher_value_valid, target_mask):
                raise RuntimeError(
                    "fine teacher values and ordered KeepMask assignment disagree"
                )
            epsilon = torch.finfo(teacher_delete_mse.dtype).tiny
            teacher_log_margin = torch.log(
                (teacher_delete_mse + epsilon)
                / tolerance.to(teacher_delete_mse.dtype).unsqueeze(-1)
            )
            teacher_risk = torch.sigmoid(
                teacher_log_margin / self.fine_teacher_temperature
            ) * target_mask.to(teacher_delete_mse.dtype)
        else:
            teacher_delete_mse = torch.zeros_like(proposals)
            teacher_log_margin = torch.zeros_like(proposals)
            teacher_risk = torch.zeros_like(proposals)

        deployment_mask = model.select_mask(context)
        if (
            not isinstance(deployment_mask, torch.Tensor)
            or deployment_mask.shape != logits.shape
            or deployment_mask.dtype != torch.bool
        ):
            raise ValueError("select_mask must return a boolean [B,K] tensor")

        def decode(mask):
            output = model.decode_subset(context, mask)
            if not isinstance(output, Mapping):
                raise ValueError("decode_subset must return a mapping")
            returned = output.get("learned_keep_mask")
            if not isinstance(returned, torch.Tensor) or not torch.equal(returned, mask):
                raise ValueError("decode_subset must preserve the requested mask")
            parameters = output.get("params")
            knots = output.get("internal_knots")
            if (
                not isinstance(parameters, torch.Tensor)
                or parameters.shape != points.shape[:2]
                or not torch.isfinite(parameters).all()
            ):
                raise ValueError("decoded params must be finite [B,M]")
            if (
                not isinstance(knots, torch.Tensor)
                or knots.shape != proposals.shape
                or not torch.isfinite(knots).all()
            ):
                raise ValueError("decoded internal_knots must be finite [B,K]")
            mse = self._fit(parameters, knots, mask, points, degree)
            return output, mse

        deployment_output, deployment_mse = decode(deployment_mask)
        labelled_output, labelled_mse = decode(target_mask)
        deployment_count = deployment_mask.sum(-1).to(dense_mse.dtype)
        target_count = labelled_counts.to(
            device=points.device, dtype=context["keep_logits"].dtype
        )

        # Direct supervised existence labels replace online mask distillation.
        element_loss = F.binary_cross_entropy_with_logits(
            logits, target_mask.to(logits.dtype), reduction="none"
        )
        positive_count = target_mask.sum(-1).clamp_min(1)
        negative_mask = ~target_mask
        distances_to_true = (
            supervised_proposals.unsqueeze(-1) - geometry_knots.unsqueeze(1)
        ).abs().masked_fill(~geometry_mask.unsqueeze(1), 1.0)
        nearest_true = distances_to_true.amin(-1)
        negative_confidence = (
            self.keep_fuzzy_negative_floor
            + (1.0 - self.keep_fuzzy_negative_floor)
            * (nearest_true / self.keep_fuzzy_negative_radius).clamp(0.0, 1.0)
        )
        if not fine_teacher_active:
            negative_confidence = torch.ones_like(negative_confidence)
        negative_weight = negative_mask.to(element_loss.dtype) * negative_confidence
        negative_count = negative_weight.sum(-1).clamp_min(1.0)
        positive_loss = (
            element_loss * target_mask.to(element_loss.dtype)
        ).sum(-1) / positive_count
        negative_loss = (
            element_loss * negative_weight
        ).sum(-1) / negative_count
        has_negative = negative_mask.any(-1)
        keep_supervision_loss = torch.where(
            has_negative,
            (
                self.false_remove_weight * positive_loss + negative_loss
            ) / (self.false_remove_weight + 1.0),
            positive_loss,
        ).mean()
        keep_dice_loss, keep_cdf_loss = self._keep_distribution_losses(
            probabilities,
            target_mask,
            supervised_proposals,
        )
        if fine_teacher_active:
            fine_teacher_loss = (
                F.softplus(-logits)[target_mask]
                * teacher_risk[target_mask].to(logits.dtype)
            ).mean()
        else:
            fine_teacher_loss = logits.new_zeros(())
        requested_score = context.get(
            "one_shot_requested_count_score", probabilities.sum(-1)
        )
        if (
            not isinstance(requested_score, torch.Tensor)
            or requested_score.shape != (batch,)
            or not requested_score.is_floating_point()
            or not torch.isfinite(requested_score).all()
        ):
            raise ValueError(
                "one_shot_requested_count_score must be finite floating-point [B]"
            )
        score_target = (target_count - 0.25).clamp_min(0.0)
        count_loss = F.smooth_l1_loss(
            torch.log1p(requested_score.clamp_min(0.0)),
            torch.log1p(score_target),
        )
        log_excess = F.relu(
            torch.log1p(requested_score.clamp_min(0.0))
            - torch.log1p(score_target)
        )
        over_count_loss = F.smooth_l1_loss(
            log_excess, torch.zeros_like(log_excess)
        )

        pair_mask = target_mask.unsqueeze(-1) & (~target_mask).unsqueeze(-2)
        ranking_loss = self._weighted_pairwise_ranking_loss(
            logits,
            target_mask,
            negative_confidence,
            margin=self.ranking_margin,
        )
        if fine_teacher_active and bool(pair_mask.any()):
            risk = teacher_risk.to(logits.dtype)
            fine_teacher_ranking_loss = self._weighted_pairwise_ranking_loss(
                logits,
                target_mask,
                negative_confidence,
                margin=self.ranking_margin,
                positive_margin=risk,
                positive_weight=risk.clamp_min(0.5),
            )
        else:
            fine_teacher_ranking_loss = logits.new_zeros(())
        entropy = -(
            probabilities * F.logsigmoid(logits)
            + (1 - probabilities) * F.logsigmoid(-logits)
        ).mean()

        def supervised_geometry(output, mask):
            warped = output["internal_knots"].clone()
            warped[geometry_valid] = self._warp_knots_to_target_parameterization(
                output["internal_knots"][geometry_valid],
                self._scale_gradient(
                    output["params"][geometry_valid],
                    self.joint_parameter_warp_gradient_scale,
                ),
                geometry_params[geometry_valid],
            )
            return self._selected_knot_loss(
                warped,
                mask,
                geometry_knots,
                geometry_mask,
                geometry_valid,
            )

        deployed_position = supervised_geometry(deployment_output, deployment_mask)
        labelled_position = supervised_geometry(labelled_output, target_mask)
        deployed_coverage = self._directed_knot_loss(
            self._warp_knots_to_target_parameterization(
                deployment_output["internal_knots"],
                self._scale_gradient(
                    deployment_output["params"],
                    self.joint_parameter_warp_gradient_scale,
                ),
                geometry_params,
            ),
            deployment_mask,
            geometry_knots,
            geometry_mask,
            geometry_valid,
        )
        labelled_coverage = self._directed_knot_loss(
            self._warp_knots_to_target_parameterization(
                labelled_output["internal_knots"],
                self._scale_gradient(
                    labelled_output["params"],
                    self.joint_parameter_warp_gradient_scale,
                ),
                geometry_params,
            ),
            target_mask,
            geometry_knots,
            geometry_mask,
            geometry_valid,
        )
        # One-to-one matching locates individual survivors; directed coverage
        # additionally penalizes any labelled knot left unmatched when the
        # deployed count is too small.  This is direct ground-truth geometry
        # supervision and introduces no subset-search/refit loop.
        selected_knot_position_loss = 0.25 * (
            deployed_position[0]
            + labelled_position[0]
            + deployed_coverage[0]
            + labelled_coverage[0]
        )
        # Diagnostics describe the actual one-shot deployment, not teacher-forced
        # geometry, while both paths contribute gradients above.
        selected_knot_nearest_mae = deployed_coverage[1]
        selected_to_true_knot_mae = deployed_position[2]

        deployment_true_parameter_loss = 0.5 * (
            F.mse_loss(deployment_output["params"], geometry_params)
            + F.mse_loss(labelled_output["params"], geometry_params)
        )
        deployment_true_parameter_mae = 0.5 * (
            (deployment_output["params"] - geometry_params).abs().mean()
            + (labelled_output["params"] - geometry_params).abs().mean()
        )
        true_parameter_loss = 0.5 * (
            proposal_true_parameter_loss + deployment_true_parameter_loss
        )
        true_parameter_mae = 0.5 * (
            proposal_true_parameter_mae + deployment_true_parameter_mae
        )
        deployed_parameter_shape = self._parameter_gap_and_bias_loss(
            deployment_output["params"], geometry_params,
        )
        labelled_parameter_shape = self._parameter_gap_and_bias_loss(
            labelled_output["params"], geometry_params,
        )
        parameter_gap_loss = (
            proposal_parameter_gap_loss
            + deployed_parameter_shape[0]
            + labelled_parameter_shape[0]
        ) / 3.0
        parameter_bias_loss = (
            proposal_parameter_bias_loss
            + deployed_parameter_shape[1]
            + labelled_parameter_shape[1]
        ) / 3.0
        parameter_bias_mae = (
            proposal_parameter_bias_mae
            + deployed_parameter_shape[2]
            + labelled_parameter_shape[2]
        ) / 3.0
        selected_fit = 0.5 * (
            self._tail_aware_mean(self._fit_penalty(deployment_mse, tolerance))
            + self._tail_aware_mean(self._fit_penalty(labelled_mse, tolerance))
        )
        loss = (
            self.fit_weight * selected_fit
            + self.dense_weight * dense_penalty
            + self.distillation_weight * keep_supervision_loss
            + self.keep_dice_weight * keep_dice_loss
            + self.keep_cdf_weight * keep_cdf_loss
            + self.fine_teacher_weight * fine_teacher_loss
            + self.fine_teacher_ranking_weight * fine_teacher_ranking_loss
            + (self.count_weight + self.supervised_count_weight) * count_loss
            + self.supervised_over_count_weight * over_count_loss
            + self.ranking_weight * ranking_loss
            + self.entropy_weight * entropy
            + self.true_parameter_weight * true_parameter_loss
            + self.proposal_knot_coverage_weight * proposal_knot_coverage_loss
            + self.proposal_knot_assignment_weight * proposal_knot_assignment_loss
            + self.proposal_multiscale_recall_weight
            * proposal_multiscale_recall_loss
            + self.selected_knot_position_weight * selected_knot_position_loss
            + self.parameter_gap_weight * parameter_gap_loss
            + self.parameter_bias_weight * parameter_bias_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite v16 supervised subset objective")

        zero = dense_mse.new_zeros(())
        supervised_count_mae = (
            deployment_count - target_count.to(deployment_count.dtype)
        ).abs().mean()
        true_positive = (deployment_mask & target_mask).sum().to(dense_mse.dtype)
        predicted_positive = deployment_mask.sum().clamp_min(1).to(dense_mse.dtype)
        labelled_positive = target_mask.sum().clamp_min(1).to(dense_mse.dtype)
        keep_precision = true_positive / predicted_positive
        keep_recall = true_positive / labelled_positive
        keep_f1 = (
            2.0 * keep_precision * keep_recall
            / (keep_precision + keep_recall).clamp_min(
                torch.finfo(dense_mse.dtype).eps
            )
        )
        critical = target_mask & (teacher_risk >= 0.75)
        critical_false_delete_rate = (
            (critical & ~deployment_mask).sum().to(dense_mse.dtype)
            / critical.sum().clamp_min(1).to(dense_mse.dtype)
        )
        metrics = {
            "loss": loss,
            "dense_mse": dense_mse.mean(),
            "dense_pass_rate": (dense_mse <= tolerance).double().mean(),
            "deployment_mse": deployment_mse.mean(),
            "deployment_pass_rate": (deployment_mse <= tolerance).double().mean(),
            "keep_count": deployment_count.mean(),
            "policy_loss": zero,
            # Compatibility name; this value is direct labelled BCE, not
            # teacher distillation. Prefer supervised_keep_loss in new reports.
            "mask_distillation_loss": keep_supervision_loss,
            "supervised_keep_loss": keep_supervision_loss,
            "keep_dice_loss": keep_dice_loss,
            "keep_cdf_loss": keep_cdf_loss,
            "keep_mask_precision": keep_precision,
            "keep_mask_recall": keep_recall,
            "keep_mask_f1": keep_f1,
            "fine_teacher_loss": fine_teacher_loss,
            "fine_teacher_ranking_loss": fine_teacher_ranking_loss,
            "fine_teacher_mean_risk": (
                teacher_risk[target_mask].mean()
                if bool(target_mask.any()) else zero
            ),
            "fine_teacher_mean_log_delete_margin": (
                teacher_log_margin[target_mask].mean()
                if bool(target_mask.any()) else zero
            ),
            "critical_false_delete_rate": critical_false_delete_rate,
            "fuzzy_negative_fraction": (
                ((negative_confidence < 1.0) & negative_mask)
                .to(dense_mse.dtype).sum()
                / negative_mask.sum().clamp_min(1).to(dense_mse.dtype)
            ),
            "structured_count_loss": count_loss,
            "structured_count_mae": (
                requested_score - score_target.to(requested_score.dtype)
            ).abs().mean(),
            "supervised_count_loss": count_loss,
            "supervised_over_count_loss": over_count_loss,
            "supervised_count_mae": supervised_count_mae,
            "supervised_excess_count": F.relu(
                deployment_count - target_count.to(deployment_count.dtype)
            ).mean(),
            "true_parameter_loss": true_parameter_loss,
            "true_parameter_mae": true_parameter_mae,
            "proposal_true_parameter_loss": proposal_true_parameter_loss,
            "proposal_true_parameter_mae": proposal_true_parameter_mae,
            "deployment_true_parameter_loss": deployment_true_parameter_loss,
            "deployment_true_parameter_mae": deployment_true_parameter_mae,
            "proposal_knot_coverage_loss": proposal_knot_coverage_loss,
            "proposal_knot_nearest_mae": proposal_knot_nearest_mae,
            "proposal_knot_assignment_loss": proposal_knot_assignment_loss,
            "proposal_knot_assignment_mae": proposal_knot_assignment_mae,
            "proposal_knot_assignment_count": proposal_knot_assignment_count,
            "proposal_multiscale_recall_loss": proposal_multiscale_recall_loss,
            "proposal_recall_at_005": proposal_recall_at_005,
            "proposal_recall_at_010": proposal_recall_at_010,
            "proposal_recall_at_020": proposal_recall_at_020,
            "parameter_gap_loss": parameter_gap_loss,
            "parameter_bias_loss": parameter_bias_loss,
            "parameter_bias_mae": parameter_bias_mae,
            "selected_knot_position_loss": selected_knot_position_loss,
            "selected_knot_coverage_loss": 0.5 * (
                deployed_coverage[0] + labelled_coverage[0]
            ),
            "selected_knot_nearest_mae": selected_knot_nearest_mae,
            "selected_to_true_knot_mae": selected_to_true_knot_mae,
            "teacher_ranking_loss": zero,
            "supervised_ranking_loss": ranking_loss,
            "proposal_feasible_fraction": (dense_mse <= tolerance).double().mean(),
            "subset_best_mse": labelled_mse.mean(),
            "subset_best_count": target_count.mean().to(dense_mse.dtype),
            "subset_best_pass_rate": (labelled_mse <= tolerance).double().mean(),
            "supervised_target_mse": labelled_mse.mean(),
            "supervised_target_pass_rate": (labelled_mse <= tolerance).double().mean(),
            "prefix_teacher_feasible_fraction": zero,
            "prefix_teacher_fallback_fraction": zero,
            "prefix_teacher_search_evaluations": zero,
            "oracle_teacher_feasible_fraction": zero,
            "oracle_teacher_selected_fraction": zero,
            "mask_entropy": entropy,
            "feasible_complexity_loss": zero,
        }
        return loss, {name: value.detach() for name, value in metrics.items()}

    def forward(
        self, model, points, *, stage="joint", mse_tolerance=None,
        complexity_scale: float = 1.0, synthetic_target_count=None,
        synthetic_target_valid=None, target_params=None,
        target_internal_knots=None, target_internal_knot_mask=None,
        target_geometry_valid=None, target_single_deletion_mse=None,
        target_single_deletion_mask=None, target_single_deletion_valid=None,
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
        fine_teacher_available = False
        if target_single_deletion_mse is None:
            geometry_single_deletion_mse = (
                torch.zeros_like(geometry_knots)
                if isinstance(geometry_knots, torch.Tensor)
                else torch.zeros_like(proposals)
            )
        else:
            if (
                not isinstance(target_single_deletion_mse, torch.Tensor)
                or target_single_deletion_mse.shape != geometry_knots.shape
                or not target_single_deletion_mse.is_floating_point()
            ):
                raise ValueError(
                    "target_single_deletion_mse must be floating-point with "
                    "the target knot shape"
                )
            geometry_single_deletion_mse = target_single_deletion_mse.to(
                device=points.device, dtype=points.dtype,
            )
            if (
                not torch.isfinite(geometry_single_deletion_mse).all()
                or torch.any(geometry_single_deletion_mse[geometry_mask] < 0)
            ):
                raise ValueError(
                    "target_single_deletion_mse must be finite and non-negative"
                )
            if (
                not isinstance(target_single_deletion_mask, torch.Tensor)
                or target_single_deletion_mask.shape != geometry_mask.shape
                or target_single_deletion_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "target_single_deletion_mask must be boolean with the "
                    "target knot shape"
                )
            if (
                not isinstance(target_single_deletion_valid, torch.Tensor)
                or target_single_deletion_valid.shape != geometry_valid.shape
                or target_single_deletion_valid.dtype != torch.bool
            ):
                raise ValueError(
                    "target_single_deletion_valid must be boolean [B]"
                )
            deletion_mask = target_single_deletion_mask.to(points.device)
            deletion_valid = target_single_deletion_valid.to(points.device)
            if bool((deletion_valid & ~geometry_valid).any()):
                raise ValueError(
                    "single-deletion teacher cannot be valid without knot labels"
                )
            if bool(
                deletion_valid.any()
                and not torch.equal(
                    deletion_mask[deletion_valid], geometry_mask[deletion_valid]
                )
            ):
                raise ValueError(
                    "single-deletion teacher mask must match true knot mask"
                )
            fine_teacher_available = bool(deletion_valid.all())
        supervised_proposals = proposals
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
            parameter_warp_scale = (
                self.proposal_parameter_warp_gradient_scale
                if stage == "proposal"
                else self.joint_parameter_warp_gradient_scale
            )
            supervised_proposals[geometry_valid] = (
                self._warp_knots_to_target_parameterization(
                    proposals[geometry_valid],
                    self._scale_gradient(
                        context["proposal_params"][geometry_valid],
                        parameter_warp_scale,
                    ),
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
            (
                proposal_knot_assignment_loss,
                proposal_knot_assignment_mae,
                proposal_knot_assignment_count,
            ) = self._proposal_ordered_assignment_loss(
                supervised_proposals,
                geometry_knots,
                geometry_mask,
                geometry_valid,
            )
            (
                proposal_multiscale_recall_loss,
                proposal_recall_at_005,
                proposal_recall_at_010,
                proposal_recall_at_020,
            ) = self._multiscale_candidate_recall(
                supervised_proposals,
                geometry_knots,
                geometry_mask,
                geometry_valid,
            )
            (
                proposal_parameter_gap_loss,
                proposal_parameter_bias_loss,
                proposal_parameter_bias_mae,
            ) = self._parameter_gap_and_bias_loss(
                context["proposal_params"][geometry_valid],
                geometry_params[geometry_valid],
            )
        else:
            proposal_true_parameter_loss = proposal_true_parameter_mae = zero
            proposal_knot_coverage_loss = proposal_knot_nearest_mae = zero
            proposal_knot_assignment_loss = proposal_knot_assignment_mae = zero
            proposal_knot_assignment_count = zero
            proposal_multiscale_recall_loss = zero
            proposal_recall_at_005 = proposal_recall_at_010 = zero
            proposal_recall_at_020 = zero
            proposal_parameter_gap_loss = proposal_parameter_bias_loss = zero
            proposal_parameter_bias_mae = zero
        dense_mse = self._fit(context["proposal_params"], proposals, torch.ones_like(proposals, dtype=torch.bool), points, degree)
        dense_penalty = self._tail_aware_mean(self._fit_penalty(dense_mse, tolerance))
        zero = dense_mse.new_zeros(())
        parameter_gap_loss = proposal_parameter_gap_loss
        parameter_bias_loss = proposal_parameter_bias_loss
        parameter_bias_mae = proposal_parameter_bias_mae
        keep_dice_loss = keep_cdf_loss = zero
        keep_precision = keep_recall = keep_f1 = zero
        fine_teacher_loss = fine_teacher_ranking_loss = zero
        fine_teacher_mean_risk = fine_teacher_mean_log_delete_margin = zero
        critical_false_delete_rate = fuzzy_negative_fraction = zero
        if stage == "proposal":
            loss = (
                self.fit_weight * dense_penalty
                + self.true_parameter_weight * proposal_true_parameter_loss
                + self.proposal_knot_coverage_weight
                * proposal_knot_coverage_loss
                + self.proposal_knot_assignment_weight
                * proposal_knot_assignment_loss
                + self.proposal_multiscale_recall_weight
                * proposal_multiscale_recall_loss
                + self.parameter_gap_weight * proposal_parameter_gap_loss
                + self.parameter_bias_weight * proposal_parameter_bias_loss
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
            oracle_teacher_feasible_fraction = zero
            oracle_teacher_selected_fraction = zero
            selected_knot_position_loss = selected_knot_nearest_mae = zero
            selected_to_true_knot_mae = zero
            deployment_true_parameter_loss = deployment_true_parameter_mae = zero
            true_parameter_loss = proposal_true_parameter_loss
            true_parameter_mae = proposal_true_parameter_mae
        else:
            if self.joint_supervision == "synthetic_ground_truth":
                return self._supervised_joint_forward(
                    model=model,
                    context=context,
                    points=points,
                    degree=degree,
                    tolerance=tolerance,
                    proposals=proposals,
                    supervised_proposals=supervised_proposals,
                    supervised_counts=supervised_counts,
                    supervised_valid=supervised_valid,
                    geometry_params=geometry_params,
                    geometry_knots=geometry_knots,
                    geometry_mask=geometry_mask,
                    geometry_valid=geometry_valid,
                    dense_mse=dense_mse,
                    dense_penalty=dense_penalty,
                    proposal_true_parameter_loss=proposal_true_parameter_loss,
                    proposal_true_parameter_mae=proposal_true_parameter_mae,
                    proposal_knot_coverage_loss=proposal_knot_coverage_loss,
                    proposal_knot_nearest_mae=proposal_knot_nearest_mae,
                    proposal_knot_assignment_loss=proposal_knot_assignment_loss,
                    proposal_knot_assignment_mae=proposal_knot_assignment_mae,
                    proposal_knot_assignment_count=proposal_knot_assignment_count,
                    proposal_multiscale_recall_loss=proposal_multiscale_recall_loss,
                    proposal_recall_at_005=proposal_recall_at_005,
                    proposal_recall_at_010=proposal_recall_at_010,
                    proposal_recall_at_020=proposal_recall_at_020,
                    proposal_parameter_gap_loss=proposal_parameter_gap_loss,
                    proposal_parameter_bias_loss=proposal_parameter_bias_loss,
                    proposal_parameter_bias_mae=proposal_parameter_bias_mae,
                    geometry_single_deletion_mse=geometry_single_deletion_mse,
                    fine_teacher_available=fine_teacher_available,
                )
            logits = context["keep_logits"]
            if logits.shape != proposals.shape or logits.device != points.device:
                raise ValueError("keep_logits must share proposal [B,K] shape and device")
            if not logits.is_floating_point() or not torch.isfinite(logits).all():
                raise ValueError("keep_logits must be finite floating-point values")
            probabilities = logits.sigmoid()
            mask = model.select_mask(context)
            if not isinstance(mask, torch.Tensor) or mask.shape != logits.shape or mask.dtype != torch.bool:
                raise ValueError("select_mask must return a boolean [B,K] tensor")
            deployment_mse = self._decode_mse(model, context, mask, points, degree)
            count = mask.sum(-1).to(dense_mse.dtype)
            oracle_teacher_feasible_fraction = zero
            oracle_teacher_selected_fraction = zero
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
                    oracle_pool_indices = []
                    if (
                        self.synthetic_geometry_oracle_teacher
                        and bool(geometry_valid.any())
                    ):
                        oracle_mask = self._synthetic_oracle_candidate_mask(
                            supervised_proposals,
                            geometry_knots,
                            geometry_mask,
                            geometry_valid,
                        )
                        # Real/uncertified rows have no oracle. Preserve their
                        # prefix teacher rows so the batched comparison cannot
                        # accidentally offer them an empty subset.
                        oracle_mask = torch.where(
                            geometry_valid.unsqueeze(-1), oracle_mask, best_mask,
                        )
                        expanded_oracle = self._expand_teacher_mask(
                            oracle_mask,
                            probabilities.detach(),
                            self.oracle_teacher_extra_knots,
                            eligible=geometry_valid,
                        )
                        for oracle_trial in (oracle_mask, expanded_oracle):
                            if any(
                                torch.equal(oracle_trial, existing)
                                for existing in compared_masks
                            ):
                                continue
                            oracle_pool_indices.append(len(compared_masks))
                            compared_masks.append(oracle_trial)
                            compared_mse.append(
                                self._decode_mse(
                                    model, context, oracle_trial, points, degree,
                                )
                            )
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
                    if oracle_pool_indices:
                        oracle_mse = candidate_mse[oracle_pool_indices]
                        oracle_teacher_feasible_fraction = (
                            (oracle_mse <= tolerance.unsqueeze(0)).any(0)
                            & geometry_valid
                        ).double().sum() / geometry_valid.double().sum().clamp_min(1)
                        chosen_oracle = torch.zeros_like(best_index, dtype=torch.bool)
                        for oracle_index in oracle_pool_indices:
                            chosen_oracle |= best_index == oracle_index
                        oracle_teacher_selected_fraction = (
                            chosen_oracle & geometry_valid
                        ).double().sum() / geometry_valid.double().sum().clamp_min(1)
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
                    oracle_teacher_feasible_fraction = zero
                    oracle_teacher_selected_fraction = zero
            # Re-decode the actual chosen set with gradients. Targets are from
            # this model/context, so relocation is trained for the selected set.
            best_mse = self._decode_mse(model, context, best_mask, points, degree)
            best_count = best_mask.sum(-1).to(dense_mse.dtype)
            target = best_mask.to(logits.dtype)
            element_weight = 1.0 + (self.false_remove_weight - 1.0) * target
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
            entropy = -(probabilities * F.logsigmoid(logits) + (1 - probabilities) * F.logsigmoid(-logits)).mean()
            # Complexity is earned only with a strict feasibility margin.  The
            # The trainer ramps ``complexity_scale`` on a deterministic Joint-
            # epoch schedule; this per-sample condition still prevents the
            # complexity reward from acting on an infeasible deployed curve.
            safe = deployment_mse.detach() <= tolerance * self.complexity_activation_ratio
            complexity = (safe * probabilities.mean(-1)).mean()
            if bool(geometry_valid.any()):
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
                if self.synthetic_count_role == "exact":
                    # Before the deployed subset is feasible, the true-count
                    # label may safely correct under-selection but must not
                    # demand still more pruning. Once feasible, use the full
                    # symmetric pull (the historical v16 behaviour).
                    count_label_active = (
                        (
                            deployment_mse.detach()[supervised_valid]
                            <= tolerance[supervised_valid]
                        )
                        | (supervised_requested < supervised_score_target)
                    )
                    if bool(count_label_active.any()):
                        supervised_count_loss = F.smooth_l1_loss(
                            torch.log1p(
                                supervised_requested[count_label_active]
                            ),
                            torch.log1p(
                                supervised_score_target[count_label_active]
                            ),
                        )
                    else:
                        supervised_count_loss = zero
                else:
                    # The source-subset certificate proves that the supplied
                    # source representation is irreducible only while its knot
                    # coordinates remain fixed. Once the survivor decoder may
                    # relocate knots, source K is a feasible upper bound, not a
                    # proof of the globally minimal continuous representation.
                    # Therefore never pull a smaller feasible teacher back up
                    # to source K; penalize only safe over-retention.
                    upper_bound_active = (
                        deployment_mse.detach()[supervised_valid]
                        <= tolerance[supervised_valid]
                    )
                    if bool(upper_bound_active.any()):
                        log_excess = F.relu(
                            torch.log1p(
                                supervised_requested[upper_bound_active]
                            )
                            - torch.log1p(
                                supervised_score_target[upper_bound_active]
                            )
                        )
                        supervised_count_loss = F.smooth_l1_loss(
                            log_excess, torch.zeros_like(log_excess),
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
                self._tail_aware_mean(self._fit_penalty(deployment_mse, tolerance))
                + self._tail_aware_mean(self._fit_penalty(best_mse, tolerance))
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
                    + self.proposal_knot_assignment_weight
                    * proposal_knot_assignment_loss
                    + self.selected_knot_position_weight
                    * selected_knot_position_loss
                    + self.proposal_multiscale_recall_weight
                    * proposal_multiscale_recall_loss
                    + self.parameter_gap_weight * parameter_gap_loss
                    + self.parameter_bias_weight * parameter_bias_loss)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite v16 subset objective")
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
            "proposal_knot_assignment_loss": proposal_knot_assignment_loss,
            "proposal_knot_assignment_mae": proposal_knot_assignment_mae,
            "proposal_knot_assignment_count": proposal_knot_assignment_count,
            "proposal_multiscale_recall_loss": proposal_multiscale_recall_loss,
            "proposal_recall_at_005": proposal_recall_at_005,
            "proposal_recall_at_010": proposal_recall_at_010,
            "proposal_recall_at_020": proposal_recall_at_020,
            "parameter_gap_loss": parameter_gap_loss,
            "parameter_bias_loss": parameter_bias_loss,
            "parameter_bias_mae": parameter_bias_mae,
            "keep_dice_loss": keep_dice_loss,
            "keep_cdf_loss": keep_cdf_loss,
            "keep_mask_precision": keep_precision,
            "keep_mask_recall": keep_recall,
            "keep_mask_f1": keep_f1,
            "fine_teacher_loss": fine_teacher_loss,
            "fine_teacher_ranking_loss": fine_teacher_ranking_loss,
            "fine_teacher_mean_risk": fine_teacher_mean_risk,
            "fine_teacher_mean_log_delete_margin": (
                fine_teacher_mean_log_delete_margin
            ),
            "critical_false_delete_rate": critical_false_delete_rate,
            "fuzzy_negative_fraction": fuzzy_negative_fraction,
            "selected_knot_position_loss": selected_knot_position_loss,
            "selected_knot_nearest_mae": selected_knot_nearest_mae,
            "selected_to_true_knot_mae": selected_to_true_knot_mae,
            "teacher_ranking_loss": ranking_loss,
            "proposal_feasible_fraction": (dense_mse <= tolerance).double().mean(),
            "subset_best_mse": best_mse.mean(), "subset_best_count": best_count.mean(),
            "subset_best_pass_rate": (best_mse <= tolerance).double().mean(),
            "prefix_teacher_feasible_fraction": prefix_feasible_fraction,
            "prefix_teacher_fallback_fraction": prefix_fallback_fraction,
            "prefix_teacher_search_evaluations": prefix_search_evaluations,
            "oracle_teacher_feasible_fraction": oracle_teacher_feasible_fraction,
            "oracle_teacher_selected_fraction": oracle_teacher_selected_fraction,
            "mask_entropy": entropy, "feasible_complexity_loss": complexity,
        }
        return loss, {name: value.detach() for name, value in metrics.items()}
