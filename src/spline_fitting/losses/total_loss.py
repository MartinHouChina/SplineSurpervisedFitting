from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LossWeights:
    fit: float = 1.0
    # Optional sparsity term: expected number of non-zero Hard-Concrete gates.
    l0: float = 0.0
    # Kept only so legacy soft-gate checkpoints/configurations remain loadable.
    activity: float = 0.0
    binary: float = 0.0
    # Deprecated: accepted only to reproduce historical checkpoints.
    orthogonal: float = 0.0
    gap: float = 0.0
    parameter_prior: float = 0.0
    true_parameter: float = 0.0
    # Supervised structure terms for independent node queries.
    existence: float = 0.0
    knot_position: float = 0.0
    count: float = 0.0
    over_count: float = 0.0


class SplineFittingLoss(nn.Module):
    """Combine fitting with either gated or count-conditioned structure loss."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        min_knot_gap: float = 1e-3,
        knot_position_beta: float = 0.02,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.min_knot_gap = min_knot_gap
        if knot_position_beta <= 0.0:
            raise ValueError("knot_position_beta must be positive")
        self.knot_position_beta = float(knot_position_beta)

    @staticmethod
    def _ordered_knot_assignment(
        predicted_knots: torch.Tensor,
        true_knots: torch.Tensor,
        true_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match each true knot to a unique sorted candidate with minimum L1 cost.

        The dynamic-programming assignment is discrete and intentionally uses
        detached positions. Gradients flow only through the selected predicted
        positions, exactly as in set-prediction matching losses.
        """
        if predicted_knots.ndim != 2:
            raise ValueError("predicted_knots must have shape [B, K]")
        if true_knots.shape != true_mask.shape:
            raise ValueError("true_knots and true_mask must have identical shape")
        if true_knots.shape[0] != predicted_knots.shape[0]:
            raise ValueError("predicted and true knots must share the batch size")
        true_mask = true_mask.to(dtype=torch.bool)

        existence_targets = torch.zeros_like(predicted_knots)
        position_targets = torch.zeros_like(predicted_knots)
        matched_mask = torch.zeros_like(predicted_knots, dtype=torch.bool)
        detached_predictions = predicted_knots.detach().cpu()
        detached_targets = true_knots.detach().cpu()
        detached_mask = true_mask.detach().cpu().to(torch.bool)

        for batch_index in range(predicted_knots.shape[0]):
            predictions = detached_predictions[batch_index].tolist()
            targets = detached_targets[batch_index][detached_mask[batch_index]].tolist()
            candidate_count = len(predictions)
            target_count = len(targets)
            if target_count > candidate_count:
                raise ValueError(
                    "the number of true knots cannot exceed the candidate count"
                )
            if target_count == 0:
                continue

            infinity = float("inf")
            costs = [
                [infinity] * (target_count + 1) for _ in range(candidate_count + 1)
            ]
            take = [[False] * (target_count + 1) for _ in range(candidate_count + 1)]
            for candidate_index in range(candidate_count + 1):
                costs[candidate_index][0] = 0.0
            for candidate_index in range(1, candidate_count + 1):
                maximum_targets = min(candidate_index, target_count)
                for target_index in range(1, maximum_targets + 1):
                    skip_cost = costs[candidate_index - 1][target_index]
                    match_cost = costs[candidate_index - 1][target_index - 1] + abs(
                        predictions[candidate_index - 1] - targets[target_index - 1]
                    )
                    if match_cost <= skip_cost:
                        costs[candidate_index][target_index] = match_cost
                        take[candidate_index][target_index] = True
                    else:
                        costs[candidate_index][target_index] = skip_cost

            candidate_index = candidate_count
            target_index = target_count
            matches: list[tuple[int, int]] = []
            while target_index > 0:
                if take[candidate_index][target_index]:
                    matches.append((candidate_index - 1, target_index - 1))
                    candidate_index -= 1
                    target_index -= 1
                else:
                    candidate_index -= 1
            for candidate_index, target_index in matches:
                existence_targets[batch_index, candidate_index] = 1.0
                position_targets[batch_index, candidate_index] = true_knots[
                    batch_index
                ][true_mask[batch_index]][target_index]
                matched_mask[batch_index, candidate_index] = True

        return existence_targets, position_targets, matched_mask

    @staticmethod
    def _fit_loss(reconstructed: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return (reconstructed - points).pow(2).sum(dim=-1).mean()

    @staticmethod
    def _orthogonal_loss(
        reconstructed: torch.Tensor,
        points: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        residual = reconstructed - points
        numerator = (residual * tangent).sum(dim=-1)
        denominator = tangent.norm(dim=-1).clamp_min(1e-8)
        return (numerator / denominator).pow(2).mean()

    def _gap_loss(
        self,
        internal_knots: torch.Tensor,
        knot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if internal_knots.shape[-1] == 0:
            return internal_knots.new_zeros(())
        if knot_mask is not None:
            losses: list[torch.Tensor] = []
            for batch_index in range(internal_knots.shape[0]):
                knots = internal_knots[batch_index, knot_mask[batch_index]]
                if knots.numel() == 0:
                    continue
                gaps = torch.cat(
                    [knots[:1], knots[1:] - knots[:-1], 1.0 - knots[-1:]]
                )
                losses.append(torch.relu(self.min_knot_gap - gaps).pow(2).mean())
            return torch.stack(losses).mean() if losses else internal_knots.new_zeros(())
        boundary_left = internal_knots[:, :1]
        interior = internal_knots[:, 1:] - internal_knots[:, :-1]
        boundary_right = 1.0 - internal_knots[:, -1:]
        gaps = torch.cat([boundary_left, interior, boundary_right], dim=-1)
        return torch.relu(self.min_knot_gap - gaps).pow(2).mean()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        points: torch.Tensor,
        chord_params: torch.Tensor | None = None,
        true_params: torch.Tensor | None = None,
        true_internal_knots: torch.Tensor | None = None,
        true_internal_knot_mask: torch.Tensor | None = None,
        l0_scale: float = 1.0,
        activity_scale: float = 1.0,
        binary_scale: float = 1.0,
        activity_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 <= activity_threshold <= 1.0:
            raise ValueError("activity_threshold must lie in [0, 1]")
        fit_loss = self._fit_loss(output["reconstructed_points"], points)
        count_conditioned = "count_logits" in output

        if count_conditioned:
            l0_probability = None
            l0_loss = output["expected_knot_count"].mean()
            activity_loss = points.new_zeros(())
            binary_loss = points.new_zeros(())
        else:
            # Historical Hard-Concrete/soft-gate objective.
            l0_probability = output.get("l0_probability", output["activity"])
            l0_loss = l0_probability.sum(dim=-1).mean()
            if output["activity"].shape[-1] == 0:
                activity_loss = output["activity"].new_zeros(())
                binary_loss = output["activity"].new_zeros(())
            else:
                activity_loss = output["activity"].mean()
                binary_loss = (
                    output["activity"] * (1.0 - output["activity"])
                ).mean()
        gap_loss = self._gap_loss(
            output["internal_knots"],
            output.get("knot_mask"),
        )

        if chord_params is None:
            parameter_prior_loss = torch.zeros(
                (), device=points.device, dtype=points.dtype
            )
        else:
            parameter_prior_loss = (output["params"] - chord_params).pow(2).mean()

        if true_params is None:
            true_parameter_loss = torch.zeros(
                (), device=points.device, dtype=points.dtype
            )
        else:
            if true_params.shape != output["params"].shape:
                raise ValueError(
                    "true_params and predicted params must have identical shape"
                )
            true_parameter_loss = (output["params"] - true_params).pow(2).mean()

        if true_internal_knots is None and true_internal_knot_mask is None:
            existence_loss = points.new_zeros(())
            knot_position_loss = points.new_zeros(())
            count_loss = points.new_zeros(())
            over_count_loss = points.new_zeros(())
            existence_true_positive_count = points.new_zeros(())
            existence_predicted_count = points.new_zeros(())
            existence_target_count = points.new_zeros(())
            count_correct = points.new_zeros(())
            count_absolute_error = points.new_zeros(())
        elif true_internal_knots is None or true_internal_knot_mask is None:
            raise ValueError(
                "true_internal_knots and true_internal_knot_mask must be provided together"
            )
        else:
            true_count = true_internal_knot_mask.to(torch.long).sum(dim=-1)
            if count_conditioned:
                existence_loss = points.new_zeros(())
                existence_true_positive_count = points.new_zeros(())
                existence_predicted_count = points.new_zeros(())
                existence_target_count = points.new_zeros(())
                if "count_ordinal_logits" in output:
                    thresholds = torch.arange(
                        1,
                        output["count_ordinal_logits"].shape[-1] + 1,
                        device=points.device,
                    )
                    ordinal_targets = (
                        true_count.unsqueeze(-1) >= thresholds.unsqueeze(0)
                    ).to(points.dtype)
                    count_loss = F.binary_cross_entropy_with_logits(
                        output["count_ordinal_logits"], ordinal_targets
                    )
                else:
                    count_loss = F.cross_entropy(output["count_logits"], true_count)
                over_count_loss = torch.relu(
                    output["expected_knot_count"] - true_count.to(points.dtype)
                ).mean()
                predicted_count = output["predicted_knot_count"].to(torch.long)
                count_correct = (predicted_count == true_count).to(points.dtype).mean()
                count_absolute_error = (
                    predicted_count - true_count
                ).abs().to(points.dtype).mean()
                selected_count = output["count_used_for_knots"].to(torch.long)
                predicted_positions: list[torch.Tensor] = []
                target_positions: list[torch.Tensor] = []
                for batch_index in range(points.shape[0]):
                    count = int(true_count[batch_index].item())
                    if count == 0 or int(selected_count[batch_index].item()) != count:
                        continue
                    predicted_positions.append(
                        output["internal_knots"][batch_index, :count]
                    )
                    target_positions.append(
                        true_internal_knots[batch_index][
                            true_internal_knot_mask[batch_index]
                        ]
                    )
                if predicted_positions:
                    knot_position_loss = F.smooth_l1_loss(
                        torch.cat(predicted_positions),
                        torch.cat(target_positions),
                        beta=self.knot_position_beta,
                    )
                else:
                    knot_position_loss = points.new_zeros(())
            else:
                existence_targets, position_targets, matched_mask = (
                    self._ordered_knot_assignment(
                        output["internal_knots"],
                        true_internal_knots,
                        true_internal_knot_mask,
                    )
                )
                if output["activity"].numel() == 0:
                    existence_loss = points.new_zeros(())
                    knot_position_loss = points.new_zeros(())
                else:
                    probability_logits = output.get("activity_probability_logits")
                    if probability_logits is None:
                        probability_logits = torch.logit(
                            output["activity"].clamp(1e-6, 1.0 - 1e-6)
                        )
                    existence_loss = F.binary_cross_entropy_with_logits(
                        probability_logits, existence_targets
                    )
                    if matched_mask.any():
                        knot_position_loss = F.smooth_l1_loss(
                            output["internal_knots"][matched_mask],
                            position_targets[matched_mask],
                            beta=self.knot_position_beta,
                        )
                    else:
                        knot_position_loss = points.new_zeros(())
                candidate_count = max(output["activity"].shape[-1], 1)
                true_count_float = true_count.to(points.dtype)
                predicted_count_float = l0_probability.sum(dim=-1)
                count_loss = (
                    (
                        (predicted_count_float - true_count_float)
                        / candidate_count
                    )
                    .pow(2)
                    .mean()
                )
                over_count_loss = points.new_zeros(())
                existence_prediction = output["activity"] >= activity_threshold
                existence_target = existence_targets.to(torch.bool)
                existence_true_positive_count = (
                    (existence_prediction & existence_target)
                    .to(points.dtype)
                    .sum(dim=-1)
                    .mean()
                )
                existence_predicted_count = (
                    existence_prediction.to(points.dtype).sum(dim=-1).mean()
                )
                existence_target_count = (
                    existence_target.to(points.dtype).sum(dim=-1).mean()
                )
                count_correct = points.new_zeros(())
                count_absolute_error = (
                    predicted_count_float - true_count_float
                ).abs().mean()

        total = (
            self.weights.fit * fit_loss
            + l0_scale * self.weights.l0 * l0_loss
            + activity_scale * self.weights.activity * activity_loss
            + binary_scale * self.weights.binary * binary_loss
            + self.weights.gap * gap_loss
            + self.weights.parameter_prior * parameter_prior_loss
            + self.weights.true_parameter * true_parameter_loss
            + self.weights.existence * existence_loss
            + self.weights.knot_position * knot_position_loss
            + self.weights.count * count_loss
            + self.weights.over_count * over_count_loss
        )
        losses = {
            "loss": total,
            "fit_loss": fit_loss,
            "l0_loss": l0_loss,
            "expected_active_count": l0_loss,
            "activity_loss": activity_loss,
            "binary_loss": binary_loss,
            "gap_loss": gap_loss,
            "parameter_prior_loss": parameter_prior_loss,
            "true_parameter_loss": true_parameter_loss,
            "existence_loss": existence_loss,
            "knot_position_loss": knot_position_loss,
            "count_loss": count_loss,
            "over_count_loss": over_count_loss,
            "existence_true_positive_count": existence_true_positive_count,
            "existence_predicted_count": existence_predicted_count,
            "existence_target_count": existence_target_count,
            "count_accuracy": count_correct,
            "count_absolute_error": count_absolute_error,
        }
        if self.weights.orthogonal != 0.0:
            if "first_derivative" not in output:
                raise KeyError(
                    "first_derivative is required only when loading a historical "
                    "checkpoint with a non-zero orthogonal loss weight"
                )
            orthogonal_loss = self._orthogonal_loss(
                output["reconstructed_points"],
                points,
                output["first_derivative"],
            )
            losses["orthogonal_loss"] = orthogonal_loss
            losses["loss"] = total + self.weights.orthogonal * orthogonal_loss
        return losses
