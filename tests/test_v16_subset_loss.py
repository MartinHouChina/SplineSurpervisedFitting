from __future__ import annotations

import itertools
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.losses.v16_subset_loss import (
    V16SubsetLoss,
    bernoulli_subset_policy_loss,
    select_best_subset,
    subset_cost,
)
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture(autouse=True)
def single_threaded():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def curve_batch():
    t = torch.linspace(0, 1, 28)
    return torch.stack([
        torch.stack([t, 0.3 * torch.sin(9 * t)], -1),
        torch.stack([t, 0.2 * torch.cos(7 * t)], -1),
    ])


def test_cost_prioritizes_feasibility_then_count_and_distinguishes_equal_count():
    tolerance = torch.tensor(1e-4, dtype=torch.float64)
    counts = torch.tensor([0, 5, 4, 4, 1])
    mse = torch.tensor([1.00001e-4, 1e-4, 9e-5, 2e-5, 1e-2], dtype=torch.float64)
    cost = subset_cost(mse, counts, tolerance, 5)
    assert cost[3] < cost[2] < cost[1] < cost[0] < cost[4]
    # Infeasible count changes alone give no incentive to discard useful knots.
    repeated_mse = torch.full((2,), 2e-4)
    assert torch.equal(subset_cost(repeated_mse, torch.tensor([0, 5]), tolerance, 5),
                       subset_cost(repeated_mse, torch.tensor([5, 0]), tolerance, 5))


def test_best_subset_is_feasibility_first_and_has_minimum_error_fallback():
    masks = torch.tensor([
        [[False, False, False], [True, True, True]],
        [[True, True, False], [True, False, False]],
        [[True, False, False], [False, True, False]],
        [[False, True, False], [False, False, False]],
    ])
    mse = torch.tensor([[1.01, 1.3], [0.1, 1.2], [0.9, 1.5], [0.7, 1.4]])
    chosen, indices = select_best_subset(masks, mse, torch.ones(2))
    assert torch.equal(indices, torch.tensor([3, 1]))
    assert torch.equal(chosen, masks[indices, torch.arange(2)])


def test_policy_gradient_matches_exact_expected_cost_gradient():
    """Enumerate every ordered pair of IID masks; LOO must introduce no bias."""
    logits = torch.tensor([[0.3, -0.7]], dtype=torch.float64, requires_grad=True)
    masks = torch.tensor(list(itertools.product([False, True], repeat=2)))
    mse = torch.tensor([8e-4, 8e-4, 1e-5, 1e-6], dtype=torch.float64)
    costs = subset_cost(mse, masks.sum(-1), torch.tensor(1e-4), 2)
    p = logits.sigmoid()[0]
    probabilities = torch.where(masks, p, 1 - p).prod(-1)
    expected = (probabilities * costs).sum()
    exact_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    enumerated_gradient = torch.zeros_like(logits)
    for first, second in itertools.product(range(4), repeat=2):
        pair = masks[[first, second]].unsqueeze(1)
        pair_cost = costs[[first, second]].unsqueeze(1).clone().requires_grad_()
        loss = bernoulli_subset_policy_loss(logits, pair, pair_cost)
        gradient, cost_gradient = torch.autograd.grad(
            loss, (logits, pair_cost), retain_graph=True, allow_unused=True,
        )
        assert cost_gradient is None
        pair_probability = (probabilities[first] * probabilities[second]).detach()
        enumerated_gradient += pair_probability * gradient
    assert torch.allclose(enumerated_gradient, exact_gradient, atol=1e-12, rtol=1e-12)
    assert enumerated_gradient[0, 0] < 0  # increase the useful first knot's probability
    assert enumerated_gradient[0, 1] > 0  # reduce the redundant second knot's probability


class PolicyOnlyModel(nn.Module):
    degree = 3

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[0.2, -0.2]]))

    def encode_candidates(self, points, mse_tolerance):
        return {
            "proposal_params": torch.linspace(0, 1, points.shape[1]).unsqueeze(0),
            "proposal_internal_knots": torch.tensor([[0.3, 0.7]]),
            "keep_logits": self.logits,
        }

    def select_mask(self, context):
        return context["keep_logits"] >= 0


class MinimumPolicyOnlyModel(PolicyOnlyModel):
    min_selected_knots = 2


class TableSubsetLoss(V16SubsetLoss):
    def _decode_mse(self, model, context, mask, points, degree):
        indices = 2 * mask[:, 0].long() + mask[:, 1].long()
        return torch.tensor([8e-4, 2e-4, 1e-5, 1e-6], dtype=torch.float64)[indices]


class PrefixPolicyModel(nn.Module):
    degree = 3
    min_selected_knots = 0
    one_shot_selection_policy = "mass_topk"

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
        ]))

    def encode_candidates(self, points, mse_tolerance):
        batch = points.shape[0]
        return {
            "proposal_params": torch.linspace(0, 1, points.shape[1]).repeat(batch, 1),
            "proposal_internal_knots": torch.tensor(
                [[0.2, 0.4, 0.6, 0.8]],
            ).repeat(batch, 1),
            "keep_logits": self.logits[:batch],
        }

    def select_mask_at_count(self, context, counts):
        logits = context["keep_logits"]
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1, order,
            torch.arange(order.shape[1], device=order.device).expand_as(order),
        )
        return rank < counts.unsqueeze(-1)

    def select_mask(self, context):
        counts = torch.full(
            (context["keep_logits"].shape[0],), 4, dtype=torch.long,
            device=context["keep_logits"].device,
        )
        return self.select_mask_at_count(context, counts)


class PrefixTableSubsetLoss(V16SubsetLoss):
    def _decode_mse(self, model, context, mask, points, degree):
        # Curve 0 first passes at K=2.  Curve 1 remains infeasible at K=4.
        table = torch.tensor([
            [8e-4, 2e-4, 8e-5, 6e-5, 1e-5],
            [9e-4, 7e-4, 5e-4, 3e-4, 2e-4],
        ], dtype=torch.float64, device=mask.device)
        rows = torch.arange(mask.shape[0], device=mask.device)
        return table[rows, mask.sum(-1)]


class GuidedNonmonotonePrefixLoss(V16SubsetLoss):
    def _decode_mse(self, model, context, mask, points, degree):
        # K=1 and K=4 pass, while the intervening counts fail.  A monotone
        # full-to-half search misses K=1 unless the certified true K is probed.
        table = torch.tensor(
            [8e-4, 8e-5, 3e-4, 2e-4, 1e-5],
            dtype=torch.float64,
            device=mask.device,
        )
        return table[mask.sum(-1)]


def test_full_objective_policy_only_updates_keep_logits_and_excludes_edited_masks():
    model = PolicyOnlyModel()
    objective = TableSubsetLoss(
        mse_tolerance=1e-4, policy_samples=2, fit_weight=0, dense_weight=0,
        distillation_weight=0, entropy_weight=0, complexity_weight=0,
    )
    draws = torch.tensor([[[1., 0.]], [[0., 1.]]])
    module = "spline_fitting.losses.v16_subset_loss"
    with patch("torch.bernoulli", return_value=draws):
        with patch(f"{module}.bernoulli_subset_policy_loss", wraps=bernoulli_subset_policy_loss) as estimator:
            loss, metrics = objective(model, curve_batch()[:1])
    loss.backward()
    assert estimator.call_count == 1
    assert torch.equal(estimator.call_args.args[1], draws.bool())
    assert model.logits.grad is not None and torch.isfinite(model.logits.grad).all()
    assert model.logits.grad[0, 0] < 0 < model.logits.grad[0, 1]
    assert metrics["subset_best_count"] == 1


def test_minimum_count_does_not_project_iid_policy_draws():
    model = MinimumPolicyOnlyModel()
    objective = TableSubsetLoss(
        mse_tolerance=1e-4, policy_samples=2, counterfactual_edits=0,
        fit_weight=0, dense_weight=0, distillation_weight=0,
        count_weight=0, ranking_weight=0, entropy_weight=0,
        complexity_weight=0,
    )
    draws = torch.tensor([[[0., 0.]], [[1., 0.]]])
    module = "spline_fitting.losses.v16_subset_loss"
    with patch("torch.bernoulli", return_value=draws):
        with patch(
            f"{module}.bernoulli_subset_policy_loss",
            wraps=bernoulli_subset_policy_loss,
        ) as estimator:
            objective(model, curve_batch()[:1])
    assert torch.equal(estimator.call_args.args[1], draws.bool())


def test_structured_teacher_losses_supervise_count_ranking_and_false_removal():
    model = PolicyOnlyModel()
    objective = TableSubsetLoss(
        mse_tolerance=1e-4, policy_samples=2, counterfactual_edits=1,
        fit_weight=0, dense_weight=0, policy_weight=0,
        distillation_weight=1, count_weight=1, ranking_weight=1,
        entropy_weight=0, complexity_weight=0, false_remove_weight=5,
    )
    draws = torch.tensor([[[1., 0.]], [[0., 1.]]])
    with patch("torch.bernoulli", return_value=draws):
        loss, metrics = objective(model, curve_batch()[:1])
    loss.backward()
    assert metrics["structured_count_loss"] > 0
    assert metrics["teacher_ranking_loss"] > 0
    assert model.logits.grad is not None and torch.isfinite(model.logits.grad).all()
    # The feasible one-knot target is the first slot; false deletion is more
    # expensive, so its logit receives the stronger upward correction.
    assert model.logits.grad[0, 0] < 0


def test_counterfactual_global_budgets_follow_scores_not_uniform_indices():
    objective = V16SubsetLoss(policy_samples=2, counterfactual_edits=2)
    mask = torch.ones(1, 6, dtype=torch.bool)
    probabilities = torch.tensor([[0.05, 0.9, 0.1, 0.8, 0.2, 0.7]])
    trials = list(objective._counterfactual_masks(mask, probabilities, minimum=1))
    # Last budget reaches the deployment minimum. It must choose the highest
    # score instead of the historical equal-index half-capacity shortcut.
    assert torch.equal(
        trials[-1], torch.tensor([[False, True, False, False, False, False]])
    )


def test_large_capacity_counterfactual_budgets_reach_compact_sets():
    objective = V16SubsetLoss(policy_samples=2, counterfactual_edits=4)
    probabilities = torch.linspace(0.0, 1.0, 96).unsqueeze(0)
    mask = torch.ones_like(probabilities, dtype=torch.bool)
    trials = list(objective._counterfactual_masks(mask, probabilities, minimum=4))
    assert [int(trial.sum()) for trial in trials[-4:]] == [44, 20, 9, 4]


def test_ranked_prefix_teacher_finds_minimum_and_full_infeasible_fallback():
    model = PrefixPolicyModel()
    objective = PrefixTableSubsetLoss(
        mse_tolerance=1e-4, policy_samples=2,
        fit_weight=0, dense_weight=0, policy_weight=0,
        distillation_weight=0, count_weight=0, ranking_weight=0,
        entropy_weight=0, complexity_weight=0,
    )
    _, metrics = objective(model, curve_batch())
    # The first curve receives its minimum feasible K=2; the second safely
    # retains the full K=4 prefix because even that prefix misses tolerance.
    assert metrics["subset_best_count"] == pytest.approx(3.0)
    assert metrics["subset_best_pass_rate"] == pytest.approx(0.5)
    assert metrics["prefix_teacher_feasible_fraction"] == pytest.approx(0.5)
    assert metrics["prefix_teacher_fallback_fraction"] == pytest.approx(0.5)
    assert metrics["prefix_teacher_search_evaluations"] == 11


def test_certified_true_count_is_an_explicit_nonmonotone_prefix_probe():
    model = PrefixPolicyModel()
    objective = GuidedNonmonotonePrefixLoss(
        mse_tolerance=1e-4,
        policy_samples=2,
        counterfactual_edits=0,
        teacher_prefix_search_steps=1,
        fit_weight=0,
        dense_weight=0,
        policy_weight=0,
        distillation_weight=0,
        count_weight=0,
        ranking_weight=0,
        entropy_weight=0,
        complexity_weight=0,
        supervised_count_weight=0,
        supervised_over_count_weight=0,
    )
    _, metrics = objective(
        model,
        curve_batch()[:1],
        synthetic_target_count=torch.tensor([1]),
        synthetic_target_valid=torch.tensor([True]),
    )
    assert metrics["subset_best_count"] == 1
    assert metrics["subset_best_pass_rate"] == 1


def test_low_count_sweep_finds_nonmonotone_simple_prefix_without_label_guide():
    model = PrefixPolicyModel()
    objective = GuidedNonmonotonePrefixLoss(
        mse_tolerance=1e-4,
        policy_samples=2,
        counterfactual_edits=0,
        teacher_prefix_search_steps=1,
        teacher_low_count_sweep=1,
        fit_weight=0,
        dense_weight=0,
        policy_weight=0,
        distillation_weight=0,
        count_weight=0,
        ranking_weight=0,
        entropy_weight=0,
        complexity_weight=0,
        supervised_count_weight=0,
        supervised_over_count_weight=0,
    )
    _, metrics = objective(model, curve_batch()[:1])
    assert metrics["subset_best_count"] == 1
    assert metrics["subset_best_pass_rate"] == 1


def test_geometry_oracle_mask_is_score_independent_and_monotone_unique():
    objective = V16SubsetLoss(policy_samples=2)
    candidates = torch.tensor([[0.10, 0.24, 0.49, 0.76, 0.90]])
    targets = torch.tensor([[0.20, 0.80, 0.0]])
    target_mask = torch.tensor([[True, True, False]])
    oracle = objective._synthetic_oracle_candidate_mask(
        candidates, targets, target_mask, torch.tensor([True]),
    )
    assert torch.equal(
        oracle, torch.tensor([[False, True, False, True, False]])
    )
    expanded = objective._expand_teacher_mask(
        oracle,
        torch.tensor([[0.1, 0.0, 0.8, 0.0, 0.9]]),
        2,
    )
    assert torch.equal(
        expanded, torch.tensor([[False, True, True, True, True]])
    )


def test_geometry_oracle_expansion_is_not_applied_to_unlabelled_rows():
    objective = V16SubsetLoss(
        policy_samples=2,
        synthetic_geometry_oracle_teacher=True,
        oracle_teacher_extra_knots=2,
    )
    oracle = torch.tensor([
        [False, True, False, True],
        [True, False, False, False],
    ])
    scores = torch.tensor([
        [0.9, 0.1, 0.8, 0.0],
        [0.0, 0.9, 0.8, 0.7],
    ])
    valid = torch.tensor([True, False])
    expanded = objective._expand_teacher_mask(
        oracle, scores, 2, eligible=valid,
    )
    assert torch.equal(expanded[0], torch.tensor([True, True, True, True]))
    assert torch.equal(expanded[1], oracle[1])


def test_certified_count_penalizes_only_safe_valid_overprediction():
    model = PolicyOnlyModel()
    objective = TableSubsetLoss(
        mse_tolerance=1e-4, policy_samples=2,
        fit_weight=0, dense_weight=0, policy_weight=0,
        distillation_weight=0, count_weight=0, ranking_weight=0,
        entropy_weight=0, complexity_weight=0,
        supervised_over_count_weight=1,
    )
    draws = torch.tensor([[[1., 0.]], [[0., 1.]]])
    with patch("torch.bernoulli", return_value=draws):
        loss, metrics = objective(
            model, curve_batch()[:1],
            synthetic_target_count=torch.tensor([0]),
            synthetic_target_valid=torch.tensor([True]),
        )
    loss.backward()
    assert metrics["supervised_over_count_loss"] > 0
    assert metrics["supervised_count_mae"] == 1
    assert metrics["supervised_excess_count"] == 1
    assert model.logits.grad is not None
    assert torch.all(model.logits.grad > 0)

    with patch("torch.bernoulli", return_value=draws):
        _, unsafe = objective(
            model, curve_batch()[:1], mse_tolerance=1e-6,
            synthetic_target_count=torch.tensor([0]),
            synthetic_target_valid=torch.tensor([True]),
        )
    assert unsafe["supervised_over_count_loss"] == 0
    assert unsafe["supervised_count_loss"] == 0


def test_certified_symmetric_count_pull_corrects_underprediction():
    model = PolicyOnlyModel()
    objective = TableSubsetLoss(
        mse_tolerance=1e-4, policy_samples=2,
        fit_weight=0, dense_weight=0, policy_weight=0,
        distillation_weight=0, count_weight=0, ranking_weight=0,
        entropy_weight=0, complexity_weight=0,
        supervised_count_weight=1, supervised_over_count_weight=0,
    )
    draws = torch.tensor([[[1., 0.]], [[0., 1.]]])
    with patch("torch.bernoulli", return_value=draws):
        loss, metrics = objective(
            model, curve_batch()[:1],
            synthetic_target_count=torch.tensor([2]),
            synthetic_target_valid=torch.tensor([True]),
        )
    loss.backward()
    assert metrics["supervised_count_loss"] > 0
    assert metrics["supervised_count_mae"] == 1
    assert model.logits.grad is not None
    assert torch.all(model.logits.grad < 0)


def test_certified_upper_bound_does_not_pull_feasible_teacher_up_to_source_k():
    model = PolicyOnlyModel()
    objective = TableSubsetLoss(
        mse_tolerance=1e-4, policy_samples=2,
        fit_weight=0, dense_weight=0, policy_weight=0,
        distillation_weight=0, count_weight=0, ranking_weight=0,
        entropy_weight=0, complexity_weight=0,
        supervised_count_weight=1, supervised_over_count_weight=0,
        synthetic_count_role="upper_bound",
    )
    draws = torch.tensor([[[1., 0.]], [[0., 1.]]])
    with patch("torch.bernoulli", return_value=draws):
        loss, metrics = objective(
            model, curve_batch()[:1],
            synthetic_target_count=torch.tensor([2]),
            synthetic_target_valid=torch.tensor([True]),
        )
    loss.backward()
    assert metrics["deployment_pass_rate"] == 1
    assert metrics["supervised_count_loss"] == 0
    assert model.logits.grad is not None
    torch.testing.assert_close(model.logits.grad, torch.zeros_like(model.logits.grad))


def test_certified_geometry_supervises_parameter_proposal_and_relocated_knots():
    torch.manual_seed(41)
    points = curve_batch()
    target_params = torch.linspace(0, 1, points.shape[1]).repeat(2, 1)
    target_knots = torch.tensor([
        [0.24, 0.71, 0.0],
        [0.0, 0.0, 0.0],
    ])
    target_mask = torch.tensor([
        [True, True, False],
        [False, False, False],
    ])
    target_valid = torch.tensor([True, False])

    proposal_model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, min_selected_knots=2,
    )
    proposal_objective = V16SubsetLoss(
        policy_samples=2, fit_weight=0, dense_weight=0,
        true_parameter_weight=1, proposal_knot_coverage_weight=1,
        selected_knot_position_weight=0,
    )
    proposal_loss, proposal_metrics = proposal_objective(
        proposal_model, points, stage="proposal",
        target_params=target_params,
        target_internal_knots=target_knots,
        target_internal_knot_mask=target_mask,
        target_geometry_valid=target_valid,
    )
    proposal_loss.backward()
    assert proposal_metrics["true_parameter_loss"] > 0
    assert proposal_metrics["proposal_knot_coverage_loss"] >= 0
    assert proposal_model.parameter_head.mlp[-1].weight.grad is not None
    assert proposal_model.candidate_head.interval_score.weight.grad is not None

    joint_model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, min_selected_knots=2,
    )
    joint_objective = V16SubsetLoss(
        policy_samples=2, counterfactual_edits=0,
        synthetic_geometry_oracle_teacher=True,
        fit_weight=0, dense_weight=0, policy_weight=0,
        distillation_weight=0, count_weight=0, ranking_weight=0,
        entropy_weight=0, complexity_weight=0,
        supervised_over_count_weight=0, true_parameter_weight=0,
        proposal_knot_coverage_weight=0, selected_knot_position_weight=1,
    )
    joint_loss, joint_metrics = joint_objective(
        joint_model, points, stage="joint",
        target_params=target_params,
        target_internal_knots=target_knots,
        target_internal_knot_mask=target_mask,
        target_geometry_valid=target_valid,
    )
    joint_loss.backward()
    assert joint_metrics["selected_knot_position_loss"] > 0
    assert 0 <= joint_metrics["oracle_teacher_feasible_fraction"] <= 1
    assert 0 <= joint_metrics["oracle_teacher_selected_fraction"] <= 1
    assert joint_model.relocation_update.weight.grad is not None
    assert joint_model.relocation_update.weight.grad.abs().sum() > 0


def test_supervised_joint_uses_ground_truth_without_online_teacher_and_three_fits():
    """Formal Joint is a direct-label objective, not an online search loop."""
    torch.manual_seed(2026)
    points = curve_batch()
    target_params = torch.linspace(0, 1, points.shape[1]).repeat(2, 1)
    target_knots = torch.tensor([
        [0.22, 0.69, 0.0],
        [0.31, 0.76, 0.0],
    ])
    target_mask = torch.tensor([
        [True, True, False],
        [True, True, False],
    ])
    target_valid = torch.ones(2, dtype=torch.bool)
    target_count = target_mask.sum(-1)

    model = V16CandidateSelectionNetwork(
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=6,
        attention_heads=2,
        selector_layers=1,
        min_selected_knots=0,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True,
    )
    objective = V16SubsetLoss(
        mse_tolerance=1e-4,
        policy_samples=2,
        counterfactual_edits=4,
        joint_supervision="synthetic_ground_truth",
        ranked_prefix_teacher=False,
        synthetic_count_role="exact",
        complexity_weight=0,
    )

    with (
        patch.object(objective, "_fit", wraps=objective._fit) as fitted,
        patch.object(
            objective,
            "_minimum_feasible_ranked_prefix",
            side_effect=AssertionError("online prefix teacher was called"),
        ),
        patch.object(
            objective,
            "_counterfactual_masks",
            side_effect=AssertionError("online counterfactual teacher was called"),
        ),
        patch(
            "torch.bernoulli",
            side_effect=AssertionError("online policy sampling was called"),
        ),
    ):
        loss, metrics = objective(
            model,
            points,
            stage="joint",
            synthetic_target_count=target_count,
            synthetic_target_valid=target_valid,
            target_params=target_params,
            target_internal_knots=target_knots,
            target_internal_knot_mask=target_mask,
            target_geometry_valid=target_valid,
        )

    # Dense proposal, deployed mask and labelled mask: no hidden fit/search loop.
    assert fitted.call_count == 3
    assert metrics["prefix_teacher_search_evaluations"] == 0
    assert metrics["oracle_teacher_selected_fraction"] == 0
    assert metrics["subset_best_count"] == pytest.approx(2.0)
    assert metrics["supervised_keep_loss"] > 0
    assert metrics["supervised_count_loss"] >= 0
    assert torch.isfinite(loss)

    loss.backward()
    for parameter in (
        model.keep_head.weight,
        model.adaptive_threshold_head[-1].weight,
        model.relocation_update.weight,
        model.candidate_head.interval_score.weight,
        model.parameter_head.mlp[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_ordered_assignment_is_one_to_one_and_keeps_coordinate_gradients():
    objective = V16SubsetLoss(policy_samples=2)
    predicted = torch.tensor(
        [[0.10, 0.48, 0.90]], requires_grad=True,
    )
    target = torch.tensor([[0.50, 0.88]])
    loss, target_nearest, predicted_nearest = objective._selected_knot_loss(
        predicted,
        torch.ones_like(predicted, dtype=torch.bool),
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.tensor([True]),
    )
    loss.backward()
    # The optimal ordered maximum-cardinality assignment skips 0.10 and pairs
    # 0.48->0.50, 0.90->0.88.  The unmatched survivor is handled by count loss.
    assert predicted.grad is not None
    assert predicted.grad[0, 0] == 0
    assert predicted.grad[0, 1:].abs().sum() > 0
    assert float(target_nearest.detach()) == pytest.approx(0.02)
    assert float(predicted_nearest.detach()) == pytest.approx(
        (0.4 + 0.02 + 0.02) / 3
    )


def test_proposal_assignment_matches_only_target_count_when_capacity_is_larger():
    objective = V16SubsetLoss(policy_samples=2)
    predicted = torch.tensor(
        [[0.10, 0.48, 0.90]], requires_grad=True,
    )
    target = torch.tensor([[0.50, 0.88]])
    loss, matched_mae, matched_count = (
        objective._proposal_ordered_assignment_loss(
            predicted,
            target,
            torch.ones_like(target, dtype=torch.bool),
            torch.tensor([True]),
        )
    )
    loss.backward()

    assert predicted.grad is not None
    assert predicted.grad[0, 0] == 0
    assert predicted.grad[0, 1:].abs().sum() > 0
    assert float(matched_mae.detach()) == pytest.approx(0.02)
    assert float(matched_count.detach()) == pytest.approx(2.0)


def test_equal_capacity_proposal_assignment_is_strictly_one_to_one():
    objective = V16SubsetLoss(policy_samples=2)
    predicted = torch.tensor(
        [[0.10, 0.48, 0.90]], requires_grad=True,
    )
    target = torch.tensor([[0.12, 0.50, 0.88]])
    loss, matched_mae, matched_count = (
        objective._proposal_ordered_assignment_loss(
            predicted,
            target,
            torch.ones_like(target, dtype=torch.bool),
            torch.tensor([True]),
        )
    )
    loss.backward()

    assert predicted.grad is not None
    assert torch.all(predicted.grad != 0)
    assert float(matched_mae.detach()) == pytest.approx(0.02)
    assert float(matched_count.detach()) == pytest.approx(3.0)


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_proposal_assignment_weight_anchors_proposals_in_both_stages(stage):
    torch.manual_seed(71)
    points = curve_batch()
    target_params = torch.linspace(0, 1, points.shape[1]).repeat(2, 1)
    target_knots = torch.tensor([
        [0.24, 0.71, 0.0],
        [0.0, 0.0, 0.0],
    ])
    target_mask = torch.tensor([
        [True, True, False],
        [False, False, False],
    ])
    target_valid = torch.tensor([True, False])
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, min_selected_knots=2,
    )
    objective = V16SubsetLoss(
        policy_samples=2, counterfactual_edits=0,
        fit_weight=0, dense_weight=0, policy_weight=0,
        distillation_weight=0, count_weight=0, ranking_weight=0,
        entropy_weight=0, complexity_weight=0,
        supervised_count_weight=0, supervised_over_count_weight=0,
        true_parameter_weight=0, proposal_knot_coverage_weight=0,
        proposal_knot_assignment_weight=1,
        selected_knot_position_weight=0,
    )
    loss, metrics = objective(
        model, points, stage=stage,
        target_params=target_params,
        target_internal_knots=target_knots,
        target_internal_knot_mask=target_mask,
        target_geometry_valid=target_valid,
    )
    assert float(loss.detach()) == pytest.approx(
        float(metrics["proposal_knot_assignment_loss"])
    )
    loss.backward()

    assert metrics["proposal_knot_assignment_count"] == 2
    assert model.candidate_head.interval_score.weight.grad is not None
    assert model.candidate_head.interval_score.weight.grad.abs().sum() > 0


def test_training_parameterization_warp_matches_piecewise_correspondence():
    knots = torch.tensor([[0.25, 0.75]], requires_grad=True)
    source = torch.tensor([[0.0, 0.5, 1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.2, 1.0]])
    warped = V16SubsetLoss._warp_knots_to_target_parameterization(
        knots, source, target,
    )
    assert torch.allclose(warped, torch.tensor([[0.1, 0.6]]))
    warped.sum().backward()
    assert knots.grad is not None and knots.grad.abs().sum() > 0
    assert source.grad is not None and source.grad.abs().sum() > 0


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_real_network_objective_backpropagates_finite_selected_geometry_gradients(stage):
    torch.manual_seed(29)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
    )
    loss, metrics = V16SubsetLoss(policy_samples=2, counterfactual_edits=3)(
        model, curve_batch(), stage=stage,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(not value.requires_grad and torch.isfinite(value).all() for value in metrics.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    required = [model.candidate_head.interval_score.weight, model.parameter_head.mlp[-1].weight]
    if stage == "joint":
        required += [model.keep_head.weight, model.parameter_update[-1].weight,
                     model.relocation_update.weight]
        # The online target pool contains the current mask, so its feasibility
        # cannot be lower on deterministic decode than current deployment.
        assert metrics["subset_best_pass_rate"] >= metrics["deployment_pass_rate"]
    else:
        assert model.keep_head.weight.grad is None
        assert metrics["keep_count"] == 6
    for parameter in required:
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_fit_matches_deployment_and_only_selected_knots_get_fit_gradients():
    points = curve_batch().double()
    parameters = torch.linspace(0, 1, points.shape[1], dtype=torch.float64).repeat(2, 1).requires_grad_()
    knots = torch.tensor([[0.15, 0.4, 0.7], [0.2, 0.5, 0.8]], dtype=torch.float64, requires_grad=True)
    masks = torch.tensor([[True, False, True], [False, False, False]])
    mse = V16SubsetLoss()._fit(parameters, knots, masks, points, 3)
    for row in range(2):
        production = refit_bspline_control_points(
            parameters.detach()[row], points[row], knots.detach()[row, masks[row]],
            degree=3, smoothness_weight=0, control_ridge=0, interpolate_endpoints=True,
        )
        assert mse[row].item() == pytest.approx(production.fit_mse.item(), rel=1e-7, abs=1e-11)
    mse.sum().backward()
    assert torch.isfinite(parameters.grad).all() and parameters.grad.abs().sum() > 0
    assert torch.isfinite(knots.grad).all() and knots.grad[masks].abs().sum() > 0
    assert torch.equal(knots.grad[~masks], torch.zeros_like(knots.grad[~masks]))


def test_fit_penalty_is_finite_for_extreme_ratios_and_rewards_error_reduction():
    mse = torch.tensor([0., 1e-20, 1e-5, 1e20], dtype=torch.float64, requires_grad=True)
    tolerance = torch.tensor(1e-10, dtype=torch.float64)
    penalties = V16SubsetLoss._fit_penalty(mse, tolerance)
    assert torch.isfinite(penalties).all() and (penalties.diff() > 0).all()
    penalties.sum().backward()
    assert torch.isfinite(mse.grad).all() and (mse.grad > 0).all()


@pytest.mark.parametrize("kwargs", [
    {"mse_tolerance": 0}, {"mse_tolerance": float("nan")},
    {"policy_samples": 1}, {"policy_samples": True}, {"policy_samples": 2.5},
    {"counterfactual_edits": -1}, {"fit_weight": -1},
    {"supervised_count_weight": -1},
    {"supervised_over_count_weight": -1}, {"ranked_prefix_teacher": 1},
    {"teacher_prefix_search_steps": 0}, {"teacher_prefix_search_steps": True},
    {"true_parameter_weight": -1}, {"proposal_knot_coverage_weight": -1},
    {"proposal_knot_assignment_weight": -1},
    {"selected_knot_position_weight": -1}, {"knot_position_beta": 0},
    {"solver_jitter": float("inf")}, {"entropy_weight": float("nan")},
])
def test_invalid_loss_configuration(kwargs):
    with pytest.raises(ValueError):
        V16SubsetLoss(**kwargs)


@pytest.mark.parametrize("tolerance", [0, -1, float("inf"), float("nan"), [1e-4, 1e-4]])
def test_invalid_curve_tolerance(tolerance):
    with pytest.raises(ValueError, match="mse_tolerance"):
        V16SubsetLoss()(PolicyOnlyModel(), curve_batch()[:1], mse_tolerance=tolerance)


@pytest.mark.parametrize("scale", [-0.1, float("nan")])
def test_invalid_complexity_scale(scale):
    with pytest.raises(ValueError, match="complexity_scale"):
        V16SubsetLoss()(PolicyOnlyModel(), curve_batch()[:1], complexity_scale=scale)


def test_complexity_scale_may_exceed_one():
    objective = TableSubsetLoss(policy_samples=2)
    loss, _ = objective(PolicyOnlyModel(), curve_batch()[:1], complexity_scale=1.1)
    assert torch.isfinite(loss)


def test_invalid_certified_count_supervision():
    objective = TableSubsetLoss(policy_samples=2)
    model = PolicyOnlyModel()
    points = curve_batch()[:1]
    with pytest.raises(ValueError, match="requires"):
        objective(model, points, synthetic_target_valid=torch.tensor([True]))
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        objective(model, points, synthetic_target_count=torch.tensor([[1]]))
    with pytest.raises(ValueError, match="boolean"):
        objective(
            model, points, synthetic_target_count=torch.tensor([1]),
            synthetic_target_valid=torch.tensor([1]),
        )
    with pytest.raises(ValueError, match="within capacity"):
        objective(
            model, points, synthetic_target_count=torch.tensor([3]),
            synthetic_target_valid=torch.tensor([True]),
        )


def test_invalid_certified_geometry_supervision():
    objective = TableSubsetLoss(policy_samples=2)
    model = PolicyOnlyModel()
    points = curve_batch()[:1]
    params = torch.linspace(0, 1, points.shape[1]).unsqueeze(0)
    knots = torch.tensor([[0.3, 0.0]])
    mask = torch.tensor([[True, False]])
    valid = torch.tensor([True])
    with pytest.raises(ValueError, match="supplied together"):
        objective(model, points, target_params=params)
    with pytest.raises(ValueError, match="boolean"):
        objective(
            model, points, stage="proposal", target_params=params,
            target_internal_knots=knots,
            target_internal_knot_mask=mask.long(),
            target_geometry_valid=valid,
        )
    with pytest.raises(ValueError, match="strictly"):
        objective(
            model, points, stage="proposal", target_params=params.flip(-1),
            target_internal_knots=knots,
            target_internal_knot_mask=mask,
            target_geometry_valid=valid,
        )


def test_invalid_stage_points_and_geometry():
    objective = V16SubsetLoss()
    with pytest.raises(ValueError, match="stage"):
        objective(PolicyOnlyModel(), curve_batch(), stage="teacher")
    with pytest.raises(ValueError, match="points"):
        objective(PolicyOnlyModel(), curve_batch()[0])
    with pytest.raises(ValueError, match="finite"):
        objective(PolicyOnlyModel(), curve_batch() * float("nan"))
    parameters = torch.linspace(0, 1, 28).unsqueeze(0)
    knots = torch.tensor([[0.3, 0.7]])
    with pytest.raises(ValueError, match="boolean"):
        objective._fit(parameters, knots, torch.ones_like(knots), curve_batch()[:1], 3)
    with pytest.raises(ValueError, match="strictly increasing"):
        objective._fit(parameters.flip(-1), knots, knots.bool(), curve_batch()[:1], 3)
    with pytest.raises(ValueError, match="strictly inside"):
        objective._fit(parameters, torch.tensor([[0., 1.]]), knots.bool(), curve_batch()[:1], 3)


def test_invalid_policy_samples_and_costs():
    logits = torch.zeros(1, 2)
    masks = torch.zeros(2, 1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="two samples"):
        bernoulli_subset_policy_loss(logits, masks[:1], torch.zeros(1, 1))
    with pytest.raises(ValueError, match="boolean"):
        bernoulli_subset_policy_loss(logits, masks.float(), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="finite"):
        bernoulli_subset_policy_loss(logits, masks, torch.full((2, 1), float("nan")))
    with pytest.raises(ValueError, match="positive integer"):
        subset_cost(torch.zeros(1), torch.zeros(1), torch.ones(1), 1.5)
