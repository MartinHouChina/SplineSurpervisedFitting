"""Targeted deep-search labels must remain feasible and mass-TopK aligned."""
from types import SimpleNamespace

import pytest
import torch

from spline_fitting.losses.v16_subset_loss import V16SubsetLoss
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork
from test_v16_compact_loss import GeometryOracle, _greedy_case, _line


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _priority_inputs():
    counts = torch.tensor([6, 7, 8, 8, 6, 8, 8, 8])
    mask = torch.arange(8).unsqueeze(0) < counts.unsqueeze(-1)
    mse = torch.tensor([.98, .1, .02, 2., .2, .99, 4., .05])
    return mask, mse, torch.ones(8)


def test_disabled_priority_is_original_first_rows_and_does_not_consume_rng():
    masks, mse, tolerance = _priority_inputs()
    loss = V16SubsetLoss(teacher_greedy_max_curves=4)
    state = torch.random.get_rng_state().clone()
    rows = loss._greedy_priority_rows(masks, mse, masks, tolerance, 0)
    assert rows == [0, 1, 2, 3]
    assert torch.equal(state, torch.random.get_rng_state())


def test_compact_priority_targets_feasible_slack_excess_and_reserves_hard_exploration():
    masks, mse, tolerance = _priority_inputs()
    loss = V16SubsetLoss(teacher_greedy_max_curves=4, teacher_greedy_priority="compact")
    rows = loss._greedy_priority_rows(
        masks, mse, masks, tolerance, 0,
        supervised_counts=torch.tensor([5, 2, 6, 6, 2, 6, 6, 6]),
        supervised_valid=torch.ones(8, dtype=torch.bool),
    )
    assert rows[:3] == [1, 2, 7]
    assert rows[3] == 6  # Hardest remaining sample is not starved.
    assert len(set(rows)) == 4
    assert all(mse[row] <= tolerance[row] for row in rows[:3])


def test_priority_invalid_source_counts_are_not_used_and_budget_is_bounded():
    masks, mse, tolerance = _priority_inputs()
    loss = V16SubsetLoss(teacher_greedy_max_curves=32, teacher_greedy_priority="compact")
    expected = loss._greedy_priority_rows(masks, mse, masks, tolerance, 0)
    actual = loss._greedy_priority_rows(
        masks, mse, masks, tolerance, 0, supervised_counts=torch.zeros(8),
        supervised_valid=torch.zeros(8, dtype=torch.bool),
    )
    assert expected == actual
    assert sorted(actual) == list(range(8))


@pytest.mark.parametrize("trajectory", [False, True])
def test_targeted_search_budget_changes_only_its_selected_rows(trajectory):
    model = GeometryOracle()
    objective = V16SubsetLoss(
        teacher_greedy_steps=1, teacher_greedy_max_curves=1,
        teacher_greedy_priority="compact",
        teacher_greedy_trajectory_checks=2 if trajectory else 0,
    )
    points = _line(3)
    context = model.context(points)
    masks = torch.ones(3, 4, dtype=torch.bool)
    output = model.decode_subset(context, masks)
    mse = torch.tensor([8e-11, 1e-12, 9e-11], dtype=torch.float64)
    result, _, _, metrics = objective._greedy_teacher(
        model, context, masks, mse, masks, output,
        points, 3, torch.full((3,), 1e-10, dtype=torch.float64), 0,
    )
    assert result[1].sum() == 3
    assert torch.equal(result[[0, 2]], masks[[0, 2]])
    assert metrics["teacher_greedy_curve_fraction"] == pytest.approx(1 / 3)
    assert metrics["teacher_greedy_candidate_refits"] == 4


def _compact_context(batch=1):
    logits = torch.full((batch, 4), 2., requires_grad=True)
    requested = torch.full((batch,), 3.5, requires_grad=True)
    return {"keep_logits": logits, "one_shot_requested_count_score": requested}


def test_infeasible_compact_mask_stays_geometry_only_with_detached_labels():
    model = GeometryOracle(drift=True)
    objective, points, context, original, tolerance, result = _greedy_case(
        model, minimum=1, teacher_greedy_priority="compact",
        teacher_compact_mask_weight=1., teacher_greedy_trajectory_checks=4,
        teacher_geometry_trajectory_targets=2,
    )
    mask, mse, targets, _ = result
    assert torch.equal(mask, original)
    assert torch.all(mse <= tolerance)
    assert not any(target["decoded_feasible"] for target in targets)
    for target in targets:
        target["params"].requires_grad_()
        target["knots"].requires_grad_()
    selector_context = _compact_context()
    compact, metrics = objective._compact_mask_distillation(
        model, selector_context, targets, original,
    )
    assert compact == 0
    assert metrics["teacher_compact_target_count"] == 0
    geometry, violation, _ = objective._greedy_geometry_distillation(
        model, context, targets, points, 3, tolerance,
    )
    assert geometry > 0 and violation > 0
    geometry.backward()
    assert torch.isfinite(model.drift.grad) and model.drift.grad.abs() > 0
    assert torch.isfinite(model.position_shift.grad) and model.position_shift.grad.abs() > 0
    assert selector_context["keep_logits"].grad is None
    for target in targets:
        assert target["params"].grad is None
        assert target["knots"].grad is None


def test_compact_feasible_label_trains_removal_ranking_and_requested_count():
    model = GeometryOracle()
    model.one_shot_selection_policy = "mass_topk"
    objective, _, _, original, _, result = _greedy_case(
        model, minimum=1, teacher_greedy_priority="compact", teacher_compact_mask_weight=1.,
    )
    _, _, targets, _ = result
    target_mask = targets[0]["mask"]
    assert target_mask.sum() == 1 and targets[0]["decoded_feasible"]
    context = _compact_context()
    loss, metrics = objective._compact_mask_distillation(model, context, targets, original)
    assert loss > 0 and torch.isfinite(loss)
    assert metrics["teacher_compact_target_count"] == 1
    assert metrics["teacher_compact_target_k_mean"] == 1
    loss.backward()
    assert context["one_shot_requested_count_score"].grad > 0  # SGD lowers excess mass.
    grad = context["keep_logits"].grad
    assert torch.isfinite(grad).all()
    assert grad[~target_mask].mean() > grad[target_mask].mean()


def test_compact_loss_ignores_failed_targets_and_is_not_diluted_by_unsearched_rows():
    model = SimpleNamespace(one_shot_selection_policy="mass_topk")
    objective = V16SubsetLoss()
    good = dict(row=0, mask=torch.tensor([[True, False, False, False]]),
                decoded_feasible=True, rank=(1, 1e-6))
    failed = dict(row=0, mask=torch.zeros(1, 4, dtype=torch.bool),
                  decoded_feasible=False, rank=(0, 1e-7))
    first, metrics = objective._compact_mask_distillation(
        model, _compact_context(), [good], torch.ones(1, 4, dtype=torch.bool),
    )
    second, other_metrics = objective._compact_mask_distillation(
        model, _compact_context(8), [good, failed], torch.ones(8, 4, dtype=torch.bool),
    )
    torch.testing.assert_close(first, second)
    assert metrics["teacher_compact_target_count"] == other_metrics["teacher_compact_target_count"] == 1


def test_compact_target_cannot_conflict_with_a_larger_configured_reserve():
    model = SimpleNamespace(one_shot_selection_policy="mass_topk", one_shot_safety_knots=2)
    target = dict(row=0, mask=torch.tensor([[True, False, False, False]]),
                  decoded_feasible=True, rank=(1, 1e-6))
    loss, metrics = V16SubsetLoss()._compact_mask_distillation(
        model, _compact_context(), [target], torch.ones(1, 4, dtype=torch.bool),
    )
    assert loss == 0 and metrics["teacher_compact_target_count"] == 0


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_targeted_compact_full_objective_has_finite_gradients(stage):
    torch.manual_seed(301)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=4,
        attention_heads=2, selector_layers=1, parameter_trust_enabled=True,
        one_shot_selection_policy="mass_topk", one_shot_adaptive_threshold=True,
    )
    objective = V16SubsetLoss(
        mse_tolerance=1e-4, policy_samples=2, counterfactual_edits=0,
        teacher_prefix_search_steps=2, teacher_greedy_steps=2,
        teacher_greedy_max_curves=2, teacher_geometry_distillation_weight=.2,
        teacher_greedy_trajectory_checks=2, teacher_geometry_trajectory_targets=2,
        teacher_greedy_priority="compact", teacher_compact_mask_weight=1.,
    )
    loss, metrics = objective(model, _line(3), stage=stage)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if stage == "proposal":
        assert metrics["teacher_compact_mask_loss"] == 0


def test_explicit_off_matches_default_objective_and_rng():
    torch.manual_seed(307)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=4,
        attention_heads=2, selector_layers=1,
    )
    options = dict(policy_samples=2, teacher_prefix_search_steps=2, teacher_greedy_steps=1)
    default = V16SubsetLoss(**options)
    explicit = V16SubsetLoss(**options, teacher_greedy_priority="sequential",
                              teacher_compact_mask_weight=0.)
    torch.manual_seed(311)
    first, first_metrics = default(model, _line())
    state = torch.random.get_rng_state()
    torch.manual_seed(311)
    second, second_metrics = explicit(model, _line())
    assert torch.equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())
    for name in first_metrics:
        assert torch.equal(first_metrics[name], second_metrics[name]), name


@pytest.mark.parametrize("options", [
    {"teacher_greedy_priority": "unknown"}, {"teacher_greedy_priority": None},
    {"teacher_compact_mask_weight": -1}, {"teacher_compact_mask_weight": float("nan")},
])
def test_targeted_options_are_validated(options):
    with pytest.raises(ValueError):
        V16SubsetLoss(**options)
