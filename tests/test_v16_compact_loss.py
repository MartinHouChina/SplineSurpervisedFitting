from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from spline_fitting.losses.v16_subset_loss import V16SubsetLoss
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _line(batch=1):
    t = torch.linspace(0, 1, 28)
    return torch.stack((t, .2 * t), -1).repeat(batch, 1, 1)


class GeometryOracle(nn.Module):
    degree = 3
    min_selected_knots = 0

    def __init__(self, drift=False, anchor=False):
        super().__init__()
        self.drift = nn.Parameter(torch.tensor(.65 if drift else 0.0))
        self.position_shift = nn.Parameter(torch.tensor(.04 if drift else 0.0))
        self.anchor = anchor

    def context(self, points):
        return {
            "proposal_params": torch.linspace(0, 1, points.shape[1]).repeat(len(points), 1),
            "proposal_internal_knots": torch.tensor([[.2, .4, .6, .8]]).repeat(len(points), 1),
        }

    def constrain_selection_mask(self, context, mask):
        result = mask.clone()
        if self.anchor:
            result[:, 0] = True
        return result

    def decode_subset(self, context, mask):
        t = context["proposal_params"]
        changed = (mask.sum(-1) < 4).to(t.dtype).unsqueeze(-1)
        return {
            "params": t + self.drift * changed * t * (1 - t),
            "internal_knots": context["proposal_internal_knots"] + changed * self.position_shift,
            "learned_keep_mask": mask,
        }


def test_feasible_fit_is_continuous_monotone_and_has_weaker_safe_gradient():
    objective = V16SubsetLoss(feasible_objective=True, feasible_fit_margin=.8,
                              feasible_fit_weight=.02)
    tolerance = torch.tensor(5e-5, dtype=torch.float64)
    ratios = torch.tensor([0., .1, .5, .8 - 1e-8, .8, .8 + 1e-8, 1., 2.], dtype=torch.float64)
    mse = (ratios * tolerance).requires_grad_()
    compact = objective._objective_fit_penalty(mse, tolerance)
    legacy = objective._fit_penalty(mse, tolerance)
    assert torch.isfinite(compact).all()
    assert (compact.diff() >= 0).all()
    assert abs(float((compact[5] - compact[3]).detach())) < 1e-7
    compact_grad = torch.autograd.grad(compact.sum(), mse, retain_graph=True)[0]
    legacy_grad = torch.autograd.grad(legacy.sum(), mse)[0]
    torch.testing.assert_close(compact_grad[1:3], .02 * legacy_grad[1:3])
    torch.testing.assert_close(compact_grad[5:], legacy_grad[5:])
    assert compact[-1] > compact[-2] > compact[4]
    original = V16SubsetLoss()
    assert torch.equal(original._objective_fit_penalty(mse, tolerance), legacy)


@pytest.mark.parametrize("sigma,reserve", [(0., 0.), (.2, 0.), (0., 2.), (.2, 2.)])
def test_soft_mask_targets_align_probability_mass_and_requested_count(sigma, reserve):
    counts = torch.tensor([0, 1, 4, 8])
    mask = torch.arange(8).unsqueeze(0) < counts.unsqueeze(-1)
    targets, feasible = V16SubsetLoss._count_aligned_targets(
        mask, sigma=sigma, reserve=reserve, dtype=torch.float64,
    )
    score = targets.sum(-1) + sigma * (targets * (1 - targets)).sum(-1).sqrt() + reserve
    expected = (counts.double() - .25).clamp_min(0)
    torch.testing.assert_close(score[feasible], expected[feasible], atol=1e-10, rtol=0)
    assert torch.equal(score[feasible].ceil().long(), counts[feasible])
    assert torch.equal(targets[~mask], torch.zeros_like(targets[~mask]))
    assert torch.equal(feasible, expected >= reserve)
    assert torch.equal(targets[~feasible], mask[~feasible].double())
    assert not targets.requires_grad


def test_aligned_bce_does_not_push_mass_to_teacher_k_plus_reserve():
    mask = torch.tensor([[True, False, True, True, False, True]])
    target, _ = V16SubsetLoss._count_aligned_targets(mask, sigma=0., reserve=2., dtype=torch.float64)
    logits = torch.logit(target.clamp(1e-12, 1 - 1e-12)).requires_grad_()
    weight = 1 + 4 * mask.double()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weight)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert gradient.abs().max() < 1e-10
    assert float(target.sum() + 2) == pytest.approx(3.75)


@pytest.mark.parametrize("minimum,anchor,expected,scans", [
    (0, False, 0, 10), (2, False, 2, 7), (0, True, 1, 9),
])
def test_fixed_geometry_exhaustively_deletes_redundancy_with_exact_budget(minimum, anchor, expected, scans):
    model = GeometryOracle(anchor=anchor)
    objective = V16SubsetLoss(teacher_greedy_steps=8)
    points = _line()
    context = model.context(points)
    mask = torch.ones(1, 4, dtype=torch.bool)
    result, mse, scan_count, verification_count = objective._fixed_geometry_greedy(
        model, context, context["proposal_params"], context["proposal_internal_knots"],
        mask, points, 3, torch.tensor([1e-10]), minimum,
    )
    assert int(result.sum()) == expected
    assert mse <= 1e-10
    assert scan_count == scans
    assert verification_count == 1 + 4 - expected
    assert not mse.requires_grad
    if anchor:
        assert result[0, 0]


def _greedy_case(model, *, batch=1, minimum=0, steps=8, max_curves=2, **options):
    objective = V16SubsetLoss(teacher_greedy_steps=steps, teacher_greedy_max_curves=max_curves,
                             **options)
    points = _line(batch)
    context = model.context(points)
    mask = torch.ones(batch, 4, dtype=torch.bool)
    output = model.decode_subset(context, mask)
    mse = objective._fit(output["params"], output["internal_knots"], mask, points, 3)
    tolerance = torch.full((batch,), 1e-10, dtype=torch.float64)
    result = objective._greedy_teacher(model, context, mask, mse, mask, output,
                                       points, 3, tolerance, minimum)
    return objective, points, context, mask, tolerance, result


def test_decoded_feasible_compact_target_can_replace_old_mask_teacher():
    _, _, _, _, _, (mask, mse, targets, metrics) = _greedy_case(GeometryOracle())
    assert not mask.any()
    assert mse <= 1e-10
    assert len(targets) == 1
    assert metrics["teacher_greedy_fixed_count"] == 0
    assert metrics["teacher_greedy_fixed_pass_rate"] == 1
    assert metrics["teacher_greedy_decoded_pass_rate"] == 1
    assert metrics["teacher_greedy_accepted_delta_k"] == 4
    assert metrics["teacher_greedy_refits"] == 16  # 10 batched systems + 5 checks + 1 decode fit.


def test_decoder_infeasible_compact_geometry_is_not_a_feasible_mask_label():
    model = GeometryOracle(drift=True)
    objective, points, context, original, tolerance, result = _greedy_case(model, minimum=1)
    mask, mse, targets, metrics = result
    assert torch.equal(mask, original)
    assert mse <= tolerance
    assert metrics["teacher_greedy_fixed_count"] == 1
    assert metrics["teacher_greedy_fixed_pass_rate"] == 1
    assert metrics["teacher_greedy_decoded_pass_rate"] == 0
    assert metrics["teacher_greedy_accepted_delta_k"] == 0
    assert not targets[0]["params"].requires_grad
    assert not targets[0]["knots"].requires_grad
    # Even a caller-supplied target carrying gradients must remain a teacher.
    targets[0]["params"].requires_grad_()
    targets[0]["knots"].requires_grad_()
    loss, violation, refits = objective._greedy_geometry_distillation(
        model, context, targets, points, 3, tolerance,
    )
    assert loss > 0 and violation > 0 and refits == 1
    loss.backward()
    assert torch.isfinite(model.drift.grad) and model.drift.grad.abs() > 0
    assert torch.isfinite(model.position_shift.grad) and model.position_shift.grad.abs() > 0
    assert targets[0]["params"].grad is None
    assert targets[0]["knots"].grad is None


def test_curve_and_step_budgets_do_not_change_unsearched_rows():
    _, _, _, original, _, (mask, _, _, metrics) = _greedy_case(
        GeometryOracle(), batch=2, steps=1, max_curves=1,
    )
    assert mask[0].sum() == 3
    assert torch.equal(mask[1], original[1])
    assert metrics["teacher_greedy_curve_fraction"] == .5
    assert metrics["teacher_greedy_candidate_refits"] == 4
    assert metrics["teacher_greedy_accepted_delta_k"] == .5


@pytest.mark.parametrize("stage", ["proposal", "joint"])
@pytest.mark.parametrize("trajectory", [False, True])
def test_compact_forward_is_finite_differentiable_and_zero_safe_by_stage(stage, trajectory):
    torch.manual_seed(101)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=4,
        attention_heads=2, selector_layers=1, parameter_trust_enabled=True,
        one_shot_selection_policy="mass_topk", one_shot_adaptive_threshold=True,
    )
    objective = V16SubsetLoss(
        mse_tolerance=1e-4, policy_samples=2, counterfactual_edits=0,
        teacher_prefix_search_steps=2, teacher_greedy_steps=2,
        teacher_greedy_max_curves=1, teacher_geometry_distillation_weight=.2,
        feasible_objective=True, count_reserve_alignment=True,
        teacher_greedy_trajectory_checks=4 if trajectory else 0,
        teacher_geometry_trajectory_targets=2 if trajectory else 0,
    )
    points = _line(2)
    loss, metrics = objective(model, points, stage=stage)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert metrics["count_reserve_infeasible_fraction"] == 0
    if stage == "proposal":
        assert metrics["teacher_greedy_refits"] == 0
        assert metrics["teacher_geometry_aux_refits"] == 0
        assert metrics["complexity_active_fraction"] == 0
        assert metrics["teacher_greedy_trajectory_checks"] == 0
    else:
        assert metrics["teacher_greedy_curve_fraction"] == .5
        assert metrics["teacher_greedy_refits"] > 0
        assert metrics["complexity_active_fraction"] == 1


def test_default_loss_never_calls_new_search_and_explicit_off_is_identical():
    torch.manual_seed(107)
    model = V16CandidateSelectionNetwork(hidden_dim=16, encoder_layers=1,
        max_internal_knots=4, attention_heads=2, selector_layers=1)
    old = V16SubsetLoss(policy_samples=2, teacher_prefix_search_steps=2)
    explicit = V16SubsetLoss(policy_samples=2, teacher_prefix_search_steps=2,
        feasible_objective=False, teacher_greedy_steps=0,
        teacher_geometry_distillation_weight=0, count_reserve_alignment=False)
    with patch.object(old, "_greedy_teacher", side_effect=AssertionError("disabled search")):
        torch.manual_seed(109)
        first, first_metrics = old(model, _line())
    torch.manual_seed(109)
    second, second_metrics = explicit(model, _line())
    assert torch.equal(first, second)
    assert first_metrics.keys() == second_metrics.keys()
    for name in first_metrics:
        assert torch.equal(first_metrics[name], second_metrics[name]), name


@pytest.mark.parametrize("options", [
    {"feasible_objective": 1}, {"count_reserve_alignment": 1},
    {"feasible_fit_margin": 0}, {"feasible_fit_margin": 1.1},
    {"feasible_fit_weight": -1}, {"feasible_fit_weight": float("nan")},
    {"teacher_greedy_steps": -1}, {"teacher_greedy_max_curves": 0},
    {"teacher_geometry_distillation_weight": -1},
    {"teacher_greedy_trajectory_checks": -1}, {"teacher_greedy_trajectory_checks": True},
    {"teacher_geometry_trajectory_targets": -1},
    {"teacher_geometry_trajectory_targets": 2},  # No trajectory decode checks.
])
def test_compact_configuration_validation(options):
    with pytest.raises(ValueError):
        V16SubsetLoss(**options)


class TrajectoryOracle(GeometryOracle):
    """The decoder is useful near its seed, but some smaller sets drift."""

    def __init__(self, bad_counts=(0, 1, 2)):
        super().__init__(drift=True)
        self.bad_counts = bad_counts

    def decode_subset(self, context, mask):
        t = context["proposal_params"]
        changed = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
        for count in self.bad_counts:
            changed |= mask.sum(-1) == count
        changed = changed.to(t.dtype).unsqueeze(-1)
        return {
            "params": t + self.drift * changed * t * (1 - t),
            "internal_knots": context["proposal_internal_knots"] + changed * self.position_shift,
            "learned_keep_mask": mask,
        }


@pytest.mark.parametrize("length,budget,expected", [
    (0, 4, []), (1, 4, [0]), (4, 0, []), (4, 1, [3]),
    (4, 2, [0, 3]), (4, 4, [0, 1, 2, 3]), (16, 4, [0, 5, 10, 15]),
])
def test_trajectory_probe_indices_are_bounded_and_cover_near_and_far(length, budget, expected):
    assert V16SubsetLoss._trajectory_check_indices(length, budget) == expected


def test_trajectory_records_only_verified_feasible_nested_deletions():
    model = GeometryOracle(anchor=True)
    objective = V16SubsetLoss(teacher_greedy_steps=3)
    points = _line()
    context = model.context(points)
    mask = torch.ones(1, 4, dtype=torch.bool)
    path = []
    result, mse, scans, checks = objective._fixed_geometry_greedy(
        model, context, context["proposal_params"], context["proposal_internal_knots"],
        mask, points, 3, torch.tensor([1e-10]), 1, trajectory=path,
    )
    assert len(path) == 3 and scans == 9 and checks == 4
    previous = mask
    for selected, error in path:
        assert selected[0, 0] and error <= 1e-10
        assert not error.requires_grad
        assert int((previous & ~selected).sum()) == 1
        assert not (selected & ~previous).any()
        previous = selected
    assert torch.equal(path[-1][0], result)
    assert torch.equal(path[-1][1], mse)


def test_unreachable_endpoint_does_not_hide_a_reachable_teacher_transition():
    model = TrajectoryOracle()
    old = _greedy_case(model)[-1]
    assert old[0].sum() == 4
    objective, points, context, _, tolerance, result = _greedy_case(
        model, teacher_greedy_trajectory_checks=4, teacher_geometry_trajectory_targets=2,
    )
    mask, mse, targets, metrics = result
    assert mask.sum() == 3 and mse <= tolerance
    assert metrics["teacher_greedy_fixed_count"] == 0
    assert metrics["teacher_greedy_decoded_pass_rate"] == 0  # Endpoint still fails.
    assert metrics["teacher_greedy_trajectory_checks"] == 4
    assert metrics["teacher_greedy_trajectory_pass_rate"] == .25
    assert metrics["teacher_greedy_accepted_delta_k"] == 1
    assert metrics["teacher_greedy_refits"] == 19  # 10 scans + 5 verifies + 4 actual decodes.
    assert [int(target["mask"].sum()) for target in targets] == [3, 2]
    assert [target["decoded_feasible"] for target in targets] == [True, False]
    assert metrics["teacher_geometry_target_count"] == 2
    assert metrics["teacher_geometry_target_k_mean"] == 2.5
    assert metrics["teacher_geometry_target_pass_rate"] == .5  # Both auxiliary targets.
    loss, violation, refits = objective._greedy_geometry_distillation(
        model, context, targets, points, 3, tolerance,
    )
    assert loss > 0 and violation > 0 and refits == 2
    loss.backward()
    assert torch.isfinite(model.drift.grad) and model.drift.grad.abs() > 0
    assert torch.isfinite(model.position_shift.grad) and model.position_shift.grad.abs() > 0
    assert all(not item["params"].requires_grad and not item["knots"].requires_grad for item in targets)


def test_all_unreachable_trajectory_uses_nearest_auxiliary_without_false_feasible_labels():
    _, _, _, original, tolerance, (mask, mse, targets, metrics) = _greedy_case(
        GeometryOracle(drift=True), teacher_greedy_trajectory_checks=4,
        teacher_geometry_trajectory_targets=2,
    )
    assert torch.equal(mask, original) and mse <= tolerance
    assert [int(target["mask"].sum()) for target in targets] == [3, 2]
    assert not any(target["decoded_feasible"] for target in targets)
    assert metrics["teacher_greedy_trajectory_pass_rate"] == 0
    assert metrics["teacher_greedy_accepted_delta_k"] == 0


def test_trajectory_checks_do_not_assume_monotonic_decoded_feasibility():
    _, _, _, _, tolerance, (mask, mse, targets, metrics) = _greedy_case(
        TrajectoryOracle(bad_counts=(0, 2)), teacher_greedy_trajectory_checks=4,
        teacher_geometry_trajectory_targets=2,
    )
    assert mask.sum() == 1 and mse <= tolerance  # K=2 failed; K=1 must still be checked.
    assert [int(target["mask"].sum()) for target in targets] == [1, 0]
    assert metrics["teacher_greedy_trajectory_pass_rate"] == .5


def test_trajectory_and_auxiliary_budgets_are_per_seed_and_per_curve():
    model = GeometryOracle()
    objective = V16SubsetLoss(teacher_greedy_steps=4, teacher_greedy_max_curves=2,
        teacher_greedy_trajectory_checks=2, teacher_geometry_trajectory_targets=2)
    points = _line(3)
    context = model.context(points)
    original = torch.ones(3, 4, dtype=torch.bool)
    deployment = original.clone()
    deployment[:, -1] = False
    output = model.decode_subset(context, deployment)
    tolerance = torch.full((3,), 1e-10, dtype=torch.float64)
    mask, _, targets, metrics = objective._greedy_teacher(
        model, context, original, torch.zeros(3, dtype=torch.float64), deployment,
        output, points, 3, tolerance, 0,
    )
    assert torch.equal(mask[-1], original[-1])
    assert metrics["teacher_greedy_trajectory_checks"] == 2 * 2 * 2  # Rows, seeds, probes.
    assert len(targets) <= 2 * 2  # Rows and max geometry targets.
    assert all(target["row"] < 2 for target in targets)


def test_explicit_trajectory_off_preserves_legacy_greedy_values():
    model = GeometryOracle(drift=True)
    old = _greedy_case(model)[-1]
    explicit = _greedy_case(model, teacher_greedy_trajectory_checks=0,
                            teacher_geometry_trajectory_targets=0)[-1]
    assert torch.equal(old[0], explicit[0]) and torch.equal(old[1], explicit[1])
    assert old[2][0]["rank"] == explicit[2][0]["rank"]
    for name in old[3]:
        assert torch.equal(old[3][name], explicit[3][name]), name


def test_geometry_distillation_balances_curves_with_unequal_target_counts():
    model = GeometryOracle(drift=True)
    objective = V16SubsetLoss(teacher_greedy_trajectory_checks=4,
                             teacher_geometry_trajectory_targets=2)
    points = _line(2)
    context = model.context(points)
    tolerance = torch.full((2,), 1e-10, dtype=torch.float64)
    targets = []
    for row, count in ((0, 3), (0, 2), (1, 4)):
        targets.append(dict(
            row=row, mask=(torch.arange(4).unsqueeze(0) < count),
            params=context["proposal_params"][row:row + 1].detach().clone(),
            knots=context["proposal_internal_knots"][row:row + 1].detach().clone(),
        ))
    individual = [objective._greedy_geometry_distillation(
        model, context, [target], points, 3, tolerance,
    ) for target in targets]
    loss, violation, refits = objective._greedy_geometry_distillation(
        model, context, targets, points, 3, tolerance,
    )
    expected_loss = ((individual[0][0] + individual[1][0]) / 2 + individual[2][0]) / 2
    expected_violation = ((individual[0][1] + individual[1][1]) / 2 + individual[2][1]) / 2
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(violation, expected_violation)
    pooled_loss = torch.stack([value[0] for value in individual]).mean()
    assert not torch.isclose(loss, pooled_loss)  # The two-target curve has no extra weight.
    assert refits == 3 and all(value[2] == 1 for value in individual)
    loss.backward()
    assert torch.isfinite(model.drift.grad) and model.drift.grad.abs() > 0
    assert torch.isfinite(model.position_shift.grad) and model.position_shift.grad.abs() > 0
