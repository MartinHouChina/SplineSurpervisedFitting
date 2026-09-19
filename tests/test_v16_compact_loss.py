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


def _greedy_case(model, *, batch=1, minimum=0, steps=8, max_curves=2):
    objective = V16SubsetLoss(teacher_greedy_steps=steps, teacher_greedy_max_curves=max_curves)
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
def test_compact_forward_is_finite_differentiable_and_zero_safe_by_stage(stage):
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
])
def test_compact_configuration_validation(options):
    with pytest.raises(ValueError):
        V16SubsetLoss(**options)
