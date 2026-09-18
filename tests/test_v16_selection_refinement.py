from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.losses.v16_subset_loss import V16SubsetLoss
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def curve_batch():
    t = torch.linspace(0, 1, 32)
    return torch.stack([torch.stack([t, 0.3 * torch.sin(9 * t)], -1)])


class SmallModel(nn.Module):
    degree = 3

    def __init__(self, logits=(1.0, 0.5, -0.5), minimum=1):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([logits]))
        self.min_selected_knots = minimum
        self.decode_grad_modes = []

    def encode_candidates(self, points, mse_tolerance):
        return {
            "proposal_params": torch.linspace(0, 1, points.shape[1]).unsqueeze(0),
            "proposal_internal_knots": torch.linspace(
                0.2, 0.7, self.logits.shape[1],
            ).unsqueeze(0),
            "keep_logits": self.logits,
        }

    def select_mask(self, context):
        return context["keep_logits"] >= 0

    def decode_subset(self, context, mask):
        self.decode_grad_modes.append(torch.is_grad_enabled())
        return {
            "params": context["proposal_params"],
            "internal_knots": context["proposal_internal_knots"],
            "learned_keep_mask": mask,
        }


class MaskTableLoss(V16SubsetLoss):
    def __init__(self, table, **kwargs):
        super().__init__(**kwargs)
        self.table = torch.tensor(table, dtype=torch.float64)
        self.evaluated_masks = []
        self.decode_grad_modes = []

    def _decode_mse(self, model, context, mask, points, degree):
        self.evaluated_masks.append(mask.clone())
        self.decode_grad_modes.append(torch.is_grad_enabled())
        codes = (mask.long() * 2 ** torch.arange(mask.shape[1])).sum(-1)
        return self.table[codes]


def refine(objective, model, mask):
    points = curve_batch()
    context = model.encode_candidates(points, mse_tolerance=1.0)
    initial_mse = objective._decode_mse(model, context, mask, points, 3)
    objective.evaluated_masks.clear()
    objective.decode_grad_modes.clear()
    return objective._refine_teacher(
        model, context, mask, initial_mse, model.logits.sigmoid(),
        points, 3, torch.ones(1), model.min_selected_knots,
    )


def test_real_endpoint_refit_swap_improves_teacher_at_minimum_without_gradients():
    t = torch.linspace(0, 1, 64, dtype=torch.float64)
    points = torch.stack([t, 10 * torch.relu(t - 0.7) ** 3], -1).unsqueeze(0)
    model = SmallModel(logits=(1.0, -1.0), minimum=1)
    context = {
        "proposal_params": t.unsqueeze(0),
        "proposal_internal_knots": torch.tensor(
            [[0.2, 0.7]], dtype=torch.float64, requires_grad=True,
        ),
    }
    objective = V16SubsetLoss(teacher_refinement_steps=1, teacher_refinement_candidates=1)
    initial = torch.tensor([[True, False]])
    initial_mse = objective._decode_mse(model, context, initial, points, 3)
    assert initial_mse.item() > 1e-10
    model.decode_grad_modes.clear()
    selected, mse, diagnostics = objective._refine_teacher(
        model, context, initial, initial_mse, model.logits.sigmoid(),
        points, 3, initial_mse.detach() * 2, minimum=1,
    )
    assert torch.equal(selected, torch.tensor([[False, True]]))
    assert selected.sum() == 1
    assert mse.item() < initial_mse.item() * 1e-5
    assert not mse.requires_grad
    assert model.decode_grad_modes and not any(model.decode_grad_modes)
    assert diagnostics["teacher_refinement_evaluations"] == 2
    assert diagnostics["teacher_refinement_improved_fraction"] == 1
    assert diagnostics["teacher_refinement_count_reduction"] == 0
    assert diagnostics["teacher_refinement_feasible_fraction"] == 1


def test_refinement_preserves_infeasible_fallback_even_if_smaller_error_is_available():
    model = SmallModel()
    objective = MaskTableLoss(
        [9, 2, 3, 4, 2, 3, 4, 8], teacher_refinement_steps=3,
    )
    initial = torch.ones(1, 3, dtype=torch.bool)
    selected, mse, diagnostics = refine(objective, model, initial)
    assert torch.equal(selected, initial)
    assert mse.item() == 8
    assert diagnostics["teacher_refinement_improved_fraction"] == 0
    assert diagnostics["teacher_refinement_steps_used"] == 1
    assert diagnostics["teacher_refinement_feasible_fraction"] == 0


def test_feasible_lower_count_wins_over_lower_error_and_rounds_refine_new_incumbent():
    model = SmallModel()
    objective = MaskTableLoss(
        [9, 0.9, 9, 0.4, 9, 9, 9, 0.01],
        teacher_refinement_steps=2, teacher_refinement_candidates=1,
    )
    initial = torch.ones(1, 3, dtype=torch.bool)
    selected, mse, diagnostics = refine(objective, model, initial)
    assert torch.equal(selected, torch.tensor([[True, False, False]]))
    assert mse.item() == pytest.approx(0.9)
    assert diagnostics["teacher_refinement_steps_used"] == 2
    assert diagnostics["teacher_refinement_count_reduction"] == 2
    assert diagnostics["teacher_refinement_mse_gain"] < 0
    assert not any(objective.decode_grad_modes)


def test_refinement_uses_actual_constrained_masks_and_respects_minimum():
    class AnchoredModel(SmallModel):
        def constrain_selection_mask(self, context, mask):
            result = mask.clone()
            result[:, 0] = True
            return result

    model = AnchoredModel(logits=(0.1, 1.0, -1.0), minimum=2)
    # The unconstrained swap {1,2} would pass but violates the anchor.
    # Actual constrained {0,1,2} has larger K and must not replace {0,1}.
    objective = MaskTableLoss(
        [9, 9, 9, 0.8, 9, 9, 0.001, 0.01],
        teacher_refinement_steps=1, teacher_refinement_candidates=1,
    )
    initial = torch.tensor([[True, True, False]])
    selected, _, diagnostics = refine(objective, model, initial)
    assert torch.equal(selected, initial)
    assert diagnostics["teacher_refinement_evaluations"] == 1  # duplicate constrained swap skipped
    assert all(bool(mask[:, 0].all()) for mask in objective.evaluated_masks)
    assert all(bool((mask.sum(-1) >= 2).all()) for mask in objective.evaluated_masks)


def test_feasible_alternative_can_rescue_an_infeasible_teacher():
    model = SmallModel(logits=(1.0, -1.0), minimum=1)
    objective = MaskTableLoss(
        [9, 2, 0.5, 0.1], teacher_refinement_steps=1, teacher_refinement_candidates=1,
    )
    selected, mse, diagnostics = refine(objective, model, torch.tensor([[True, False]]))
    assert torch.equal(selected, torch.tensor([[False, True]]))
    assert mse.item() == 0.5
    assert diagnostics["teacher_refinement_feasible_fraction"] == 1


def test_boundary_ranking_focuses_weakest_keep_strongest_reject_and_correct_gradients():
    objective = V16SubsetLoss(boundary_ranking_candidates=1)
    logits = torch.tensor([[5.0, -1.0, 2.0, -5.0]], requires_grad=True)
    teacher = torch.tensor([[True, True, False, False]])
    loss = objective._boundary_ranking_loss(logits, teacher, torch.tensor([True]))
    loss.backward()
    assert logits.grad[0, 1] < 0  # protect weak kept knot
    assert logits.grad[0, 2] > 0  # demote strongest incorrect competitor
    assert logits.grad[0, 0] == logits.grad[0, 3] == 0


def test_joint_auxiliary_uses_refined_feasible_teacher_and_configured_weight():
    model = SmallModel()
    objective = MaskTableLoss(
        [9, 2, 9, 0.8, 9, 0.5, 9, 0.1],
        mse_tolerance=1.0, policy_samples=2, counterfactual_edits=0,
        fit_weight=0, dense_weight=0, policy_weight=0, distillation_weight=0,
        count_weight=0, ranking_weight=0, entropy_weight=0, complexity_weight=0,
        teacher_refinement_steps=1, teacher_refinement_candidates=1,
        boundary_ranking_weight=0.7, boundary_ranking_candidates=1,
    )
    loss, metrics = objective(model, curve_batch())
    loss.backward()
    assert loss.item() == pytest.approx(0.7 * metrics["teacher_boundary_ranking_loss"].item())
    assert metrics["teacher_refinement_improved_fraction"] == 1
    assert metrics["teacher_keep_recall"] == 0.5
    assert metrics["teacher_false_remove_rate"] == 0.5
    # The accepted swap replaces slot 1 by slot 2. Only that hardest boundary
    # receives this auxiliary's gradient; source geometry remains a real refit.
    assert model.logits.grad[0, 1] > 0
    assert model.logits.grad[0, 2] < 0
    assert model.logits.grad[0, 0] == 0


@pytest.mark.parametrize("teacher", [[True, False], [True, True], [False, False]])
def test_boundary_ranking_excludes_infeasible_teachers_and_empty_boundaries(teacher):
    objective = V16SubsetLoss()
    logits = torch.tensor([[0.2, -0.2]], requires_grad=True)
    loss = objective._boundary_ranking_loss(
        logits, torch.tensor([teacher]), torch.tensor([False]),
    )
    loss.backward()
    assert loss == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    if all(teacher) or not any(teacher):
        assert objective._boundary_ranking_loss(
            logits, torch.tensor([teacher]), torch.tensor([True]),
        ) == 0


def test_teacher_keep_metrics_measure_feasible_teacher_only():
    metrics = V16SubsetLoss._teacher_match_metrics(
        torch.tensor([[True, False, True], [False, False, False]]),
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.tensor([True, False]),
    )
    for name in (
        "teacher_keep_precision", "teacher_keep_recall", "teacher_keep_f1",
        "teacher_false_remove_rate", "teacher_match_valid_fraction",
    ):
        assert metrics[name] == 0.5


def test_disabled_options_leave_loss_gradients_rng_and_original_metrics_exactly_unchanged():
    results = []
    for kwargs in ({}, {"teacher_refinement_steps": 0, "boundary_ranking_weight": 0.0}):
        torch.manual_seed(79)
        model = SmallModel()
        objective = MaskTableLoss([9, 0.8, 2, 0.5, 4, 2, 1.5, 0.1], **kwargs)
        with patch.object(objective, "_refine_teacher", side_effect=AssertionError("disabled")):
            with patch.object(objective, "_boundary_ranking_loss", side_effect=AssertionError("disabled")):
                loss, metrics = objective(model, curve_batch(), mse_tolerance=1.0)
        loss.backward()
        results.append((loss.detach(), metrics, model.logits.grad, torch.get_rng_state()))
    for index in (0, 2, 3):
        assert torch.equal(results[0][index], results[1][index])
    assert results[0][1].keys() == results[1][1].keys()
    assert all(torch.equal(results[0][1][name], results[1][1][name]) for name in results[0][1])


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_tiny_real_network_forward_backward_has_finite_gradients_and_metrics(stage):
    torch.manual_seed(101)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
    )
    objective = V16SubsetLoss(
        policy_samples=2, counterfactual_edits=2, teacher_refinement_steps=2,
        teacher_refinement_candidates=2, boundary_ranking_weight=0.3,
        boundary_ranking_candidates=2,
    )
    loss, metrics = objective(model, curve_batch(), stage=stage)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() and not value.requires_grad for value in metrics.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if stage == "proposal":
        assert all(value == 0 for name, value in metrics.items() if name.startswith("teacher_"))
    else:
        assert model.keep_head.weight.grad is not None
        assert model.relocation_update.weight.grad is not None
        assert model.parameter_update[-1].weight.grad is not None
        assert metrics["teacher_refinement_evaluations"] <= 3 * 2 * 2
        assert metrics["subset_best_pass_rate"] >= metrics["deployment_pass_rate"]


@pytest.mark.parametrize("kwargs", [
    {"teacher_refinement_steps": -1}, {"teacher_refinement_steps": True},
    {"teacher_refinement_steps": 1.5}, {"teacher_refinement_candidates": 0},
    {"teacher_refinement_candidates": True}, {"boundary_ranking_weight": -1},
    {"boundary_ranking_weight": float("nan")}, {"boundary_ranking_candidates": 0},
    {"boundary_ranking_candidates": True},
])
def test_invalid_refinement_and_boundary_options(kwargs):
    with pytest.raises(ValueError):
        V16SubsetLoss(**kwargs)
