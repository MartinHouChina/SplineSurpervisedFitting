from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve  # noqa: E402
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    build_open_knot_vector,
)
from spline_fitting.losses import (  # noqa: E402
    CandidatePruningLoss,
    CandidatePruningLossWeights,
)
from spline_fitting.models import SplineFittingNetwork  # noqa: E402
from spline_fitting.training import (  # noqa: E402
    OneShotTeacherConfig,
    build_one_shot_teacher_batch,
    load_one_shot_teacher_cache,
)
from spline_fitting.training import one_shot_teacher as teacher_module  # noqa: E402
from scripts.train_candidate_pruning import (  # noqa: E402
    _set_one_shot_trainable,
)


def _offset_single_knot_curve() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    parameters = torch.linspace(0.0, 1.0, 121, dtype=dtype).unsqueeze(0)
    controls_x = torch.linspace(-1.0, 1.0, 5, dtype=dtype)
    controls = torch.stack(
        [
            controls_x,
            controls_x.square() + 0.7 * torch.sin(3.1 * controls_x),
        ],
        dim=-1,
    )
    points = evaluate_bspline_curve(
        parameters[0],
        controls,
        build_open_knot_vector(torch.tensor([0.5], dtype=dtype), degree=3),
        degree=3,
    ).unsqueeze(0)
    candidates = torch.tensor([[0.12, 0.42, 0.68, 0.88]], dtype=dtype)
    return parameters, points, candidates


def test_delete_then_relax_moves_survivors_and_can_delete_beyond_greedy() -> None:
    parameters, points, candidates = _offset_single_knot_curve()
    config = OneShotTeacherConfig(
        error_tolerance=1e-3,
        smoothness_weight=0.0,
        relocation_strategy="delete_then_relax",
        relocation_rounds=2,
        relocation_sweeps=2,
        relocation_grid_size=7,
        relocation_restarts=1,
        relocation_min_gap=1e-3,
    )

    teacher = build_one_shot_teacher_batch(
        parameters,
        points,
        candidates,
        sample_indices=[0],
        config=config,
    )

    assert int(teacher.teacher_count[0]) < int(teacher.teacher_greedy_count[0])
    assert int(teacher.teacher_extra_deleted_after_relocation[0]) > 0
    assert float(teacher.teacher_relocation_mean_abs[0]) > 0.01
    assert float(teacher.teacher_fit_rms[0]) < float(teacher.teacher_greedy_fit_rms[0])
    assert bool(teacher.teacher_threshold_satisfied[0])
    torch.testing.assert_close(
        teacher.teacher_fit_mse,
        teacher.teacher_fit_rms.square(),
    )

    retained_slots = teacher.teacher_retained_mask[0]
    count = int(teacher.teacher_count[0])
    original_survivors = candidates[0, retained_slots]
    relocated_survivors = teacher.teacher_internal_knots[0, :count]
    assert not torch.allclose(relocated_survivors, original_survivors)
    assert float((relocated_survivors - original_survivors).abs().max()) <= 0.15 + 1e-9


def test_v2_cache_loads_as_zero_relocation_compatibility(tmp_path: Path) -> None:
    parameters, points, candidates = _offset_single_knot_curve()
    config = OneShotTeacherConfig(error_tolerance=5e-3, smoothness_weight=0.0)
    teacher = build_one_shot_teacher_batch(
        parameters,
        points,
        candidates,
        sample_indices=[0],
        config=config,
    )
    legacy_config = config.as_dict()
    for name in (
        "relocation_strategy",
        "relocation_rounds",
        "relocation_sweeps",
        "relocation_grid_size",
        "relocation_restarts",
        "relocation_min_gap",
        "relocation_max_shift",
    ):
        legacy_config.pop(name)
    legacy_keys = (
        "teacher_retained_mask",
        "teacher_soft_keep_risk",
        "teacher_internal_knots",
        "teacher_internal_knot_mask",
        "teacher_count",
        "teacher_fit_rms",
        "teacher_threshold_satisfied",
        "teacher_deletion_order",
        "teacher_single_deletion_rms",
    )
    path = tmp_path / "legacy-v2.pt"
    torch.save(
        {
            "format": "spline_fitting.one_shot_teacher",
            "version": 2,
            "config": legacy_config,
            "config_fingerprint": teacher_module._config_fingerprint(
                legacy_config,
                version=2,
            ),
            "sample_indices": torch.tensor([0]),
            "input_shapes": {
                "parameters": list(parameters.shape),
                "points": list(points.shape),
                "candidate_knots": list(candidates.shape),
            },
            "labels": {name: getattr(teacher, name).cpu() for name in legacy_keys},
        },
        path,
    )

    loaded = load_one_shot_teacher_cache(
        path,
        expected_config=config,
        expected_sample_indices=[0],
        expected_input_shapes={
            "parameters": parameters.shape,
            "points": points.shape,
            "candidate_knots": candidates.shape,
        },
    )

    torch.testing.assert_close(loaded.teacher_greedy_count, loaded.teacher_count)
    torch.testing.assert_close(
        loaded.teacher_extra_deleted_after_relocation,
        torch.zeros_like(loaded.teacher_count),
    )
    torch.testing.assert_close(
        loaded.teacher_relocation_mean_abs,
        torch.zeros_like(loaded.teacher_fit_rms),
    )


def test_max_shift_projection_never_replaces_a_better_input_fit(monkeypatch) -> None:
    dtype = torch.float64
    initial_fit = SimpleNamespace(
        internal_knots=torch.tensor([0.5], dtype=dtype),
        fit_mse=torch.tensor(0.1, dtype=dtype),
    )
    unconstrained_fit = SimpleNamespace(
        internal_knots=torch.tensor([0.8], dtype=dtype),
        fit_mse=torch.tensor(0.01, dtype=dtype),
    )
    projected_worse_fit = SimpleNamespace(
        internal_knots=torch.tensor([0.65], dtype=dtype),
        fit_mse=torch.tensor(0.2, dtype=dtype),
    )
    monkeypatch.setattr(
        teacher_module,
        "refine_knot_positions",
        lambda *args, **kwargs: SimpleNamespace(
            initial_fit=initial_fit,
            final_fit=unconstrained_fit,
        ),
    )
    monkeypatch.setattr(
        teacher_module,
        "refit_bspline_control_points",
        lambda *args, **kwargs: projected_worse_fit,
    )

    result = teacher_module._relax_teacher_state(
        torch.linspace(0.0, 1.0, 5, dtype=dtype),
        torch.zeros(5, 2, dtype=dtype),
        initial_fit.internal_knots,
        torch.tensor([0.5], dtype=dtype),
        OneShotTeacherConfig(
            error_tolerance=5e-3,
            relocation_strategy="delete_then_relax",
            relocation_max_shift=0.15,
        ),
    )

    assert result is initial_fit


@pytest.mark.parametrize(
    ("predicted_mask", "expect_position", "expect_spacing"),
    [
        ([False, True, True], True, True),
        ([True, True, True], True, False),
        ([False, True, False], True, False),
        ([False, False, False], False, False),
    ],
)
def test_relocated_teacher_matches_actual_survivors_for_any_count(
    predicted_mask: list[bool],
    expect_position: bool,
    expect_spacing: bool,
) -> None:
    dtype = torch.float64
    refined = torch.tensor([[0.30, 0.55, 0.70]], dtype=dtype, requires_grad=True)
    logits = torch.tensor([[0.4, -0.2, 0.1]], dtype=dtype, requires_grad=True)
    probability = logits.sigmoid()
    hard_mask = torch.tensor([predicted_mask])
    points = torch.zeros(1, 9, 2, dtype=dtype)
    params = torch.linspace(0.0, 1.0, 9, dtype=dtype).unsqueeze(0)
    output = {
        "candidate_knots": torch.tensor([[0.25, 0.50, 0.75]], dtype=dtype),
        "proposal_internal_knots": torch.tensor([[0.25, 0.50, 0.75]], dtype=dtype),
        "internal_knots": refined,
        "keep_logits": logits,
        "keep_probability": probability,
        "reconstructed_points": points.clone(),
        "params": params,
        "final_hard_keep_mask": hard_mask,
        "final_hard_st_keep_gate": (
            hard_mask.to(dtype) + probability - probability.detach()
        ),
    }
    weights = CandidatePruningLossWeights(
        fit=0.0,
        threshold_violation=0.0,
        true_parameter=0.0,
        candidate_coverage=0.0,
        candidate_repulsion=0.0,
        keep=0.0,
        remove_action=0.0,
        knot_position=1.0,
        deletion_cost=0.0,
        teacher_distribution=1.0,
    )
    loss_fn = CandidatePruningLoss(
        weights,
        candidate_match_tolerance=0.1,
        exact_deletion_supervision=False,
        position_aware_distribution=True,
        teacher_relocation_supervision=True,
        teacher_survivor_spacing_weight=1.0,
    )
    losses = loss_fn(
        output,
        points,
        true_internal_knots=torch.tensor([[0.2, 0.8, 0.0]], dtype=dtype),
        true_internal_knot_mask=torch.tensor([[True, True, False]]),
        # Deliberately different from the deployed mask: slot identity must
        # not decide which positions receive the relocation loss.
        teacher_retained_mask=torch.tensor([[True, True, False]]),
        teacher_internal_knots=torch.tensor([[0.18, 0.84, 0.0]], dtype=dtype),
        teacher_internal_knot_mask=torch.tensor([[True, True, False]]),
        teacher_count=torch.tensor([2]),
    )

    assert bool(float(losses["teacher_anchor_position_loss"].detach()) > 0.0) is (
        expect_position
    )
    assert bool(float(losses["teacher_survivor_spacing_loss"].detach()) > 0.0) is (
        expect_spacing
    )
    assert torch.isfinite(losses["loss"])
    assert float(losses["teacher_set_coverage_loss"].detach()) > 0.0

    if expect_position:
        position_gradient = torch.autograd.grad(
            losses["teacher_anchor_position_loss"],
            refined,
            retain_graph=True,
        )[0]
        selected = hard_mask.to(position_gradient.device)
        assert position_gradient[selected].abs().sum() > 0
        torch.testing.assert_close(
            position_gradient[~selected],
            torch.zeros_like(position_gradient[~selected]),
        )
    losses["loss"].backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_teacher_set_distribution_is_slot_invariant_and_uses_relocated_targets() -> (
    None
):
    dtype = torch.float64
    logits = torch.tensor([[0.3, -0.1, 0.2, -0.4]], dtype=dtype, requires_grad=True)
    positions = torch.tensor(
        [[0.12, 0.31, 0.66, 0.88]],
        dtype=dtype,
        requires_grad=True,
    )
    points = torch.zeros(1, 10, 2, dtype=dtype)
    teacher_knots = torch.tensor([[0.28, 0.72, 0.0, 0.0]], dtype=dtype)
    teacher_knot_mask = torch.tensor([[True, True, False, False]])

    def distribution_loss(
        teacher_slots: torch.Tensor,
        source_knots: torch.Tensor,
    ) -> torch.Tensor:
        probability = logits.sigmoid()
        output = {
            "candidate_knots": positions.detach(),
            "proposal_internal_knots": positions,
            "internal_knots": positions,
            "keep_logits": logits,
            "keep_probability": probability,
            "reconstructed_points": points.clone(),
            "params": torch.linspace(0.0, 1.0, 10, dtype=dtype).unsqueeze(0),
            "final_hard_keep_mask": torch.tensor([[False, True, True, False]]),
        }
        losses = CandidatePruningLoss(
            CandidatePruningLossWeights(),
            candidate_match_tolerance=0.1,
            exact_deletion_supervision=False,
            position_aware_distribution=True,
        )(
            output,
            points,
            true_internal_knots=source_knots,
            true_internal_knot_mask=torch.tensor([[True, True, False, False]]),
            teacher_retained_mask=teacher_slots,
            teacher_internal_knots=teacher_knots,
            teacher_internal_knot_mask=teacher_knot_mask,
            teacher_count=torch.tensor([2]),
        )
        return losses["teacher_distribution_loss"]

    first = distribution_loss(
        torch.tensor([[True, False, True, False]]),
        torch.tensor([[0.05, 0.95, 0.0, 0.0]], dtype=dtype),
    )
    second = distribution_loss(
        torch.tensor([[False, True, False, True]]),
        torch.tensor([[0.15, 0.85, 0.0, 0.0]], dtype=dtype),
    )

    torch.testing.assert_close(first, second)
    probability_gradient = torch.autograd.grad(first, logits, retain_graph=True)[0]
    position_gradient = torch.autograd.grad(first, positions)[0]
    assert probability_gradient.abs().sum() > 0
    assert position_gradient.abs().sum() > 0


def test_survivor_calibration_updates_preliminary_position_keep_feedback() -> None:
    torch.manual_seed(73)
    model = SplineFittingNetwork(
        point_dim=2,
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=3,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
        one_shot_selection_policy="mass_topk",
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=True,
        one_shot_max_position_shift=0.15,
    )
    trainable = _set_one_shot_trainable(model, calibrate_positions=True)
    head = model.pruning_head
    preliminary = head.joint_preliminary_position_head.weight
    feedback = head.joint_position_to_keep_feedback[-1].weight
    trainable_ids = {id(parameter) for parameter in trainable}
    assert id(preliminary) in trainable_ids
    assert id(feedback) in trainable_ids

    before_preliminary = preliminary.detach().clone()
    before_feedback = feedback.detach().clone()
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    output = model(torch.randn(2, 16, 2))
    objective = (
        output["provisional_candidate_positions"].mean()
        + output["keep_probability"].mean()
    )
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    assert preliminary.grad is not None and preliminary.grad.abs().sum() > 0
    assert feedback.grad is not None and feedback.grad.abs().sum() > 0
    optimizer.step()

    assert not torch.equal(preliminary.detach(), before_preliminary)
    assert not torch.equal(feedback.detach(), before_feedback)
