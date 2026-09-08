from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (  # noqa: E402
    CANDIDATE_PRUNING_OBJECTIVE_VERSION,
    ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.losses import (  # noqa: E402
    CandidatePruningLoss,
    CandidatePruningLossWeights,
)
from spline_fitting.models import SplineFittingNetwork  # noqa: E402


def _model() -> SplineFittingNetwork:
    return SplineFittingNetwork(
        point_dim=2,
        hidden_dim=32,
        encoder_layers=1,
        max_internal_knots=6,
        structure_mode="candidate_pruning",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
    )


def _one_shot_model() -> SplineFittingNetwork:
    return SplineFittingNetwork(
        point_dim=2,
        hidden_dim=32,
        encoder_layers=1,
        max_internal_knots=6,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
    )


def _v9_model() -> SplineFittingNetwork:
    return SplineFittingNetwork(
        point_dim=2,
        hidden_dim=32,
        encoder_layers=1,
        max_internal_knots=6,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
    )


def _stable_one_shot_model() -> SplineFittingNetwork:
    return SplineFittingNetwork(
        point_dim=2,
        hidden_dim=32,
        encoder_layers=1,
        max_internal_knots=12,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
        one_shot_selection_policy="mass_topk",
        one_shot_selector_layers=2,
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=True,
        stable_pilot_descriptors=True,
    )


def test_stable_pilot_is_batch_companion_invariant_in_train_and_eval() -> None:
    torch.manual_seed(211)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _stable_one_shot_model().to(device)
    points = torch.randn(9, 64, 2, device=device)
    first_batch = torch.cat([points[:1], points[1:5]], dim=0)
    second_batch = torch.cat([points[:1], points[5:]], dim=0)

    for training in (False, True):
        model.train(training)
        with torch.no_grad():
            single = model(points[:1])
            with_first_companions = model(first_batch)
            with_second_companions = model(second_batch)

        for key, absolute_tolerance in (
            ("params", 1e-5),
            ("candidate_knots", 1e-5),
            ("proposal_internal_knots", 1e-5),
            ("keep_probability", 2e-4),
            ("deployment_internal_knots", 1e-5),
        ):
            torch.testing.assert_close(
                single[key][0],
                with_first_companions[key][0],
                rtol=1e-5,
                atol=absolute_tolerance,
            )
            torch.testing.assert_close(
                single[key][0],
                with_second_companions[key][0],
                rtol=1e-5,
                atol=absolute_tolerance,
            )
        assert torch.equal(
            single["final_hard_keep_mask"][0],
            with_first_companions["final_hard_keep_mask"][0],
        )
        assert torch.equal(
            single["final_hard_keep_mask"][0],
            with_second_companions["final_hard_keep_mask"][0],
        )


def test_candidate_pruning_network_output_and_backward() -> None:
    torch.manual_seed(23)
    model = _model()
    points = torch.randn(2, 32, 2)
    output = model(points)

    assert output["candidate_knots"].shape == (2, 6)
    assert output["internal_knots"].shape == (2, 6)
    assert output["remove_stop_logits"].shape == (2, 7)
    assert output["analytic_drop_objective_delta"].shape == (2, 6)
    assert output["reconstructed_points"].shape == points.shape
    assert torch.all(output["internal_knots"][:, 1:] > output["internal_knots"][:, :-1])
    # The training fit deliberately keeps every candidate open.  Learned keep
    # predictions are diagnostics/order proposals, not a differentiable mask.
    assert torch.all(output["fit_activity_gate"] == 1)

    true_knots = torch.tensor([[0.25, 0.65, 0.0], [0.35, 0.0, 0.0]], dtype=points.dtype)
    true_mask = torch.tensor([[True, True, False], [True, False, False]])
    loss = CandidatePruningLoss()(
        output,
        points,
        true_params=torch.linspace(0.0, 1.0, 32).expand(2, -1),
        true_internal_knots=true_knots,
        true_internal_knot_mask=true_mask,
    )
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    assert model.candidate_head.interval_score.weight.grad is not None
    assert model.pruning_head.remove_action_head.weight.grad is not None


def test_candidate_pruning_checkpoint_round_trip_is_strict() -> None:
    torch.manual_seed(31)
    reference = _model().eval()
    config = {
        "point_dim": 2,
        "hidden_dim": 32,
        "encoder_layers": 1,
        "max_internal_knots": 6,
        "structure_mode": "candidate_pruning",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
    }
    checkpoint = {
        "objective_version": CANDIDATE_PRUNING_OBJECTIVE_VERSION,
        "model_config": config,
        "model_state_dict": reference.state_dict(),
    }
    restored, migrated, legacy = build_model_from_checkpoint(checkpoint)
    assert not legacy
    assert migrated["structure_mode"] == "candidate_pruning"
    loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)
    assert assumed
    assert loss_config["weights"]["count_consistency"] == 0.0
    assert loss_config["weights"]["remove_action"] == 1.0
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        expected = reference(points)
        actual = restored.eval()(points)
    torch.testing.assert_close(actual["candidate_knots"], expected["candidate_knots"])
    torch.testing.assert_close(
        actual["remove_stop_logits"], expected["remove_stop_logits"]
    )


def test_one_shot_network_uses_discrete_straight_through_mask_and_teacher_loss() -> (
    None
):
    torch.manual_seed(37)
    model = _one_shot_model()
    points = torch.randn(2, 32, 2)
    output = model(points)

    assert output["adaptive_keep_threshold"].shape == (2,)
    assert output["learned_keep_mask"].shape == (2, 6)
    torch.testing.assert_close(
        output["fit_activity_gate"].detach(),
        output["learned_keep_mask"].to(points.dtype),
    )
    teacher_mask = torch.tensor(
        [
            [True, False, True, False, False, True],
            [False, True, False, True, False, False],
        ]
    )
    teacher_risk = teacher_mask.to(points.dtype) * 0.8 + 0.1
    weights = CandidatePruningLossWeights(
        fit=0.25,
        threshold_violation=1.0,
        true_parameter=0.0,
        candidate_coverage=1.0,
        candidate_repulsion=0.0,
        keep=1.0,
        remove_action=0.0,
        knot_position=0.0,
        count_consistency=0.0,
        deletion_cost=0.0,
        teacher_risk=0.5,
        teacher_ranking=1.0,
        teacher_distribution=1.0,
        teacher_critical_recall=1.0,
        teacher_count=0.25,
        complexity=0.05,
    )
    true_knots = torch.tensor([[0.25, 0.65], [0.35, 0.75]], dtype=points.dtype)
    true_mask = torch.ones_like(true_knots, dtype=torch.bool)
    losses = CandidatePruningLoss(
        weights,
        fit_tolerance=1.0,
        exact_deletion_supervision=False,
    )(
        output,
        points,
        true_internal_knots=true_knots,
        true_internal_knot_mask=true_mask,
        teacher_retained_mask=teacher_mask,
        teacher_soft_keep_risk=teacher_risk,
        teacher_count=teacher_mask.sum(dim=-1),
    )
    assert torch.isfinite(losses["loss"])
    assert losses["teacher_risk_loss"] > 0
    assert losses["teacher_ranking_loss"] > 0
    assert losses["teacher_distribution_loss"] > 0
    assert losses["teacher_critical_recall_loss"] > 0
    assert losses["canonical_position_loss"] > 0
    assert losses["keep_dice_loss"] > 0
    torch.testing.assert_close(
        losses["structured_active_count"],
        output["learned_keep_mask"].to(points.dtype).sum(dim=-1).mean(),
    )
    losses["loss"].backward()
    assert model.pruning_head.adaptive_threshold_head[-1].weight.grad is not None
    assert model.pruning_head.keep_context_projection[0].weight.grad is not None


def test_position_aware_distribution_backpropagates_to_probability_and_position() -> (
    None
):
    probability = torch.tensor(
        [[0.85, 0.30, 0.65, 0.20]],
        requires_grad=True,
    )
    refined_positions = torch.tensor(
        [[0.08, 0.31, 0.57, 0.78]],
        requires_grad=True,
    )
    points = torch.zeros(1, 12, 2)
    output = {
        "candidate_knots": refined_positions.detach().clone(),
        "proposal_internal_knots": refined_positions,
        "internal_knots": refined_positions,
        "keep_logits": torch.logit(probability),
        "keep_probability": probability,
        "reconstructed_points": torch.zeros_like(points),
        "params": torch.linspace(0.0, 1.0, points.shape[1]).unsqueeze(0),
    }
    losses = CandidatePruningLoss(
        CandidatePruningLossWeights(),
        candidate_match_tolerance=0.1,
        exact_deletion_supervision=False,
        position_aware_distribution=True,
    )(
        output,
        points,
        true_internal_knots=torch.tensor([[0.24, 0.84]]),
        true_internal_knot_mask=torch.tensor([[True, True]]),
        teacher_retained_mask=torch.tensor([[True, False, True, False]]),
    )

    probability_gradient, position_gradient = torch.autograd.grad(
        losses["teacher_distribution_loss"],
        (probability, refined_positions),
    )
    assert torch.isfinite(probability_gradient).all()
    assert torch.isfinite(position_gradient).all()
    assert probability_gradient.abs().sum() > 0
    assert position_gradient.abs().sum() > 0


def test_candidate_pruning_loss_reports_multiscale_recall_counts() -> None:
    candidate_positions = torch.tensor([[0.100, 0.204, 0.419]])
    keep_probability = torch.full_like(candidate_positions, 0.75)
    points = torch.zeros(1, 12, 2)
    output = {
        "candidate_knots": candidate_positions,
        "proposal_internal_knots": candidate_positions,
        "internal_knots": candidate_positions,
        "keep_logits": torch.logit(keep_probability),
        "keep_probability": keep_probability,
        "reconstructed_points": torch.zeros_like(points),
        "params": torch.linspace(0.0, 1.0, points.shape[1]).unsqueeze(0),
    }

    losses = CandidatePruningLoss(
        CandidatePruningLossWeights(),
        candidate_match_tolerance=0.01,
        candidate_coverage_tolerances=(0.005, 0.01, 0.02),
        exact_deletion_supervision=False,
    )(
        output,
        points,
        true_internal_knots=torch.tensor([[0.1, 0.2, 0.4]]),
        true_internal_knot_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(
        losses["candidate_match_count_at_0p005"], points.new_tensor(2.0)
    )
    torch.testing.assert_close(
        losses["candidate_match_count_at_0p01"], points.new_tensor(2.0)
    )
    torch.testing.assert_close(
        losses["candidate_match_count_at_0p02"], points.new_tensor(3.0)
    )


def test_one_shot_checkpoint_round_trip_and_default_loss_config() -> None:
    reference = _one_shot_model().eval()
    config = {
        "point_dim": 2,
        "hidden_dim": 32,
        "encoder_layers": 1,
        "max_internal_knots": 6,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
    }
    checkpoint = {
        "objective_version": ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        "model_config": config,
        "model_state_dict": reference.state_dict(),
    }
    restored, migrated, legacy = build_model_from_checkpoint(checkpoint)
    assert not legacy
    assert migrated["structure_mode"] == "candidate_pruning_one_shot"
    loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)
    assert assumed
    assert loss_config["one_shot_teacher"] is True
    assert loss_config["weights"]["teacher_risk"] == 0.5
    assert loss_config["weights"]["teacher_ranking"] == 1.0
    assert loss_config["weights"]["fit"] == 0.0
    assert loss_config["weights"]["threshold_violation"] == 0.0
    assert loss_config["weights"]["candidate_coverage"] == 0.0
    assert loss_config["weights"]["keep"] == 1.0
    assert loss_config["weights"]["knot_position"] == 4.0
    assert loss_config["weights"]["teacher_count"] == 2.0
    assert loss_config["weights"]["complexity"] == 0.25
    assert loss_config["positive_keep_weight"] == 1.0
    assert loss_config["teacher_ranking_margin"] == 1.0
    assert loss_config["candidate_match_tolerance"] == 0.01
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        expected = reference(points)
        actual = restored.eval()(points)
    torch.testing.assert_close(actual["keep_probability"], expected["keep_probability"])
    torch.testing.assert_close(
        actual["adaptive_keep_threshold"], expected["adaptive_keep_threshold"]
    )


def test_v9_checkpoint_round_trip_uses_fixed_geometry_selector() -> None:
    reference = _v9_model().eval()
    config = {
        "point_dim": 2,
        "hidden_dim": 32,
        "encoder_layers": 1,
        "max_internal_knots": 6,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
        "one_shot_fixed_proposal_geometry": True,
    }
    checkpoint = {
        "objective_version": V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        "model_config": config,
        "model_state_dict": reference.state_dict(),
    }
    restored, migrated, legacy = build_model_from_checkpoint(checkpoint)
    assert not legacy
    assert migrated["one_shot_fixed_proposal_geometry"] is True
    loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)
    assert assumed
    assert loss_config["fixed_proposal_geometry"] is True
    assert loss_config["weights"]["knot_position"] == 0.0
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        expected = reference(points)
        actual = restored.eval()(points)
    torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])
    torch.testing.assert_close(actual["keep_probability"], expected["keep_probability"])


def test_v10_checkpoint_uses_mass_topk_and_deeper_selector() -> None:
    config = {
        "point_dim": 2,
        "hidden_dim": 32,
        "encoder_layers": 1,
        "max_internal_knots": 6,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": "mass_topk",
        "one_shot_safety_sigma": 0.5,
        "one_shot_selector_layers": 2,
        "one_shot_coverage_bins": 4,
    }
    reference = SplineFittingNetwork(**config).eval()
    checkpoint = {
        "objective_version": V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        "model_config": config,
        "model_state_dict": reference.state_dict(),
    }

    restored, migrated, legacy = build_model_from_checkpoint(checkpoint)
    assert not legacy
    assert migrated["one_shot_selection_policy"] == "mass_topk"
    assert migrated["one_shot_selector_layers"] == 2
    assert migrated["one_shot_coverage_bins"] == 4
    loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)
    assert assumed
    assert loss_config["weights"]["teacher_distribution"] == 2.0
    assert loss_config["weights"]["teacher_critical_recall"] == 1.0
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        expected = reference(points)
        actual = restored.eval()(points)
    torch.testing.assert_close(actual["keep_probability"], expected["keep_probability"])
    assert torch.equal(actual["learned_keep_mask"], expected["learned_keep_mask"])


def test_v11_checkpoint_joint_refinement_and_v10_neutral_initialization() -> None:
    base_config = {
        "point_dim": 2,
        "hidden_dim": 32,
        "encoder_layers": 1,
        "max_internal_knots": 6,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": "mass_topk",
        "one_shot_safety_sigma": 0.25,
        "one_shot_selector_layers": 2,
        "one_shot_coverage_bins": 0,
        "candidate_local_attention_bandwidth": 0.0,
    }
    torch.manual_seed(107)
    v10 = SplineFittingNetwork(**base_config).eval()
    v11_config = {
        **base_config,
        "one_shot_joint_position_refinement": True,
        "one_shot_max_position_shift": 0.05,
    }
    v11 = SplineFittingNetwork(**v11_config).eval()
    v11.load_state_dict(v10.state_dict(), strict=True)

    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        expected = v10(points)
        neutral = v11(points)
    torch.testing.assert_close(
        neutral["proposal_internal_knots"], expected["internal_knots"]
    )
    torch.testing.assert_close(neutral["internal_knots"], expected["internal_knots"])
    torch.testing.assert_close(
        neutral["keep_probability"], expected["keep_probability"]
    )
    assert torch.equal(neutral["learned_keep_mask"], expected["learned_keep_mask"])

    checkpoint = {
        "objective_version": V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        "model_config": v11_config,
        "model_state_dict": v11.state_dict(),
    }
    restored, migrated, legacy = build_model_from_checkpoint(checkpoint)
    assert not legacy
    assert migrated["one_shot_joint_position_refinement"] is True
    loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)
    assert assumed
    assert loss_config["weights"]["teacher_false_positive"] == 1.0
    assert loss_config["weights"]["policy_count"] == 4.0
    assert loss_config["position_aware_distribution"] is True
    with torch.no_grad():
        actual = restored.eval()(points)
    torch.testing.assert_close(actual["internal_knots"], neutral["internal_knots"])


def _teacher_feasibility_loss_inputs() -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    dtype = torch.float64
    logits = torch.tensor(
        [[1.2, -0.8, 0.7, -1.4], [3.0, 2.5, 2.0, 1.5]],
        dtype=dtype,
        requires_grad=True,
    )
    probability = logits.sigmoid()
    refined = torch.tensor(
        [[0.15, 0.38, 0.62, 0.85], [0.10, 0.30, 0.60, 0.90]],
        dtype=dtype,
        requires_grad=True,
    )
    hard_keep = torch.tensor(
        [[True, False, True, False], [True, True, True, True]]
    )
    remove_logits = torch.tensor(
        [[-0.5, 0.7, -0.2, 0.3, -1.0], [0.8, 0.4, 0.2, 0.1, -0.6]],
        dtype=dtype,
        requires_grad=True,
    )
    predicted_cost = torch.tensor(
        [[0.2, -0.3, 0.4, -0.1], [2.0, 2.0, 2.0, 2.0]],
        dtype=dtype,
        requires_grad=True,
    )
    points = torch.zeros(2, 12, 2, dtype=dtype)
    output = {
        "candidate_knots": refined,
        "proposal_internal_knots": refined,
        "internal_knots": refined,
        "keep_logits": logits,
        "keep_probability": probability,
        "reconstructed_points": points.clone(),
        "params": torch.linspace(0.0, 1.0, 12, dtype=dtype).expand(2, -1),
        "final_hard_keep_mask": hard_keep,
        "final_hard_st_keep_gate": (
            hard_keep.to(dtype) + probability - probability.detach()
        ),
        "one_shot_requested_count_score": probability.sum(dim=-1),
        "remove_stop_logits": remove_logits,
        "predicted_log_deletion_cost": predicted_cost,
    }
    teacher_kwargs = {
        "true_internal_knots": torch.tensor(
            [[0.17, 0.65, 0.0, 0.0], [0.20, 0.80, 0.0, 0.0]], dtype=dtype
        ),
        "true_internal_knot_mask": torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        ),
        # The second row represents the failure mode being guarded against:
        # the search missed epsilon and therefore returned an all-keep mask.
        "teacher_retained_mask": torch.tensor(
            [[True, False, True, False], [True, True, True, True]]
        ),
        "teacher_soft_keep_risk": torch.tensor(
            [[0.9, 0.1, 0.8, 0.2], [1.0, 1.0, 1.0, 1.0]], dtype=dtype
        ),
        "teacher_internal_knots": torch.tensor(
            [[0.18, 0.66, 0.0, 0.0], [0.10, 0.30, 0.60, 0.90]], dtype=dtype
        ),
        "teacher_internal_knot_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
        "teacher_count": torch.tensor([2, 4]),
        "teacher_fit_rms": torch.tensor([0.04, 0.20], dtype=dtype),
        "teacher_single_deletion_rms": torch.tensor(
            [[0.20, 0.02, 0.30, 0.03], [0.40, 0.40, 0.40, 0.40]], dtype=dtype
        ),
    }
    return output, points, teacher_kwargs


def _teacher_only_loss() -> CandidatePruningLoss:
    return CandidatePruningLoss(
        CandidatePruningLossWeights(
            fit=0.0,
            threshold_violation=0.0,
            true_parameter=0.0,
            candidate_coverage=0.0,
            candidate_repulsion=0.0,
            keep=1.0,
            remove_action=1.0,
            knot_position=1.0,
            count_consistency=1.0,
            deletion_cost=1.0,
            teacher_risk=1.0,
            teacher_ranking=1.0,
            teacher_distribution=1.0,
            teacher_critical_recall=1.0,
            teacher_false_positive=1.0,
            teacher_count=1.0,
            policy_count=1.0,
            canonical_selection=0.0,
            complexity=1.0,
        ),
        fit_tolerance=0.1,
        candidate_match_tolerance=0.1,
        exact_deletion_supervision=False,
        position_aware_distribution=True,
        joint_position_supervision=True,
        teacher_relocation_supervision=True,
    )


def test_infeasible_offline_teacher_rows_do_not_supervise_student() -> None:
    output, points, teacher_kwargs = _teacher_feasibility_loss_inputs()
    loss_fn = _teacher_only_loss()
    mixed = loss_fn(
        output,
        points,
        teacher_threshold_satisfied=torch.tensor([True, False]),
        **teacher_kwargs,
    )
    first_output = {key: value[:1] for key, value in output.items()}
    first_teacher = {key: value[:1] for key, value in teacher_kwargs.items()}
    feasible_only = loss_fn(
        first_output,
        points[:1],
        teacher_threshold_satisfied=torch.tensor([True]),
        **first_teacher,
    )

    teacher_loss_keys = (
        "keep_bce_loss",
        "keep_dice_loss",
        "keep_loss",
        "remove_action_loss",
        "deletion_cost_loss",
        "teacher_risk_loss",
        "teacher_ranking_loss",
        "teacher_distribution_loss",
        "teacher_set_coverage_loss",
        "teacher_critical_recall_loss",
        "teacher_false_positive_loss",
        "teacher_count_loss",
        "policy_count_loss",
        "complexity_loss",
        "teacher_anchor_position_loss",
        "teacher_survivor_spacing_loss",
        "joint_deployment_position_loss",
        "knot_position_loss",
        "loss",
    )
    for key in teacher_loss_keys:
        torch.testing.assert_close(mixed[key], feasible_only[key])

    gradients = torch.autograd.grad(
        mixed["loss"],
        (
            output["keep_logits"],
            output["internal_knots"],
            output["remove_stop_logits"],
            output["predicted_log_deletion_cost"],
        ),
    )
    for gradient in gradients:
        assert gradient[0].abs().sum() > 0
        torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]))


def test_teacher_feasibility_defaults_to_all_rows_and_all_false_is_zero() -> None:
    output, points, teacher_kwargs = _teacher_feasibility_loss_inputs()
    loss_fn = _teacher_only_loss()
    implicit_all_feasible = loss_fn(output, points, **teacher_kwargs)
    explicit_all_feasible = loss_fn(
        output,
        points,
        teacher_threshold_satisfied=torch.tensor([True, True]),
        **teacher_kwargs,
    )
    torch.testing.assert_close(
        implicit_all_feasible["loss"], explicit_all_feasible["loss"]
    )

    all_infeasible = loss_fn(
        output,
        points,
        teacher_threshold_satisfied=torch.tensor([False, False]),
        **teacher_kwargs,
    )
    assert float(all_infeasible["loss"].detach()) == 0.0
    for key in (
        "keep_loss",
        "remove_action_loss",
        "deletion_cost_loss",
        "teacher_risk_loss",
        "teacher_ranking_loss",
        "teacher_distribution_loss",
        "teacher_critical_recall_loss",
        "teacher_false_positive_loss",
        "teacher_count_loss",
        "policy_count_loss",
        "complexity_loss",
        "knot_position_loss",
    ):
        assert float(all_infeasible[key].detach()) == 0.0


def test_redundant_candidate_targets_split_largest_intervals_one_to_one() -> None:
    dtype = torch.float64
    true_knots = torch.tensor([[0.25, 0.75, 0.0, 0.0]], dtype=dtype)
    true_mask = torch.tensor([[True, True, False, False]])
    expected = torch.tensor(
        [[0.125, 0.25, 0.375, 0.50, 0.75]],
        dtype=dtype,
    )
    actual = CandidatePruningLoss._redundant_candidate_targets(
        true_knots,
        true_mask,
        candidate_count=5,
    )
    torch.testing.assert_close(actual, expected)

    points = torch.zeros(1, 10, 2, dtype=dtype)
    duplicated_candidates = torch.tensor(
        [[0.25, 0.25, 0.50, 0.75, 0.75]],
        dtype=dtype,
        requires_grad=True,
    )
    output = {
        "candidate_knots": duplicated_candidates,
        "proposal_internal_knots": duplicated_candidates,
        "internal_knots": duplicated_candidates,
        "keep_logits": torch.zeros_like(duplicated_candidates),
        "keep_probability": torch.full_like(duplicated_candidates, 0.5),
        "reconstructed_points": points.clone(),
        "params": torch.linspace(0.0, 1.0, 10, dtype=dtype).unsqueeze(0),
    }
    weights = CandidatePruningLossWeights(
        fit=0.0,
        threshold_violation=0.0,
        true_parameter=0.0,
        candidate_coverage=0.0,
        redundant_candidate=1.0,
        candidate_repulsion=0.0,
        keep=0.0,
        remove_action=0.0,
        knot_position=0.0,
        count_consistency=0.0,
        deletion_cost=0.0,
        teacher_risk=0.0,
        teacher_ranking=0.0,
        teacher_distribution=0.0,
        teacher_critical_recall=0.0,
        teacher_false_positive=0.0,
        teacher_count=0.0,
        policy_count=0.0,
        canonical_selection=0.0,
        complexity=0.0,
    )
    losses = CandidatePruningLoss(
        weights,
        exact_deletion_supervision=False,
    )(
        output,
        points,
        true_internal_knots=true_knots,
        true_internal_knot_mask=true_mask,
    )
    assert losses["redundant_candidate_loss"] > 0
    torch.testing.assert_close(losses["loss"], losses["redundant_candidate_loss"])
    gradient = torch.autograd.grad(
        losses["redundant_candidate_loss"], duplicated_candidates
    )[0]
    assert gradient.abs().sum() > 0


def test_true_parameter_loss_uses_configured_error_scale_and_reports_raw_mse() -> (
    None
):
    dtype = torch.float64
    points = torch.zeros(1, 8, 2, dtype=dtype)
    true_params = torch.linspace(0.0, 1.0, 8, dtype=dtype).unsqueeze(0)
    predicted_params = true_params + 0.02
    candidates = torch.tensor([[0.25, 0.50, 0.75]], dtype=dtype)
    output = {
        "candidate_knots": candidates,
        "proposal_internal_knots": candidates,
        "internal_knots": candidates,
        "keep_logits": torch.zeros_like(candidates),
        "keep_probability": torch.full_like(candidates, 0.5),
        "reconstructed_points": points.clone(),
        "params": predicted_params,
    }
    labels = {
        "true_params": true_params,
        "true_internal_knots": torch.tensor([[0.5]], dtype=dtype),
        "true_internal_knot_mask": torch.tensor([[True]]),
    }
    raw = CandidatePruningLoss(exact_deletion_supervision=False)(
        output,
        points,
        **labels,
    )
    scaled = CandidatePruningLoss(
        exact_deletion_supervision=False,
        true_parameter_error_scale=0.02,
    )(
        output,
        points,
        **labels,
    )

    torch.testing.assert_close(
        raw["raw_true_parameter_loss"],
        torch.tensor(0.02**2, dtype=dtype),
    )
    torch.testing.assert_close(raw["true_parameter_loss"], raw["raw_true_parameter_loss"])
    torch.testing.assert_close(
        scaled["raw_true_parameter_loss"], raw["raw_true_parameter_loss"]
    )
    torch.testing.assert_close(
        scaled["true_parameter_loss"], torch.tensor(1.0, dtype=dtype)
    )

    dual_output = {
        **output,
        "proposal_params": true_params + 0.04,
    }
    dual = CandidatePruningLoss(
        exact_deletion_supervision=False,
        true_parameter_error_scale=0.02,
    )(
        dual_output,
        points,
        **labels,
    )
    torch.testing.assert_close(
        dual["raw_proposal_parameter_loss"],
        torch.tensor(0.04**2, dtype=dtype),
    )
    torch.testing.assert_close(
        dual["true_parameter_loss"],
        torch.tensor(2.5, dtype=dtype),
    )


def test_true_parameter_gap_loss_supervises_local_log_intervals() -> None:
    dtype = torch.float64
    points = torch.zeros(1, 4, 2, dtype=dtype)
    true_params = torch.tensor([[0.0, 0.1, 0.4, 1.0]], dtype=dtype)
    predicted_params = torch.tensor(
        [[0.0, 0.2, 0.5, 1.0]],
        dtype=dtype,
        requires_grad=True,
    )
    candidates = torch.tensor([[0.5]], dtype=dtype)
    output = {
        "candidate_knots": candidates,
        "proposal_internal_knots": candidates,
        "internal_knots": candidates,
        "keep_logits": torch.zeros_like(candidates),
        "keep_probability": torch.full_like(candidates, 0.5),
        "reconstructed_points": points.clone(),
        "proposal_params": true_params.clone(),
        "params": predicted_params,
    }
    labels = {
        "true_params": true_params,
        "true_internal_knots": torch.tensor([[0.5]], dtype=dtype),
        "true_internal_knot_mask": torch.tensor([[True]]),
    }
    base_weights = CandidatePruningLossWeights()
    base_weights.true_parameter_gap = 0.0
    weighted_weights = CandidatePruningLossWeights()
    weighted_weights.true_parameter_gap = 2.0
    base = CandidatePruningLoss(
        base_weights,
        exact_deletion_supervision=False,
    )(output, points, **labels)
    weighted = CandidatePruningLoss(
        weighted_weights,
        exact_deletion_supervision=False,
    )(output, points, **labels)

    expected_output_gap_loss = torch.nn.functional.smooth_l1_loss(
        torch.log(torch.diff(predicted_params, dim=-1)),
        torch.log(torch.diff(true_params, dim=-1)),
        beta=0.1,
    )
    torch.testing.assert_close(
        weighted["raw_true_parameter_gap_loss"],
        expected_output_gap_loss,
    )
    torch.testing.assert_close(
        weighted["raw_proposal_parameter_gap_loss"],
        torch.zeros((), dtype=dtype),
    )
    torch.testing.assert_close(
        weighted["true_parameter_gap_loss"],
        0.5 * expected_output_gap_loss,
    )
    torch.testing.assert_close(
        weighted["loss"] - base["loss"],
        2.0 * weighted["true_parameter_gap_loss"],
    )
    gradient = torch.autograd.grad(
        weighted["true_parameter_gap_loss"], predicted_params
    )[0]
    assert gradient.abs().sum() > 0


def test_teacher_positions_are_warped_from_proposal_t0_to_calibrated_t1() -> None:
    dtype = torch.float64
    proposal_params = torch.linspace(0.0, 1.0, 9, dtype=dtype).unsqueeze(0)
    calibrated_params = proposal_params.square()
    proposal_knots = torch.tensor([[0.5]], dtype=dtype)
    deployed_knots = torch.tensor([[0.25]], dtype=dtype)
    points = torch.zeros(1, 9, 2, dtype=dtype)
    output = {
        "candidate_knots": proposal_knots,
        "proposal_internal_knots": proposal_knots,
        "internal_knots": deployed_knots,
        "keep_logits": torch.full_like(proposal_knots, 4.0),
        "keep_probability": torch.sigmoid(torch.full_like(proposal_knots, 4.0)),
        "final_hard_keep_mask": torch.ones_like(proposal_knots, dtype=torch.bool),
        "reconstructed_points": points.clone(),
        "proposal_params": proposal_params,
        "params": calibrated_params,
    }
    weights = CandidatePruningLossWeights(
        fit=0.0,
        threshold_violation=0.0,
        true_parameter=0.0,
        candidate_coverage=0.0,
        redundant_candidate=0.0,
        candidate_repulsion=0.0,
        keep=0.0,
        remove_action=0.0,
        knot_position=1.0,
        count_consistency=0.0,
        deletion_cost=0.0,
        teacher_risk=0.0,
        teacher_ranking=0.0,
        teacher_distribution=0.0,
        teacher_critical_recall=0.0,
        teacher_false_positive=0.0,
        teacher_count=0.0,
        policy_count=0.0,
        canonical_selection=0.0,
        complexity=0.0,
    )
    losses = CandidatePruningLoss(
        weights,
        exact_deletion_supervision=False,
        teacher_relocation_supervision=True,
    )(
        output,
        points,
        true_internal_knots=proposal_knots,
        true_internal_knot_mask=torch.ones_like(proposal_knots, dtype=torch.bool),
        teacher_retained_mask=torch.ones_like(proposal_knots, dtype=torch.bool),
        teacher_internal_knots=proposal_knots,
        teacher_internal_knot_mask=torch.ones_like(proposal_knots, dtype=torch.bool),
        teacher_count=torch.ones(1, dtype=torch.long),
        teacher_threshold_satisfied=torch.ones(1, dtype=torch.bool),
    )

    torch.testing.assert_close(
        losses["teacher_anchor_position_loss"], points.new_zeros(())
    )
    torch.testing.assert_close(losses["knot_position_loss"], points.new_zeros(()))
    torch.testing.assert_close(losses["loss"], points.new_zeros(()))


def test_canonical_proposal_targets_are_warped_from_true_domain_to_t0() -> None:
    dtype = torch.float64
    true_params = torch.linspace(0.0, 1.0, 9, dtype=dtype).unsqueeze(0)
    proposal_params = true_params.square()
    # The generating-domain knot t=0.5 denotes the same sampled location as
    # t0=0.25 under this deliberately non-identity proposal parameterization.
    true_knots = torch.tensor([[0.5]], dtype=dtype)
    proposal_knots = torch.tensor([[0.25]], dtype=dtype)
    points = torch.zeros(1, 9, 2, dtype=dtype)
    output = {
        "candidate_knots": proposal_knots,
        "proposal_internal_knots": proposal_knots,
        "internal_knots": proposal_knots,
        "keep_logits": torch.full_like(proposal_knots, 4.0),
        "keep_probability": torch.sigmoid(torch.full_like(proposal_knots, 4.0)),
        "reconstructed_points": points.clone(),
        "proposal_params": proposal_params,
        "params": proposal_params,
    }

    losses = CandidatePruningLoss(
        CandidatePruningLossWeights(redundant_candidate=1.0),
        exact_deletion_supervision=False,
        candidate_match_tolerance=0.01,
    )(
        output,
        points,
        true_params=true_params,
        true_internal_knots=true_knots,
        true_internal_knot_mask=torch.ones_like(true_knots, dtype=torch.bool),
    )

    torch.testing.assert_close(
        losses["candidate_nearest_mae"], points.new_zeros(())
    )
    torch.testing.assert_close(
        losses["candidate_coverage_loss"], points.new_zeros(())
    )
    torch.testing.assert_close(
        losses["redundant_candidate_loss"], points.new_zeros(())
    )
    torch.testing.assert_close(losses["canonical_position_loss"], points.new_zeros(()))


def test_redundant_targets_are_boehm_refined_before_nonlinear_t0_warp() -> None:
    dtype = torch.float64
    true_params = torch.linspace(0.0, 1.0, 9, dtype=dtype).unsqueeze(0)
    proposal_params = true_params.square()
    true_knots = torch.tensor([[0.5]], dtype=dtype)
    true_mask = torch.ones_like(true_knots, dtype=torch.bool)
    points = torch.zeros(1, 9, 2, dtype=dtype)

    # True-domain refinement gives [0.25, 0.5, 0.75], which is then warped
    # into t0 as [0.0625, 0.25, 0.5625].  Refining the already-warped knot
    # would instead produce [0.25, 0.4375, 0.625].
    expected = torch.tensor([[0.0625, 0.25, 0.5625]], dtype=dtype)
    wrong_warp_then_refine = CandidatePruningLoss._redundant_candidate_targets(
        torch.tensor([[0.25]], dtype=dtype),
        true_mask,
        candidate_count=3,
    )
    assert not torch.allclose(expected, wrong_warp_then_refine)

    def redundant_loss(candidates: torch.Tensor) -> torch.Tensor:
        output = {
            "candidate_knots": candidates,
            "proposal_internal_knots": candidates,
            "internal_knots": candidates,
            "keep_logits": torch.full_like(candidates, 4.0),
            "keep_probability": torch.sigmoid(torch.full_like(candidates, 4.0)),
            "reconstructed_points": points.clone(),
            "proposal_params": proposal_params,
            "params": proposal_params,
        }
        return CandidatePruningLoss(
            CandidatePruningLossWeights(redundant_candidate=1.0),
            exact_deletion_supervision=False,
            candidate_match_tolerance=0.01,
        )(
            output,
            points,
            true_params=true_params,
            true_internal_knots=true_knots,
            true_internal_knot_mask=true_mask,
        )["redundant_candidate_loss"]

    torch.testing.assert_close(redundant_loss(expected), points.new_zeros(()))
    assert redundant_loss(wrong_warp_then_refine) > 0


def test_joint_position_targets_are_warped_from_true_domain_to_t1() -> None:
    dtype = torch.float64
    true_params = torch.linspace(0.0, 1.0, 9, dtype=dtype).unsqueeze(0)
    corrected_params = true_params.square()
    true_knots = torch.tensor([[0.5]], dtype=dtype)
    proposal_knots = true_knots.clone()
    deployed_knots = torch.tensor([[0.25]], dtype=dtype)
    points = torch.zeros(1, 9, 2, dtype=dtype)
    output = {
        "candidate_knots": proposal_knots,
        "proposal_internal_knots": proposal_knots,
        "internal_knots": deployed_knots,
        "keep_logits": torch.full_like(proposal_knots, 4.0),
        "keep_probability": torch.sigmoid(torch.full_like(proposal_knots, 4.0)),
        "final_hard_keep_mask": torch.ones_like(proposal_knots, dtype=torch.bool),
        "reconstructed_points": points.clone(),
        "proposal_params": true_params,
        "params": corrected_params,
    }

    losses = CandidatePruningLoss(
        exact_deletion_supervision=False,
        joint_position_supervision=True,
    )(
        output,
        points,
        true_params=true_params,
        true_internal_knots=true_knots,
        true_internal_knot_mask=torch.ones_like(true_knots, dtype=torch.bool),
        teacher_retained_mask=torch.ones_like(proposal_knots, dtype=torch.bool),
        teacher_threshold_satisfied=torch.ones(1, dtype=torch.bool),
    )

    torch.testing.assert_close(
        losses["joint_deployment_position_loss"], points.new_zeros(())
    )
    torch.testing.assert_close(losses["knot_position_loss"], points.new_zeros(()))
