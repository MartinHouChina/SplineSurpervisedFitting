from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    CANDIDATE_PRUNING_OBJECTIVE_VERSION,
    ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.losses import CandidatePruningLoss, CandidatePruningLossWeights
from spline_fitting.models import SplineFittingNetwork


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
