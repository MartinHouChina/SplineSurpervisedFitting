from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    CANDIDATE_PRUNING_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.losses import CandidatePruningLoss
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

    true_knots = torch.tensor(
        [[0.25, 0.65, 0.0], [0.35, 0.0, 0.0]], dtype=points.dtype
    )
    true_mask = torch.tensor(
        [[True, True, False], [True, False, False]]
    )
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
    torch.testing.assert_close(actual["remove_stop_logits"], expected["remove_stop_logits"])
