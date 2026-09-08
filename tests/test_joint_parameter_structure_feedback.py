from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.losses import (  # noqa: E402
    ParameterFeedbackLoss,
    ParameterFeedbackLossWeights,
)
from spline_fitting.checkpointing import (  # noqa: E402
    V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
    V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_model_config,
)
from spline_fitting.models import (  # noqa: E402
    JointParameterStructureHead,
    SplineFittingNetwork,
)
from spline_fitting.models import spline_network as spline_network_module  # noqa: E402


def _joint_head_inputs() -> dict[str, torch.Tensor]:
    batch, point_count, candidate_count, hidden_dim = 2, 19, 6, 16
    params = torch.linspace(0.0, 1.0, point_count).expand(batch, -1)
    candidates = torch.linspace(0.08, 0.92, candidate_count).expand(batch, -1)
    preliminary_mask = torch.tensor(
        [[True, False, True, True, False, True]],
    ).expand(batch, -1)
    return {
        "local_features": torch.randn(batch, point_count, hidden_dim),
        "corrected_params": params,
        "candidate_tokens": torch.randn(batch, candidate_count, hidden_dim),
        "candidate_positions": candidates,
        "preliminary_keep_logits": torch.randn(batch, candidate_count),
        "deletion_delta": torch.rand(batch, candidate_count),
        "candidate_local_residual": torch.rand(batch, candidate_count),
        "preliminary_position_residual": 0.01 * torch.randn(batch, candidate_count),
        "preliminary_keep_mask": preliminary_mask,
        "candidate_mask": torch.ones_like(preliminary_mask),
    }


def test_joint_head_is_zero_initialised_and_locally_attends() -> None:
    torch.manual_seed(401)
    head = JointParameterStructureHead(
        hidden_dim=16,
        attention_heads=4,
        local_bandwidth=0.08,
    )
    inputs = _joint_head_inputs()
    output = head(**inputs, return_attention_weights=True)

    torch.testing.assert_close(
        output["joint_structure_tokens"], inputs["candidate_tokens"]
    )
    torch.testing.assert_close(
        output["joint_keep_logits"], inputs["preliminary_keep_logits"]
    )
    assert torch.count_nonzero(output["joint_keep_logit_delta"]) == 0
    assert torch.count_nonzero(output["joint_raw_position_signal"]) == 0
    assert output["joint_local_attention_weights"].shape == (2, 6, 19)
    assert torch.isfinite(output["joint_local_attention_weights"]).all()


def test_joint_terminal_heads_receive_gradients() -> None:
    torch.manual_seed(409)
    head = JointParameterStructureHead(hidden_dim=16, attention_heads=4)
    output = head(**_joint_head_inputs())
    objective = (
        output["joint_keep_logits"].square().sum()
        + output["joint_raw_position_signal"].sum()
        + output["joint_structure_tokens"].square().sum()
    )
    objective.backward()

    for terminal in (
        head.keep_delta_head,
        head.position_delta_head,
        head.token_delta_head,
    ):
        assert terminal.weight.grad is not None
        assert float(terminal.weight.grad.abs().sum()) > 0.0


def _joint_network() -> SplineFittingNetwork:
    return SplineFittingNetwork(
        point_dim=2,
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=6,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
        one_shot_selection_policy="threshold",
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=True,
        parameter_feedback_fusion=True,
        parameter_feedback_fusion_mode="cross_attention",
        joint_parameter_structure_feedback=True,
        joint_parameter_structure_max_keep_logit_shift=20.0,
    )


def test_joint_network_accepts_coverage_and_order_constraints() -> None:
    torch.manual_seed(417)
    model = SplineFittingNetwork(
        point_dim=2,
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=12,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
        one_shot_selection_policy="mass_topk",
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=True,
        parameter_feedback_fusion=True,
        parameter_feedback_fusion_mode="cross_attention",
        joint_parameter_structure_feedback=True,
        candidate_interval_logit_limit=0.5,
        candidate_position_parameterization="bounded_anchor_residual",
        enforce_ordered_joint_candidates=True,
        parameter_gap_reference="chord_residual",
        parameter_residual_logit_limit=0.5,
    ).eval()
    points = torch.randn(2, 32, 2)
    with torch.no_grad():
        output = model.forward_deployment(points)

    torch.testing.assert_close(
        output["proposal_params"],
        model._chord_length_parameters(points),
    )
    corrected = output["joint_corrected_candidate_knots"]
    pre_relocation = output["joint_pre_relocation_candidate_knots"]
    assert torch.all(pre_relocation[:, 1:] - pre_relocation[:, :-1] >= 1e-3 - 1e-6)
    assert torch.all((pre_relocation - corrected).abs() <= 0.05 + 1e-6)
    assert torch.all(output["internal_knots"][:, 1:] > output["internal_knots"][:, :-1])
    for batch_index in range(2):
        mask = output["learned_keep_mask"][batch_index]
        retained = output["internal_knots"][batch_index, mask]
        assert torch.all(retained[1:] - retained[:-1] >= 1e-3 - 1e-6)


def test_uniform_parameter_reference_does_not_compute_chord_parameters() -> None:
    model = SplineFittingNetwork(
        point_dim=2,
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=6,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        parameter_gap_reference="uniform_residual",
    ).eval()
    points = torch.randn(2, 9, 2)
    local_features, global_features = model.encoder(points)

    with mock.patch.object(
        model,
        "_chord_length_parameters",
        side_effect=AssertionError("uniform mode must not construct chord parameters"),
    ):
        output = model._parameter_head_forward(
            points,
            local_features,
            global_features,
        )

    expected = torch.linspace(0.0, 1.0, 9).expand(2, -1)
    torch.testing.assert_close(output["params"], expected, atol=1e-7, rtol=0.0)


def test_joint_network_reselects_then_relocates_only_final_survivors() -> None:
    torch.manual_seed(419)
    model = _joint_network().eval()
    with torch.no_grad():
        model.joint_parameter_structure_head.keep_delta_head.bias.fill_(-8.0)
        output = model(torch.randn(2, 25, 2))

    assert "joint_preliminary_hard_keep_mask" in output
    assert torch.count_nonzero(output["learned_keep_mask"]) == 0
    assert torch.equal(output["relocation_selected_mask"], output["learned_keep_mask"])
    assert torch.count_nonzero(output["relocation_attention_weights"]) == 0
    assert torch.count_nonzero(output["relocation_position_residual"]) == 0
    assert torch.isfinite(output["internal_knots"]).all()
    assert torch.all(output["params"][:, 1:] > output["params"][:, :-1])


def test_joint_deployment_still_uses_one_internal_spline_solve() -> None:
    torch.manual_seed(421)
    model = _joint_network().eval()
    original_solve = spline_network_module.solve_coefficients
    solve_count = 0

    def counted_solve(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        nonlocal solve_count
        solve_count += 1
        return original_solve(*args, **kwargs)

    with mock.patch.object(
        spline_network_module,
        "solve_coefficients",
        side_effect=counted_solve,
    ):
        with torch.no_grad():
            output = model.forward_deployment(torch.randn(1, 24, 2))

    assert solve_count == 1
    assert "reconstructed_points" not in output
    assert "joint_final_deployment_knots" in output
    torch.testing.assert_close(
        output["internal_knots"],
        output["parameter_feedback_transport_internal_knots"],
    )
    assert torch.equal(
        output["learned_keep_mask"],
        output["joint_preliminary_hard_keep_mask"],
    )


def test_joint_feedback_loss_supervises_mask_count_and_warped_set() -> None:
    points = torch.tensor(
        [[[0.0, 0.0], [0.4, 0.2], [0.7, -0.1], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    keep_logits = torch.tensor([[1.0, -0.5, 0.3]], requires_grad=True)
    deployed_knots = torch.tensor([[0.2, 0.5, 0.8]], requires_grad=True)
    params = torch.tensor([[0.0, 0.2, 0.65, 1.0]], requires_grad=True)
    output = {
        "params": params,
        "proposal_params": torch.tensor([[0.0, 0.3, 0.7, 1.0]]),
        "reconstructed_points": points + 0.001,
        "keep_logits": keep_logits,
        "keep_probability": torch.sigmoid(keep_logits),
        "candidate_mask": torch.ones(1, 3, dtype=torch.bool),
        "learned_keep_mask": torch.tensor([[True, False, True]]),
        "deployment_internal_knots": deployed_knots,
    }
    loss_fn = ParameterFeedbackLoss(
        ParameterFeedbackLossWeights(
            fit=0.0,
            threshold_violation=0.0,
            true_parameter=0.0,
            chord_prior=0.0,
            identity=0.0,
            joint_keep=1.0,
            joint_position=1.0,
            joint_count=1.0,
        )
    )
    result = loss_fn(
        output,
        points,
        teacher_retained_mask=torch.tensor([[True, False, True]]),
        teacher_internal_knots=torch.tensor([[0.25, 0.75, 0.0]]),
        teacher_internal_knot_mask=torch.tensor([[True, True, False]]),
        teacher_count=torch.tensor([2]),
    )
    result["loss"].backward()

    assert result["joint_keep_loss"] > 0.0
    assert result["joint_knot_position_loss"] > 0.0
    assert keep_logits.grad is not None
    assert float(keep_logits.grad.abs().sum()) > 0.0
    assert deployed_knots.grad is not None
    assert float(deployed_knots.grad.abs().sum()) > 0.0


def test_joint_checkpoint_is_explicit_and_legacy_v14_stays_disabled() -> None:
    legacy_config, _ = migrate_model_config(
        {
            "objective_version": V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
            "model_config": {
                "point_dim": 2,
                "degree": 3,
                "hidden_dim": 16,
                "encoder_layers": 1,
                "max_internal_knots": 6,
                "gap_parameterization": "strict",
            },
        }
    )
    assert legacy_config["parameter_feedback_fusion_mode"] == "fast_global"
    assert legacy_config["joint_parameter_structure_feedback"] is False

    new_model_config = {
        "point_dim": 2,
        "hidden_dim": 16,
        "encoder_layers": 1,
        "max_internal_knots": 6,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": "threshold",
        "one_shot_joint_position_refinement": True,
        "one_shot_survivor_relocation": True,
        "parameter_feedback_fusion": True,
        "parameter_feedback_fusion_mode": "cross_attention",
        "joint_parameter_structure_feedback": True,
        "joint_parameter_structure_max_keep_logit_shift": 20.0,
        "parameter_gap_reference": "chord_residual",
        "parameter_residual_logit_limit": 0.5,
        "candidate_interval_logit_limit": 0.5,
        "candidate_position_parameterization": "bounded_anchor_residual",
        "enforce_ordered_joint_candidates": True,
    }
    model = SplineFittingNetwork(**new_model_config).eval()
    checkpoint = {
        "objective_version": V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
        "model_config": new_model_config,
        "model_state_dict": model.state_dict(),
    }
    restored, restored_config, _ = build_model_from_checkpoint(checkpoint)
    assert restored_config["joint_parameter_structure_feedback"] is True
    assert hasattr(restored, "joint_parameter_structure_head")
    assert restored_config["parameter_gap_reference"] == "chord_residual"
    assert restored.parameter_head.gap_reference == "chord_residual"
    assert restored.candidate_head.interval_logit_limit == 0.5
    assert (
        restored.candidate_head.position_parameterization
        == "bounded_anchor_residual"
    )
    assert restored.enforce_ordered_joint_candidates is True
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        expected = model.forward_deployment(points)
        actual = restored.eval().forward_deployment(points)
    torch.testing.assert_close(actual["proposal_params"], expected["proposal_params"])
    torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])
    assert torch.equal(actual["learned_keep_mask"], expected["learned_keep_mask"])
