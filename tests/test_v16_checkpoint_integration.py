from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spline_fitting.checkpointing import (
    LATEST_OBJECTIVE_VERSION,
    V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
    V16_ADAPTIVE_SELECTION_REVISION,
    V16_CERTIFIED_SYNTHETIC_CONTRACT,
    V16_FORMAL_PASS_RATE,
    V16_JOINT_CHECKPOINT_QUALITY,
    V16_SIMPLIFICATION_CONTRACT,
    V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
    assess_v16_checkpoint,
    build_model_from_checkpoint,
    migrate_model_config,
)
from spline_fitting.models.spline_network import SplineFittingNetwork


def make_v16_checkpoint():
    from spline_fitting.models.v16_network import V16CandidateSelectionNetwork

    model = V16CandidateSelectionNetwork(
        point_dim=2, hidden_dim=16, encoder_layers=1,
        max_internal_knots=4, attention_heads=4, selector_layers=1,
    ).eval()
    return model, {
        "objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        "model_config": model.get_config(),
        "model_state_dict": model.state_dict(),
    }


def qualified_v16_metadata(*, target=0.90, observed=0.92):
    return {
        "objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        "architecture_revision": V16_ADAPTIVE_SELECTION_REVISION,
        "simplification_contract": V16_SIMPLIFICATION_CONTRACT,
        "simplification_ready": True,
        "simplification_curriculum": {"aggregate_pass_feedback": False},
        "checkpoint_selection": "mean_per_curve_subset_cost_v1",
        "synthetic_data_contract": V16_CERTIFIED_SYNTHETIC_CONTRACT,
        "dataset_config": {
            "certified_minimal_source": True,
            "canonical_knot_tolerance": 0.005,
            "minimality_margin": 0.2,
            "minimality_audit_points": 512,
            "min_control_points": 8,
            "max_control_points": 60,
            "knot_min_span": 0.01,
        },
        "model_config": {
            "one_shot_selection_policy": "mass_topk",
            "one_shot_adaptive_threshold": True,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
            "max_internal_knots": 56,
        },
        "stage": "joint",
        "training_config": {
            "real_fraction": 0.0,
            "proposal_pass_target": target,
            "deployment_pass_target": target,
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "allow_infeasible_proposals": False,
            "final_safety_sigma": 0.05,
            "final_safety_knots": 0,
            "proposal_high_k_fraction": 0.5,
            "proposal_high_k_min_knots": 40,
        },
        "validation_metrics": {
            "worst_dense_pass_rate": 0.98,
            "worst_deployment_pass_rate": observed,
            "qualification_dense_pass_rate": 0.96,
            "qualification_deployment_pass_rate": observed,
            "keep_count": 8.0,
            "synthetic_count_mae": 0.75,
            "synthetic_knot_match_f1": 0.8,
            "synthetic_knot_matched_mae": 0.004,
            "synthetic_boundary_knot_count": 56,
            "synthetic_boundary_sample_count": 32,
            "synthetic_boundary_dense_pass_rate": 0.96,
            "synthetic_boundary_deployment_pass_rate": 0.92,
        },
        "deployment_config": {
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "one_shot_selection_policy": "mass_topk",
            "adaptive_keep_threshold": True,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
        },
        "loss_config": {
            "joint_supervision": "synthetic_ground_truth",
            "online_teacher": False,
            "ranked_prefix_teacher": False,
            "synthetic_count_role": "exact",
            "weights": {
                "fit_weight": 1.0,
                "distillation_weight": 2.0,
                "count_weight": 2.0,
                "supervised_count_weight": 1.0,
                "supervised_over_count_weight": 1.0,
                "complexity_weight": 0.0,
                "true_parameter_weight": 0.1,
                "proposal_knot_coverage_weight": 1.0,
                "proposal_knot_assignment_weight": 1.0,
                "selected_knot_position_weight": 1.0,
            },
        },
        "proposal_ready": True,
        "best_deployment_pass_constraint_satisfied": observed >= target,
        "checkpoint_quality": V16_JOINT_CHECKPOINT_QUALITY,
    }


def test_v16_formal_qualification_checks_metrics_and_configured_target():
    result = assess_v16_checkpoint(
        qualified_v16_metadata(), required_mse_tolerance=2.5e-5,
    )
    assert result["formal_reporting_eligible"]
    assert result["required_reporting_pass_rate"] == V16_FORMAL_PASS_RATE
    assert result["configured_target_met"]
    assert result["configured_proposal_high_k_fraction"] == pytest.approx(0.5)
    assert result["configured_proposal_high_k_min_knots"] == 40
    assert result["proposal_knot_assignment_weight"] == pytest.approx(1.0)

    relaxed = assess_v16_checkpoint(
        qualified_v16_metadata(target=0.85, observed=0.50)
    )
    assert relaxed["formal_reporting_eligible"]
    assert relaxed["pass_rate_reference_only"]
    assert not relaxed["pass_rate_reference_met"]


@pytest.mark.parametrize(
    "mutation, expected_reason",
    [
        (lambda value: value.update(stage="proposal"), "not from the joint stage"),
        (
            lambda value: value["simplification_curriculum"].update(
                aggregate_pass_feedback=True
            ),
            "aggregate pass-rate feedback",
        ),
        (
            lambda value: value.update(checkpoint_selection="legacy_pass_gate"),
            "mean per-curve subset cost",
        ),
        (
            lambda value: value.update(checkpoint_quality="deployment_target_met"),
            "joint checkpoint quality",
        ),
    ],
)
def test_v16_formal_qualification_rejects_unsafe_or_inconsistent_metadata(
    mutation, expected_reason,
):
    checkpoint = qualified_v16_metadata()
    mutation(checkpoint)
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any(expected_reason in reason for reason in result["reasons"])


def test_v16_formal_qualification_checks_exact_reporting_tolerance():
    result = assess_v16_checkpoint(
        qualified_v16_metadata(), required_mse_tolerance=1e-5,
    )
    assert not result["formal_reporting_eligible"]
    assert any("MSE tolerance" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda checkpoint: checkpoint["training_config"].update(
                proposal_high_k_fraction=0.0
            ),
            "50%",
        ),
        (
            lambda checkpoint: checkpoint["training_config"].update(
                proposal_high_k_min_knots=48
            ),
            "K=40",
        ),
        (
            lambda checkpoint: checkpoint["loss_config"]["weights"].update(
                proposal_knot_assignment_weight=0.0
            ),
            "proposal_knot_assignment_weight",
        ),
    ],
)
def test_v16_formal_qualification_requires_the_new_proposal_curriculum(
    mutation, reason,
):
    checkpoint = qualified_v16_metadata()
    mutation(checkpoint)
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any(reason in item for item in result["reasons"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda checkpoint: checkpoint["training_config"].update(
                real_fraction=0.35
            ),
            "labelled synthetic curves only",
        ),
        (
            lambda checkpoint: checkpoint["loss_config"].update(
                joint_supervision="online_teacher",
                online_teacher=True,
                ranked_prefix_teacher=True,
            ),
            "direct synthetic ground truth",
        ),
        (
            lambda checkpoint: checkpoint["loss_config"].update(
                synthetic_count_role="upper_bound"
            ),
            "exact label",
        ),
    ],
)
def test_v16_formal_qualification_requires_synthetic_labelled_joint(
    mutation, reason,
):
    checkpoint = qualified_v16_metadata()
    mutation(checkpoint)
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any(reason in item for item in result["reasons"])


def test_v16_configured_target_uses_boundary_aware_qualification_rate():
    checkpoint = qualified_v16_metadata(target=0.90, observed=0.95)
    checkpoint["validation_metrics"].update(
        synthetic_boundary_deployment_pass_rate=0.85,
        qualification_deployment_pass_rate=0.85,
    )
    checkpoint["best_deployment_pass_constraint_satisfied"] = False
    result = assess_v16_checkpoint(checkpoint)
    assert not result["configured_target_met"]
    assert not result["configured_checkpoint_accepted"]
    assert result["formal_reporting_eligible"]
    assert not result["pass_rate_reference_met"]


def test_v16_formal_qualification_rejects_legacy_fixed_threshold_revision():
    checkpoint = qualified_v16_metadata()
    checkpoint.pop("architecture_revision")
    checkpoint["model_config"] = {
        "one_shot_selection_policy": "threshold",
        "one_shot_adaptive_threshold": False,
    }
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any("adaptive-beta mass-TopK" in reason for reason in result["reasons"])
    assert any("not mass_topk" in reason for reason in result["reasons"])


def test_v16_formal_qualification_reports_degenerate_all_keep_solution():
    checkpoint = qualified_v16_metadata()
    checkpoint["validation_metrics"]["keep_count"] = 56.0
    result = assess_v16_checkpoint(checkpoint)
    assert result["formal_reporting_eligible"]
    assert result["observed_mean_retained_knots"] == 56.0


def test_v16_formal_qualification_rejects_uncertified_synthetic_contract():
    checkpoint = qualified_v16_metadata()
    checkpoint["synthetic_data_contract"] = "random_source_uncertified"
    checkpoint["dataset_config"]["certified_minimal_source"] = False
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any("synthetic data contract" in reason for reason in result["reasons"])
    assert any("certified-minimal" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda checkpoint: checkpoint.pop("simplification_contract"),
            "synthetic-ground-truth",
        ),
        (
            lambda checkpoint: checkpoint.update(simplification_ready=False),
            "curriculum",
        ),
        (
            lambda checkpoint: checkpoint["model_config"].update(
                one_shot_safety_knots=1
            ),
            "safety-knot",
        ),
    ],
)
def test_v16_formal_qualification_requires_simplification_contract(
    mutation, reason,
):
    checkpoint = qualified_v16_metadata()
    mutation(checkpoint)
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any(reason in item for item in result["reasons"])


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("canonical_knot_tolerance", 2.5e-5, "sqrt(MSE tolerance)"),
        ("minimality_margin", 0.0, "margin is below"),
        ("minimality_audit_points", 192, "fewer than 512"),
    ],
)
def test_v16_formal_qualification_checks_minimality_certificate_strength(
    field, value, reason,
):
    checkpoint = qualified_v16_metadata()
    checkpoint["dataset_config"][field] = value
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any(reason in item for item in result["reasons"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda checkpoint: checkpoint["dataset_config"].update(
                min_control_points=9
            ),
            "minimum must be 8 control points",
        ),
        (
            lambda checkpoint: checkpoint["dataset_config"].update(
                max_control_points=59
            ),
            "maximum must be 60 control points",
        ),
        (
            lambda checkpoint: checkpoint["dataset_config"].update(
                knot_min_span=0.02
            ),
            "knot_min_span must equal 0.01",
        ),
        (
            lambda checkpoint: checkpoint["model_config"].update(
                max_internal_knots=64
            ),
            "capacity must equal 56",
        ),
        (
            lambda checkpoint: checkpoint["validation_metrics"].update(
                synthetic_boundary_sample_count=0
            ),
            "at least 32 synthetic samples",
        ),
    ],
)
def test_v16_formal_qualification_enforces_k4_56_protocol(
    mutation, reason,
):
    checkpoint = qualified_v16_metadata()
    mutation(checkpoint)
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any(reason in item for item in result["reasons"])


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("synthetic_count_mae", 2.01, "count MAE exceeds"),
        ("synthetic_knot_match_f1", 0.59, "F1 is below"),
        ("synthetic_knot_matched_mae", 0.0051, "matched-knot MAE exceeds"),
        ("worst_dense_pass_rate", 0.89, "dense proposal pass rate"),
    ],
)
def test_v16_formal_qualification_reports_geometry_quality(field, value, reason):
    checkpoint = qualified_v16_metadata()
    checkpoint["validation_metrics"][field] = value
    result = assess_v16_checkpoint(checkpoint)
    assert result["formal_reporting_eligible"]
    assert not any(reason in item for item in result["reasons"])


def test_v16_formal_qualification_requires_active_losses_and_match_protocol():
    checkpoint = qualified_v16_metadata()
    checkpoint["loss_config"]["weights"]["supervised_count_weight"] = 0.0
    checkpoint["deployment_config"]["knot_match_tolerance"] = 0.02
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any("supervised_count_weight" in item for item in result["reasons"])
    assert any("match@0.01" in item for item in result["reasons"])


def test_v16_formal_qualification_rejects_large_final_safety_reserve():
    checkpoint = qualified_v16_metadata()
    checkpoint["training_config"]["final_safety_sigma"] = 0.10
    checkpoint["model_config"]["one_shot_safety_sigma"] = 0.10
    checkpoint["deployment_config"]["one_shot_safety_sigma"] = 0.10
    result = assess_v16_checkpoint(checkpoint)
    assert not result["formal_reporting_eligible"]
    assert any("formal simplification allowance" in item for item in result["reasons"])


def test_v16_config_bypasses_legacy_head_defaults():
    raw = {"point_dim": 2, "max_internal_knots": 4}
    config, legacy = migrate_model_config({
        "objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        "model_config": raw,
    })
    assert not legacy
    assert config == {**raw, "structure_mode": "candidate_pruning_one_shot"}
    assert "structure_mode" not in raw
    assert LATEST_OBJECTIVE_VERSION == V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION


def test_v16_config_rejects_legacy_structure():
    with pytest.raises(ValueError, match="v16 requires"):
        migrate_model_config({
            "objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
            "model_config": {"structure_mode": "hard_concrete"},
        })


def test_v16_checkpoint_round_trip_preserves_deployment():
    reference, checkpoint = make_v16_checkpoint()
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored, config, legacy = build_model_from_checkpoint(saved)
    assert type(restored) is type(reference)
    assert not legacy
    assert config == reference.get_config()
    points = torch.randn(2, 24, 2)
    with torch.inference_mode():
        expected = reference.forward_deployment(points, mse_tolerance=1e-5)
        actual = restored.eval().forward_deployment(points, mse_tolerance=1e-5)
    for key in ("params", "internal_knots", "learned_keep_mask", "keep_probability"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_v16_checkpoint_does_not_silently_drop_unknown_state():
    _, checkpoint = make_v16_checkpoint()
    checkpoint["model_state_dict"]["legacy_unused_parameter"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        build_model_from_checkpoint(checkpoint)


def test_v16_checkpoint_rejects_legacy_configuration_fields():
    _, checkpoint = make_v16_checkpoint()
    checkpoint["model_config"]["gate_mode"] = "hard_concrete"
    with pytest.raises(TypeError, match="gate_mode"):
        build_model_from_checkpoint(checkpoint)


def test_actual_v15_checkpoint_retains_original_class_and_every_weight():
    path = ROOT / "outputs/checkpoints/candidate_pruning_one_shot_v15.pt"
    if not path.is_file():
        pytest.skip("Local trained v15 checkpoint is not distributed with the repository")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    original_config = deepcopy(checkpoint["model_config"])
    model, config, legacy = build_model_from_checkpoint(checkpoint)
    assert type(model) is SplineFittingNetwork
    assert not legacy
    assert checkpoint["model_config"] == original_config
    assert config["structure_mode"] == "candidate_pruning_one_shot"
    restored_state = model.state_dict()
    assert restored_state.keys() == checkpoint["model_state_dict"].keys()
    for key, tensor in restored_state.items():
        torch.testing.assert_close(tensor, checkpoint["model_state_dict"][key], rtol=0, atol=0)


def test_legacy_evaluator_explains_v16_entry_points(monkeypatch, capsys):
    import evaluate_checkpoint

    monkeypatch.setattr(sys, "argv", ["evaluate_checkpoint.py", "--checkpoint", "v16.pt"])
    monkeypatch.setattr(evaluate_checkpoint.torch, "load", lambda *args, **kwargs: {
        "objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
    })
    with pytest.raises(SystemExit) as error:
        evaluate_checkpoint.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "benchmark_v16_datasets.py" in message
    assert "fit_v16_point_cloud.py" in message
