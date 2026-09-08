from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (  # noqa: E402
    V13_SET_RELOCATION_OBJECTIVE_VERSION,
)
from spline_fitting.models import SplineFittingNetwork  # noqa: E402
from spline_fitting.data.real_world import write_curve_manifest  # noqa: E402
from spline_fitting.losses import ParameterFeedbackLoss  # noqa: E402


def _load_script():
    path = ROOT / "scripts" / "finetune_parameter_feedback.py"
    spec = importlib.util.spec_from_file_location(
        "test_finetune_parameter_feedback_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feedback_script = _load_script()


def _source_checkpoint() -> dict[str, object]:
    model_config: dict[str, object] = {
        "point_dim": 2,
        "degree": 3,
        "hidden_dim": 16,
        "encoder_layers": 1,
        "max_internal_knots": 5,
        "min_parameter_gap": 1e-4,
        "min_knot_gap": 1e-3,
        "gap_parameterization": "strict",
        "lambda_poly": 1e-6,
        "lambda_knot": 1e-5,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
        "pruning_residual_bandwidth": 0.05,
        "pruning_initial_keep_probability": 0.55,
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": "mass_topk",
        "one_shot_safety_sigma": 0.25,
        "one_shot_selector_layers": 2,
        "one_shot_coverage_bins": 0,
        "candidate_local_attention_bandwidth": 0.08,
        "one_shot_joint_position_refinement": True,
        "one_shot_survivor_relocation": True,
        "one_shot_max_position_shift": 0.15,
        "stable_pilot_descriptors": True,
        "compute_first_derivative": False,
    }
    source_model = SplineFittingNetwork(**model_config)
    return {
        "objective_version": V13_SET_RELOCATION_OBJECTIVE_VERSION,
        "model_config": model_config,
        "model_state_dict": source_model.state_dict(),
        "dataset_config": {
            "num_points": 24,
            "point_dim": 2,
            "min_control_points": 5,
            "max_control_points": 9,
            "noise_std": 0.001,
            "knot_nonuniformity": 0.65,
            "sampling_nonuniformity": 0.45,
            "turn_strength": 0.45,
            "canonical_knot_tolerance": 0.005,
            "normalize": True,
            "return_ground_truth": True,
        },
    }


def test_help_command_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/finetune_parameter_feedback.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "No offline teacher" in completed.stdout


def test_legacy_no_joint_path_allows_only_new_feedback_keys() -> None:
    checkpoint = _source_checkpoint()
    source_state = checkpoint["model_state_dict"]
    assert isinstance(source_state, dict)
    model, config, missing = feedback_script.build_feedback_model_from_checkpoint(
        checkpoint,
        attention_heads=None,
        max_logit_shift=0.4,
        joint_parameter_structure_feedback=False,
    )

    assert config["parameter_feedback_fusion"] is True
    assert config["parameter_feedback_attention_heads"] == 4
    assert config["parameter_feedback_max_logit_shift"] == 0.4
    assert config["stable_pilot_descriptors"] is True
    assert missing
    assert all(key.startswith("parameter_feedback_head.") for key in missing)
    restored = model.state_dict()
    for key, expected in source_state.items():
        torch.testing.assert_close(restored[key], expected)

    trainable = feedback_script.freeze_except_parameter_feedback(model)
    assert trainable
    trainable_ids = {id(parameter) for parameter in trainable}
    for name, parameter in model.named_parameters():
        del name
        assert parameter.requires_grad == (id(parameter) in trainable_ids)


def test_old_checkpoint_rejects_missing_nonfeedback_weight() -> None:
    checkpoint = _source_checkpoint()
    state = dict(checkpoint["model_state_dict"])
    state.pop("parameter_head.mlp.0.weight")
    checkpoint["model_state_dict"] = state

    with pytest.raises(RuntimeError, match="invalid missing keys"):
        feedback_script.build_feedback_model_from_checkpoint(
            checkpoint,
            attention_heads=4,
            max_logit_shift=0.5,
        )


def test_existing_v14_feedback_checkpoint_can_continue_calibration() -> None:
    source = _source_checkpoint()
    first_model, first_config, _ = feedback_script.build_feedback_model_from_checkpoint(
        source,
        attention_heads=4,
        max_logit_shift=0.5,
        joint_parameter_structure_feedback=False,
    )
    with torch.no_grad():
        first_model.parameter_feedback_head.chord_blend_weight.fill_(0.37)
    continued_checkpoint = {
        **source,
        "model_config": first_config,
        "model_state_dict": first_model.state_dict(),
    }

    continued, config, missing = feedback_script.build_feedback_model_from_checkpoint(
        continued_checkpoint,
        attention_heads=None,
        max_logit_shift=0.5,
        joint_parameter_structure_feedback=False,
    )

    assert missing == []
    assert config["parameter_feedback_fusion"] is True
    torch.testing.assert_close(
        continued.parameter_feedback_head.chord_blend_weight,
        torch.tensor(0.37),
    )


def test_generic_structure_checkpoint_can_reuse_proposal_metadata() -> None:
    metadata = _source_checkpoint()
    attached, _, _ = feedback_script.build_feedback_model_from_checkpoint(
        metadata,
        attention_heads=4,
        max_logit_shift=0.5,
    )
    with torch.no_grad():
        attached.parameter_feedback_head.chord_blend_weight.fill_(0.9)
        attached.parameter_head.mlp[0].weight.add_(0.125)
    generic = {
        "model_state_dict": attached.state_dict(),
        "stage": "one_shot_selector_calibration",
        "epoch": 100,
    }

    recovered = feedback_script.compose_structure_checkpoint(generic, metadata)

    assert recovered["stage"] == "one_shot_selector_calibration"
    assert recovered["epoch"] == 100
    assert recovered["recovered_structure_checkpoint"] is True
    assert recovered["model_config"]["parameter_feedback_fusion"] is False
    assert not any(
        name.startswith("parameter_feedback_head.")
        for name in recovered["model_state_dict"]
    )
    model, _, missing = feedback_script.build_feedback_model_from_checkpoint(
        recovered,
        attention_heads=4,
        max_logit_shift=0.5,
    )
    assert missing
    torch.testing.assert_close(
        model.parameter_head.mlp[0].weight,
        attached.parameter_head.mlp[0].weight,
    )
    assert float(model.parameter_feedback_head.chord_blend_weight.detach()) == 0.0


def test_default_joint_upgrade_attaches_cross_attention_and_joint_heads() -> None:
    checkpoint = _source_checkpoint()
    source_state = checkpoint["model_state_dict"]
    assert isinstance(source_state, dict)

    model, config, missing = feedback_script.build_feedback_model_from_checkpoint(
        checkpoint,
        attention_heads=None,
        max_logit_shift=0.5,
    )

    assert config["parameter_feedback_fusion_mode"] == "cross_attention"
    assert config["joint_parameter_structure_feedback"] is True
    assert hasattr(model, "joint_parameter_structure_head")
    assert any(key.startswith("parameter_feedback_head.") for key in missing)
    assert any(key.startswith("joint_parameter_structure_head.") for key in missing)
    for key, expected in source_state.items():
        torch.testing.assert_close(model.state_dict()[key], expected)

    primary, relocation = feedback_script.feedback_training_parameter_groups(model)
    assert primary
    assert relocation
    primary_ids = {id(parameter) for parameter in primary}
    relocation_ids = {id(parameter) for parameter in relocation}
    assert primary_ids.isdisjoint(relocation_ids)
    assert all(parameter.requires_grad for parameter in (*primary, *relocation))


def test_legacy_v14_fast_global_is_neutrally_reattached_for_joint_upgrade() -> None:
    source = _source_checkpoint()
    legacy_model, legacy_config, _ = (
        feedback_script.build_feedback_model_from_checkpoint(
            source,
            attention_heads=4,
            max_logit_shift=0.5,
            joint_parameter_structure_feedback=False,
        )
    )
    with torch.no_grad():
        legacy_model.parameter_feedback_head.chord_blend_weight.fill_(0.75)
    legacy_checkpoint = {
        **source,
        "model_config": legacy_config,
        "model_state_dict": legacy_model.state_dict(),
    }

    upgraded, config, missing = feedback_script.build_feedback_model_from_checkpoint(
        legacy_checkpoint,
        attention_heads=4,
        max_logit_shift=0.5,
    )

    assert config["parameter_feedback_fusion_mode"] == "cross_attention"
    assert config["joint_parameter_structure_feedback"] is True
    assert any(key.startswith("parameter_feedback_head.") for key in missing)
    assert any(key.startswith("joint_parameter_structure_head.") for key in missing)
    assert float(upgraded.parameter_feedback_head.chord_blend_weight.detach()) == 0.0


def test_real_world_joint_loss_has_no_label_supervision() -> None:
    parser = feedback_script._build_parser()
    args = parser.parse_args(["--checkpoint", "unused.pt"])

    weights = feedback_script.build_feedback_loss_weights(
        args,
        real_world_unlabeled=True,
    )

    assert weights.fit == 0.0
    assert weights.threshold_violation == 0.0
    assert weights.deployment_fit > 0.0
    assert weights.deployment_threshold_violation > 0.0
    assert weights.true_parameter == 0.0
    assert weights.joint_keep == 0.0
    assert weights.joint_position == 0.0
    assert weights.joint_count == 0.0


def test_adapter_maps_synthetic_true_knots_but_never_unlabeled_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []
    loss = ParameterFeedbackLoss()

    def capture(
        output: dict[str, torch.Tensor],
        points: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del output
        captured.append(kwargs)
        return {"loss": points.new_zeros(())}

    monkeypatch.setattr(loss, "forward", capture)
    output = {
        "proposal_internal_knots": torch.tensor([[0.1, 0.3, 0.6, 0.9]]),
    }
    points = torch.zeros(1, 8, 2)
    true_knots = torch.tensor([[0.28, 0.88, 0.0]])
    true_mask = torch.tensor([[True, True, False]])

    supervised = feedback_script._TrainerCompatibleFeedbackLoss(
        loss,
        canonical_structure_supervision=True,
    )
    supervised(
        output,
        points,
        true_internal_knots=true_knots,
        true_internal_knot_mask=true_mask,
    )
    keep = captured[-1]["teacher_retained_mask"]
    assert isinstance(keep, torch.Tensor)
    assert keep.tolist() == [[False, True, False, True]]
    assert captured[-1]["teacher_count"].tolist() == [2]

    unlabeled = feedback_script._TrainerCompatibleFeedbackLoss(
        loss,
        canonical_structure_supervision=False,
    )
    unlabeled(output, points)
    assert "teacher_retained_mask" not in captured[-1]


def test_exact_deployment_selection_preserves_identity_on_regression() -> None:
    identity = {
        "threshold_satisfied_rate": 0.97,
        "standard_bspline_mse_p95": 3e-5,
        "standard_bspline_mse_mean": 2e-5,
        "parameter_rmse": 0.02,
        "structure_fingerprint": "same",
    }
    improved = {
        **identity,
        "standard_bspline_mse_p95": 2.5e-5,
        "standard_bspline_mse_mean": 1.8e-5,
    }
    regressed = {
        **identity,
        "threshold_satisfied_rate": 0.96,
        "standard_bspline_mse_mean": 1e-5,
    }
    changed_structure = {**improved, "structure_fingerprint": "different"}

    assert feedback_script.choose_feedback_candidate(
        identity,
        improved,
        pass_rate_target=0.97,
    )
    assert not feedback_script.choose_feedback_candidate(
        identity,
        regressed,
        pass_rate_target=0.97,
    )
    assert not feedback_script.choose_feedback_candidate(
        identity,
        changed_structure,
        pass_rate_target=0.97,
    )
    assert feedback_script.choose_feedback_candidate(
        identity,
        changed_structure,
        pass_rate_target=0.97,
        preserve_structure=False,
    )


def test_real_world_split_is_filtered_resampled_and_capped(tmp_path: Path) -> None:
    records = []
    for index, split in enumerate(("train", "train", "val")):
        points = np.stack(
            [np.linspace(0.0, 1.0, 9), np.linspace(0.0, 1.0, 9) ** 2],
            axis=-1,
        ).astype(np.float32)
        path = tmp_path / f"curve_{index}.npy"
        np.save(path, points)
        records.append(
            {
                "sample_id": f"sample-{index}",
                "source_dataset": "fixture",
                "group_id": f"group-{index}",
                "split": split,
                "points_path": path.name,
                "num_points": 9,
                "point_dim": 2,
                "has_knot_labels": False,
                "metadata": {},
            }
        )
    manifest = write_curve_manifest(records, tmp_path / "manifest.jsonl")

    train = feedback_script.build_real_world_split(
        [manifest],
        split="train",
        num_points=12,
        maximum_size=1,
        seed=7,
    )
    validation = feedback_script.build_real_world_split(
        [manifest],
        split="val",
        num_points=12,
        maximum_size=4,
        seed=8,
    )

    assert len(train) == 1
    assert len(validation) == 1
    assert validation[0]["points"].shape == (12, 2)
    assert validation[0]["has_knot_labels"] is False
