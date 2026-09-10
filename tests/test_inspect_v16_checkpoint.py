from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import inspect_v16_checkpoint as entry
from spline_fitting.checkpointing import (
    V16_ADAPTIVE_SELECTION_REVISION,
    V16_CERTIFIED_SYNTHETIC_CONTRACT,
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_SIMPLIFICATION_CONTRACT,
)


def _checkpoint(*, configured: float = 0.90, observed: float = 0.92) -> dict:
    accepted = observed >= configured
    return {
        "objective_version": V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "architecture_revision": V16_ADAPTIVE_SELECTION_REVISION,
        "simplification_contract": V16_SIMPLIFICATION_CONTRACT,
        "simplification_ready": True,
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
            "max_internal_knots": 56,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
        },
        "stage": "joint",
        "epoch": 73,
        "training_config": {
            "proposal_pass_target": configured,
            "deployment_pass_target": configured,
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "allow_infeasible_proposals": False,
            "final_safety_sigma": 0.05,
            "final_safety_knots": 0,
            "proposal_high_k_fraction": 0.5,
            "proposal_high_k_min_knots": 40,
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
            "ranked_prefix_teacher": True,
            "weights": {
                "count_weight": 2.0,
                "supervised_count_weight": 1.0,
                "supervised_over_count_weight": 1.0,
                "complexity_weight": 0.05,
                "true_parameter_weight": 0.1,
                "proposal_knot_coverage_weight": 1.0,
                "proposal_knot_assignment_weight": 1.0,
                "selected_knot_position_weight": 1.0,
            },
        },
        "validation_metrics": {
            "dense_pass_rate": 0.99,
            "deployment_pass_rate": 0.985,
            "worst_dense_pass_rate": 0.98,
            "worst_deployment_pass_rate": observed,
            "qualification_dense_pass_rate": 0.96,
            "qualification_deployment_pass_rate": observed,
            "keep_count": 8.0,
            "synthetic_count_mae": 0.75,
            "synthetic_knot_match_f1": 0.82,
            "synthetic_knot_matched_mae": 0.003,
            "synthetic_boundary_knot_count": 56,
            "synthetic_boundary_sample_count": 32,
            "synthetic_boundary_dense_pass_rate": 0.96,
            "synthetic_boundary_deployment_pass_rate": observed,
            "by_source": {
                "Synthetic": {
                    "dense_pass_rate": 1.0,
                    "deployment_pass_rate": 0.99,
                    "dense_mse": 1e-6,
                    "deployment_mse": 2e-6,
                    "keep_count": 9.5,
                    "target_count_mean": 9.0,
                    "count_mae": 0.75,
                    "count_bias": 0.5,
                    "knot_match_f1": 0.82,
                    "knot_matched_mae": 0.003,
                }
            },
        },
        "proposal_ready": True,
        "best_deployment_pass_constraint_satisfied": accepted,
        "checkpoint_quality": "deployment_target_met" if accepted else "target_not_met",
    }


def test_inspector_reports_metrics_and_returns_qualification_status(tmp_path, capsys):
    qualified = tmp_path / "qualified.pt"
    torch.save(_checkpoint(), qualified)
    assert entry.main(["--checkpoint", str(qualified), "--mse-tolerance", "2.5e-5"]) == 0
    output = capsys.readouterr().out
    assert "stage: joint" in output and "epoch: 73" in output
    assert "certified minimal source: True" in output
    assert "source control-point range: 8..60" in output
    assert "source internal-knot range: K=4..56" in output
    assert "knot min span: 0.01" in output
    assert "high-K synthetic allocation: 50.000%" in output
    assert "high-K stratum begins at internal K: 40" in output
    assert "ordered assignment weight: 1" in output
    assert "overall dense: 99.000%" in output
    assert "synthetic boundary: K=56, n=32" in output
    assert "qualification dense/deployment" in output
    assert "Synthetic: dense=100.000%" in output
    assert "target_K=9" in output and "knot_F1=0.82" in output
    assert "eligible: YES" in output and "reasons: none" in output

    relaxed = tmp_path / "relaxed.pt"
    torch.save(_checkpoint(configured=0.85), relaxed)
    assert entry.main(["--checkpoint", str(relaxed)]) == 2
    output = capsys.readouterr().out
    assert "eligible: NO" in output
    assert "configured deployment pass target 85.000% is below" in output


def test_inspector_routes_bad_input_through_argparse(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        entry.main(["--checkpoint", str(tmp_path / "missing.pt")])
    assert error.value.code == 2
    assert "checkpoint does not exist" in capsys.readouterr().err
