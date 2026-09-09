from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import visualize_v16_real_deployments as entry  # noqa: E402
import visualize_v16_ours_cases as ours_entry  # noqa: E402
from spline_fitting.checkpointing import (  # noqa: E402
    V16_ADAPTIVE_SELECTION_REVISION,
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_SIMPLIFICATION_CONTRACT,
)
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)


def checkpoint(*, stage="joint", quality="deployment_target_met", met=True,
               target=0.97, observed=0.98):
    return {
        "objective_version": V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "architecture_revision": V16_ADAPTIVE_SELECTION_REVISION,
        "simplification_contract": V16_SIMPLIFICATION_CONTRACT,
        "simplification_ready": True,
        "synthetic_data_contract": "source_subset_threshold_minimal_v1",
        "dataset_config": {
            "certified_minimal_source": True,
            "canonical_knot_tolerance": 0.005,
            "minimality_margin": 0.2,
            "minimality_audit_points": 512,
        },
        "model_config": {
            "one_shot_selection_policy": "mass_topk",
            "one_shot_adaptive_threshold": True,
            "max_internal_knots": 16,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
        },
        "stage": stage,
        "training_config": {
            "proposal_pass_target": target,
            "deployment_pass_target": target,
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "allow_infeasible_proposals": False,
            "final_safety_sigma": 0.05,
            "final_safety_knots": 0,
        },
        "validation_metrics": {
            "worst_dense_pass_rate": 0.98,
            "worst_deployment_pass_rate": observed,
            "keep_count": 8.0,
            "synthetic_count_mae": 0.75,
            "synthetic_knot_match_f1": 0.8,
            "synthetic_knot_matched_mae": 0.004,
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
                "selected_knot_position_weight": 1.0,
            },
        },
        "proposal_ready": True,
        "checkpoint_quality": quality,
        "best_deployment_pass_constraint_satisfied": met,
    }


def test_qualified_joint_checkpoint_has_no_watermark():
    assert entry.validate_checkpoint_for_visualization(
        checkpoint(), allow_unqualified_diagnostic=False,
    ) is False


@pytest.mark.parametrize(
    "payload",
    [
        checkpoint(stage="proposal", quality="target_not_met", met=False),
        checkpoint(stage="joint", quality="target_not_met", met=False, observed=0.50),
        checkpoint(target=0.85, observed=0.91),
    ],
)
def test_unqualified_checkpoint_is_rejected_by_default(payload):
    with pytest.raises(ValueError, match="allow-unqualified-diagnostic"):
        entry.validate_checkpoint_for_visualization(
            payload, allow_unqualified_diagnostic=False,
        )
    assert entry.validate_checkpoint_for_visualization(
        payload, allow_unqualified_diagnostic=True,
    ) is True


def test_non_v16_checkpoint_is_never_accepted_as_diagnostic():
    payload = checkpoint()
    payload["objective_version"] = "candidate_pruning_one_shot_teacher_v15"
    with pytest.raises(ValueError, match="requires objective_version"):
        entry.validate_checkpoint_for_visualization(
            payload, allow_unqualified_diagnostic=True,
        )


def test_four_methods_have_distinct_curve_colors():
    assert set(entry.PLOT_COLORS) == set(entry.METHODS)
    assert len(set(entry.PLOT_COLORS.values())) == len(entry.METHODS)


def simple_case_and_result():
    parameters = torch.linspace(0, 1, 32, dtype=torch.float64)
    points = torch.stack(
        [parameters, 0.18 * torch.sin(2 * torch.pi * parameters)], dim=-1,
    )
    fit = refit_bspline_control_points(
        parameters, points, torch.tensor([0.3, 0.68], dtype=torch.float64),
        degree=3, smoothness_weight=0.0, control_ridge=0.0,
        interpolate_endpoints=True,
    )
    case = {
        "dataset": "fixture",
        "sample_id": "curve-1",
        "points": points.float(),
        "reference": points,
    }
    result = {
        "status": "ok",
        "fit": fit,
        "final_k": 2,
        "mse": float(fit.fit_mse),
        "reference_mse": float(fit.fit_mse),
        "network_ms": 1.25,
        "total_ms": 1.75,
        "fit_pass": True,
    }
    return case, result


def test_ours_geometry_legend_distinguishes_every_spline_element():
    case, result = simple_case_and_result()
    figure, axis = plt.subplots()
    try:
        entry._plot_geometry(
            axis, case=case, result=result, color=entry.PLOT_COLORS["ours"],
            annotate_indices=True, ours_label=True,
        )
        labels = set(axis.get_legend_handles_labels()[1])
        assert labels == {
            "Original reference polyline (evaluation only)",
            "Input sampled points",
            "Ours deployed B-spline",
            "Control polygon",
            "Control vertices $P_i$",
            "Internal knots $C(u_i)$",
        }
    finally:
        plt.close(figure)


def test_ours_case_png_contains_explicit_knot_value_strip(tmp_path):
    case, result = simple_case_and_result()
    figure, axis = plt.subplots()
    try:
        entry._plot_knot_strip(axis, result["fit"])
        strip_text = "\n".join(text.get_text() for text in axis.texts)
        assert "u1=0.3000" in strip_text
        assert "u2=0.6800" in strip_text
    finally:
        plt.close(figure)

    output = tmp_path / "ours.png"
    entry.plot_ours_case(
        output, case=case, result=result, tolerance=2.5e-5,
        diagnostic=False, dpi=72,
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000

    overview = tmp_path / "overview.png"
    entry.plot_ours_overview(
        overview, items=[(case, result), (case | {"sample_id": "curve-2"}, result)],
        tolerance=2.5e-5, diagnostic=False, dpi=72,
    )
    assert overview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert overview.stat().st_size > 10_000


def test_ours_entry_point_supplies_current_checkpoint_and_output_defaults():
    arguments = ours_entry.with_ours_defaults(["--real-samples-per-dataset", "1"])
    assert "--ours-only" in arguments
    assert str(ours_entry.DEFAULT_CHECKPOINT) in arguments
    assert str(ours_entry.DEFAULT_OUTPUT_DIR) in arguments
    assert ours_entry.DEFAULT_CHECKPOINT.name == (
        "candidate_selection_v16_simplified_certified_k96.pt"
    )
