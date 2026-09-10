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
    V16_CERTIFIED_SYNTHETIC_CONTRACT,
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
        "stage": stage,
        "training_config": {
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
            "qualification_dense_pass_rate": 0.98,
            "qualification_deployment_pass_rate": observed,
            "keep_count": 8.0,
            "synthetic_count_mae": 0.75,
            "synthetic_knot_match_f1": 0.8,
            "synthetic_knot_matched_mae": 0.004,
            "synthetic_boundary_knot_count": 56,
            "synthetic_boundary_sample_count": 32,
            "synthetic_boundary_dense_pass_rate": 0.98,
            "synthetic_boundary_deployment_pass_rate": observed,
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


def test_six_requested_methods_have_distinct_curve_colors():
    assert entry.METHODS == (
        "ours",
        "park_dominant_point_2007_adaptation",
        "liang_feature_iki_2017_adaptation",
        "dung_direct_knot_2017_adaptation",
        "kang_sparse_2015_adaptation",
        "luo_linf_de_2022_adaptation",
    )
    assert set(entry.PLOT_COLORS) == set(entry.METHODS)
    assert len(set(entry.PLOT_COLORS.values())) == len(entry.METHODS)


def test_visualization_cli_exposes_every_published_method_option_for_1e4_cuda():
    args = entry.parser().parse_args([
        "--mse-tolerance", "1e-4",
        "--device", "cuda",
        "--park-shape-weight", "0.7",
        "--liang-dense-knots", "64",
        "--liang-initial-knots", "3",
        "--liang-curvature-weight", "0.6",
        "--liang-feature-samples", "513",
        "--dung-scan-intervals", "12",
        "--dung-optimization-iterations", "9",
        "--luo-eta", "0.4",
        "--luo-de-population", "12",
        "--luo-de-iterations", "60",
        "--luo-seed", "7",
    ])
    assert args.mse_tolerance == pytest.approx(1e-4)
    assert args.device == "cuda"
    assert args.park_shape_weight == pytest.approx(0.7)
    assert args.liang_dense_knots == 64
    assert args.dung_scan_intervals == 12
    assert args.luo_de_population == 12


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


def test_six_method_case_writes_one_panel_per_method(tmp_path):
    case, base_result = simple_case_and_result()
    results = [
        base_result | {
            "method": method,
            "network_ms": 1.25 if method == "ours" else None,
        }
        for method in entry.METHODS
    ]
    output = tmp_path / "six.png"
    entry.plot_case(
        output, case=case, results=results, tolerance=1e-4,
        diagnostic=True, dpi=72,
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000

    with pytest.raises(ValueError, match="exactly this order"):
        entry.plot_case(
            tmp_path / "invalid.png", case=case, results=results[:-1],
            tolerance=1e-4, diagnostic=False, dpi=72,
        )


def test_ours_entry_point_supplies_current_checkpoint_and_output_defaults():
    arguments = ours_entry.with_ours_defaults(["--real-samples-per-dataset", "1"])
    assert "--ours-only" in arguments
    assert str(ours_entry.DEFAULT_CHECKPOINT) in arguments
    assert str(ours_entry.DEFAULT_OUTPUT_DIR) in arguments
    assert ours_entry.DEFAULT_CHECKPOINT.name == (
        "candidate_selection_v16_mse1e-4_k56_ordered_highk.pt"
    )
