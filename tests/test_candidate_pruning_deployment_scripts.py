from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_problem() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    parameters = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    candidates = torch.tensor([0.2, 0.45, 0.75], dtype=torch.float64)
    seed_points = torch.stack(
        [parameters, parameters.square() + 0.1 * torch.sin(8.0 * parameters)],
        dim=-1,
    )
    generating_fit = refit_bspline_control_points(
        parameters,
        seed_points,
        candidates,
        smoothness_weight=0.0,
    )
    points = generating_fit.reconstructed_points.unsqueeze(0)
    output = {
        "params": parameters.unsqueeze(0),
        "internal_knots": candidates.unsqueeze(0),
        # Deliberately contradictory: deployment must ignore this learned hint.
        "keep_probability": torch.zeros(1, candidates.numel(), dtype=torch.float64),
    }
    return output, points


def _one_shot_problem() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    output, points = _candidate_problem()
    output["keep_probability"] = torch.tensor([[0.9, 0.2, 0.7]], dtype=torch.float64)
    output["learned_keep_mask"] = torch.tensor([[True, False, False]], dtype=torch.bool)
    output["adaptive_keep_threshold"] = torch.tensor([0.63], dtype=torch.float64)
    output["keep_importance_logits"] = torch.tensor(
        [[2.0, -1.0, 0.4]], dtype=torch.float64
    )
    return output, points


def test_fit_tolerance_resolution_prefers_cli_then_deployment_then_dataset() -> None:
    module = _load_script("evaluate_checkpoint")
    checkpoint = {
        "deployment_config": {"error_tolerance": 0.03},
        "dataset_config": {"canonical_knot_tolerance": 0.02},
    }

    assert module.resolve_fit_tolerance(checkpoint, 0.04) == 0.04
    assert module.resolve_fit_tolerance(checkpoint, None) == 0.03
    del checkpoint["deployment_config"]
    assert module.resolve_fit_tolerance(checkpoint, None) == 0.02


def test_evaluation_candidate_batch_uses_hard_final_fit_not_keep_probability() -> None:
    module = _load_script("evaluate_checkpoint")
    output, points = _candidate_problem()

    deployed, results = module.prune_candidate_output_batch(
        output,
        points,
        error_tolerance=1.0,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )

    assert len(deployed) == len(results) == 1
    assert results[0].final_count == 0
    assert deployed[0].retained_count == results[0].final_count
    assert deployed[0].spline is results[0].final_fit
    assert bool(results[0].threshold_satisfied)


def test_point_cloud_and_visualization_adapters_replay_relative_deletions() -> None:
    evaluate_module = _load_script("evaluate_checkpoint")
    point_cloud_module = _load_script("fit_point_cloud")
    visualize_module = _load_script("visualize_result")
    output, points = _candidate_problem()
    _, results = evaluate_module.prune_candidate_output_batch(
        output,
        points,
        error_tolerance=0.02,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )
    result = results[0]

    point_cloud_fit = point_cloud_module.pruning_result_as_deployed_fit(result)
    visualization_fit = visualize_module.pruning_result_as_deployed_fit(result)

    assert int(point_cloud_fit.retained_mask.sum()) == result.final_count
    assert torch.equal(
        point_cloud_fit.retained_mask,
        visualization_fit.retained_mask,
    )
    assert torch.equal(
        point_cloud_fit.retained_internal_knots,
        result.final_internal_knots,
    )


def test_visualization_refits_all_and_empty_learned_candidate_masks() -> None:
    module = _load_script("visualize_result")
    output, points = _candidate_problem()
    candidates = output["internal_knots"][0]
    parameters = output["params"][0]

    all_fit = module.refit_candidate_mask_as_deployed_fit(
        parameters,
        points[0],
        candidates,
        torch.ones_like(candidates, dtype=torch.bool),
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )
    empty_fit = module.refit_candidate_mask_as_deployed_fit(
        parameters,
        points[0],
        candidates,
        torch.zeros_like(candidates, dtype=torch.bool),
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )

    assert all_fit.retained_count == candidates.numel()
    assert torch.equal(all_fit.retained_internal_knots, candidates)
    assert empty_fit.retained_count == 0
    assert empty_fit.control_points.shape[0] == 4
    assert torch.isfinite(empty_fit.reconstructed_points).all()
    assert torch.allclose(empty_fit.reconstructed_points[0], points[0, 0])
    assert torch.allclose(empty_fit.reconstructed_points[-1], points[0, -1])


def test_visualization_pruning_view_cli_lists_all_modes() -> None:
    source = (ROOT / "scripts" / "visualize_result.py").read_text(encoding="utf-8")

    assert '"--pruning-view"' in source
    assert 'choices=("all", "learned", "hard", "comparison")' in source


def test_v8_mode_detection_uses_structure_or_objective_feature() -> None:
    module = _load_script("evaluate_checkpoint")

    assert module.candidate_mode_flags(
        {}, {"structure_mode": "candidate_pruning_one_shot"}
    ) == (True, True)
    assert module.candidate_mode_flags(
        {"objective_version": "candidate_pruning_one_shot_v8"},
        {"structure_mode": "candidate_pruning"},
    ) == (True, True)
    assert module.candidate_mode_flags({}, {"structure_mode": "candidate_pruning"}) == (
        True,
        False,
    )


def test_v8_selection_prefers_learned_mask_and_only_reports_adaptive_threshold() -> (
    None
):
    module = _load_script("evaluate_checkpoint")
    output, _ = _one_shot_problem()

    mask, adaptive, source = module.one_shot_selection(output)

    assert source == "learned_keep_mask"
    assert torch.equal(mask, output["learned_keep_mask"])
    torch.testing.assert_close(adaptive, output["adaptive_keep_threshold"])
    # Candidate 2 has keep_probability > 0.5 but the authoritative learned
    # mask removes it. The raw adaptive threshold must not be reapplied here.
    assert not bool(mask[0, 2])


def test_v8_probability_fallback_uses_centered_half_not_raw_adaptive_threshold() -> (
    None
):
    module = _load_script("evaluate_checkpoint")
    output, _ = _one_shot_problem()
    del output["learned_keep_mask"]
    output["keep_probability"] = torch.tensor([[0.4, 0.6, 0.7]], dtype=torch.float64)
    output["adaptive_keep_threshold"] = torch.tensor([0.8], dtype=torch.float64)

    mask, _, source = module.one_shot_selection(output)

    assert source == "keep_probability>=0.5_fallback"
    assert torch.equal(mask, torch.tensor([[False, True, True]]))


def test_v8_evaluation_refits_once_without_calling_hard_pruner(monkeypatch) -> None:
    module = _load_script("evaluate_checkpoint")
    output, points = _one_shot_problem()

    def forbidden_hard_pruning(*args, **kwargs):
        raise AssertionError("v8 deployment must not run hard pruning")

    monkeypatch.setattr(module, "prune_knots_to_rms_tolerance", forbidden_hard_pruning)
    deployed, mask, adaptive, source = module.refit_one_shot_output_batch(
        output,
        points,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )

    assert len(deployed) == 1
    assert deployed[0].retained_count == 1
    assert torch.equal(mask, output["learned_keep_mask"])
    assert float(adaptive[0]) == 0.63
    assert source == "learned_keep_mask"
    assert torch.allclose(deployed[0].reconstructed_points[0], points[0, 0])
    assert torch.allclose(deployed[0].reconstructed_points[-1], points[0, -1])


def test_v8_point_cloud_refit_uses_source_resolution_and_learned_mask() -> None:
    module = _load_script("fit_point_cloud")
    output, points = _one_shot_problem()
    source_parameters = output["params"][0]

    deployed, mask, adaptive, source = module.refit_one_shot_full_resolution(
        output,
        source_parameters,
        points,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )

    assert deployed.reconstructed_points.shape == points[0].shape
    assert deployed.retained_count == int(output["learned_keep_mask"].sum())
    assert torch.equal(mask, output["learned_keep_mask"][0])
    assert float(adaptive) == 0.63
    assert source == "learned_keep_mask"


@pytest.mark.parametrize(
    "script_name",
    ("evaluate_checkpoint", "fit_point_cloud", "visualize_result"),
)
def test_v8_selection_accepts_only_boolean_or_strict_binary_masks(
    script_name: str,
) -> None:
    module = _load_script(script_name)
    output, _ = _one_shot_problem()
    output["learned_keep_mask"] = torch.tensor([[1.0, 0.0, 1.0]])

    mask, _, source = module.one_shot_selection(output)

    assert source == "learned_keep_mask"
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([[True, False, True]]))

    output["learned_keep_mask"] = torch.tensor([[1.0, 0.25, 0.0]])
    with pytest.raises(ValueError, match="strictly 0/1"):
        module.one_shot_selection(output)

    output["learned_keep_mask"] = torch.tensor([[1.0, float("nan"), 0.0]])
    with pytest.raises(ValueError, match="finite"):
        module.one_shot_selection(output)


def test_v8_reports_distinguish_proxy_rms_and_offline_view_semantics() -> None:
    evaluate_source = (ROOT / "scripts" / "evaluate_checkpoint.py").read_text(
        encoding="utf-8"
    )
    fit_source = (ROOT / "scripts" / "fit_point_cloud.py").read_text(encoding="utf-8")
    visualize_source = (ROOT / "scripts" / "visualize_result.py").read_text(
        encoding="utf-8"
    )

    assert '"network_objective_role"' in evaluate_source
    assert "inference_proxy_teacher_terms_unavailable" in evaluate_source
    assert '"standard_bspline_refit_rms_euclidean_pooled"' in evaluate_source
    assert '"standard_bspline_refit_rms_euclidean_mean_curve"' in evaluate_source
    assert '"threshold_satisfied_fraction"' in evaluate_source
    assert '"learned_one_shot_mask" if candidate_one_shot' in fit_source
    assert "selected offline diagnostic" in visualize_source
    assert "teacher terms unavailable" in visualize_source
