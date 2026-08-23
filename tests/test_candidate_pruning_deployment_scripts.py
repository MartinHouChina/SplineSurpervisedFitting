from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
