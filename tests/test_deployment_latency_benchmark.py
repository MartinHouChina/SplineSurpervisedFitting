from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_deployment_latency.py"
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.models.spline_network import SplineFittingNetwork  # noqa: E402


def _load_script():
    module_name = "test_deployment_latency_benchmark_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_script()


def test_percentile_interpolates_without_rounding_to_a_sample() -> None:
    assert benchmark._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert benchmark._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_measure_separates_warmup_from_reported_repeats() -> None:
    calls = 0

    def measured() -> int:
        nonlocal calls
        calls += 1
        return calls

    result, timing = benchmark._measure(
        measured,
        synchronization_device=torch.device("cpu"),
        warmup=2,
        repeats=3,
    )

    assert result == calls == 5
    assert len(timing["repeat_ms"]) == 3
    assert timing["p50_ms"] == timing["median_ms"]
    assert timing["minimum_ms"] <= timing["p50_ms"] <= timing["maximum_ms"]


def test_deployment_geometries_preserve_full_batch() -> None:
    proposal = torch.tensor([[0.2, 0.7], [0.1, 0.8]])
    deployment = torch.tensor([[0.25, 0.65], [0.15, 0.75]])
    mask = torch.tensor([[True, False], [False, True]])

    actual = benchmark._deployment_geometries(
        {
            "internal_knots": deployment,
            "proposal_internal_knots": proposal,
            "deployment_internal_knots": deployment,
            "keep_probability": torch.full_like(proposal, 0.5),
            "learned_keep_mask": mask,
        }
    )

    torch.testing.assert_close(actual[0], proposal)
    torch.testing.assert_close(actual[1], deployment)
    torch.testing.assert_close(actual[2], mask)


def test_refit_modes_are_explicitly_separate_from_end_to_end_modes() -> None:
    assert "network-forward" in benchmark.DEPLOYMENT_MODES
    assert "refit-network" in benchmark.DEPLOYMENT_MODES
    assert "refit-chord" in benchmark.DEPLOYMENT_MODES
    assert "learned-network" in benchmark.DEPLOYMENT_MODES
    assert "learned-chord" in benchmark.DEPLOYMENT_MODES


def test_structure_only_forward_preserves_deployment_outputs() -> None:
    torch.manual_seed(17)
    model = SplineFittingNetwork(
        point_dim=2,
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=4,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        one_shot_fixed_proposal_geometry=True,
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=True,
    ).eval()
    parameter = torch.linspace(0.0, 1.0, 24)
    points = torch.stack([parameter, parameter.square()], dim=-1).unsqueeze(0)

    with torch.inference_mode():
        full = model(points)
        structure = model.forward_deployment(points)

    for key in (
        "params",
        "proposal_internal_knots",
        "deployment_internal_knots",
        "keep_probability",
        "learned_keep_mask",
        "predicted_knot_count",
    ):
        torch.testing.assert_close(structure[key], full[key], rtol=0.0, atol=0.0)
    assert "reconstructed_points" in full
    assert "reconstructed_points" not in structure
    assert "coefficients" not in structure
