from __future__ import annotations

# The test loads a repository script directly after adding ``src`` to sys.path.
# ruff: noqa: E402

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points


def _load_script():
    path = ROOT / "scripts" / "fit_point_cloud.py"
    spec = importlib.util.spec_from_file_location(
        "test_hybrid_fit_point_cloud_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("requested", "model_device", "expected"),
    (
        ("auto", torch.device("cuda:0"), torch.device("cpu")),
        ("cpu", torch.device("cuda:0"), torch.device("cpu")),
        ("model", torch.device("cuda:0"), torch.device("cuda:0")),
        ("auto", torch.device("cpu"), torch.device("cpu")),
    ),
)
def test_verified_refit_device_resolution(
    requested: str,
    model_device: torch.device,
    expected: torch.device,
) -> None:
    module = _load_script()

    assert (
        module.resolve_verified_refit_device(
            requested,
            model_device=model_device,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("candidate_pruning", "candidate_one_shot", "expected"),
    (
        (False, False, "legacy"),
        (True, False, "hard"),
        (True, True, "learned"),
    ),
)
def test_checkpoint_mode_preserves_checkpoint_family_default(
    candidate_pruning: bool,
    candidate_one_shot: bool,
    expected: str,
) -> None:
    module = _load_script()

    actual = module.resolve_deployment_mode(
        "checkpoint",
        candidate_pruning=candidate_pruning,
        candidate_one_shot=candidate_one_shot,
    )

    assert actual == expected


def test_quality_modes_reject_incompatible_checkpoint_families() -> None:
    module = _load_script()

    for mode in ("verified", "hybrid"):
        with pytest.raises(ValueError, match=rf"{mode}.*one-shot"):
            module.resolve_deployment_mode(
                mode,
                candidate_pruning=True,
                candidate_one_shot=False,
            )
    with pytest.raises(ValueError, match="hard.*candidate-pruning"):
        module.resolve_deployment_mode(
            "hard",
            candidate_pruning=False,
            candidate_one_shot=False,
        )


def test_verified_mode_is_available_only_for_one_shot_checkpoints() -> None:
    module = _load_script()

    assert (
        module.resolve_deployment_mode(
            "verified",
            candidate_pruning=True,
            candidate_one_shot=True,
        )
        == "verified"
    )


def test_hybrid_wrapper_uses_source_resolution_and_squares_rms_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source_parameters = torch.linspace(0.0, 1.0, 47, dtype=torch.float64)
    full_resolution_points = torch.stack(
        [source_parameters, source_parameters.square()], dim=-1
    ).unsqueeze(0)
    proposal = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float64)
    deployment = torch.tensor([[0.21, 0.48, 0.79]], dtype=torch.float64)
    learned_mask = torch.tensor([True, False, True])
    output = {
        "proposal_internal_knots": proposal,
        "deployment_internal_knots": deployment,
        "internal_knots": deployment,
    }
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_search(parameters, points, proposal_knots, **kwargs):
        captured.update(
            parameters=parameters,
            points=points,
            proposal_knots=proposal_knots,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(module, "hybrid_minimal_knot_search", fake_search)

    result = module.run_hybrid_search_full_resolution(
        output,
        source_parameters,
        full_resolution_points,
        learned_mask,
        fit_tolerance_rms=0.02,
        degree=3,
        smoothness_weight=1e-6,
        control_ridge=0.0,
        beam_width=7,
        branch_factor=0,
        position_sweeps=3,
        position_grid_size=9,
        position_restarts=2,
        min_gap=2e-4,
    )

    assert result is sentinel
    assert captured["parameters"] is source_parameters
    torch.testing.assert_close(captured["points"], full_resolution_points[0])
    torch.testing.assert_close(captured["proposal_knots"], proposal[0])
    torch.testing.assert_close(captured["deployment_knots"], deployment[0])
    assert captured["points"].shape[0] == 47
    assert captured["learned_mask"] is learned_mask
    assert captured["mse_tolerance"] == pytest.approx(0.02**2)
    assert captured["beam_width"] == 7
    assert captured["branch_factor"] == 0
    assert captured["position_sweeps"] == 3
    assert captured["position_grid_size"] == 9
    assert captured["position_restarts"] == 2
    assert captured["min_gap"] == pytest.approx(2e-4)


def test_hybrid_adapter_preserves_proposal_mask_and_final_fit() -> None:
    module = _load_script()
    parameters = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    points = torch.stack([parameters, parameters.square()], dim=-1)
    final_fit = refit_bspline_control_points(
        parameters,
        points,
        torch.tensor([0.45], dtype=torch.float64),
        smoothness_weight=0.0,
    )
    retained_mask = torch.tensor([False, True, False])
    result = SimpleNamespace(
        proposal_count=3,
        final_count=1,
        retained_proposal_mask=retained_mask,
        final_fit=final_fit,
    )

    deployed = module.hybrid_result_as_deployed_fit(result)

    assert deployed.candidate_count == 3
    assert deployed.retained_count == 1
    assert deployed.spline is final_fit
    assert torch.equal(deployed.retained_mask, retained_mask)
    assert torch.equal(deployed.retained_internal_knots, final_fit.internal_knots)


def test_verified_wrapper_uses_full_resolution_and_forwards_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source_parameters = torch.linspace(0.0, 1.0, 47, dtype=torch.float64)
    source_chord_parameters = source_parameters.sqrt()
    full_resolution_points = torch.stack(
        [source_parameters, source_parameters.square()], dim=-1
    ).unsqueeze(0)
    proposal = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float64)
    # Only selected deployment entries must be ordered; the complete v12
    # deployment vector may contain unconstrained removed slots.
    deployment = torch.tensor([[0.7, 0.5, 0.1]], dtype=torch.float64)
    keep_probability = torch.tensor([[0.1, 0.8, 0.2]], dtype=torch.float64)
    learned_mask = torch.tensor([False, True, False])
    output = {
        "proposal_internal_knots": proposal,
        "deployment_internal_knots": deployment,
        "internal_knots": deployment,
        "keep_probability": keep_probability,
    }
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_repair(
        parameters, points, proposal_knots, deployment_knots, mask, scores, **kwargs
    ):
        captured.update(
            parameters=parameters,
            points=points,
            proposal_knots=proposal_knots,
            deployment_knots=deployment_knots,
            mask=mask,
            scores=scores,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(module, "verified_confidence_repair", fake_repair)

    result = module.run_verified_repair_full_resolution(
        output,
        source_parameters,
        full_resolution_points,
        learned_mask,
        source_chord_parameters=source_chord_parameters,
        fit_tolerance_rms=0.02,
        degree=3,
        smoothness_weight=1e-6,
        control_ridge=2e-7,
        min_internal_knots=2,
        compact=False,
        hard_fallback=False,
        residual_fallback=False,
        max_residual_insertions=3,
        residual_min_gap=0.02,
        refit_device=torch.device("cpu"),
        parameterization_policy="chord-fallback",
    )

    assert result is sentinel
    torch.testing.assert_close(captured["parameters"], source_parameters)
    torch.testing.assert_close(captured["points"], full_resolution_points[0])
    torch.testing.assert_close(captured["proposal_knots"], proposal[0])
    torch.testing.assert_close(captured["deployment_knots"], deployment[0])
    torch.testing.assert_close(captured["scores"], keep_probability[0])
    torch.testing.assert_close(captured["mask"], learned_mask)
    for key in (
        "parameters",
        "points",
        "proposal_knots",
        "deployment_knots",
        "mask",
        "scores",
    ):
        assert captured[key].device.type == "cpu"
    assert captured["fit_tolerance_rms"] == pytest.approx(0.02)
    assert captured["min_internal_knots"] == 2
    assert captured["degree"] == 3
    assert captured["smoothness_weight"] == pytest.approx(1e-6)
    assert captured["control_ridge"] == pytest.approx(2e-7)
    assert captured["interpolate_endpoints"] is True
    assert captured["compact"] is False
    assert captured["hard_fallback"] is False
    assert captured["residual_fallback"] is False
    assert captured["max_residual_insertions"] == 3
    assert captured["residual_min_gap"] == pytest.approx(0.02)
    torch.testing.assert_close(
        captured["alternate_parameters"], source_chord_parameters
    )
    assert captured["parameterization_policy"] == "chord-fallback"


def test_verified_adapter_preserves_proposal_identity_mask_and_final_fit() -> None:
    module = _load_script()
    parameters = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    points = torch.stack([parameters, parameters.square()], dim=-1)
    final_fit = refit_bspline_control_points(
        parameters,
        points,
        torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64),
        smoothness_weight=0.0,
    )
    retained_mask = torch.tensor([True, False, True])
    result = SimpleNamespace(
        final_count=3,
        retained_proposal_mask=retained_mask,
        deployment_candidate_count=4,
        deployment_retained_mask=torch.tensor([True, False, True, True]),
        final_fit=final_fit,
    )

    deployed = module.verified_result_as_deployed_fit(result)

    assert deployed.candidate_count == 4
    assert deployed.retained_count == 3
    assert deployed.spline is final_fit
    assert torch.equal(deployed.retained_mask, result.deployment_retained_mask)
    assert torch.equal(deployed.retained_internal_knots, final_fit.internal_knots)


def test_original_scale_mse_uses_scale_squared() -> None:
    module = _load_script()

    mse, rms = module.scale_normalized_fit_errors(0.04, 0.2, 3.0)

    assert mse == pytest.approx(0.36)
    assert rms == pytest.approx(0.6)


def test_hybrid_cli_and_schema_five_report_contract_are_present() -> None:
    source = (ROOT / "scripts" / "fit_point_cloud.py").read_text(encoding="utf-8")

    assert 'choices=("checkpoint", "learned", "verified", "hybrid", "hard")' in source
    for option in (
        "--verified-compact",
        "--verified-hard-fallback",
        "--verified-residual-fallback",
        "--verified-max-residual-insertions",
        "--verified-residual-min-gap",
        "--verified-refit-device",
        "--verified-parameterization",
    ):
        assert option in source
    assert source.count("action=argparse.BooleanOptionalAction") >= 2
    assert '"verified_deployment"' in source
    assert '"prefix_counts_evaluated"' in source
    assert '"direct_standard_refit_count"' in source
    assert '"exact_fit_evaluation_count"' in source
    assert '"postprocess_time_ms"' in source
    assert '"verified_refit_device"' in source
    assert '"model_device"' in source
    for option in (
        "--hybrid-beam-width",
        "--hybrid-branch-factor",
        "--hybrid-position-sweeps",
        "--hybrid-position-grid-size",
        "--hybrid-position-restarts",
        "--hybrid-position-refine-count-margin",
        "--hybrid-min-gap",
    ):
        assert option in source
    assert '"schema_version": 6' in source
    assert '"hybrid_deployment"' in source
    assert '"initial_learned_knot_count"' in source
    assert '"greedy_knot_count"' in source
    assert '"final_knot_count"' in source
    assert '"search_refit_count"' in source
    assert '"visited_state_count"' in source
    assert '"search_time_ms"' in source
    assert '"global_minimum_guaranteed": False' in source
