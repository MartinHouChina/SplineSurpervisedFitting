from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_checkpoint.py"


def _load_script():
    module_name = "test_hybrid_checkpoint_evaluation_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load_script()


@pytest.mark.parametrize(
    ("requested", "model_device", "expected"),
    (
        ("auto", torch.device("cuda:0"), torch.device("cpu")),
        ("cpu", torch.device("cuda:0"), torch.device("cpu")),
        ("model", torch.device("cuda:0"), torch.device("cuda:0")),
        ("auto", torch.device("cpu"), torch.device("cpu")),
        ("model", torch.device("cpu"), torch.device("cpu")),
    ),
)
def test_verified_refit_device_resolution(
    requested: str,
    model_device: torch.device,
    expected: torch.device,
) -> None:
    assert (
        evaluation.resolve_verified_refit_device(
            requested,
            model_device=model_device,
        )
        == expected
    )


def test_verified_refit_device_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="auto, cpu, model"):
        evaluation.resolve_verified_refit_device(
            "gpu",
            model_device=torch.device("cpu"),
        )


def _polynomial_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    parameters = torch.linspace(0.0, 1.0, 25, dtype=torch.float64)
    points = torch.stack(
        [parameters, 0.2 - 0.3 * parameters + 0.4 * parameters**3], dim=-1
    ).unsqueeze(0)
    proposal = torch.tensor([[0.22, 0.51, 0.79]], dtype=torch.float64)
    deployment = torch.tensor([[0.19, 0.54, 0.82]], dtype=torch.float64)
    output = {
        "params": parameters.unsqueeze(0),
        "internal_knots": deployment,
        "deployment_internal_knots": deployment,
        "proposal_internal_knots": proposal,
        "keep_probability": torch.tensor([[0.8, 0.3, 0.7]], dtype=torch.float64),
        "learned_keep_mask": torch.tensor([[True, False, True]]),
    }
    return output, points


def test_deployment_mode_resolution_preserves_legacy_and_defaults_one_shot() -> None:
    assert (
        evaluation.resolve_deployment_mode("checkpoint", candidate_one_shot=True)
        == "learned"
    )
    assert (
        evaluation.resolve_deployment_mode("checkpoint", candidate_one_shot=False)
        == "checkpoint"
    )
    for mode in ("learned", "verified", "hybrid", "hard"):
        assert evaluation.resolve_deployment_mode(mode, candidate_one_shot=True) == mode
        with pytest.raises(ValueError, match="one-shot candidate checkpoint"):
            evaluation.resolve_deployment_mode(mode, candidate_one_shot=False)
    with pytest.raises(ValueError, match="must be one of"):
        evaluation.resolve_deployment_mode("unknown", candidate_one_shot=True)


def test_hybrid_adapter_squares_rms_tolerance_and_uses_final_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, points = _polynomial_batch()
    final_fit = evaluation.refit_model_output_as_bsplines(
        {
            "params": output["params"],
            "internal_knots": torch.empty((1, 0), dtype=torch.float64),
            "knot_mask": torch.empty((1, 0), dtype=torch.bool),
        },
        points,
        smoothness_weight=0.0,
    )[0].spline
    captured: dict[str, object] = {}

    def fake_search(parameters, observed, proposal, **kwargs):
        captured["parameters"] = parameters
        captured["observed"] = observed
        captured["proposal"] = proposal
        captured.update(kwargs)
        return SimpleNamespace(
            retained_proposal_mask=torch.zeros_like(proposal, dtype=torch.bool),
            final_count=0,
            final_fit=final_fit,
        )

    monkeypatch.setattr(evaluation, "hybrid_minimal_knot_search", fake_search)
    deployed, results, elapsed = evaluation.hybrid_deploy_output_batch(
        output,
        points,
        fit_tolerance_rms=0.005,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
        beam_width=1,
        branch_factor=1,
        position_sweeps=0,
        position_grid_size=3,
    )

    assert captured["mse_tolerance"] == pytest.approx(2.5e-5)
    torch.testing.assert_close(
        captured["proposal"], output["proposal_internal_knots"][0]
    )
    torch.testing.assert_close(
        captured["deployment_knots"], output["deployment_internal_knots"][0]
    )
    torch.testing.assert_close(captured["learned_mask"], output["learned_keep_mask"][0])
    assert len(results) == len(deployed) == len(elapsed) == 1
    assert deployed[0].retained_count == 0
    assert deployed[0].candidate_count == 3
    assert deployed[0].spline is final_fit
    assert elapsed[0] >= 0.0


def test_real_hybrid_adapter_returns_threshold_feasible_deployment() -> None:
    output, points = _polynomial_batch()

    deployed, results, _ = evaluation.hybrid_deploy_output_batch(
        output,
        points,
        fit_tolerance_rms=1e-8,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
        beam_width=1,
        branch_factor=1,
        position_sweeps=0,
        position_grid_size=3,
    )

    assert len(deployed) == len(results) == 1
    result = results[0]
    assert result.greedy_threshold_satisfied
    assert result.threshold_satisfied
    assert result.final_count <= result.greedy_count
    assert float(deployed[0].fit_mse) <= 1e-16
    assert deployed[0].retained_count == result.final_count
    torch.testing.assert_close(
        deployed[0].retained_mask,
        result.retained_proposal_mask,
    )


def test_real_verified_adapter_returns_exactly_checked_deployment() -> None:
    output, points = _polynomial_batch()

    deployed, results, elapsed = evaluation.verified_deploy_output_batch(
        output,
        points,
        fit_tolerance_rms=1e-8,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
        compact=True,
        hard_fallback=True,
        refit_device=torch.device("cpu"),
    )

    assert len(deployed) == len(results) == len(elapsed) == 1
    result = results[0]
    assert result.threshold_satisfied
    assert result.final_source == "learned_verified_compact"
    assert result.cleanup_used
    assert not result.fallback_used
    assert deployed[0].retained_count == result.final_count
    assert deployed[0].spline is result.final_fit
    assert elapsed[0] >= 0.0


def test_verified_adapter_moves_every_exact_refit_input_to_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, points = _polynomial_batch()
    captured: dict[str, object] = {}
    chord_parameters = output["params"].sqrt()
    final_fit = evaluation.refit_model_output_as_bsplines(
        {
            "params": output["params"],
            "internal_knots": output["deployment_internal_knots"],
            "knot_mask": output["learned_keep_mask"],
        },
        points,
        smoothness_weight=0.0,
    )[0].spline

    def fake_repair(parameters, observed, proposal, deployment, mask, scores, **kwargs):
        captured.update(
            parameters=parameters,
            observed=observed,
            proposal=proposal,
            deployment=deployment,
            mask=mask,
            scores=scores,
            alternate_parameters=kwargs["alternate_parameters"],
            parameterization_policy=kwargs["parameterization_policy"],
        )
        return SimpleNamespace(
            final_count=int(mask.sum()),
            retained_proposal_mask=mask,
            deployment_candidate_count=int(mask.numel()),
            deployment_retained_mask=mask,
            final_fit=final_fit,
        )

    monkeypatch.setattr(evaluation, "verified_confidence_repair", fake_repair)

    deployed, _, _ = evaluation.verified_deploy_output_batch(
        output,
        points,
        chord_parameters=chord_parameters,
        parameterization_policy="chord-fallback",
        fit_tolerance_rms=0.005,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
        refit_device=torch.device("cpu"),
    )

    for key, tensor in captured.items():
        if key != "parameterization_policy":
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device == torch.device("cpu")
    torch.testing.assert_close(captured["alternate_parameters"], chord_parameters[0])
    assert captured["parameterization_policy"] == "chord-fallback"
    assert deployed[0].reconstructed_points.device == torch.device("cpu")


def test_hard_adapter_starts_from_immutable_proposal_geometry() -> None:
    output, points = _polynomial_batch()

    deployed, results = evaluation.prune_candidate_output_batch(
        output,
        points,
        error_tolerance=1.0,
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )

    assert len(deployed) == len(results) == 1
    torch.testing.assert_close(
        results[0].initial_internal_knots,
        output["proposal_internal_knots"][0],
    )
    assert not torch.equal(
        results[0].initial_internal_knots,
        output["deployment_internal_knots"][0],
    )


def test_hybrid_quality_cli_exposes_boundary_focused_search_controls() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for option in (
        "--deployment-mode",
        "--hybrid-beam-width",
        "--hybrid-branch-factor",
        "--hybrid-position-sweeps",
        "--hybrid-position-grid-size",
        "--hybrid-position-restarts",
        "--hybrid-position-refine-count-margin",
        "--hybrid-position-refine-candidate-multiplier",
        "--hybrid-min-gap",
    ):
        assert option in source
    assert '"hybrid_quality_deployment"' in source
    assert '"global_minimum_guaranteed": False' in source


def test_verified_quality_cli_exposes_repair_controls_and_report() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for option in (
        '"verified"',
        "--verified-compact",
        "--verified-hard-fallback",
        "--verified-residual-fallback",
        "--verified-max-residual-insertions",
        "--verified-residual-min-gap",
        "--verified-refit-device",
        "--verified-parameterization",
    ):
        assert option in source
    assert '"verified_quality_deployment"' in source
    assert '"accepted_states_are_exactly_refit": True' in source
    assert '"residual_insertion_fraction"' in source
    assert '"verified_refit_device"' in source
    assert '"model_device"' in source
