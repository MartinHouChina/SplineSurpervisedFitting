from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import (  # noqa: E402
    SyntheticCubicBSplineDataset,
    evaluate_bspline_curve,
)
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    build_open_knot_vector,
)
from spline_fitting.models.spline_network import SplineFittingNetwork  # noqa: E402
from spline_fitting.training import OneShotTeacherConfig  # noqa: E402


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proposal_semantic_mismatches_detect_geometry_changes_only() -> None:
    module = _load_script("train_candidate_pruning")
    requested = {
        "point_dim": 2,
        "hidden_dim": 64,
        "structure_attention_heads": 4,
        "min_knot_gap": 1e-3,
        "one_shot_selection_policy": "mass_topk",
    }

    assert module._proposal_semantic_mismatches(requested, dict(requested)) == []

    saved = {
        **requested,
        "structure_attention_heads": 8,
        "min_knot_gap": 2e-3,
        # Selection-only settings must not invalidate immutable proposal geometry.
        "one_shot_selection_policy": "threshold",
    }
    mismatches = module._proposal_semantic_mismatches(requested, saved)
    mismatched_keys = {message.split(":", 1)[0] for message in mismatches}

    assert mismatched_keys == {"structure_attention_heads", "min_knot_gap"}


def test_new_candidate_training_uses_wider_representable_logit_limits() -> None:
    module = _load_script("train_candidate_pruning")

    assert module._NEW_TRAINING_PARAMETER_RESIDUAL_LOGIT_LIMIT == 2.5
    assert module._NEW_TRAINING_CANDIDATE_INTERVAL_LOGIT_LIMIT == 1.75
    assert module._NEW_TRAINING_TRUE_PARAMETER_WEIGHT == 2.0
    assert module._NEW_TRAINING_REDUNDANT_CANDIDATE_WEIGHT == 2.5


def test_proposal_selection_uses_teacher_fallback_feasibility_target() -> None:
    module = _load_script("train_candidate_pruning")

    assert module._required_proposal_pass_rate(
        deployment_pass_rate_target=0.97,
        max_teacher_fallback_fraction=0.03,
    ) == pytest.approx(0.97)
    assert module._required_proposal_pass_rate(
        deployment_pass_rate_target=0.97,
        max_teacher_fallback_fraction=0.0,
    ) == pytest.approx(1.0)
    assert module._required_proposal_pass_rate(
        deployment_pass_rate_target=0.95,
        max_teacher_fallback_fraction=0.02,
    ) == pytest.approx(0.98)

    source = (ROOT / "scripts" / "train_candidate_pruning.py").read_text(
        encoding="utf-8"
    )
    assert "deployment_pass_rate_target=required_proposal_pass" in source
    assert (
        "trainer.deployment_pass_rate_target = args.deployment_pass_rate_target"
        in source
    )


def test_proposal_parameter_warmup_preserves_pretrain_epoch_budget() -> None:
    module = _load_script("train_candidate_pruning")

    assert module._resolve_proposal_parameter_warmup_epochs(60, None) == 15
    assert module._resolve_proposal_parameter_warmup_epochs(20, None) == 6
    assert module._resolve_proposal_parameter_warmup_epochs(5, None) == 1
    assert module._resolve_proposal_parameter_warmup_epochs(2, None) == 0
    assert module._resolve_proposal_parameter_warmup_epochs(1, None) == 0
    assert module._resolve_proposal_parameter_warmup_epochs(0, None) == 0
    assert module._resolve_proposal_parameter_warmup_epochs(20, 0) == 0
    assert module._resolve_proposal_parameter_warmup_epochs(20, 7) == 7
    with pytest.raises(ValueError, match="smaller than candidate"):
        module._resolve_proposal_parameter_warmup_epochs(20, 20)
    with pytest.raises(ValueError, match="must be zero"):
        module._resolve_proposal_parameter_warmup_epochs(0, 1)


def test_proposal_fingerprint_separates_stable_pilot_semantics() -> None:
    module = _load_script("train_candidate_pruning")
    common = {
        "point_dim": 2,
        "hidden_dim": 16,
        "encoder_layers": 1,
        "max_internal_knots": 4,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
        "one_shot_fixed_proposal_geometry": True,
    }
    legacy = SplineFittingNetwork(**common, stable_pilot_descriptors=False)
    stable = SplineFittingNetwork(**common, stable_pilot_descriptors=True)
    stable.load_state_dict(legacy.state_dict(), strict=True)

    assert module._model_fingerprint(legacy) != module._model_fingerprint(stable)


def test_proposal_stage_resets_and_freezes_late_feedback_heads() -> None:
    module = _load_script("train_candidate_pruning")
    model = SplineFittingNetwork(
        point_dim=2,
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=4,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=True,
        parameter_feedback_fusion=True,
        joint_parameter_structure_feedback=True,
    )
    with torch.no_grad():
        model.parameter_feedback_head.chord_blend_weight.fill_(0.7)
        model.parameter_feedback_head.gap_logit_head.weight.fill_(1.0)
        model.parameter_feedback_head.gap_logit_head.bias.fill_(1.0)
        model.joint_parameter_structure_head.keep_delta_head.weight.fill_(1.0)
        model.joint_parameter_structure_head.keep_delta_head.bias.fill_(1.0)
        model.joint_parameter_structure_head.relocation_scale.fill_(0.5)

    module._reset_and_freeze_late_feedback_heads(model)

    assert float(model.parameter_feedback_head.chord_blend_weight) == 0.0
    assert not bool(model.parameter_feedback_head.gap_logit_head.weight.any())
    assert not bool(model.parameter_feedback_head.gap_logit_head.bias.any())
    assert not bool(model.joint_parameter_structure_head.keep_delta_head.weight.any())
    assert not bool(model.joint_parameter_structure_head.keep_delta_head.bias.any())
    assert float(model.joint_parameter_structure_head.relocation_scale) == 0.0
    assert all(
        not parameter.requires_grad
        for parameter in model.parameter_feedback_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.joint_parameter_structure_head.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.parameter_head.parameters())


def test_offline_teacher_requires_a_proposal_that_met_exact_validation() -> None:
    module = _load_script("train_candidate_pruning")

    accepted = module._require_feasible_proposal_checkpoint(
        {"best_threshold_satisfied_rate": 0.97},
        deployment_pass_rate_target=0.97,
        max_teacher_fallback_fraction=0.03,
    )
    assert accepted == pytest.approx(0.97)

    with pytest.raises(RuntimeError, match=r"18\.600% < 97\.000%"):
        module._require_feasible_proposal_checkpoint(
            {"best_threshold_satisfied_rate": 0.186},
            deployment_pass_rate_target=0.97,
            max_teacher_fallback_fraction=0.03,
        )

    with pytest.raises(RuntimeError, match="no exact all-candidate"):
        module._require_feasible_proposal_checkpoint(
            {},
            deployment_pass_rate_target=0.97,
            max_teacher_fallback_fraction=0.03,
        )


def test_zero_teacher_fallback_requires_perfect_proposal_validation() -> None:
    module = _load_script("train_candidate_pruning")

    with pytest.raises(RuntimeError, match=r"99\.900% < 100\.000%"):
        module._require_feasible_proposal_checkpoint(
            {"best_threshold_satisfied_rate": 0.999},
            deployment_pass_rate_target=0.97,
            max_teacher_fallback_fraction=0.0,
        )

    accepted = module._require_feasible_proposal_checkpoint(
        {"best_threshold_satisfied_rate": 1.0},
        deployment_pass_rate_target=0.97,
        max_teacher_fallback_fraction=0.0,
    )
    assert accepted == pytest.approx(1.0)


def test_teacher_fallback_budget_can_dominate_deployment_target() -> None:
    module = _load_script("train_candidate_pruning")

    with pytest.raises(RuntimeError, match=r"97\.500% < 98\.000%"):
        module._require_feasible_proposal_checkpoint(
            {"best_threshold_satisfied_rate": 0.975},
            deployment_pass_rate_target=0.95,
            max_teacher_fallback_fraction=0.02,
        )

    accepted = module._require_feasible_proposal_checkpoint(
        {"best_threshold_satisfied_rate": 0.98},
        deployment_pass_rate_target=0.95,
        max_teacher_fallback_fraction=0.02,
    )
    assert accepted == pytest.approx(0.98)


def test_canonical_boehm_teacher_does_not_require_feasible_learned_proposal() -> None:
    module = _load_script("train_candidate_pruning")

    assert module._required_proposal_pass_rate(
        deployment_pass_rate_target=0.97,
        max_teacher_fallback_fraction=0.0,
        teacher_start_mode="canonical_boehm",
    ) == pytest.approx(0.97)
    observed = module._require_feasible_proposal_checkpoint(
        {"best_threshold_satisfied_rate": 0.186},
        deployment_pass_rate_target=0.97,
        max_teacher_fallback_fraction=0.0,
        teacher_start_mode="canonical_boehm",
    )
    assert observed == pytest.approx(0.186)
    assert math.isnan(
        module._require_feasible_proposal_checkpoint(
            {},
            deployment_pass_rate_target=0.97,
            max_teacher_fallback_fraction=0.0,
            teacher_start_mode="canonical_boehm",
        )
    )
    summary = module._validate_teacher_start_domain_counts(
        {"canonical_boehm": 10},
        sample_count=10,
        max_fallback_fraction=0.0,
        start_mode="canonical_boehm",
    )
    assert summary["fallback_fraction"] == 0.0


def test_canonical_boehm_teacher_start_is_exact_redundant_refinement() -> None:
    module = _load_script("train_candidate_pruning")
    dtype = torch.float64
    parameters = torch.linspace(0.0, 1.0, 81, dtype=dtype)
    true_knots = torch.tensor([0.25, 0.75], dtype=dtype)
    true_mask = torch.ones(2, dtype=torch.bool)
    controls_x = torch.linspace(-1.0, 1.0, 6, dtype=dtype)
    controls = torch.stack(
        [controls_x, torch.sin(3.0 * controls_x) + 0.1 * controls_x.square()],
        dim=-1,
    )
    points = evaluate_bspline_curve(
        parameters,
        controls,
        build_open_knot_vector(true_knots, degree=3),
        degree=3,
    )
    config = OneShotTeacherConfig(
        error_tolerance=1e-8,
        smoothness_weight=0.0,
        relocation_strategy="none",
    )

    selected_parameters, candidates, rms = module._canonical_boehm_teacher_start(
        parameters,
        points,
        true_knots,
        true_mask,
        5,
        config,
    )

    torch.testing.assert_close(selected_parameters, parameters)
    torch.testing.assert_close(
        candidates,
        torch.tensor([0.125, 0.25, 0.375, 0.5, 0.75], dtype=dtype),
    )
    assert rms <= config.error_tolerance


def test_canonical_boehm_cache_records_supervision_provenance(tmp_path: Path) -> None:
    module = _load_script("train_candidate_pruning")
    dataset = SyntheticCubicBSplineDataset(
        size=2,
        num_points=64,
        min_control_points=5,
        max_control_points=8,
        noise_std=1e-3,
        canonical_knot_tolerance=5e-3,
        seed=123,
    )
    model = SplineFittingNetwork(
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=6,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
    )
    config = OneShotTeacherConfig(
        error_tolerance=5e-3,
        smoothness_weight=1e-6,
        proposal_fingerprint="proposal",
        dataset_fingerprint="dataset",
    )
    cache_path = tmp_path / "teacher.pt"

    teacher = module._build_or_load_teacher_cache(
        model=model,
        dataset=dataset,
        cache_path=cache_path,
        config=config,
        batch_size=2,
        device=torch.device("cpu"),
        reuse=False,
        max_fallback_fraction=0.0,
        teacher_start_mode="canonical_boehm",
    )

    assert bool(teacher.teacher_threshold_satisfied.all())
    metadata = json.loads(
        module._teacher_start_domain_metadata_path(cache_path).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["version"] == 2
    assert metadata["teacher_start_mode"] == "canonical_boehm"
    assert metadata["counts"] == {"canonical_boehm": 2}
    assert metadata["fallback_fraction"] == 0.0
    assert metadata["slot_alignment"] == "ordered_rank"
    assert metadata["position_storage_domain"] == "fixed_proposal_t0"
    rank_diagnostics = metadata["ordered_rank_anchor_diagnostics"]
    assert rank_diagnostics["mean_abs_shift"] >= 0.0
    assert rank_diagnostics["max_abs_shift"] >= rank_diagnostics["mean_abs_shift"]
    assert 0.0 <= rank_diagnostics["over_relocation_limit_fraction"] <= 1.0
    assert rank_diagnostics["relocation_limit"] == config.relocation_max_shift

    reused = module._build_or_load_teacher_cache(
        model=model,
        dataset=dataset,
        cache_path=cache_path,
        config=config,
        batch_size=2,
        device=torch.device("cpu"),
        reuse=True,
        max_fallback_fraction=0.0,
        teacher_start_mode="canonical_boehm",
    )
    torch.testing.assert_close(
        reused.teacher_internal_knots,
        teacher.teacher_internal_knots,
    )


def test_teacher_start_falls_back_to_true_parameters_before_caching() -> None:
    module = _load_script("train_candidate_pruning")
    dtype = torch.float64
    proposal_parameters = torch.linspace(0.0, 1.0, 81, dtype=dtype)
    true_parameters = proposal_parameters.square()
    true_knots = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=dtype)
    proposal_knots = true_knots.sqrt()
    controls_x = torch.linspace(-1.0, 1.0, 8, dtype=dtype)
    controls = torch.stack(
        [
            controls_x,
            0.5 * torch.sin(5.0 * controls_x) + controls_x.square(),
        ],
        dim=-1,
    )
    points = evaluate_bspline_curve(
        true_parameters,
        controls,
        build_open_knot_vector(true_knots, degree=3),
        degree=3,
    )
    config = OneShotTeacherConfig(
        error_tolerance=1e-5,
        smoothness_weight=0.0,
        relocation_strategy="none",
    )

    proposal_fit, proposal_feasible, _ = (
        module._full_candidate_standard_bspline_fit(
            proposal_parameters,
            points,
            proposal_knots,
            config,
        )
    )
    selected_parameters, selected_knots, source, selected_rms = (
        module._select_feasible_teacher_start(
            proposal_parameters,
            points,
            proposal_knots,
            config,
            true_parameters=true_parameters,
            chord_parameters=None,
        )
    )

    assert not proposal_feasible
    assert float(proposal_fit.fit_rmse) > config.error_tolerance
    assert source == "true"
    torch.testing.assert_close(selected_parameters, true_parameters)
    torch.testing.assert_close(selected_knots, true_knots, rtol=0.0, atol=5e-5)
    assert selected_rms <= config.error_tolerance


def test_teacher_start_uses_calibrated_parameters_and_warped_fixed_slots() -> None:
    module = _load_script("train_candidate_pruning")
    dtype = torch.float64
    proposal_parameters = torch.linspace(0.0, 1.0, 81, dtype=dtype)
    calibrated_parameters = proposal_parameters.square()
    calibrated_knots = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=dtype)
    proposal_knots = calibrated_knots.sqrt()
    controls_x = torch.linspace(-1.0, 1.0, 8, dtype=dtype)
    controls = torch.stack(
        [controls_x, torch.sin(3.0 * controls_x) + 0.2 * controls_x.square()],
        dim=-1,
    )
    points = evaluate_bspline_curve(
        calibrated_parameters,
        controls,
        build_open_knot_vector(calibrated_knots, degree=3),
        degree=3,
    )
    config = OneShotTeacherConfig(
        error_tolerance=1e-5,
        smoothness_weight=0.0,
        relocation_strategy="none",
    )

    selected_parameters, selected_knots, source, selected_rms = (
        module._select_feasible_teacher_start(
            proposal_parameters,
            points,
            proposal_knots,
            config,
            calibrated_parameters=calibrated_parameters,
            true_parameters=None,
            chord_parameters=None,
        )
    )

    assert source == "calibrated"
    torch.testing.assert_close(selected_parameters, calibrated_parameters)
    torch.testing.assert_close(selected_knots, calibrated_knots, rtol=0.0, atol=5e-5)
    assert selected_rms <= config.error_tolerance


def test_teacher_fallback_fraction_is_a_hard_gate() -> None:
    module = _load_script("train_candidate_pruning")

    accepted = module._validate_teacher_start_domain_counts(
        {"calibrated": 9, "true": 1, "chord": 0},
        sample_count=10,
        max_fallback_fraction=0.10,
    )
    assert accepted["fallback_fraction"] == pytest.approx(0.10)

    with pytest.raises(RuntimeError, match="fallback fraction exceeds"):
        module._validate_teacher_start_domain_counts(
            {"calibrated": 8, "true": 1, "chord": 1},
            sample_count=10,
            max_fallback_fraction=0.10,
        )


def test_formal_teacher_training_defaults_to_no_fallback_domain() -> None:
    source = (ROOT / "scripts" / "train_candidate_pruning.py").read_text(
        encoding="utf-8"
    )
    flag = source.index('"--max-teacher-fallback-fraction"')

    assert "default=0.0" in source[flag : flag + 650]
    assert "not guaranteed feasible under learned deployment" in source[flag : flag + 650]


def test_teacher_start_rejects_infeasible_all_candidate_label() -> None:
    module = _load_script("train_candidate_pruning")
    parameters = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    points = torch.stack(
        [parameters, torch.sin(50.0 * parameters)],
        dim=-1,
    )
    knots = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    config = OneShotTeacherConfig(
        error_tolerance=1e-10,
        smoothness_weight=0.0,
        relocation_strategy="none",
    )

    with pytest.raises(RuntimeError, match="refusing to cache an all-keep"):
        module._select_feasible_teacher_start(
            parameters,
            points,
            knots,
            config,
            true_parameters=None,
            chord_parameters=None,
        )


def test_v13_calibration_keeps_exact_validation_fallbacks_and_surrogate_opt_in() -> (
    None
):
    source = (ROOT / "scripts" / "train_candidate_pruning.py").read_text(
        encoding="utf-8"
    )

    fit_flag = source.index('"--lambda-joint-fit"')
    threshold_flag = source.index('"--lambda-joint-threshold-violation"')
    assert "default=0.0" in source[fit_flag : fit_flag + 300]
    assert "default=0.0" in source[threshold_flag : threshold_flag + 350]
    assert "candidate_checkpoints = [distill_path]" in source
    assert "candidate_checkpoints.append(calibration_path)" in source
    assert "candidate_checkpoints = [calibration_path]" not in source
    assert "deployment_validation=True" in source


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
    assert 'choices=("all", "learned", "hard", "hybrid", "comparison")' in source


def test_v8_mode_detection_uses_structure_or_objective_feature() -> None:
    module = _load_script("evaluate_checkpoint")

    assert module.candidate_mode_flags(
        {}, {"structure_mode": "candidate_pruning_one_shot"}
    ) == (True, True)
    assert module.candidate_mode_flags(
        {"objective_version": "candidate_pruning_one_shot_v8"},
        {"structure_mode": "candidate_pruning"},
    ) == (True, True)
    for script_name in ("evaluate_checkpoint", "fit_point_cloud", "visualize_result"):
        script_module = _load_script(script_name)
        for objective_version in (
            "candidate_pruning_coupled_relocation_teacher_v12",
            "candidate_pruning_one_shot_set_relocation_teacher_v13",
        ):
            assert script_module.candidate_mode_flags(
                {"objective_version": objective_version},
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


def test_comparison_figure_contains_four_curve_panels_knots_and_timing() -> None:
    module = _load_script("visualize_result")
    source = (ROOT / "scripts" / "visualize_result.py").read_text(encoding="utf-8")
    knot_vector = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0])

    assert module.compact_knot_vector(knot_vector, 3) == ("U=[0x4, 0.250, 0.750, 1x4]")
    assert '"--timing-repeats"' in source
    assert "(a) Original/source data" in source
    assert "(b) Redundant all-candidate fit" in source
    assert "(c) Learned one-shot deployment" in source
    assert "(d) Offline Hard-RMS deletion" in source
    assert "internal knots C(u)" in source
