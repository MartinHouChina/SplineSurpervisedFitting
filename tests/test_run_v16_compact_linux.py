"""Compact is explicit, warm-start-only, and retains the overnight report path."""
from __future__ import annotations

import subprocess

import pytest

from test_run_v16_1070_overnight_linux import (
    MISSING_PYTHON, ROOT, RUN_NAME, SCRIPT, STAGES,
    _bash, _bash_path, _commands, _parsed, _run,
    preflight, resume_bundle,
)


@pytest.fixture
def selected_best(tmp_path):
    checkpoint = tmp_path / "overnight_reliable_3090_r1.pt"
    checkpoint.write_bytes(b"selected best E5 placeholder; dry-run must not load it")
    return checkpoint


def _compact(tmp_path, selected_best, *options):
    return _run(tmp_path, "--compact-selection", "--warm-start-checkpoint",
                _bash_path(selected_best), *options)


def test_compact_profile_wires_every_public_training_option(tmp_path, selected_best):
    before = selected_best.read_bytes()
    commands = _commands(_compact(tmp_path, selected_best))
    tokens = commands["train_fresh"]
    train = _parsed(tokens)
    assert (train.epochs, train.proposal_epochs) == (24, 4)
    assert (train.train_size, train.val_size, train.batch_size) == (1500, 500, 32)
    assert (train.candidate_knots, train.min_control_points, train.max_control_points) == (64, 8, 28)
    assert train.mse_tolerance == pytest.approx(5e-5)
    assert train.real_fraction == 0 and train.real_val_size == 32
    assert len(train.real_manifest) == 3
    assert train.simplification_controller == "per_curve"
    assert train.complexity_max_scale == 1.0
    assert train.feasible_objective
    assert (train.feasible_fit_margin, train.feasible_fit_weight) == (.8, .02)
    assert (train.teacher_greedy_steps, train.teacher_greedy_max_curves) == (16, 2)
    assert train.teacher_geometry_distillation_weight == .2
    assert train.count_reserve_alignment
    assert (train.synthetic_simple_fraction, train.synthetic_shape_fraction) == (.35, .25)
    assert train.resample_train_each_epoch
    assert tokens.count("--resample-train-each-epoch") == 1
    assert "--no-resample-train-each-epoch" not in tokens
    assert (train.one_shot_safety_knots, train.final_safety_knots) == (0, 0)
    assert (train.one_shot_safety_sigma, train.final_safety_sigma) == (0, 0)
    assert (train.lr, train.joint_lr) == (2e-5, 3e-5)
    assert (train.joint_proposal_lr_scale, train.joint_decoder_lr_scale,
            train.joint_final_lr_ratio) == (.1, .25, .25)
    assert (train.policy_samples, train.counterfactual_edits, train.teacher_prefix_search_steps) == (2, 4, 6)
    assert (train.teacher_refinement_steps, train.teacher_refinement_candidates) == (1, 3)
    assert (train.boundary_ranking_weight, train.boundary_ranking_candidates) == (.5, 4)
    assert train.parameter_trust_enabled and train.parameter_trust_initial == .25
    assert train.teacher_geometry_candidates == 4
    assert train.parameter_counterfactual_weight == .25
    assert train.local_fit_weight == .1 and train.proposal_ordered_weight == .5
    assert train.allow_infeasible_proposals
    assert train.warm_start_checkpoint.as_posix() == _bash_path(selected_best)
    assert train.init_checkpoint is None and train.resume is None
    assert "--validate-real-splits" in commands["check_data_and_provenance"]
    assert "--warm-start-checkpoint" in commands["check_data_and_provenance"]
    assert tuple(stage for stage in commands if stage in STAGES) == STAGES
    assert selected_best.read_bytes() == before
    assert list(tmp_path.iterdir()) == [selected_best]


def test_compact_default_run_name_and_help_are_public(tmp_path, selected_best):
    result = subprocess.run(
        [_bash(), _bash_path(SCRIPT), "--dry-run", "--python", MISSING_PYTHON,
         "--compact-selection", "--warm-start-checkpoint", _bash_path(selected_best),
         "--output-root", _bash_path(tmp_path / "new outputs")],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
    )
    assert _parsed(_commands(result)["train_fresh"]).output.name == "overnight_compact_3090_r1.pt"
    help_result = _run(tmp_path, "--help")
    assert help_result.returncode == 0
    assert "--compact-selection" in help_result.stdout
    assert list(tmp_path.iterdir()) == [selected_best]


def test_compact_keeps_six_methods_four_metrics_and_three_real_test_sources(tmp_path, selected_best):
    commands = _commands(_compact(tmp_path, selected_best))
    default = _commands(_run(tmp_path))
    for stage in ("benchmark_six_methods", "plot_four_metrics", "plot_ours_cases",
                  "plot_six_method_real_cases", "inspect_checkpoint"):
        assert commands[stage] == default[stage]
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        args = _parsed(commands[stage])
        assert {value.split("=", 1)[0] for value in args.manifest} == {"UJI", "NaturalEarth", "USGS", "IndustrialOffset"}
        assert args.published_feasibility_safeguard
    assert _parsed(commands["plot_four_metrics"]).method_set == "published"


def test_compact_overrides_and_native_baselines_remain_explicit(tmp_path, selected_best):
    commands = _commands(_compact(tmp_path, selected_best,
        "--epochs", "3", "--proposal-epochs", "1", "--train-size", "8",
        "--val-size", "7", "--batch-size", "2", "--real-val-size", "3",
        "--benchmark-profile", "quick", "--native-baselines"))
    train = _parsed(commands["train_fresh"])
    assert (train.epochs, train.proposal_epochs, train.train_size, train.val_size,
            train.batch_size, train.real_val_size) == (3, 1, 8, 7, 2, 3)
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        args = _parsed(commands[stage])
        assert args.force_diagnostic and not args.published_feasibility_safeguard


@pytest.mark.parametrize("options", [
    ("--compact-selection", "--enhanced-selection"),
    ("--enhanced-selection", "--compact-selection"),
    ("--compact-selection", "--reliable-selection"),
    ("--reliable-selection", "--compact-selection"),
])
def test_compact_is_mutually_exclusive_with_other_learning_profiles(tmp_path, options):
    result = _run(tmp_path, *options)
    assert result.returncode != 0
    assert "choose only one" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [(), ("--no-init-checkpoint",),
                                         ("--init-checkpoint", "proposal.pt")])
def test_compact_fresh_run_never_silently_uses_old_proposal_or_scratch(tmp_path, options):
    result = _run(tmp_path, "--compact-selection", *options)
    assert result.returncode != 0
    assert "explicit --warm-start-checkpoint" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("conflict", [
    ("--resume-run",), ("--no-init-checkpoint",),
    ("--init-checkpoint", "proposal.pt"), ("--checkpoint", "evaluation.pt"),
])
def test_compact_full_warm_start_cannot_mix_initialization_modes(tmp_path, selected_best, conflict):
    result = _compact(tmp_path, selected_best, *conflict)
    assert result.returncode != 0
    assert list(tmp_path.iterdir()) == [selected_best]


def test_compact_rejects_missing_initializer_and_existing_outputs(tmp_path, selected_best):
    result = _run(tmp_path, "--compact-selection", "--warm-start-checkpoint",
                  _bash_path(tmp_path / "missing.pt"))
    assert result.returncode != 0 and "checkpoint is missing" in result.stderr
    output = tmp_path / "results with spaces/checkpoints" / f"{RUN_NAME}.pt"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"previous experiment must survive")
    before = set(tmp_path.rglob("*"))
    result = _compact(tmp_path, selected_best)
    assert result.returncode != 0 and "refusing to overwrite" in result.stderr
    assert set(tmp_path.rglob("*")) == before
    assert output.read_bytes() == b"previous experiment must survive"


def test_compact_evaluation_does_not_require_initialization(tmp_path, selected_best):
    commands = _commands(_run(tmp_path, "--compact-selection", "--checkpoint",
                              _bash_path(selected_best)))
    assert "train_fresh" not in commands and "train_resume" not in commands
    assert "--warm-start-checkpoint" not in commands["check_data_and_provenance"]
    assert _parsed(commands["benchmark_six_methods"]).checkpoint.as_posix() == _bash_path(selected_best)


def test_compact_resume_keeps_new_objective_and_does_not_reinitialize(tmp_path):
    last = tmp_path / "results with spaces/checkpoints" / f"{RUN_NAME}.last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"dry-run only")
    before = set(tmp_path.rglob("*"))
    commands = _commands(_run(tmp_path, "--compact-selection", "--resume-run"))
    assert "train_fresh" not in commands
    train = _parsed(commands["train_resume"])
    assert (train.epochs, train.proposal_epochs) == (24, 4)
    assert train.simplification_controller == "per_curve"
    assert train.feasible_objective and train.count_reserve_alignment
    assert train.teacher_greedy_steps == 16 and train.resample_train_each_epoch
    assert train.resume.as_posix() == _bash_path(last)
    assert train.warm_start_checkpoint is None and train.init_checkpoint is None
    checked = commands["check_resume_status"]
    assert checked[checked.index("--") + 1:] == commands["train_resume"][2:]
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("option,value", [
    ("simplification_controller", "per_curve"), ("feasible_objective", True),
    ("feasible_fit_margin", .7), ("feasible_fit_weight", .03),
    ("teacher_greedy_steps", 16), ("teacher_greedy_max_curves", 3),
    ("teacher_geometry_distillation_weight", .2), ("count_reserve_alignment", True),
    ("synthetic_simple_fraction", .35), ("synthetic_shape_fraction", .25),
])
def test_completed_resume_cannot_silently_drop_compact_options(
    preflight, resume_bundle, option, value,
):
    import torch

    payload = dict(resume_bundle["payload"])
    payload["training_config"] = {**payload["training_config"], option: value}
    torch.save(payload, resume_bundle["last"])
    with pytest.raises(ValueError, match=option):
        preflight.resume_status(resume_bundle["last"], resume_bundle["arguments"])


@pytest.mark.parametrize("status,expected_training", [("completed", False), ("needed", True)])
def test_compact_resume_status_controls_training_and_preserves_reports(
    tmp_path, resume_bundle, status, expected_training,
):
    data = tmp_path / "fixture data"
    for relative in ("splits/uji_pen_v2.jsonl",
                     "processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl",
                     "processed/usgs_contours/large_scale/manifest.jsonl",
                     "processed/industrial_offsets/v1/manifest.jsonl"):
        manifest = data / relative
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("", encoding="utf-8")
    executable = tmp_path / "stub python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        f"  *' --resume-status '*) printf '%s\\n' '{status}' ;;\n"
        "  *) : ;;\n"
        "esac\n", encoding="utf-8",
    )
    executable.chmod(0o755)
    commands = _commands(_run(tmp_path, "--compact-selection", "--resume-run",
        "--python", _bash_path(executable), "--data-root", _bash_path(data), dry_run=False))
    assert ("train_resume" in commands) == expected_training
    assert "train_fresh" not in commands
    assert {"benchmark_six_methods", "plot_four_metrics", "plot_ours_cases",
            "plot_six_method_real_cases"} <= commands.keys()


def test_old_default_training_command_tokens_are_unchanged(tmp_path):
    command = _commands(_run(tmp_path))["train_fresh"]
    expected = [MISSING_PYTHON, "scripts/train_v16.py",
        "--epochs", "64", "--proposal-epochs", "4",
        "--train-size", "1500", "--val-size", "500", "--batch-size", "32",
        "--num-points", "192", "--min-control-points", "8", "--max-control-points", "28",
        "--candidate-knots", "64", "--mse-tolerance", "5e-5", "--real-fraction", "0",
        "--proposal-pass-target", "0.90", "--deployment-pass-target", "0.90",
        "--policy-samples", "2", "--counterfactual-edits", "2", "--teacher-prefix-search-steps", "6",
        "--one-shot-safety-sigma", "0.2", "--safety-anneal-epochs", "8",
        "--complexity-ramp-epochs", "8", "--no-resample-train-each-epoch", "--num-workers", "0",
        "--torch-num-threads", "4", "--device", "cuda", "--output",
        _bash_path(tmp_path / "results with spaces/checkpoints" / f"{RUN_NAME}.pt"),
        "--init-checkpoint", _bash_path(ROOT / "outputs/checkpoints/candidate_selection_v16.proposal.pt")]
    assert command == expected
