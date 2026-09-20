"""The short anchored profile retains all datasets/reports and never overwrites."""
import pytest

from test_run_v16_1070_overnight_linux import RUN_NAME, _bash_path, _commands, _parsed, _run


@pytest.fixture
def best(tmp_path):
    checkpoint = tmp_path / "selected_k32.pt"
    checkpoint.write_bytes(b"dry-run fixture; no checkpoint loading")
    return checkpoint


def test_anchored_short_profile_preserves_six_method_pipeline(tmp_path, best):
    commands = _commands(_run(tmp_path, "--anchored-selection", "--warm-start-checkpoint", _bash_path(best)))
    train = _parsed(commands["train_fresh"])
    assert (train.epochs, train.proposal_epochs, train.joint_geometry_calibration_epochs) == (16, 4, 6)
    assert train.candidate_knots == 32 and train.mse_tolerance == 5e-5
    assert train.subset_geometry_mode == "anchored" and train.subset_geometry_residual_scale == .25
    assert train.teacher_greedy_priority == "compact" and train.teacher_greedy_max_curves == 4
    assert train.teacher_compact_mask_weight == .5 and train.teacher_geometry_distillation_weight == .4
    assert train.joint_decoder_lr_scale == 1. and train.joint_proposal_lr_scale == .1
    assert train.complexity_ramp_epochs == 4 and train.safety_anneal_epochs == 4
    assert train.real_fraction == 0 and len(train.real_manifest) == 3
    assert train.warm_start_checkpoint.as_posix() == _bash_path(best)
    assert train.resume is None and train.init_checkpoint is None
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        args = _parsed(commands[stage])
        assert {value.split("=", 1)[0] for value in args.manifest} == {
            "UJI", "NaturalEarth", "USGS", "IndustrialOffset"}
        assert args.max_internal_knots == 32
        assert args.published_feasibility_safeguard
    assert {"plot_four_metrics", "inspect_checkpoint"} <= commands.keys()
    assert list(tmp_path.iterdir()) == [best]


@pytest.mark.parametrize("other", ["--stable-selection", "--compact-selection", "--reliable-selection", "--enhanced-selection"])
def test_anchored_is_exclusive(tmp_path, other):
    result = _run(tmp_path, "--anchored-selection", other)
    assert result.returncode != 0 and "choose only one" in result.stderr


def test_anchored_rejects_missing_warmstart_or_overlong_calibration(tmp_path, best):
    result = _run(tmp_path, "--anchored-selection")
    assert result.returncode != 0 and "explicit --warm-start-checkpoint" in result.stderr
    result = _run(tmp_path, "--anchored-selection", "--warm-start-checkpoint", _bash_path(best),
                  "--epochs", "6", "--proposal-epochs", "4")
    assert result.returncode != 0 and "cannot exceed total Joint" in result.stderr


def test_anchored_override_and_resume_keep_functional_configuration(tmp_path, best):
    commands = _commands(_run(tmp_path, "--anchored-selection", "--warm-start-checkpoint", _bash_path(best),
        "--epochs", "6", "--proposal-epochs", "2", "--joint-geometry-calibration-epochs", "2",
        "--candidate-knots", "48"))
    train = _parsed(commands["train_fresh"])
    assert (train.epochs, train.proposal_epochs, train.joint_geometry_calibration_epochs) == (6, 2, 2)
    assert train.candidate_knots == 48
    last = tmp_path / "results with spaces/checkpoints" / f"{RUN_NAME}.last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"dry-run resume placeholder")
    commands = _commands(_run(tmp_path, "--anchored-selection", "--resume-run"))
    resumed = _parsed(commands["train_resume"])
    assert resumed.subset_geometry_mode == "anchored" and resumed.joint_geometry_calibration_epochs == 6
    assert resumed.teacher_greedy_priority == "compact"
    assert resumed.resume.as_posix() == _bash_path(last)
    assert resumed.warm_start_checkpoint is None
