"""Stable adds bounded decoded trajectory supervision to compact, not new deployment."""
import subprocess

import pytest

from test_run_v16_1070_overnight_linux import (
    MISSING_PYTHON, ROOT, RUN_NAME, SCRIPT, _bash, _bash_path, _commands, _parsed, _run,
)


@pytest.fixture
def compact_best(tmp_path):
    checkpoint = tmp_path / "overnight_compact_3090_r1.pt"
    checkpoint.write_bytes(b"best E7 fixture; dry-run never loads")
    return checkpoint


def test_stable_only_adds_two_learning_flags_to_compact(tmp_path, compact_best):
    common = ("--warm-start-checkpoint", _bash_path(compact_best))
    stable = _commands(_run(tmp_path, "--stable-selection", *common))
    compact = _commands(_run(tmp_path, "--compact-selection", *common))
    tokens = list(stable["train_fresh"])
    for option, value in (("--teacher-greedy-trajectory-checks", "4"),
                          ("--teacher-geometry-trajectory-targets", "2")):
        index = tokens.index(option)
        assert tokens[index + 1] == value
        del tokens[index:index + 2]
    assert tokens == compact["train_fresh"]
    train = _parsed(stable["train_fresh"])
    assert (train.epochs, train.proposal_epochs) == (24, 4)
    assert len(train.real_manifest) == 3
    assert train.teacher_greedy_trajectory_checks == 4
    assert train.teacher_geometry_trajectory_targets == 2
    assert train.warm_start_checkpoint.as_posix() == _bash_path(compact_best)
    assert train.init_checkpoint is None and train.resume is None
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        assert stable[stage] == compact[stage]
        assert {value.split("=", 1)[0] for value in _parsed(stable[stage]).manifest} == {
            "UJI", "NaturalEarth", "USGS", "IndustrialOffset"}
    assert list(tmp_path.iterdir()) == [compact_best]


def test_stable_default_name_and_help(tmp_path, compact_best):
    result = subprocess.run([_bash(), _bash_path(SCRIPT), "--dry-run", "--python", MISSING_PYTHON,
        "--stable-selection", "--warm-start-checkpoint", _bash_path(compact_best),
        "--output-root", _bash_path(tmp_path / "new outputs")], cwd=ROOT,
        capture_output=True, text=True, timeout=30, check=False)
    assert _parsed(_commands(result)["train_fresh"]).output.name == "overnight_stable_3090_r1.pt"
    assert "--stable-selection" in _run(tmp_path, "--help").stdout


@pytest.mark.parametrize("profile", ["--compact-selection", "--reliable-selection", "--enhanced-selection"])
@pytest.mark.parametrize("stable_first", [True, False])
def test_stable_is_mutually_exclusive(tmp_path, profile, stable_first):
    flags = ("--stable-selection", profile) if stable_first else (profile, "--stable-selection")
    result = _run(tmp_path, *flags)
    assert result.returncode != 0 and "choose only one" in result.stderr
    assert not list(tmp_path.iterdir())


def test_stable_new_run_requires_explicit_best_warm_start(tmp_path):
    result = _run(tmp_path, "--stable-selection")
    assert result.returncode != 0 and "explicit --warm-start-checkpoint" in result.stderr
    assert not list(tmp_path.iterdir())


def test_stable_resume_preserves_trajectory_settings_without_warm_start(tmp_path):
    last = tmp_path / "results with spaces/checkpoints" / f"{RUN_NAME}.last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"resume dry-run placeholder")
    commands = _commands(_run(tmp_path, "--stable-selection", "--resume-run"))
    train = _parsed(commands["train_resume"])
    assert "train_fresh" not in commands
    assert train.teacher_greedy_trajectory_checks == 4 and train.teacher_geometry_trajectory_targets == 2
    assert train.resume.as_posix() == _bash_path(last)
    assert train.warm_start_checkpoint is None and train.init_checkpoint is None


def test_stable_evaluation_can_reuse_existing_weights_without_retraining(tmp_path, compact_best):
    result = _run(tmp_path, "--stable-selection", "--checkpoint", _bash_path(compact_best))
    commands = _commands(result)
    assert "train_fresh" not in commands and "train_resume" not in commands
    assert "Evaluation-only:" in result.stdout and "no training" in result.stdout
    assert "Proposal 4 + Joint" not in result.stdout
    assert len(_parsed(commands["benchmark_six_methods"]).manifest) == 4
