"""Experiment orchestration tests never train models or claim benchmark outcomes."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v16_granularity_experiments as entry


def options(*extra):
    return entry.parser().parse_args([
        "--warm-start-checkpoint", "a checkpoint with spaces.pt", "--run-prefix", "test_r1", *extra,
    ])


def value(command, name):
    return command[command.index(name) + 1]


def test_default_plan_is_twelve_paired_controls_same_seed_and_synthetic_only(tmp_path):
    plan = entry.build_plan(options(), root=tmp_path)
    assert plan["run_count"] == 12
    assert len({run["run_name"] for run in plan["runs"]}) == 12
    assert plan["seed"] == 42
    assert plan["evaluation"]["synthetic_source_k"] == [4, 24]
    assert len(plan["evaluation"]["sources"]) == 5
    assert len(plan["evaluation"]["methods"]) == 6
    for run in plan["runs"]:
        command = run["command"]
        assert value(command, "--training-source") == "synthetic"
        assert value(command, "--epochs") == "24"
        assert value(command, "--proposal-epochs") == "4"
        assert value(command, "--joint-geometry-calibration-epochs") == "4"
        assert value(command, "--mse-tolerance") == "5e-5"
        assert value(command, "--warm-start-checkpoint").endswith("a checkpoint with spaces.pt")
        assert value(command, "--real-samples-per-dataset") == "20"
        assert value(command, "--visual-samples-per-dataset") == "6"
        assert "--resume-run" not in command
        assert "--prepare-real-data" not in command
    assert not list(tmp_path.iterdir())


def test_depth_ablation_changes_only_added_depth_and_optional_peak_loss(tmp_path):
    plan = entry.build_plan(options("--route", "depth"), root=tmp_path)
    runs = plan["runs"]
    assert [run["run_name"] for run in runs] == ["test_r1_depth_" + v for v in entry.DEPTH_VARIANTS]
    for variant, run in zip(entry.DEPTH_VARIANTS, runs):
        command = run["command"]
        deep = variant in ("deep", "deep_peak")
        peak = variant in ("peak", "deep_peak")
        for flag in ("--candidate-refinement-layers", "--selection-refinement-layers", "--decoder-refinement-layers"):
            assert value(command, flag) == ("2" if deep else "0")
        assert value(command, "--max-point-error-weight") == ("0.05" if peak else "0")
        if peak:
            assert value(command, "--max-point-error-tolerance") == "5e-4"
        else:
            assert "--max-point-error-tolerance" not in command
        assert value(command, "--candidate-knots") == "32"
        assert value(command, "--source-max-knots") == "24"
        assert value(command, "--validation-source") == "all"
        assert "--resize-candidate-warm-start" not in command


def test_domain_pairs_hold_source_range_fixed_and_explicitly_resize_only_small_model(tmp_path):
    runs = entry.build_plan(options("--route", "specialists"), root=tmp_path)["runs"]
    for offset, domain in enumerate(entry.DOMAINS):
        small, control = runs[2 * offset:2 * offset + 2]
        shape, cap, _ = entry.DOMAIN_SETTINGS[domain]
        assert small["candidate_internal_knots"] == cap
        assert control["candidate_internal_knots"] == 32
        assert small["source_internal_knots"] == control["source_internal_knots"] == [4, cap]
        for run in (small, control):
            command = run["command"]
            assert value(command, "--validation-source") == domain
            assert value(command, "--synthetic-shape-domain") == shape
            assert value(command, "--synthetic-shape-fraction") == "0.5"
            assert value(command, "--synthetic-simple-fraction") == "0.25"
            assert value(command, "--max-point-error-weight") == "0.05"
            assert value(command, "--candidate-refinement-layers") == "0"
        assert "--resize-candidate-warm-start" in small["command"]
        assert "--resize-candidate-warm-start" not in control["command"]


def test_user_subsets_and_all_requested_settings_are_forwarded(tmp_path):
    args = options("--route", "all", "--depth-variants", "deep_peak", "--domains", "USGS",
                   "--epochs", "60", "--proposal-epochs", "12",
                   "--joint-geometry-calibration-epochs", "6", "--train-size", "100",
                   "--val-size", "50", "--batch-size", "8", "--prepare-real-data",
                   "--benchmark-profile", "quick", "--device", "cpu", "--data-root", str(tmp_path / "data tree"),
                   "--real-samples-per-dataset", "4", "--visual-samples-per-dataset", "2")
    plan = entry.build_plan(args, root=tmp_path)
    assert plan["run_count"] == 3
    for run in plan["runs"]:
        command = run["command"]
        for flag, expected in (("--epochs", "60"), ("--proposal-epochs", "12"),
                               ("--joint-geometry-calibration-epochs", "6"), ("--train-size", "100"),
                               ("--val-size", "50"), ("--batch-size", "8"),
                               ("--benchmark-profile", "quick"), ("--device", "cpu"),
                               ("--real-samples-per-dataset", "4"), ("--visual-samples-per-dataset", "2")):
            assert value(command, flag) == expected
        assert "--prepare-real-data" in command


@pytest.mark.parametrize("extra", [
    ["--run-prefix", "../escape"], ["--run-prefix", "bad name"],
    ["--epochs", "0"], ["--epochs", "4"], ["--proposal-epochs", "-1"], ["--proposal-epochs", "0"],
    ["--joint-geometry-calibration-epochs", "25"], ["--batch-size", "0"],
    ["--depth-variants", "deep", "deep"], ["--domains", "UJI", "UJI"],
])
def test_invalid_plan_is_rejected(extra):
    with pytest.raises(ValueError):
        entry.build_plan(options(*extra))


def test_dry_run_does_not_read_checkpoint_write_or_execute(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(entry, "ROOT", tmp_path)

    def forbidden(*a, **kw):
        raise AssertionError("dry-run must be purely declarative")

    monkeypatch.setattr(entry, "validate_initializer", forbidden)
    monkeypatch.setattr(entry, "check_artifact_conflicts", forbidden)
    monkeypatch.setattr(entry.subprocess, "run", forbidden)
    monkeypatch.setattr(entry.shutil, "which", forbidden)
    result = entry.main(["--warm-start-checkpoint", "missing.pt", "--run-prefix", "preview_r1", "--dry-run"])
    assert result == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("capacity,depth,valid", [(32, 0, True), (64, 0, False), (24, 0, False), (32, 1, False)])
def test_real_execution_checks_common_shallow_k32_initializer(tmp_path, capacity, depth, valid):
    checkpoint = tmp_path / "initial.pt"
    torch.save({"model_config": {"max_internal_knots": capacity,
                                 "proposal_refinement_layers": depth}}, checkpoint)
    if valid:
        entry.validate_initializer(checkpoint)
    else:
        with pytest.raises(ValueError):
            entry.validate_initializer(checkpoint)


@pytest.fixture
def fake_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_v16_1070_overnight_linux.sh").write_text("# test fixture; not executed\n")
    monkeypatch.setattr(entry.shutil, "which", lambda _: "/usr/bin/bash")
    monkeypatch.setattr(entry, "validate_initializer", lambda _: None)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", run)
    return tmp_path, calls


def test_execution_is_serial_argv_only_with_exclusive_plan_and_no_overwrite(fake_execution):
    root, calls = fake_execution
    argv = ["--warm-start-checkpoint", "old.pt", "--run-prefix", "execute_r1", "--route", "depth"]
    assert entry.main(argv) == 0
    assert len(calls) == 4
    for (command, kwargs), variant in zip(calls, entry.DEPTH_VARIANTS):
        assert isinstance(command, list) and command[0] == "/usr/bin/bash"
        assert value(command, "--run-name") == "execute_r1_depth_" + variant
        assert kwargs == dict(cwd=str(root), shell=False, check=False)
    manifest = root / "outputs/experiment_plans/execute_r1.json"
    old_content = manifest.read_bytes()
    assert json.loads(old_content)["status"] == "planned"
    assert entry.main(argv) == 2
    assert len(calls) == 4 and manifest.read_bytes() == old_content


def test_failure_stops_before_later_models(monkeypatch, fake_execution, capsys):
    root, calls = fake_execution

    def fail_second(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=17 if len(calls) == 2 else 0)

    monkeypatch.setattr(entry.subprocess, "run", fail_second)
    result = entry.main(["--warm-start-checkpoint", "old.pt", "--run-prefix", "fail_r1", "--route", "depth"])
    assert result == 17 and len(calls) == 2
    assert "later models were not started" in capsys.readouterr().err


def test_existing_later_run_artifact_aborts_before_any_model(fake_execution):
    root, calls = fake_execution
    artifact = root / "outputs/checkpoints/collision_r1_depth_peak.last.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"USER CHECKPOINT MUST BE PRESERVED")
    result = entry.main(["--warm-start-checkpoint", "old.pt", "--run-prefix", "collision_r1", "--route", "depth"])
    assert result == 2 and not calls
    assert artifact.read_bytes() == b"USER CHECKPOINT MUST BE PRESERVED"
    assert not (root / "outputs/experiment_plans").exists()
