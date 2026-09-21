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


def test_default_plan_is_exactly_two_general_models_same_seed_and_synthetic_only(tmp_path):
    plan = entry.build_plan(options(), root=tmp_path)
    assert plan["schema_version"] == 2
    assert plan["route"] == "capacity"
    assert plan["run_count"] == 2
    assert [run["run_name"] for run in plan["runs"]] == ["test_r1_m16", "test_r1_m32"]
    assert [run["candidate_internal_knots"] for run in plan["runs"]] == [16, 32]
    assert [run["source_internal_knots"] for run in plan["runs"]] == [[4, 16], [4, 24]]
    assert "general-purpose" in plan["training_data"]
    assert plan["source_range_policy"] == "capacity_matched"
    assert not plan["capacity_only_ablation"]
    assert plan["seed"] == 42
    assert plan["evaluation"]["synthetic_source_k"] == [4, 24]
    assert set(plan["evaluation"]["sources"]) == {"Synthetic", "UJI", "NaturalEarth", "USGS", "IndustrialOffset"}
    assert set(plan["evaluation"]["methods"]) == {"Ours", "Park", "Liang", "Dung", "Kang", "Luo"}
    for run in plan["runs"]:
        command = run["command"]
        assert value(command, "--training-source") == "synthetic"
        assert value(command, "--validation-source") == "all"
        assert value(command, "--synthetic-shape-domain") == "mixed"
        assert "--synthetic-simple-fraction" not in command
        assert "--synthetic-shape-fraction" not in command
        assert value(command, "--candidate-knots") == str(run["candidate_internal_knots"])
        assert value(command, "--source-max-knots") == str(run["source_internal_knots"][1])
        assert ("--resize-candidate-warm-start" in command) == (run["candidate_internal_knots"] == 16)
        for flag in ("--candidate-refinement-layers", "--selection-refinement-layers", "--decoder-refinement-layers"):
            assert value(command, flag) == "2"
        assert value(command, "--max-point-error-weight") == "0.05"
        assert value(command, "--max-point-error-tolerance") == "5e-4"
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


@pytest.mark.parametrize("source_max", [4, 12, 16])
def test_capacity_only_ablation_uses_shared_training_range_but_keeps_common_test_range(tmp_path, source_max):
    plan = entry.build_plan(options("--shared-source-max-knots", str(source_max)), root=tmp_path)
    assert plan["run_count"] == 2
    assert plan["capacity_only_ablation"] and plan["source_range_policy"] == "shared"
    assert plan["evaluation"]["synthetic_source_k"] == [4, 24]
    assert [run["source_internal_knots"] for run in plan["runs"]] == [[4, source_max], [4, source_max]]
    assert [value(run["command"], "--source-max-knots") for run in plan["runs"]] == [str(source_max)] * 2


@pytest.mark.parametrize("variant", entry.DEPTH_VARIANTS)
def test_capacity_variant_applies_the_same_architecture_and_loss_to_both_models(tmp_path, variant):
    runs = entry.build_plan(options("--capacity-variant", variant), root=tmp_path)["runs"]
    assert len(runs) == 2
    for run in runs:
        assert run["extra_depth_per_module"] == (2 if variant in ("deep", "deep_peak") else 0)
        assert run["point_error_loss_enabled"] == (variant in ("peak", "deep_peak"))


def test_user_subsets_and_all_requested_settings_are_forwarded(tmp_path):
    args = options("--route", "capacity", "--capacity-variant", "peak",
                   "--epochs", "60", "--proposal-epochs", "12",
                   "--joint-geometry-calibration-epochs", "6", "--train-size", "100",
                   "--val-size", "50", "--batch-size", "8", "--prepare-real-data",
                   "--benchmark-profile", "quick", "--device", "cpu", "--data-root", str(tmp_path / "data tree"),
                   "--real-samples-per-dataset", "4", "--visual-samples-per-dataset", "2")
    plan = entry.build_plan(args, root=tmp_path)
    assert plan["run_count"] == 2
    for run in plan["runs"]:
        command = run["command"]
        for flag, expected in (("--epochs", "60"), ("--proposal-epochs", "12"),
                               ("--joint-geometry-calibration-epochs", "6"), ("--train-size", "100"),
                               ("--val-size", "50"), ("--batch-size", "8"),
                               ("--benchmark-profile", "quick"), ("--device", "cpu"),
                               ("--real-samples-per-dataset", "4"), ("--visual-samples-per-dataset", "2")):
            assert value(command, flag) == expected
        assert "--prepare-real-data" in command


def test_explicit_depth_subset_does_not_schedule_capacity_models(tmp_path):
    plan = entry.build_plan(options("--route", "depth", "--depth-variants", "deep_peak"), root=tmp_path)
    assert plan["run_count"] == 1
    assert plan["runs"][0]["run_name"] == "test_r1_depth_deep_peak"


@pytest.mark.parametrize("obsolete", [
    ["--route", "all"], ["--route", "specialists"], ["--domains", "UJI"],
    ["--route", "capacity", "--domains", "IndustrialOffset"],
])
def test_obsolete_specialist_commands_are_rejected_not_silently_reinterpreted(obsolete):
    with pytest.raises(SystemExit) as error:
        options(*obsolete)
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [
    ["--run-prefix", "../escape"], ["--run-prefix", "bad name"],
    ["--epochs", "0"], ["--epochs", "4"], ["--proposal-epochs", "-1"], ["--proposal-epochs", "0"],
    ["--joint-geometry-calibration-epochs", "25"], ["--batch-size", "0"],
    ["--route", "depth", "--depth-variants", "deep", "deep"],
    ["--depth-variants", "deep_peak"],
    ["--shared-source-max-knots", "3"], ["--shared-source-max-knots", "17"],
    ["--route", "depth", "--shared-source-max-knots", "16"],
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


def test_default_capacity_execution_starts_exactly_m16_then_m32(fake_execution):
    root, calls = fake_execution
    assert entry.main(["--warm-start-checkpoint", "old.pt", "--run-prefix", "capacity_r1"]) == 0
    assert [value(command, "--run-name") for command, _ in calls] == ["capacity_r1_m16", "capacity_r1_m32"]
    assert all(kwargs == dict(cwd=str(root), shell=False, check=False) for _, kwargs in calls)
    recorded = json.loads((root / "outputs/experiment_plans/capacity_r1.json").read_text())
    assert recorded["schema_version"] == 2 and recorded["run_count"] == 2


def test_capacity_collision_in_second_model_preserves_interrupted_artifacts(fake_execution):
    root, calls = fake_execution
    artifact = root / "outputs/checkpoints/two_r1_m32.last.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"INTERRUPTED TRAINING MUST NOT BE OVERWRITTEN")
    result = entry.main(["--warm-start-checkpoint", "old.pt", "--run-prefix", "two_r1"])
    assert result == 2 and not calls
    assert artifact.read_bytes() == b"INTERRUPTED TRAINING MUST NOT BE OVERWRITTEN"
    assert not (root / "outputs/experiment_plans").exists()


@pytest.mark.parametrize("capacity", [16, 32])
def test_two_model_commands_expand_to_valid_training_and_five_source_reports(tmp_path, capacity):
    """Use the real Bash dry run and real CLI parsers without executing Python."""
    from test_run_v16_1070_overnight_linux import MISSING_PYTHON, _bash_path, _commands, _parsed, _run

    checkpoint = tmp_path / "initial k32.pt"
    checkpoint.write_bytes(b"dry-run fixture, never deserialized")
    args = options("--warm-start-checkpoint", str(checkpoint), "--python", MISSING_PYTHON)
    plan = entry.build_plan(args)
    run = next(item for item in plan["runs"] if item["candidate_internal_knots"] == capacity)
    arguments = list(run["command"][2:])
    for flag in ("--data-root", "--warm-start-checkpoint"):
        index = arguments.index(flag) + 1
        arguments[index] = _bash_path(Path(arguments[index]))
    commands = _commands(_run(tmp_path, *arguments))
    train = _parsed(commands["train_fresh"])
    assert train.candidate_knots == capacity
    assert (train.min_control_points, train.max_control_points) == (8, 20 if capacity == 16 else 28)
    assert (train.epochs, train.proposal_epochs, train.joint_geometry_calibration_epochs) == (24, 4, 4)
    assert train.resize_candidate_warm_start == (capacity == 16)
    assert train.real_fraction == 0 and len(train.real_manifest) == 3
    assert train.synthetic_shape_domain == "mixed"
    assert train.synthetic_simple_fraction == .35 and train.synthetic_shape_fraction == .25
    assert train.proposal_refinement_layers == train.selection_refinement_layers == train.survivor_refinement_layers == 2
    assert train.max_point_error_weight == .05 and train.max_point_error_tolerance == 5e-4
    assert train.mse_tolerance == 5e-5
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        parsed = _parsed(commands[stage])
        assert parsed.max_internal_knots == capacity
        assert {item.split("=", 1)[0] for item in parsed.manifest} == {
            "UJI", "NaturalEarth", "USGS", "IndustrialOffset"}
    benchmark = _parsed(commands["benchmark_six_methods"])
    assert not benchmark.skip_synthetic
    assert (benchmark.min_knot_count, benchmark.max_knot_count, benchmark.synthetic_source_max_knots) == (4, 24, 24)
    assert benchmark.method_set == "published"
    assert {"inspect_checkpoint", "plot_four_metrics"} <= commands.keys()
    assert list(tmp_path.iterdir()) == [checkpoint]
