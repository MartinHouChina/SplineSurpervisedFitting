"""CPU-only contract tests for the historical GTX-1070 Linux launcher.

Dry runs intentionally use a nonexistent Python executable: these tests must
never train, download data, load a checkpoint, or require CUDA.  CLI validation
compiles the repository's real argparse factories without importing torch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_v16_1070_overnight_linux.sh"
RUN_NAME = "pytest_1070_linux"
MISSING_PYTHON = "/definitely-missing-python-for-dry-run"
STAGES = (
    "train_fresh",
    "inspect_checkpoint",
    "benchmark_six_methods",
    "plot_four_metrics",
    "plot_ours_cases",
    "plot_six_method_real_cases",
)


def _bash() -> str:
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is not available")
    return executable


def _bash_path(path: Path) -> str:
    """Use a genuine POSIX absolute path when testing under Windows Git Bash."""
    value = path.resolve().as_posix()
    if re.match(r"^[A-Za-z]:/", value):
        return f"/{value[0].lower()}/{value[3:]}"
    return value


def _run(tmp_path: Path, *options: str, dry_run: bool = True):
    arguments = [
        _bash(), _bash_path(SCRIPT),
        "--python", MISSING_PYTHON,
        "--output-root", _bash_path(tmp_path / "results with spaces"),
        "--run-name", RUN_NAME,
    ]
    if dry_run:
        arguments.append("--dry-run")
    arguments.extend(options)
    return subprocess.run(
        arguments,
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _commands(completed) -> dict[str, list[str]]:
    assert completed.returncode == 0, completed.stdout + completed.stderr
    commands: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        match = re.match(r"^\[([^\]]+)\]\s+(.*)$", line)
        if match is None:
            continue
        tokens = shlex.split(match.group(2))
        if not any(token.endswith(".py") for token in tokens):
            continue
        stage = match.group(1)
        assert stage not in commands, f"duplicate stage command: {line}"
        commands[stage] = tokens
    return commands


def test_enhanced_profile_increases_learning_without_changing_benchmark(tmp_path):
    commands = _commands(_run(tmp_path, "--enhanced-selection"))
    train = _parsed(commands["train_fresh"])
    assert (train.epochs, train.proposal_epochs) == (96, 12)
    assert (train.policy_samples, train.counterfactual_edits, train.teacher_prefix_search_steps) == (4, 6, 8)
    assert (train.teacher_refinement_steps, train.teacher_refinement_candidates) == (1, 3)
    assert (train.boundary_ranking_weight, train.boundary_ranking_candidates) == (0.5, 4)
    assert train.lr == train.joint_lr == pytest.approx(5e-5)
    assert (train.joint_proposal_lr_scale, train.joint_decoder_lr_scale,
            train.joint_final_lr_ratio) == (0.25, 0.5, 0.25)
    assert train.candidate_knots == 64 and train.mse_tolerance == pytest.approx(5e-5)
    benchmark = _parsed(commands["benchmark_six_methods"])
    assert benchmark.paper_admm_iterations == 400 and benchmark.luo_de_iterations == 50
    assert benchmark.method_set == "published" and benchmark.max_internal_knots == 64
    assert not list(tmp_path.iterdir())


def test_full_model_warm_start_is_not_proposal_only_initialization(tmp_path):
    checkpoint = tmp_path / "good overnight.pt"
    checkpoint.write_bytes(b"dry-run only, never deserialize")
    commands = _commands(_run(tmp_path, "--enhanced-selection", "--warm-start-checkpoint",
                              _bash_path(checkpoint), "--epochs", "8", "--proposal-epochs", "2"))
    train = _parsed(commands["train_fresh"])
    assert train.warm_start_checkpoint.as_posix() == _bash_path(checkpoint)
    assert train.init_checkpoint is None and train.resume is None
    assert (train.epochs, train.proposal_epochs) == (8, 2)
    assert "--warm-start-checkpoint" in commands["check_data_and_provenance"]
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("conflict", [
    ("--resume-run",), ("--no-init-checkpoint",), ("--init-checkpoint", "proposal.pt"),
    ("--checkpoint", "evaluation.pt"),
])
def test_full_warm_start_rejects_conflicting_modes(tmp_path, conflict):
    checkpoint = tmp_path / "warm.pt"
    checkpoint.write_bytes(b"placeholder")
    result = _run(tmp_path, "--warm-start-checkpoint", _bash_path(checkpoint), *conflict)
    assert result.returncode != 0
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_resume_preflight_rejects_silently_disabling_enhanced_loss(preflight, resume_bundle):
    import torch

    payload = dict(resume_bundle["payload"])
    payload["training_config"] = {**payload["training_config"], "teacher_refinement_steps": 1}
    torch.save(payload, resume_bundle["last"])
    with pytest.raises(ValueError, match="teacher_refinement_steps"):
        preflight.resume_status(resume_bundle["last"], resume_bundle["arguments"])


def _script_arguments(command: list[str]) -> tuple[str, list[str]]:
    for index, token in enumerate(command):
        if token.endswith(".py"):
            return token.replace("\\", "/").rsplit("/", 1)[-1], command[index + 1:]
    raise AssertionError(f"no Python entry point in {command!r}")


def _real_parser(filename: str) -> argparse.ArgumentParser:
    """Exercise real CLI definitions without importing model/training modules."""
    if filename == "benchmark_v16_datasets.py":
        filename = "benchmark_v15_datasets.py"
    elif filename == "visualize_v16_ours_cases.py":
        filename = "visualize_v16_real_deployments.py"
    source = ROOT / "scripts" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    namespace = {
        "argparse": argparse, "Path": Path, "ROOT": ROOT,
        "math": math, "__doc__": ast.get_docstring(tree),
    }
    # Constants are read from the old checkout, not copied from current main.
    for path in (ROOT / "src/spline_fitting/checkpointing.py", source):
        for statement in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(statement, ast.Assign):
                try:
                    value = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    continue
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        namespace[target.id] = value
    factories = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("parser", "_unit_interval", "_positive_float")
    ]
    if not any(node.name == "parser" for node in factories):
        # The report renderer constructs its argparse parser inside main().
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        body = []
        for statement in main.body:
            if (isinstance(statement, ast.Assign)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr == "parse_args"):
                break
            body.append(statement)
        body.append(ast.Return(value=ast.Name(id="parser", ctx=ast.Load())))
        factories.append(ast.FunctionDef(
            name="parser", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                kw_defaults=[], defaults=[]), body=body, decorator_list=[],
        ))
    module = ast.fix_missing_locations(ast.Module(body=factories, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["parser"]()


def _parsed(command: list[str]):
    filename, arguments = _script_arguments(command)
    return _real_parser(filename).parse_args(arguments)


def test_shell_syntax_and_default_dry_run_have_no_side_effects(tmp_path: Path):
    syntax = subprocess.run([_bash(), "-n", _bash_path(SCRIPT)],
                            capture_output=True, text=True, timeout=15)
    assert syntax.returncode == 0, syntax.stderr
    completed = _run(tmp_path)
    commands = _commands(completed)
    assert tuple(stage for stage in commands if stage in STAGES) == STAGES
    assert not list(tmp_path.iterdir()), "--dry-run must not create output directories"
    assert all(MISSING_PYTHON in command for command in commands.values())


def test_default_training_matches_original_1070_profile(tmp_path: Path):
    commands = _commands(_run(tmp_path))
    args = _parsed(commands["train_fresh"])
    assert (args.epochs, args.proposal_epochs, args.batch_size) == (64, 4, 32)
    assert (args.train_size, args.val_size, args.num_points) == (1500, 500, 192)
    assert (args.min_control_points, args.max_control_points) == (8, 28)
    assert args.candidate_knots == 64
    assert args.mse_tolerance == pytest.approx(5e-5)
    assert args.device == "cuda"
    assert args.resample_train_each_epoch is False
    assert args.certified_minimal_source is True
    assert args.teacher_prefix_search_steps == 6
    assert args.counterfactual_edits == 2
    assert args.policy_samples == 2
    assert args.one_shot_safety_sigma == pytest.approx(0.2)
    assert (args.safety_anneal_epochs, args.complexity_ramp_epochs) == (8, 8)
    assert args.real_fraction == 0.0
    assert args.init_checkpoint.name == "candidate_selection_v16.proposal.pt"
    assert args.resume is None
    assert args.output.name == f"{RUN_NAME}.pt"
    assert "results with spaces" in str(args.output)
    assert not any("teacher-cache" in token for token in commands["train_fresh"])


def test_all_evaluation_commands_share_capacity_tolerance_and_data(tmp_path: Path):
    commands = _commands(_run(tmp_path))
    parsed = {stage: _parsed(command) for stage, command in commands.items()
              if stage in STAGES}
    train = parsed["train_fresh"]
    benchmark = parsed["benchmark_six_methods"]
    assert (benchmark.min_knot_count, benchmark.max_knot_count) == (4, 24)
    assert benchmark.skip_synthetic is False and benchmark.skip_real is False
    assert benchmark.samples_per_knot_count == 2
    assert benchmark.real_samples_per_dataset == 8
    assert (benchmark.network_repeats, benchmark.end_to_end_repeats) == (100, 3)
    assert benchmark.paper_admm_iterations == 400
    assert benchmark.luo_de_iterations == 50
    assert benchmark.method_set == "published"
    assert benchmark.force_diagnostic is False
    manifest_sets = []
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        args = parsed[stage]
        assert args.checkpoint == train.output
        assert args.mse_tolerance == train.mse_tolerance
        assert args.device == train.device
        assert args.max_internal_knots == 64
        assert args.paper_initial_knots == 64
        assert args.liang_dense_knots == 64
        assert args.allow_unqualified_diagnostic is True
        assert args.force_diagnostic is False
        manifest_sets.append(dict(item.split("=", 1) for item in args.manifest))
    assert manifest_sets[0] == manifest_sets[1] == manifest_sets[2]
    assert set(manifest_sets[0]) == {"UJI", "NaturalEarth", "USGS"}
    assert parsed["plot_ours_cases"].real_samples_per_dataset == 2
    assert parsed["plot_six_method_real_cases"].method_set == "published"
    inspect = parsed["inspect_checkpoint"]
    assert inspect.checkpoint == train.output
    assert inspect.mse_tolerance == train.mse_tolerance
    plot = parsed["plot_four_metrics"]
    assert plot.method_set == "published"
    assert plot.input == benchmark.output_dir / "comparison.json"
    assert plot.allow_unqualified_diagnostic is True


def test_quick_profile_is_explicitly_diagnostic(tmp_path: Path):
    completed = _run(tmp_path, "--benchmark-profile", "quick")
    commands = _commands(completed)
    args = _parsed(commands["benchmark_six_methods"])
    assert (args.samples_per_knot_count, args.real_samples_per_dataset) == (1, 2)
    assert (args.network_repeats, args.end_to_end_repeats) == (5, 1)
    assert args.paper_admm_iterations < 400
    assert args.luo_de_iterations < 50
    assert "DIAGNOSTIC" in completed.stdout.upper()
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        assert "--force-diagnostic" in commands[stage]
        assert "--allow-unqualified-diagnostic" in commands[stage]
    for stage in ("plot_ours_cases", "plot_six_method_real_cases"):
        assert _parsed(commands[stage]).real_samples_per_dataset == 1


def test_explicit_overrides_survive_shell_quoting(tmp_path: Path):
    data_root = tmp_path / "datasets with spaces"
    commands = _commands(_run(
        tmp_path, "--device", "cpu", "--epochs", "9", "--proposal-epochs", "2",
        "--batch-size", "3", "--train-size", "7", "--val-size", "5",
        "--num-workers", "2", "--mse-tolerance", "0.0001",
        "--data-root", _bash_path(data_root), "--no-init-checkpoint",
        "--benchmark-profile", "quick", "--synthetic-samples-per-k", "3",
        "--real-samples-per-dataset", "4", "--visual-samples-per-dataset", "2",
        "--network-repeats", "7", "--end-to-end-repeats", "2",
    ))
    train = _parsed(commands["train_fresh"])
    assert (train.epochs, train.proposal_epochs, train.batch_size) == (9, 2, 3)
    assert (train.train_size, train.val_size, train.num_workers) == (7, 5, 2)
    assert train.init_checkpoint is None
    for stage in ("train_fresh", "benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        args = _parsed(commands[stage])
        assert args.device == "cpu"
        assert args.mse_tolerance == pytest.approx(1e-4)
    benchmark = _parsed(commands["benchmark_six_methods"])
    assert (benchmark.samples_per_knot_count, benchmark.real_samples_per_dataset) == (3, 4)
    assert (benchmark.network_repeats, benchmark.end_to_end_repeats) == (7, 2)
    for item in benchmark.manifest:
        assert item.split("=", 1)[1].startswith(_bash_path(data_root) + "/")
    assert not list(tmp_path.iterdir())


def test_external_checkpoint_skips_training_and_is_never_an_output(tmp_path: Path):
    checkpoint = tmp_path / "external checkpoint.pt"
    checkpoint.write_bytes(b"dry-run placeholder; must not be loaded or overwritten")
    before = checkpoint.read_bytes()
    commands = _commands(_run(tmp_path, "--checkpoint", _bash_path(checkpoint)))
    assert "train_fresh" not in commands and "train_resume" not in commands
    for stage in ("inspect_checkpoint", "benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        args = _parsed(commands[stage])
        assert args.checkpoint.as_posix() == _bash_path(checkpoint)
    assert checkpoint.read_bytes() == before
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_custom_initializer_is_only_a_warm_start(tmp_path: Path):
    checkpoint = tmp_path / "custom proposal checkpoint.pt"
    checkpoint.write_bytes(b"dry-run placeholder; no checkpoint load is permitted")
    commands = _commands(_run(tmp_path, "--init-checkpoint", _bash_path(checkpoint)))
    args = _parsed(commands["train_fresh"])
    assert args.init_checkpoint.as_posix() == _bash_path(checkpoint)
    assert args.resume is None
    assert args.output != args.init_checkpoint
    assert args.resample_train_each_epoch is False
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_explicit_resume_dry_run_reuses_only_the_same_run(tmp_path: Path):
    last = tmp_path / "results with spaces" / "checkpoints" / f"{RUN_NAME}.last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"dry-run placeholder; do not deserialize")
    before = set(tmp_path.rglob("*"))
    commands = _commands(_run(tmp_path, "--resume-run"))
    assert "train_fresh" not in commands
    args = _parsed(commands["train_resume"])
    assert args.resume.as_posix() == _bash_path(last)
    assert args.init_checkpoint is None
    assert args.output.name == f"{RUN_NAME}.pt"
    assert args.epochs == 64
    assert set(tmp_path.rglob("*")) == before
    assert last.read_bytes() == b"dry-run placeholder; do not deserialize"


def test_resume_without_same_run_checkpoint_fails(tmp_path: Path):
    completed = _run(tmp_path, "--resume-run")
    assert completed.returncode != 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative", [
    f"checkpoints/{RUN_NAME}.pt", f"checkpoints/{RUN_NAME}.last.pt",
    f"checkpoints/{RUN_NAME}.proposal.pt", f"checkpoints/{RUN_NAME}.history.json",
    f"logs/{RUN_NAME}/existing.log", f"comparisons/{RUN_NAME}/existing.json",
    f"figures/{RUN_NAME}/existing.png",
])
def test_fresh_run_rejects_existing_artifacts_without_overwriting(tmp_path: Path, relative: str):
    artifact = tmp_path / "results with spaces" / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"preserve this existing run")
    completed = _run(tmp_path)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert artifact.read_bytes() == b"preserve this existing run"


@pytest.mark.parametrize("options", [
    ("--device", "bad"), ("--benchmark-profile", "bad"),
    ("--epochs", "0"), ("--epochs", "4", "--proposal-epochs", "4"),
    ("--batch-size", "0"), ("--train-size", "0"), ("--val-size", "0"),
    ("--num-workers", "-1"), ("--mse-tolerance", "nan"),
    ("--mse-tolerance", "0"), ("--mse-tolerance", "inf"),
    ("--network-repeats", "0"), ("--end-to-end-repeats", "0"),
    ("--run-name", "../escape"), ("--run-name", "."), ("--run-name", ".."),
    ("--unknown-option",), ("--epochs",),
])
def test_invalid_arguments_fail_without_output_writes(tmp_path: Path, options: tuple[str, ...]):
    completed = _run(tmp_path, *options)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert not list(tmp_path.iterdir())


def test_checkpoint_and_resume_are_mutually_exclusive(tmp_path: Path):
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"placeholder")
    completed = _run(tmp_path, "--checkpoint", _bash_path(checkpoint), "--resume-run")
    assert completed.returncode != 0
    assert not (tmp_path / "results with spaces").exists()


def test_preparation_dry_run_prints_all_three_download_commands(tmp_path: Path):
    completed = _run(tmp_path, "--prepare-real-data", "--data-root", _bash_path(tmp_path / "absent data"))
    commands = _commands(completed)
    filenames = {_script_arguments(command)[0] for command in commands.values()}
    assert {"prepare_uji_pen.py", "prepare_natural_earth.py", "prepare_usgs_contours.py"} <= filenames
    assert not list(tmp_path.iterdir())


def test_default_run_name_is_isolated_from_historical_checkpoint_names(tmp_path: Path):
    completed = subprocess.run(
        [_bash(), _bash_path(SCRIPT), "--dry-run", "--python", MISSING_PYTHON,
         "--output-root", _bash_path(tmp_path / "new results")],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
    )
    commands = _commands(completed)
    assert _parsed(commands["train_fresh"]).output.name == "overnight_1070_arch_3090_r1.pt"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("filename, expected", [
    ("src/spline_fitting/models/v16_network.py", "f56bd584e1c6d8009ae4f3860455a79bd47713fd5e447a0244badd6b4dfa42b3"),
    ("src/spline_fitting/checkpointing.py", "50d1510b0e7e4ecad99f3e87e3ac9d6ab9099080100ba99a504a24ecb773e7d7"),
])
def test_historical_model_and_checkpoint_contract_remain_unchanged(filename: str, expected: str):
    # read_text normalizes CRLF so this guard works in both Windows and Linux.
    original = (ROOT / filename).read_text(encoding="utf-8").encode("utf-8")
    assert hashlib.sha256(original).hexdigest() == expected


def test_runner_does_not_delegate_to_a_new_model_version():
    source = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"(?:train|benchmark|visualize)_v(?:1[7-9]|[2-9][0-9])", source)
    assert "train_v16.py" in source
    assert "benchmark_v16_datasets.py" in source


@pytest.fixture
def preflight(monkeypatch):
    # Importing this orchestration module alone does not import torch or train.
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = ROOT / "scripts/overnight_linux_preflight.py"
    spec = importlib.util.spec_from_file_location("historical_overnight_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preflight_missing_data_fails_before_writing_a_report(tmp_path: Path, preflight):
    target = tmp_path / "audit" / "preflight.json"
    with pytest.raises(FileNotFoundError, match="manifest"):
        preflight.main(["--device", "cpu", "--data-root", str(tmp_path / "absent data"),
                        "--output", str(target)])
    assert not target.parent.exists()


def test_preflight_rejects_manifest_without_its_relative_point_files(tmp_path: Path, preflight):
    manifest = tmp_path / "data/splits/uji_pen_v2.jsonl"
    manifest.parent.mkdir(parents=True)
    row = {
        "sample_id": "held_out_curve", "source_dataset": "UJI", "group_id": "test_writer",
        "split": "test", "points_path": "../processed/uji_pen_v2/missing.npy",
        "num_points": 192, "point_dim": 2, "has_knot_labels": False, "metadata": {},
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="copy the complete data tree"):
        preflight.main(["--device", "cpu", "--data-root", str(tmp_path / "data")])


def test_preflight_keeps_initializer_provenance_instead_of_claiming_scratch(tmp_path: Path, preflight):
    import torch

    path = tmp_path / "legacy_initializer.pt"
    config = {"real_fraction": 0.5, "real_manifest": ["legacy/uji.jsonl"]}
    provenance = [{"source": "UJI", "manifest_sha256": "a" * 64}]
    torch.save({"epoch": 4, "stage": "proposal", "objective_version": "legacy_source",
                "training_config": config, "real_data_provenance": provenance}, path)
    before = path.read_bytes()
    record = preflight.checkpoint_record(path, deployment=False, tolerance=5e-5)
    assert record["training_config"] == config
    assert record["real_data_provenance"] == provenance
    assert record["stage"] == "proposal" and record["epoch"] == 4
    assert record["sha256"] == hashlib.sha256(before).hexdigest()
    assert "does not erase prior exposure" in record["note"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("capacity, tolerance, expected_error", [
    (48, 5e-5, "64-internal-candidate"),
    (64, 1e-4, "MSE tolerance"),
    (64, 5e-5, None),
])
def test_preflight_deployment_requires_old_objective_equal_capacity_and_tolerance(
    tmp_path: Path, preflight, monkeypatch, capacity, tolerance, expected_error,
):
    import torch
    import spline_fitting.checkpointing as checkpointing

    path = tmp_path / "deployment.pt"
    torch.save({"objective_version": checkpointing.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                "training_config": {"mse_tolerance": tolerance}}, path)
    monkeypatch.setattr(checkpointing, "build_model_from_checkpoint",
                        lambda payload: (None, {"max_internal_knots": capacity}, None))
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            preflight.checkpoint_record(path, deployment=True, tolerance=5e-5)
    else:
        record = preflight.checkpoint_record(path, deployment=True, tolerance=5e-5)
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_rejects_foreign_checkpoint_objective(tmp_path: Path, preflight):
    import torch

    path = tmp_path / "foreign.pt"
    torch.save({"objective_version": "not_the_historical_v16"}, path)
    with pytest.raises(ValueError, match="historical native v16 objective"):
        preflight.checkpoint_record(path, deployment=True, tolerance=5e-5)


@pytest.fixture
def resume_bundle(tmp_path: Path, preflight):
    import torch
    import train_v16
    import spline_fitting.checkpointing as checkpointing
    from spline_fitting.models.v16_network import V16CandidateSelectionNetwork

    output = tmp_path / "results with spaces/checkpoints" / f"{RUN_NAME}.pt"
    output.parent.mkdir(parents=True)
    last = output.with_name(output.stem + ".last.pt")
    proposal = output.with_name(output.stem + ".proposal.pt")
    arguments = ["--epochs", "64", "--proposal-epochs", "4", "--real-fraction", "0",
                 "--candidate-knots", "64", "--mse-tolerance", "5e-5",
                 "--hidden-dim", "16", "--encoder-layers", "1", "--selector-layers", "1",
                 "--no-resample-train-each-epoch", "--device", "cpu",
                 "--output", str(output), "--resume", str(last)]
    args = train_v16.parser().parse_args(arguments)
    train_v16.validate_args(args)
    config = train_v16.serial_args(args)
    config.update(train_seed=args.seed, train_seed_stride=train_v16.EPOCH_SEED_STRIDE)
    model = V16CandidateSelectionNetwork(hidden_dim=16, encoder_layers=1,
                                        selector_layers=1, max_internal_knots=64,
                                        mse_tolerance=5e-5)
    payload = {
        "objective_version": checkpointing.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "architecture_revision": checkpointing.V16_ADAPTIVE_SELECTION_REVISION,
        "simplification_contract": checkpointing.V16_SIMPLIFICATION_CONTRACT,
        "training_config": config, "real_data_provenance": [], "epoch": 64, "stage": "joint",
        "model_config": model.get_config(), "model_state_dict": model.state_dict(),
    }
    torch.save(payload, last)
    torch.save({**payload, "epoch": 62}, output)
    torch.save({**payload, "epoch": 4, "stage": "proposal"}, proposal)
    return {"last": last, "output": output, "proposal": proposal,
            "arguments": arguments, "payload": payload}


def test_completed_resume_status_is_read_only_and_needs_no_data_or_cuda(
    preflight, resume_bundle, capsys,
):
    files = [resume_bundle[key] for key in ("last", "output", "proposal")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    # --device defaults to cuda in preflight, but this status-only path is CPU-only.
    status = preflight.main(["--resume-status", "--checkpoint", str(resume_bundle["last"]),
                             "--", *resume_bundle["arguments"]])
    assert status == "completed"
    assert capsys.readouterr().out == "completed\n"
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files} == before


def test_resume_status_allows_total_epoch_extension_and_runtime_changes(preflight, resume_bundle):
    arguments = list(resume_bundle["arguments"])
    arguments[arguments.index("--epochs") + 1] = "80"
    arguments.extend(["--device", "cuda", "--num-workers", "4"])
    assert preflight.resume_status(resume_bundle["last"], arguments) == "needed"


@pytest.mark.parametrize("option, value", [
    ("--batch-size", "32"), ("--train-size", "1501"), ("--proposal-epochs", "5"),
    ("--mse-tolerance", "0.0001"), ("--candidate-knots", "56"),
    ("--max-control-points", "28"), ("--policy-samples", "2"),
])
def test_completed_resume_cannot_bypass_training_config_checks(preflight, resume_bundle, option, value):
    arguments = [*resume_bundle["arguments"], option, value]
    with pytest.raises(ValueError, match="mismatch|tolerance|Kc64"):
        preflight.resume_status(resume_bundle["last"], arguments)


@pytest.mark.parametrize("field, value, message", [
    ("epoch", -1, "completed epoch"), ("epoch", "64", "completed epoch"),
    ("epoch", 65, "saved training budget"),
    ("objective_version", "foreign", "objective_version"),
    ("architecture_revision", "foreign", "architecture_revision"),
    ("simplification_contract", "foreign", "simplification_contract"),
    ("stage", "unknown", "training stage"),
    ("real_data_provenance", [{"source": "other"}], "provenance"),
])
def test_resume_status_rejects_invalid_or_foreign_metadata(
    preflight, resume_bundle, field, value, message,
):
    import torch

    torch.save({**resume_bundle["payload"], field: value}, resume_bundle["last"])
    with pytest.raises(ValueError, match=message):
        preflight.resume_status(resume_bundle["last"], resume_bundle["arguments"])


@pytest.mark.parametrize("artifact", ["last", "output", "proposal"])
def test_resume_status_rejects_relocated_windows_output_in_every_artifact(
    preflight, resume_bundle, artifact,
):
    import torch

    payload = dict(resume_bundle["payload"])
    payload["training_config"] = {**payload["training_config"],
                                  "output": "Z:\\original_windows_run\\historic.pt"}
    torch.save(payload, resume_bundle[artifact])
    with pytest.raises(ValueError, match="original absolute --output"):
        preflight.resume_status(resume_bundle["last"], resume_bundle["arguments"])


@pytest.mark.parametrize("artifact", ["output", "proposal"])
def test_completed_resume_requires_best_and_proposal_artifacts(preflight, resume_bundle, artifact):
    resume_bundle[artifact].unlink()
    with pytest.raises(ValueError, match="saved checkpoint artifact"):
        preflight.resume_status(resume_bundle["last"], resume_bundle["arguments"])


def test_resume_status_strictly_restores_old_model_weights(preflight, resume_bundle):
    import torch

    payload = dict(resume_bundle["payload"])
    payload["model_state_dict"] = {}
    torch.save(payload, resume_bundle["last"])
    with pytest.raises(RuntimeError, match="Missing key"):
        preflight.resume_status(resume_bundle["last"], resume_bundle["arguments"])


@pytest.mark.parametrize("status, expected_training", [("completed", False), ("needed", True)])
def test_shell_resume_status_controls_training_but_continues_reports(
    tmp_path: Path, resume_bundle, status, expected_training,
):
    data = tmp_path / "fixture data"
    for relative in ("splits/uji_pen_v2.jsonl",
                     "processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl",
                     "processed/usgs_contours/large_scale/manifest.jsonl"):
        manifest = data / relative
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("", encoding="utf-8")
    # Stub expensive commands, testing the real Bash control flow separately
    # from the strict checkpoint validator exercised above.
    executable = tmp_path / "stub python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        f"  *' --resume-status '*) printf '%s\\n' '{status}' ;;\n"
        "  *) : ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    completed = _run(tmp_path, "--resume-run", "--python", _bash_path(executable),
                     "--data-root", _bash_path(data), dry_run=False)
    commands = _commands(completed)
    assert ("train_resume" in commands) == expected_training
    assert "train_fresh" not in commands
    assert "check_resume_status" in commands
    assert {"benchmark_six_methods", "plot_four_metrics", "plot_ours_cases",
            "plot_six_method_real_cases"} <= commands.keys()
    log = tmp_path / "results with spaces/logs" / RUN_NAME / "check_resume_status.log"
    assert "--resume-status" in log.read_text(encoding="utf-8")
    if not expected_training:
        assert "skip training" in completed.stdout
