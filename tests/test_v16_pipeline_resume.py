from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import train_v16 as train  # noqa: E402
import v16_pipeline_resume as recovery  # noqa: E402


@pytest.fixture
def training_run(tmp_path, monkeypatch):
    output = tmp_path / "model.pt"
    last = tmp_path / "model.last.pt"
    argv = ["--epochs", "4", "--proposal-epochs", "2", "--selector-warmup-epochs", "0",
            "--output", str(output), "--resume", str(last), "--device", "cpu",
            "--joint-supervision", "offline_feasible_teacher", "--no-resample-train-each-epoch",
            "--feasible-teacher-cache-dir", str(tmp_path / "teacher"),
            "--synthetic-count-role", "reference_only", "--proposal-joint-lr", "0",
            "--parameter-joint-lr", "0"]
    args = train.parser().parse_args(argv)
    train.validate_args(args)
    saved = {
        "epoch": 4, "stage": "joint", "training_config": train.serial_args(args),
        "synthetic_data_contract": train.synthetic_data_contract(args),
        "objective_version": train.V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
        "simplification_contract": train.V16_FEASIBLE_TEACHER_CONTRACT,
        "architecture_revision": train.V16_ADAPTIVE_SELECTION_REVISION,
        "loss_semantics_revision": train.V16_OFFLINE_LOSS_SEMANTICS_REVISION,
        "best_joint_rank": [1.0], "best_proposal_rank": [1.0],
        "real_data_provenance": [],
        "model_config": {"max_internal_knots": 72, "one_shot_safety_sigma": 0.0,
                         "one_shot_safety_knots": 0},
    }
    artifacts = {last: saved, output: copy.deepcopy(saved),
                 tmp_path / "model.proposal.pt": copy.deepcopy(saved)}
    artifacts[tmp_path / "model.proposal.pt"]["stage"] = "proposal"
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: artifacts[Path(path)])
    monkeypatch.setattr(train, "build_model_from_checkpoint", lambda payload: None)
    return argv, saved, artifacts


def test_completed_training_is_validated_and_skipped(training_run):
    argv, _, _ = training_run
    assert recovery.training_status(argv) == "complete"


def test_unfinished_training_and_explicit_extension_use_native_resume(training_run):
    argv, saved, _ = training_run
    saved["epoch"] = 3
    assert recovery.training_status(argv) == "resume"
    saved["epoch"] = 4
    assert recovery.training_status([*argv, "--epochs", "5"]) == "resume"


@pytest.mark.parametrize("override,expected", [
    (["--epochs", "3"], "cannot reduce"),
    (["--candidate-knots", "80"], "configuration mismatch"),
    (["--max-control-points", "56"], "configuration mismatch"),
    (["--train-size", "10"], "configuration mismatch"),
])
def test_completed_training_rejects_changed_experiment(training_run, override, expected):
    argv, _, _ = training_run
    with pytest.raises(ValueError, match=expected):
        recovery.training_status([*argv, *override])


def test_completed_training_allows_runtime_changes_and_proposal_safety_schedule(training_run):
    argv, saved, artifacts = training_run
    saved["training_config"]["device"] = "cuda"
    saved["loss_semantics_migration"] = {"reason": "historical completed migration"}
    proposal = next(value for path, value in artifacts.items() if path.name.endswith(".proposal.pt"))
    proposal["model_config"]["one_shot_safety_sigma"] = 0.4
    proposal["model_config"]["one_shot_safety_knots"] = 2
    assert recovery.training_status([*argv, "--num-workers", "0"]) == "complete"
    proposal["model_config"]["max_internal_knots"] = 56
    with pytest.raises(ValueError, match="model configuration mismatch"):
        recovery.training_status(argv)


def test_completed_training_rejects_proposal_copied_over_best_model(training_run):
    argv, _, artifacts = training_run
    best = next(value for path, value in artifacts.items() if path.name == "model.pt")
    best["stage"] = "proposal"
    with pytest.raises(ValueError, match="expected joint checkpoint"):
        recovery.training_status(argv)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def benchmark_run(tmp_path, monkeypatch):
    import benchmark_v15_datasets as benchmark
    directory = tmp_path / "comparison"
    directory.mkdir()
    command = [sys.executable, "scripts/benchmark_v16_datasets.py", "--output-dir", str(directory)]
    files = [ROOT / "scripts/benchmark_v15_datasets.py", ROOT / "scripts/benchmark_v16_datasets.py",
             *(ROOT / "src").rglob("*.py")]
    identity = {"checkpoint_sha256": "c" * 64, "configuration": {"candidate_knots": 24}}
    metadata = {**identity, "code_sha256": {
        str(path.relative_to(ROOT)): recovery.digest(path) for path in files}}
    metadata["fingerprint"] = recovery.json_digest(metadata)
    rows = [{"dataset": "Synthetic", "sample_id": "k4", "method": "ours"}]
    summary = [{"n": 1}]
    _write_json(directory / "experiment.json", metadata)
    _write_json(directory / "comparison.json", {"metadata": metadata, "measurements": rows, "summary": summary})
    (directory / "measurements.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    for name in ("report.md", "summary.csv", "measurements.csv"):
        (directory / name).write_text("complete", encoding="utf-8")
    monkeypatch.setattr(benchmark, "summarize", lambda rows: summary)
    monkeypatch.setattr(recovery, "benchmark_context", lambda command: (identity, {("Synthetic", "k4", "ours")}))
    calls = []
    monkeypatch.setattr(recovery.subprocess, "call", lambda command: calls.append(command) or 0)
    return directory, command, metadata, identity, calls


def test_complete_benchmark_is_reused_without_solver_or_file_writes(benchmark_run):
    directory, command, _, _, calls = benchmark_run
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert recovery.run_benchmark(command, resume=True) == 0
    assert calls == []
    assert original == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_partial_benchmark_uses_native_fingerprint_checked_resume(benchmark_run):
    directory, command, metadata, _, calls = benchmark_run
    _write_json(directory / "comparison.json", {"metadata": metadata, "measurements": [], "summary": []})
    assert recovery.run_benchmark(command, resume=True) == 0
    assert calls == [[*command, "--resume"]]


def test_benchmark_never_ignores_changed_code_or_requested_configuration(benchmark_run):
    directory, command, metadata, identity, calls = benchmark_run
    identity["configuration"]["candidate_knots"] = 56
    with pytest.raises(ValueError, match="fingerprint inputs changed"):
        recovery.run_benchmark(command, resume=True)
    identity["configuration"]["candidate_knots"] = 24
    metadata["code_sha256"]["scripts/benchmark_v15_datasets.py"] = "0" * 64
    metadata["fingerprint"] = recovery.json_digest({key: value for key, value in metadata.items() if key != "fingerprint"})
    _write_json(directory / "experiment.json", metadata)
    with pytest.raises(ValueError, match="source fingerprint changed"):
        recovery.run_benchmark(command, resume=True)
    assert not calls


def test_existing_outputs_still_require_explicit_resume(benchmark_run):
    _, command, _, _, calls = benchmark_run
    with pytest.raises(ValueError, match="without --resume-run"):
        recovery.run_benchmark(command, resume=False)
    assert not calls


def test_plot_retry_preserves_partial_png_and_complete_receipt_reuses(tmp_path, monkeypatch):
    directory = tmp_path / "figures"
    command = [sys.executable, "plot.py", "--output-dir", str(directory)]
    monkeypatch.setattr(recovery, "derived_signature", lambda command, kind: {"request": "same"})
    calls = []

    def plot(command):
        target = Path(recovery.command_option(command, "--output-dir"))
        calls.append(target)
        (target / "v16_published_methods_input.png").write_bytes(b"original" if len(calls) == 1 else b"retry")
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(recovery.subprocess, "call", plot)
    assert recovery.run_derived(command, kind="plot", resume=False) == 1
    assert recovery.run_derived(command, kind="plot", resume=True) == 0
    assert (directory / "v16_published_methods_input.png").read_bytes() == b"original"
    assert calls[1].parent == directory and calls[1] != directory
    assert recovery.run_derived(command, kind="plot", resume=True) == 0
    assert len(calls) == 2
    receipt = recovery.read_json(directory / ".pipeline-stage.json")
    assert [attempt["state"] for attempt in receipt["attempts"]] == ["interrupted", "complete"]


def test_unowned_figure_files_are_preserved_on_resume(tmp_path, monkeypatch):
    (tmp_path / "user.png").write_bytes(b"untouched")
    monkeypatch.setattr(recovery, "derived_signature", lambda command, kind: {})
    def plot(command):
        target = Path(recovery.command_option(command, "--output-dir"))
        assert target.parent == tmp_path and target != tmp_path
        (target / "v16_published_methods_input.png").write_bytes(b"result")
        return 0
    monkeypatch.setattr(recovery.subprocess, "call", plot)
    command = [sys.executable, "plot.py", "--output-dir", str(tmp_path)]
    with pytest.raises(ValueError, match="without --resume-run"):
        recovery.run_derived(command, kind="plot", resume=False)
    assert recovery.run_derived(command, kind="plot", resume=True) == 0
    assert (tmp_path / "user.png").read_bytes() == b"untouched"


def test_visualization_receipt_requires_every_selected_case_image(tmp_path):
    image = tmp_path / "case.png"
    image.write_bytes(b"png")
    report = {"dataset_provenance": [{"selected_count": 2}],
              "records": [{"image": str(image)}]}
    _write_json(tmp_path / "deployment_visualizations.json", report)
    with pytest.raises(ValueError, match="all selected cases"):
        recovery.checked_artifacts(tmp_path, "visualize")
    report["dataset_provenance"][0]["selected_count"] = 1
    _write_json(tmp_path / "deployment_visualizations.json", report)
    assert set(recovery.checked_artifacts(tmp_path, "visualize")) == {"case.png", "deployment_visualizations.json"}


def test_stage_cli_does_not_swallow_resume_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(recovery, "run_benchmark", lambda command, resume: seen.append((command, resume)) or 0)
    assert recovery.main(["benchmark", "--resume-run", "--", "python", "script.py"]) == 0
    assert seen == [(["python", "script.py"], True)]


def test_paired_receipt_does_not_hash_itself_and_reuses_complete_report(tmp_path, monkeypatch):
    target = tmp_path / "paired_native_stage" / "paired_native.json"
    command = [sys.executable, "paired.py", "--output-json", str(target)]
    monkeypatch.setattr(recovery, "derived_signature", lambda command, kind: {})
    calls = []
    def paired(command):
        calls.append(command)
        _write_json(Path(recovery.command_option(command, "--output-json")), {"complete": True})
        return 0
    monkeypatch.setattr(recovery.subprocess, "call", paired)
    assert recovery.run_derived(command, kind="paired", resume=False) == 0
    assert recovery.run_derived(command, kind="paired", resume=True) == 0
    assert len(calls) == 1
    receipt = recovery.read_json(target.parent / ".pipeline-stage.json")
    assert set(receipt["attempts"][0]["artifacts"]) == {"paired_native.json"}


def test_paired_signature_detects_changed_curve_content_with_same_manifest(tmp_path, monkeypatch):
    import quick_compare_v16_checkpoints as paired
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("unchanged", encoding="utf-8")
    command = [sys.executable, "scripts/quick_compare_v16_checkpoints.py",
               "--old-checkpoint", str(checkpoint), "--new-checkpoint", str(checkpoint),
               "--manifest", f"Real={manifest}", "--device", "cpu",
               "--output-json", str(tmp_path / "paired_native.json")]
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model_config": {}})
    monkeypatch.setattr(paired, "_dataset_config_from_checkpoint", lambda *args: {"num_points": 2, "point_dim": 2})
    case = {"dataset": "Real", "case_id": "curve", "points": torch.zeros(2, 2)}
    monkeypatch.setattr(paired, "_real_cases", lambda *args, **kwargs: [case])
    before = recovery.derived_signature(command, "paired")
    case["points"][0, 0] = 1
    after = recovery.derived_signature(command, "paired")
    assert before["inputs"] == after["inputs"]
    assert before["real_cases"] != after["real_cases"]
