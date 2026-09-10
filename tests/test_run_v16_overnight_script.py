from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v16_overnight_12h.ps1"


def test_overnight_script_has_the_audited_training_and_benchmark_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    required_training_fragments = (
        '[int]$TargetEpochs = 56',
        '[int]$FallbackEpochs = 64',
        '[double]$BudgetHours = 12.0',
        '"--train-size", "1500"',
        '"--val-size", "500"',
        '"--min-control-points", "8"',
        '"--max-control-points", "28"',
        '"--candidate-knots", "64"',
        '"--mse-tolerance", "5e-5"',
        '"--no-resample-train-each-epoch"',
    )
    for fragment in required_training_fragments:
        assert fragment in text

    required_benchmark_fragments = (
        '[int]$SyntheticSamplesPerK = 2',
        '[int]$RealSamplesPerDataset = 8',
        '[int]$KangAdmmIterations = 400',
        '[int]$LuoDeIterations = 50',
        '"--gradient-steps", "12"',
        '"--paper-lambda-bisections", "8"',
        '"--paper-relocation-iterations", "8"',
        '"--liang-feature-samples", "1025"',
        '"--dung-scan-intervals", "10"',
        '"--dung-optimization-iterations", "10"',
        '"--luo-de-population", "10"',
        '"--network-repeats", "20"',
        '"--end-to-end-repeats", "1"',
        '"--method-set", "all"',
    )
    for fragment in required_benchmark_fragments:
        assert fragment in text

    assert text.index('"scripts/train_v16.py"') < text.index(
        '"scripts/benchmark_v16_datasets.py"'
    )
    assert text.index('"scripts/benchmark_v16_datasets.py"') < text.index(
        '"scripts/plot_v16_method_comparison.py"'
    )
    assert text.index('"scripts/plot_v16_method_comparison.py"') < text.index(
        '"scripts/visualize_v16_ours_cases.py"'
    )


def test_overnight_script_waits_safely_and_marks_diagnostic_outputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Get-CimInstance Win32_Process" in text
    assert "Wait-Process -Id $Target.ProcessId -Timeout 45" in text
    assert "Test-ProcessTargetsCheckpoint" in text
    assert "Get-ResumeArguments" in text
    assert "payload['training_config']" in text
    assert "Another GPU compute process is active" in text
    assert "Stop-Process" not in text
    assert "Start-Process" in text
    assert "-WindowStyle Hidden" in text
    assert '@("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand"' in text
    assert "[switch]$Detach" in text
    assert '$env:PYTHONUNBUFFERED = "1"' in text
    assert '$env:MPLBACKEND = "Agg"' in text
    assert '"--allow-unqualified-diagnostic"' in text
    assert "DIAGNOSTIC NOT FINAL" in text
    assert "eligible:\\s+NO" in text
    assert "formal_" in text and "diagnostic_" in text


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell.exe") or shutil.which(
        "powershell"
    )


def test_dry_run_materializes_the_serial_plan(tmp_path: Path) -> None:
    executable = _powershell()
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    completed = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-DryRun",
            "-OutputRoot",
            str(tmp_path),
            "-RunName",
            "pytest_v16_overnight",
            "-InitCheckpoint",
            str(tmp_path / "not-present.pt"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    manifest_path = (
        tmp_path
        / "logs"
        / "pytest_v16_overnight"
        / "overnight_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert manifest["status"] == "completed"
    assert manifest["requested_profile"] == {
        "candidate_internal_knots": 64,
        "source_internal_knots": "4..24",
        "source_control_points": "8..28",
        "largest_source_full_knot_vector": 32,
        "train_size": 1500,
        "validation_size": 500,
        "mse_tolerance": 5e-5,
        "fresh_batch_size": 64,
        "target_epochs": 56,
        "fallback_epochs": 64,
        "budget_hours": 12,
        "device": "cuda",
    }
    assert list(manifest["phases"]) == [
        "train_fresh",
        "benchmark_all_methods",
        "plot_method_comparison",
        "plot_ours_cases",
    ]
    benchmark = manifest["phases"]["benchmark_all_methods"]["command"]
    assert "--samples-per-knot-count 2" in benchmark
    assert "--real-samples-per-dataset 8" in benchmark
    assert "--paper-admm-iterations 400" in benchmark
    assert "--luo-de-iterations 50" in benchmark


def test_resume_dry_run_replays_saved_config_and_only_extends_epochs(
    tmp_path: Path,
) -> None:
    executable = _powershell()
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    output = checkpoint_dir / "resume_contract.pt"
    last = checkpoint_dir / "resume_contract.last.pt"
    history = checkpoint_dir / "resume_contract.history.json"
    torch.save(
        {
            "training_config": {
                "epochs": 16,
                "proposal_epochs": 4,
                "train_size": 1500,
                "val_size": 500,
                "batch_size": 32,
                "candidate_knots": 64,
                "min_control_points": 8,
                "max_control_points": 28,
                "mse_tolerance": 5e-5,
                "resample_train_each_epoch": False,
                "certified_minimal_source": True,
                "device": "cuda",
                "output": str(output.resolve()),
                "resume": None,
                "init_checkpoint": "ignored-on-resume.pt",
                "train_seed": 42,
                "train_seed_stride": 1_000_003,
            }
        },
        last,
    )
    history.write_text('[{"epoch": 16, "stage": "joint"}]', encoding="utf-8")

    completed = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-DryRun",
            "-OutputRoot",
            str(tmp_path),
            "-RunName",
            "resume_contract",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads(
        (
            tmp_path
            / "logs"
            / "resume_contract"
            / "overnight_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    assert "train_fresh" not in manifest["phases"]
    command = manifest["phases"]["train_resume"]["command"]
    assert "--batch-size 32" in command
    assert "--proposal-epochs 4" in command
    assert "--no-resample-train-each-epoch" in command
    assert "--epochs 56" in command
    assert "--batch-size 64" not in command
    assert "--proposal-epochs 12" not in command
    assert f"--resume {last}" in command
    assert f"--output {output}" in command
