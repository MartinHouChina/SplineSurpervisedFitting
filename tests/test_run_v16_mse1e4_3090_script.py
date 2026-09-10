from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v16_mse1e-4_3090.ps1"


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell.exe") or shutil.which(
        "powershell"
    )


def _run_dry(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    executable = _powershell()
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    return subprocess.run(
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
            "pytest_v16_mse1e4",
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_script_encodes_the_requested_training_and_fair_comparison_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    training_fragments = (
        '[string]$RunName = "candidate_selection_v16_mse1e-4_k56"',
        '"candidate_selection_v16_mse5e-5_k64.proposal.pt"',
        '[int]$Epochs = 80',
        '[int]$ProposalEpochs = 16',
        '[int]$TrainSize = 3000',
        '[int]$ValSize = 600',
        '[int]$RealValSize = 100',
        '[double]$RealFraction = 0.35',
        '[int]$BatchSize = 64',
        '"--train-size", [string]$TrainSize',
        '"--val-size", [string]$ValSize',
        '"--synthetic-boundary-val-size", "32"',
        '"--real-val-size", [string]$RealValSize',
        '"--min-control-points", "8"',
        '"--max-control-points", "60"',
        '"--candidate-knots", "56"',
        '"--knot-min-span", "0.01"',
        '"--mse-tolerance", "1e-4"',
        '"--initial-keep-fraction", "0.5357142857142857"',
        '"--teacher-low-count-sweep", "16"',
        '"--synthetic-count-role", "upper_bound"',
        '"--synthetic-geometry-oracle-teacher"',
        '"--oracle-teacher-extra-knots", "2"',
        '"--one-shot-coverage-bins", "0"',
        '"--complexity-max-scale", "6.0"',
        '"--complexity-ramp-epochs", "12"',
        '"--policy-samples", "2"',
        '"--counterfactual-edits", "4"',
        '"--final-safety-sigma", "0.03"',
        '"--final-safety-knots", "0"',
        '"--real-fraction", ([string]::Format(',
    )
    for fragment in training_fragments:
        assert fragment in text

    comparison_fragments = (
        '[int]$SyntheticSamplesPerK = 5',
        '[int]$RealSamplesPerDataset = 20',
        '[int]$KangAdmmIterations = 1000',
        '[int]$LuoDePopulation = 20',
        '[int]$LuoDeIterations = 100',
        '"--method-set", "published"',
        '"--max-knot-count", "56"',
        '"--max-internal-knots", "56"',
        '"--paper-initial-knots", "56"',
        '"--paper-lambda-bisections", "10"',
        '"--paper-relocation-iterations", "12"',
        '"--liang-dense-knots", "56"',
        '"--network-warmups", "10"',
        '"--reference"',
    )
    for fragment in comparison_fragments:
        assert fragment in text

    assert text.index('"scripts/train_v16.py"') < text.index(
        '"scripts/inspect_v16_checkpoint.py"'
    )
    assert text.index('"scripts/inspect_v16_checkpoint.py"') < text.index(
        '"scripts/benchmark_v16_datasets.py"'
    )
    assert text.index('"scripts/benchmark_v16_datasets.py"') < text.index(
        '"scripts/plot_v16_method_comparison.py"'
    )
    assert text.index('"scripts/plot_v16_method_comparison.py"') < text.index(
        '"scripts/visualize_v16_real_deployments.py"'
    )
    assert '"--resume"' not in text
    assert '"--overwrite"' not in text


def test_default_initializer_is_proposal_only_and_missing_file_is_explicit(
    tmp_path: Path,
) -> None:
    completed = _run_dry(
        tmp_path,
        "-InitCheckpoint",
        str(tmp_path / "missing_proposal.pt"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    combined = completed.stdout + completed.stderr
    assert "initializer was not found" in combined
    assert "train all modules from scratch" in combined

    manifest = json.loads(
        (
            tmp_path
            / "logs"
            / "pytest_v16_mse1e4"
            / "pipeline_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    assert manifest["status"] == "completed"
    assert manifest["checkpoint"].endswith("pytest_v16_mse1e4.pt")
    assert manifest["requested_profile"] == {
        "mse_tolerance": 1e-4,
        "candidate_internal_knots": 56,
        "full_cubic_knot_vector_size_at_all_keep": 64,
        "source_internal_knots": "4..56",
        "source_control_points": "8..60",
        "knot_min_span": 0.01,
        "points": 192,
        "epochs": 80,
        "proposal_epochs": 16,
        "train_size": 3000,
        "validation_size": 600,
        "synthetic_boundary_validation_size": 32,
        "real_validation_size_per_source": 100,
        "real_fraction": 0.35,
        "batch_size": 64,
        "selection_policy": "mass_topk",
        "initial_keep_fraction": 30 / 56,
        "teacher_low_count_sweep": 16,
        "synthetic_count_role": "upper_bound",
        "synthetic_geometry_oracle_teacher": True,
        "oracle_teacher_extra_knots": 2,
        "coverage_bins": 0,
        "synthetic_samples_per_k": 5,
        "real_samples_per_dataset": 20,
        "device": "cuda",
    }
    assert list(manifest["phases"]) == [
        "train_fresh",
        "inspect_checkpoint",
        "benchmark_six_methods",
        "plot_four_metrics",
        "visualize_six_method_real_cases",
    ]
    train = manifest["phases"]["train_fresh"]["command"]
    assert "--init-checkpoint" not in train
    assert "--resume" not in train
    benchmark = manifest["phases"]["benchmark_six_methods"]["command"]
    assert "--samples-per-knot-count 5" in benchmark
    assert "--min-knot-count 4" in benchmark
    assert "--max-knot-count 56" in benchmark
    assert "--real-samples-per-dataset 20" in benchmark
    assert "--method-set published" in benchmark
    assert "--paper-admm-iterations 1000" in benchmark
    assert "--luo-de-population 20" in benchmark
    assert "--luo-de-iterations 100" in benchmark


def test_initializer_is_forwarded_as_a_warm_start_without_resume(
    tmp_path: Path,
) -> None:
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"dry-run placeholder")
    completed = _run_dry(tmp_path, "-InitCheckpoint", str(initializer))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads(
        (
            tmp_path
            / "logs"
            / "pytest_v16_mse1e4"
            / "pipeline_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    command = manifest["phases"]["train_fresh"]["command"]
    assert f"--init-checkpoint {initializer}" in command
    assert "--resume" not in command
    assert "Warm-starting encoder, ParameterHead and CandidateHead only" in (
        completed.stdout + completed.stderr
    )


def test_diagnostic_dry_run_is_isolated_and_marks_all_consumers(
    tmp_path: Path,
) -> None:
    completed = _run_dry(
        tmp_path,
        "-Diagnostic",
        "-Epochs",
        "12",
        "-ProposalEpochs",
        "4",
        "-TrainSize",
        "256",
        "-ValSize",
        "64",
        "-RealValSize",
        "16",
        "-RealFraction",
        "0.25",
        "-InitCheckpoint",
        str(tmp_path / "missing.pt"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads(
        (
            tmp_path
            / "logs"
            / "pytest_v16_mse1e4"
            / "pipeline_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    assert manifest["diagnostic_not_final"] is True
    assert "diagnostic_000000000000" in manifest["comparison_directory"]
    training = manifest["phases"]["train_fresh"]["command"]
    assert "--epochs 12" in training
    assert "--proposal-epochs 4" in training
    assert "--train-size 256" in training
    assert "--val-size 64" in training
    assert "--real-val-size 16" in training
    assert "--real-fraction 0.25" in training
    assert "--allow-infeasible-proposals" in training
    for phase in (
        "benchmark_six_methods",
        "plot_four_metrics",
        "visualize_six_method_real_cases",
    ):
        assert "--allow-unqualified-diagnostic" in manifest["phases"][phase][
            "command"
        ]


def test_existing_checkpoint_is_never_overwritten(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "pytest_v16_mse1e4.pt"
    original = b"preserve me"
    checkpoint.write_bytes(original)

    completed = _run_dry(
        tmp_path,
        "-InitCheckpoint",
        str(tmp_path / "missing.pt"),
    )
    assert completed.returncode != 0
    assert "Refusing to overwrite existing run artifacts" in (
        completed.stdout + completed.stderr
    )
    assert checkpoint.read_bytes() == original
