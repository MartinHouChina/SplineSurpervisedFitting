from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v16_mse1e-4_3090.sh"


def test_linux_runner_encodes_current_training_and_evaluation_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    required = (
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        'RUN_NAME="candidate_selection_v16_mse1e-4_sourcek56_kc72_feasible_teacher_linux_r1"',
        'SIMPLIFICATION_CONTRACT="fixed_proposal_offline_feasible_subset_v1"',
        "Checkpoint selection is empirical",
        "offline feasible-subset Teacher on a fixed Proposal frame",
        "EPOCHS=128",
        "PROPOSAL_EPOCHS=64",
        "SELECTOR_WARMUP_EPOCHS=8",
        "SELECTOR_LR=2e-4",
        "PROPOSAL_JOINT_LR=0",
        "PARAMETER_JOINT_LR=0",
        "DECODER_JOINT_LR=5e-5",
        "--min-control-points 8",
        "--max-control-points 60",
        "--candidate-knots 72",
        "--knot-min-span 0.01",
        "--synthetic-boundary-val-size 32",
        "--proposal-high-k-fraction \"$PROPOSAL_HIGH_K_FRACTION\"",
        "--proposal-high-k-min-knots \"$PROPOSAL_HIGH_K_MIN_KNOTS\"",
        "--proposal-knot-assignment-weight 1.0",
        "--proposal-multiscale-recall-weight 0.25",
        "--fine-teacher-weight 0.5",
        "--parameter-gap-weight 0.05",
        '--selector-warmup-epochs "$SELECTOR_WARMUP_EPOCHS"',
        '--selector-lr "$SELECTOR_LR"',
        '--proposal-joint-lr "$PROPOSAL_JOINT_LR"',
        '--parameter-joint-lr "$PARAMETER_JOINT_LR"',
        '--decoder-joint-lr "$DECODER_JOINT_LR"',
        "--joint-supervision offline_feasible_teacher",
        '--feasible-teacher-cache-dir "$TEACHER_DIRECTORY"',
        "--synthetic-count-role reference_only",
        "--no-resample-train-each-epoch",
        "--no-synthetic-geometry-oracle-teacher",
        "--real-fraction 0",
        "--prepare-real-data",
        "scripts/prepare_industrial_offsets.py",
        'IndustrialOffset=$INDUSTRIAL_OFFSET_MANIFEST',
        "--mse-tolerance 1e-4",
        "--initial-keep-fraction 0.4166666666666667",
        "--min-knot-count 4 --max-knot-count 56",
        "--max-internal-knots 56",
        "--paper-initial-knots 56",
        "--liang-dense-knots 56",
        "--method-set published",
        "scripts/inspect_v16_checkpoint.py",
        "scripts/plot_v16_method_comparison.py",
        "scripts/visualize_v16_real_deployments.py",
        'PROPOSAL_FINAL_PATH="$CHECKPOINT_DIRECTORY/$RUN_NAME.proposal.final.pt"',
        '"$PROPOSAL_FINAL_PATH"',
        'TEACHER_DIRECTORY="$OUTPUT_ROOT/teachers/$RUN_NAME"',
        '"$TEACHER_DIRECTORY"',
        '--benchmark-profile quick|full',
    )
    for fragment in required:
        assert fragment in source

    assert source.index("scripts/train_v16.py") < source.index(
        "scripts/inspect_v16_checkpoint.py"
    )
    assert source.index("scripts/inspect_v16_checkpoint.py") < source.index(
        "scripts/benchmark_v16_datasets.py"
    )
    assert source.index("scripts/benchmark_v16_datasets.py") < source.index(
        "scripts/plot_v16_method_comparison.py"
    )
    assert source.index("scripts/plot_v16_method_comparison.py") < source.index(
        "scripts/visualize_v16_real_deployments.py"
    )
    assert '--resume "$LAST_PATH"' in source
    assert '--feasible-teacher-batch-size "$FEASIBLE_TEACHER_BATCH_SIZE"' in source
    assert '[[ -f "$LAST_PATH" ]]' in source
    assert 'TRAIN_PHASE=train_resume' in source
    assert "--overwrite" not in source
    assert "--proposal-pass-target" not in source
    assert "--deployment-pass-target" not in source
    assert "--complexity-pass-margin" not in source
    assert "--allow-infeasible-proposals" not in source
    assert "--required-pass-rate" not in source
    assert "--teacher-low-count-sweep" not in source
    assert "--teacher-prefix-search-steps" not in source
    assert "--counterfactual-edits" not in source
    assert "--policy-samples" not in source
    assert "--synthetic-geometry-oracle-teacher\n" not in source


def test_linux_runner_refuses_unknown_options_and_documents_help() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "unknown option" in source
    assert "--no-init-checkpoint" in source
    assert "rerun with --prepare-real-data" in source
    assert "--diagnostic" in source
    assert "--dry-run" in source
    assert "--benchmark-profile" in source
    assert "--resume-run" in source
    assert "--feasible-teacher-batch-size" in source
    assert "--python PATH" in source
    assert "--proposal-high-k-fraction X" in source
    assert "--proposal-high-k-min-knots N" in source
    assert "data/processed/industrial_offsets/v1/manifest.jsonl" in source


def test_linux_runner_has_valid_bash_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None or not Path(bash).as_posix().startswith("/"):
        pytest.skip("native /bin/bash is unavailable on this platform")
    completed = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_linux_runner_resume_dry_run_requires_last_and_keeps_run_name(
    request: pytest.FixtureRequest,
) -> None:
    bash = shutil.which("bash")
    if bash is None or not Path(bash).as_posix().startswith("/"):
        pytest.skip("native /bin/bash is unavailable on this platform")
    tmp_path = request.getfixturevalue("tmp_path")
    run_name = "resume_smoke"
    output_root = tmp_path / "outputs"
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    command = [
        bash, str(SCRIPT), "--dry-run", "--benchmark-profile", "quick",
        "--device", "cpu", "--output-root", str(output_root),
        "--run-name", run_name, "--resume-run",
    ]
    missing = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        timeout=10, check=False,
    )
    assert missing.returncode != 0
    assert "requires the existing last checkpoint" in missing.stderr

    last = checkpoint_dir / f"{run_name}.last.pt"
    proposal = checkpoint_dir / f"{run_name}.proposal.pt"
    last.write_bytes(b"dry-run placeholder")
    proposal.write_bytes(b"dry-run placeholder")
    resumed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        timeout=10, check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "[train_resume]" in resumed.stdout
    assert "--resume" in resumed.stdout
    assert str(last) in resumed.stdout
    assert str(checkpoint_dir / f"{run_name}.pt") in resumed.stdout
    assert "[train_fresh]" not in resumed.stdout
    assert "--init-checkpoint" not in resumed.stdout
