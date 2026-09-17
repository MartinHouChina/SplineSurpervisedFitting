from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v16_mse1e-4_3090.sh"


def _bash():
    path = Path(r"C:\Program Files\Git\bin\bash.exe") if os.name == "nt" else shutil.which("bash")
    if not path or not Path(path).is_file():
        pytest.skip("Bash is unavailable")
    return str(path)


def _shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


def _command(output, phase):
    prefix = f"[{phase}] "
    line = next(line for line in output.splitlines() if line.startswith(prefix))
    return shlex.split(line[len(prefix):])


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
        "MAX_CONTROL_POINTS=60",
        "CANDIDATE_KNOTS=72",
        '--max-control-points "$MAX_CONTROL_POINTS"',
        '--candidate-knots "$CANDIDATE_KNOTS"',
        "--knot-min-span 0.01",
        "SYNTHETIC_BOUNDARY_VAL_SIZE=32",
        '--synthetic-boundary-val-size "$SYNTHETIC_BOUNDARY_VAL_SIZE"',
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
        "MSE_TOLERANCE=1e-4",
        '--mse-tolerance "$MSE_TOLERANCE"',
        "--initial-keep-fraction 0.4166666666666667",
        "--min-knot-count 4 --max-knot-count 56",
        "BASELINE_CAP=56",
        '--max-internal-knots "$BASELINE_CAP"',
        '--paper-initial-knots "$BASELINE_CAP"',
        '--liang-dense-knots "$BASELINE_CAP"',
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
    assert source.count('--mse-tolerance "$MSE_TOLERANCE"') == 5
    assert "--mse-tolerance 1e-4" not in source


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
    assert "--mse-tolerance X" in source
    assert "--proposal-high-k-fraction X" in source
    assert "--proposal-high-k-min-knots N" in source
    assert "--pilot-kc56" in source
    assert "data/processed/industrial_offsets/v1/manifest.jsonl" in source


def test_linux_runner_has_valid_bash_syntax_when_bash_is_available() -> None:
    bash = _bash()
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
    bash = _bash()
    tmp_path = request.getfixturevalue("tmp_path")
    run_name = "resume_smoke"
    output_root = tmp_path / "outputs"
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    command = [
        bash, str(SCRIPT), "--dry-run", "--benchmark-profile", "quick",
        "--device", "cpu", "--output-root", _shell_path(output_root),
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
    assert _shell_path(last) in resumed.stdout
    assert _shell_path(checkpoint_dir / f"{run_name}.pt") in resumed.stdout
    assert "[train_fresh]" not in resumed.stdout
    assert "--init-checkpoint" not in resumed.stdout


def test_linux_pilot_kc56_dry_run_preserves_formal_defaults_and_marks_stress_as_diagnostic() -> None:
    bash = _bash()
    pilot = subprocess.run(
        [
            bash, str(SCRIPT), "--dry-run", "--pilot-kc56", "--device", "cpu",
            "--run-name", "pilot_kc56_dry_run",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert pilot.returncode == 0, pilot.stdout + pilot.stderr
    assert "PILOT profile" in pilot.stdout
    assert "Kc=56, training source K=4..44" in pilot.stdout
    assert "K=45..56 out-of-training-range stress cases" in pilot.stdout
    assert "--candidate-knots 56" in pilot.stdout
    assert "--max-control-points 48" in pilot.stdout
    assert "--epochs 72" in pilot.stdout
    assert "--proposal-epochs 48" in pilot.stdout
    assert "--train-size 1500" in pilot.stdout
    assert "--val-size 300" in pilot.stdout
    assert "--real-val-size 30" in pilot.stdout
    assert "--feasible-teacher-batch-size 8" in pilot.stdout
    assert "--feasible-teacher-strategy synthetic_anchor_first" in pilot.stdout
    assert "--feasible-teacher-anchor-match-tolerance 0.02" in pilot.stdout
    assert "--count-structure-coupling" in pilot.stdout
    assert "--max-knot-count 56" in pilot.stdout
    assert "--synthetic-test-max-control-points 60" in pilot.stdout
    assert "--force-diagnostic" in pilot.stdout
    assert "--init-checkpoint" not in pilot.stdout

    formal = subprocess.run(
        [
            bash, str(SCRIPT), "--dry-run", "--device", "cpu",
            "--run-name", "formal_dry_run",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert formal.returncode == 0, formal.stdout + formal.stderr
    assert "Kc=72, training source K=4..56" in formal.stdout
    assert "--candidate-knots 72" in formal.stdout
    assert "--max-control-points 60" in formal.stdout
    assert "--max-internal-knots 56" in formal.stdout
    assert "--paper-initial-knots 56" in formal.stdout
    assert "--liang-dense-knots 56" in formal.stdout
    assert "--epochs 128" in formal.stdout
    assert "--proposal-epochs 64" in formal.stdout
    assert "--synthetic-test-max-control-points" not in formal.stdout
    assert "--feasible-teacher-strategy synthetic_anchor_first" not in formal.stdout
    assert "--count-structure-coupling" not in formal.stdout


def test_linux_pilot_kc56_refuses_existing_checkpoint_artifact(tmp_path: Path) -> None:
    bash = _bash()
    output_root = tmp_path / "outputs"
    checkpoint_directory = output_root / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    (checkpoint_directory / "pilot_collision.pt").write_bytes(b"existing user checkpoint")
    completed = subprocess.run(
        [
            bash, str(SCRIPT), "--dry-run", "--pilot-kc56", "--device", "cpu",
            "--output-root", _shell_path(output_root), "--run-name", "pilot_collision",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert completed.returncode != 0
    assert "refusing to overwrite existing run artifact" in completed.stderr


def test_linux_keep_swap_highk72_profile_is_opt_in_and_complete(tmp_path: Path) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "--keep-swap-highk72", "KEEP_SWAP_HIGHK72=1",
        "PROPOSAL_EPOCHS=64", "EPOCHS=96", "TRAIN_SIZE=1500",
        "VAL_SIZE=400", "REAL_VAL_SIZE=40", "CANDIDATE_KNOTS=72",
        "MAX_CONTROL_POINTS=60", "BASELINE_CAP=72",
        "PROPOSAL_HIGH_K_FRACTION=0.65",
        "--joint-high-k-fraction 0.50", "--joint-high-k-min-knots 45",
        "--synthetic-high-k-val-size 32",
        "--synthetic-high-k-val-min-knots 45",
        "--feasible-teacher-strategy synthetic_anchor_counterfactual",
        "--feasible-teacher-counterfactual-max-probes 16",
        "--count-structure-coupling", "DIAGNOSTIC=1",
    ):
        assert fragment in source
    bash = _bash()
    command = [
        bash, str(SCRIPT), "--dry-run", "--device", "cpu",
        "--output-root", _shell_path(tmp_path / "outputs"),
        "--run-name", "keep_swap_highk72_dry_run", "--keep-swap-highk72",
    ]
    run = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        timeout=10, check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    for fragment in (
        "KEEP_SWAP_HIGHK72 profile", "Kc=72, training source K=4..56",
        "--candidate-knots 72", "--max-control-points 60",
        "--epochs 96", "--proposal-epochs 64", "--train-size 1500",
        "--val-size 400", "--real-val-size 40", "--batch-size 64",
        "--proposal-high-k-fraction 0.65",
        "--joint-high-k-fraction 0.50", "--joint-high-k-min-knots 45",
        "--synthetic-high-k-val-size 32",
        "--feasible-teacher-strategy synthetic_anchor_counterfactual",
        "--feasible-teacher-counterfactual-max-probes 16",
        "--count-structure-coupling", "--max-knot-count 56",
        "--max-internal-knots 72", "--paper-initial-knots 72",
        "--liang-dense-knots 72",
        "--force-diagnostic", "[train_fresh]", "[benchmark_six_methods_plus_verified]",
        "[plot_four_metrics]", "[visualize_six_method_real_cases]",
    ):
        assert fragment in run.stdout
    assert "--synthetic-test-max-control-points" not in run.stdout
    assert "--init-checkpoint" not in run.stdout

    incompatible = subprocess.run(
        command + ["--pilot-kc56"], cwd=ROOT,
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert incompatible.returncode != 0
    assert "mutually exclusive" in incompatible.stderr


@pytest.mark.parametrize("profile", [
    None, "--pilot-kc56", "--keep-swap-highk72", "--keep-state-recovery",
])
@pytest.mark.parametrize("override", [None, "7.5e-5"])
def test_legacy_profiles_share_default_or_explicit_tolerance_across_all_phases(
    tmp_path, profile, override,
):
    options = [profile] if profile is not None else []
    phases = [
        "train_fresh", "inspect_checkpoint", "benchmark_six_methods_plus_verified",
        "visualize_six_method_real_cases",
    ]
    if profile == "--keep-state-recovery":
        reference = tmp_path / "recovery_reference.pt"
        reference.write_bytes(b"dry-run placeholder")
        options.extend(["--init-full-checkpoint", _shell_path(reference)])
        phases.append("paired_native_deployment")
    if override is not None:
        options.extend(["--mse-tolerance", override])
    result = subprocess.run(
        [
            _bash(), str(SCRIPT), "--dry-run", "--device", "cpu",
            "--output-root", _shell_path(tmp_path / "outputs"),
            "--run-name", "legacy_tolerance_test", *options,
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = float(override) if override is not None else 1e-4
    for phase in phases:
        command = _command(result.stdout, phase)
        assert command.count("--mse-tolerance") == 1
        assert float(command[command.index("--mse-tolerance") + 1]) == pytest.approx(expected)
    assert not (tmp_path / "outputs").exists()
