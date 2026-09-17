from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v16_mse1e-4_3090.sh"


def _bash():
    candidate = (
        Path(r"C:\Program Files\Git\bin\bash.exe")
        if os.name == "nt" else shutil.which("bash")
    )
    if not candidate or not Path(candidate).is_file():
        pytest.skip("Bash is unavailable")
    return str(candidate)


def _shell_path(path):
    path = Path(path).resolve().as_posix()
    if os.name == "nt":
        return "/" + path[0].lower() + path[2:]
    return path


def _run(tmp_path, *options):
    return subprocess.run(
        [
            _bash(), str(SCRIPT), "--dry-run", "--device", "cpu",
            "--output-root", _shell_path(tmp_path / "outputs"),
            "--run-name", "keep_state_test", *options,
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=20, check=False,
    )


def _command(output, phase):
    prefix = f"[{phase}] "
    line = next(line for line in output.splitlines() if line.startswith(prefix))
    return shlex.split(line[len(prefix):])


def _parse_training(command):
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_v16

    args = train_v16.parser().parse_args(command[2:])
    train_v16.validate_args(args)
    return args


def test_recovery_dry_run_uses_complete_warmstart_and_real_trainer_contract(tmp_path):
    source = tmp_path / "r2_reference.pt"
    source.write_bytes(b"dry-run placeholder")
    result = _run(
        tmp_path, "--keep-state-recovery", "--benchmark-profile", "full",
        "--init-full-checkpoint", _shell_path(source),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KEEP_STATE_RECOVERY profile" in result.stdout
    assert "Benchmark profile=quick" in result.stdout
    assert "Mixed-source warm-start provenance remains diagnostic" in result.stdout
    assert not (tmp_path / "outputs").exists()
    args = _parse_training(_command(result.stdout, "train_fresh"))
    assert args.init_full_checkpoint.as_posix() == _shell_path(source)
    assert args.init_checkpoint is None and args.resume is None
    assert (args.epochs, args.proposal_epochs, args.selector_warmup_epochs) == (32, 8, 4)
    assert (args.train_size, args.val_size, args.real_val_size, args.batch_size) == (600, 160, 20, 32)
    assert (args.candidate_knots, args.min_control_points, args.max_control_points) == (56, 8, 60)
    assert args.keep_state_interaction and args.count_structure_coupling
    assert args.proposal_global_warp_limit == pytest.approx(1.0)
    assert args.keep_boundary_ranking_weight == pytest.approx(1.0)
    assert args.lr == args.selector_lr == pytest.approx(5e-5)
    assert args.decoder_joint_lr == pytest.approx(1e-5)
    assert args.proposal_joint_lr == args.parameter_joint_lr == 0
    assert args.synthetic_boundary_val_size == args.synthetic_high_k_val_size == 32
    assert args.joint_high_k_fraction == pytest.approx(0.4)
    assert args.joint_high_k_min_knots == args.synthetic_high_k_val_min_knots == 45
    assert args.joint_supervision == "offline_feasible_teacher"
    assert args.feasible_teacher_strategy == "synthetic_anchor_counterfactual"
    assert args.feasible_teacher_counterfactual_max_probes == 8
    assert not args.resample_train_each_epoch
    assert args.one_shot_safety_sigma == args.final_safety_sigma == pytest.approx(0.03)
    assert args.one_shot_safety_knots == args.final_safety_knots == 0
    assert args.safety_anneal_epochs == 1
    assert args.complexity_ramp_epochs == 0
    assert args.complexity_weight == 0

    benchmark = _command(result.stdout, "benchmark_six_methods_plus_verified")
    assert "--force-diagnostic" in benchmark
    assert "--include-verified-ours" in benchmark
    assert benchmark[benchmark.index("--method-set") + 1] == "published"
    _command(result.stdout, "plot_four_metrics")
    _command(result.stdout, "visualize_six_method_real_cases")
    paired = _command(result.stdout, "paired_native_deployment")
    assert paired[1] == "scripts/quick_compare_v16_checkpoints.py"
    assert paired[paired.index("--old-checkpoint") + 1] == _shell_path(source)
    assert paired[paired.index("--min-source-k") + 1] == "4"
    assert paired[paired.index("--max-source-k") + 1] == "56"
    assert paired[paired.index("--samples-per-k") + 1] == "1"
    assert paired[paired.index("--real-samples-per-dataset") + 1] == "5"
    assert paired.count("--manifest") == 3
    assert "--include-verified-ours" not in paired
    assert "--repair" not in paired


def test_recovery_default_reference_is_full_r2_and_bash_syntax_is_valid():
    source = SCRIPT.read_text(encoding="utf-8")
    assert (
        'INIT_FULL_CHECKPOINT="outputs/checkpoints/'
        'candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.pt"'
    ) in source
    result = subprocess.run(
        [_bash(), "-n", str(SCRIPT)], cwd=ROOT, capture_output=True,
        text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_recovery_is_opt_in_and_formal_profile_defaults_are_unchanged(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    args = _parse_training(_command(result.stdout, "train_fresh"))
    assert (args.epochs, args.proposal_epochs, args.candidate_knots) == (128, 64, 72)
    assert args.lr == args.selector_lr == pytest.approx(2e-4)
    assert args.decoder_joint_lr == pytest.approx(5e-5)
    assert not args.keep_state_interaction
    assert args.proposal_global_warp_limit == 0
    assert args.init_full_checkpoint is None
    assert args.one_shot_safety_sigma == args.final_safety_sigma == 0
    assert args.safety_anneal_epochs == args.complexity_ramp_epochs == 12
    assert "[paired_native_deployment]" not in result.stdout


@pytest.mark.parametrize("conflict", ["--pilot-kc56", "--keep-swap-highk72"])
def test_recovery_rejects_conflicting_profiles(tmp_path, conflict):
    result = _run(tmp_path, "--keep-state-recovery", conflict)
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_recovery_rejects_competing_initializers_and_explicit_initializer_with_resume(tmp_path):
    result = _run(tmp_path, "--keep-state-recovery", "--init-checkpoint", "proposal.pt")
    assert result.returncode != 0 and "mutually exclusive" in result.stderr
    result = _run(
        tmp_path, "--keep-state-recovery", "--resume-run",
        "--init-full-checkpoint", "source.pt",
    )
    assert result.returncode != 0
    assert "--resume-run cannot be combined with --init-full-checkpoint" in result.stderr


def test_recovery_requires_initializer_instead_of_silently_training_from_scratch(tmp_path):
    result = _run(
        tmp_path, "--keep-state-recovery", "--init-full-checkpoint",
        _shell_path(tmp_path / "missing.pt"),
    )
    assert result.returncode != 0
    assert "full-model initializer is missing" in result.stderr


def test_recovery_resume_uses_last_checkpoint_without_reinitializing(tmp_path):
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for suffix in (".last.pt", ".proposal.pt"):
        (checkpoint_dir / f"keep_state_test{suffix}").write_bytes(b"dry-run placeholder")
    reference = tmp_path / "original_reference.pt"
    reference.write_bytes(b"dry-run placeholder")
    result = _run(
        tmp_path, "--keep-state-recovery", "--resume-run",
        "--paired-reference-checkpoint", _shell_path(reference),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    command = _command(result.stdout, "train_resume")
    assert "--resume" in command
    assert "--init-full-checkpoint" not in command
    assert "--init-checkpoint" not in command
    paired = _command(result.stdout, "paired_native_deployment")
    assert paired[paired.index("--old-checkpoint") + 1] == _shell_path(reference)
