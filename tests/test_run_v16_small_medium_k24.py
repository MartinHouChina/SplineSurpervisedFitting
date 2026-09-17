from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_v16_mse1e-4_3090.sh"
WRAPPER = ROOT / "scripts/run_v16_small_medium_k24_3090.sh"


def _bash():
    path = Path(r"C:\Program Files\Git\bin\bash.exe") if os.name == "nt" else shutil.which("bash")
    if not path or not Path(path).is_file():
        pytest.skip("Bash is unavailable")
    return str(path)


def _shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


def _run(tmp_path, *options, wrapper=True, run_name="small_medium_test"):
    run_name_args = ["--run-name", run_name] if run_name is not None else []
    return subprocess.run(
        [
            _bash(), str(WRAPPER if wrapper else SCRIPT),
            "--dry-run", "--device", "cpu", *run_name_args,
            "--output-root", _shell_path(tmp_path / "outputs"), *options,
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=20, check=False,
    )


def _command(output, phase):
    prefix = f"[{phase}] "
    line = next(line for line in output.splitlines() if line.startswith(prefix))
    return shlex.split(line[len(prefix):])


def _training(command):
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_v16

    args = train_v16.parser().parse_args(command[2:])
    train_v16.validate_args(args)
    return args


def _assert_mse_tolerance(output, expected):
    for phase in (
        "train_fresh", "inspect_checkpoint", "benchmark_six_methods_plus_verified",
        "visualize_six_method_real_cases",
    ):
        command = _command(output, phase)
        assert command.count("--mse-tolerance") == 1
        assert float(command[command.index("--mse-tolerance") + 1]) == pytest.approx(expected)


def test_k24_wrapper_dry_run_passes_actual_training_parser_and_preserves_scope(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMALL_MEDIUM_K24 profile" in result.stdout
    assert "MSE=5e-5" in result.stdout
    assert "Proposal 64 + Joint 64 = 128 total epochs (Joint includes 4 warmup epochs)" in result.stdout
    assert "full cubic knot vector=32" in result.stdout
    assert "does not establish that Keep selection has been fixed" in result.stdout
    assert not (tmp_path / "outputs").exists()
    args = _training(_command(result.stdout, "train_fresh"))
    assert args.study_scope == "small_medium_k24"
    assert (args.min_control_points, args.max_control_points, args.candidate_knots) == (8, 24, 24)
    assert (args.epochs, args.proposal_epochs, args.selector_warmup_epochs) == (128, 64, 4)
    assert args.epochs - args.proposal_epochs == 64
    assert (args.train_size, args.val_size, args.real_val_size, args.batch_size) == (600, 160, 20, 32)
    assert args.init_checkpoint is None and args.init_full_checkpoint is None and args.resume is None
    assert args.proposal_high_k_fraction == args.joint_high_k_fraction == 0
    assert args.synthetic_high_k_val_size == 0
    assert args.synthetic_boundary_val_size == 16
    assert args.max_control_points - 4 == 20
    assert args.mse_tolerance == pytest.approx(5e-5)
    _assert_mse_tolerance(result.stdout, 5e-5)
    assert args.lr == pytest.approx(2e-4)
    assert args.selector_lr == pytest.approx(5e-5)
    assert args.decoder_joint_lr == pytest.approx(1e-5)
    assert args.proposal_joint_lr == args.parameter_joint_lr == 0
    assert args.initial_keep_fraction == pytest.approx(0.5)
    assert args.relocation_blend == pytest.approx(0.03)
    assert args.keep_state_interaction and args.count_structure_coupling
    assert args.proposal_global_warp_limit == pytest.approx(1.0)
    assert args.keep_boundary_ranking_weight == pytest.approx(1.0)
    assert args.joint_supervision == "offline_feasible_teacher"
    assert args.feasible_teacher_strategy == "synthetic_anchor_counterfactual"
    assert args.feasible_teacher_counterfactual_max_probes == 8
    assert not args.resample_train_each_epoch
    assert args.one_shot_safety_sigma == args.final_safety_sigma == pytest.approx(0.03)
    assert args.one_shot_safety_knots == args.final_safety_knots == 0
    assert args.safety_anneal_epochs == 1
    assert args.complexity_ramp_epochs == args.complexity_weight == 0
    assert "[paired_native_deployment]" not in result.stdout

    benchmark = _command(result.stdout, "benchmark_six_methods_plus_verified")
    assert benchmark[benchmark.index("--min-knot-count") + 1] == "4"
    assert benchmark[benchmark.index("--max-knot-count") + 1] == "20"
    assert "--synthetic-test-max-control-points" not in benchmark
    for option in ("--max-internal-knots", "--paper-initial-knots", "--liang-dense-knots"):
        assert benchmark[benchmark.index(option) + 1] == "24"
    assert "--force-diagnostic" in benchmark
    _command(result.stdout, "plot_four_metrics")
    cases = _command(result.stdout, "visualize_six_method_real_cases")
    assert cases[cases.index("--max-internal-knots") + 1] == "24"
    assert "--force-diagnostic" in cases


def test_k24_default_run_name_separates_new_tolerance_and_schedule_artifacts(tmp_path):
    result = _run(tmp_path, run_name=None)
    assert result.returncode == 0, result.stdout + result.stderr
    run_name = "candidate_selection_v16_mse5e-5_small_medium_sourcek20_kc24_p64_j64_linux_r1"
    assert f"/checkpoints/{run_name}.pt" in result.stdout
    assert f"/teachers/{run_name}" in result.stdout
    assert "candidate_selection_v16_mse1e-4_small_medium" not in result.stdout
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("tolerance", ["2.5e-5", "1e-4", "0.000075"])
@pytest.mark.parametrize("profile_first", [False, True])
def test_k24_explicit_mse_tolerance_overrides_profile_in_either_argument_order(
    tmp_path, tolerance, profile_first,
):
    options = ["--mse-tolerance", tolerance]
    options.insert(0 if profile_first else len(options), "--small-medium-k24")
    result = _run(tmp_path, *options, wrapper=False)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_mse_tolerance(result.stdout, float(tolerance))
    args = _training(_command(result.stdout, "train_fresh"))
    assert args.mse_tolerance == pytest.approx(float(tolerance))
    assert (args.epochs, args.proposal_epochs, args.selector_warmup_epochs) == (128, 64, 4)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("tolerance", [
    "0", "0.0", "-1e-4", "nan", "NaN", "inf", "Infinity", "1e9999", "1e-9999",
    "not-a-number", "1e-4junk",
])
def test_k24_rejects_invalid_mse_tolerance_before_creating_artifacts(tmp_path, tolerance):
    result = _run(tmp_path, "--mse-tolerance", tolerance)
    assert result.returncode != 0
    assert "MSE_TOLERANCE" in result.stderr
    assert "[train_fresh]" not in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_k24_explicit_schedule_still_overrides_profile_defaults(tmp_path):
    result = _run(tmp_path, "--epochs", "80", "--proposal-epochs", "32", "--selector-warmup-epochs", "6")
    assert result.returncode == 0, result.stdout + result.stderr
    args = _training(_command(result.stdout, "train_fresh"))
    assert (args.epochs, args.proposal_epochs, args.selector_warmup_epochs) == (80, 32, 6)
    _assert_mse_tolerance(result.stdout, 5e-5)


def test_k24_profile_allows_larger_full_benchmark_but_never_formal(tmp_path):
    result = _run(tmp_path, "--benchmark-profile", "full")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Benchmark profile=full" in result.stdout
    benchmark = _command(result.stdout, "benchmark_six_methods_plus_verified")
    assert benchmark[benchmark.index("--samples-per-knot-count") + 1] == "5"
    assert benchmark[benchmark.index("--max-knot-count") + 1] == "20"
    assert "--force-diagnostic" in benchmark
    assert "--allow-unqualified-diagnostic" in benchmark
    assert "diagnostic_000000000000" in result.stdout
    _assert_mse_tolerance(result.stdout, 5e-5)


@pytest.mark.parametrize("option", ["--init-checkpoint", "--init-full-checkpoint"])
def test_k24_rejects_external_initializers_before_attempting_load(tmp_path, option):
    result = _run(tmp_path, option, "old_k56.pt")
    assert result.returncode != 0
    assert "--small-medium-k24 rejects" in result.stderr
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("option,value", [
    ("--proposal-high-k-fraction", "0.1"),
    ("--joint-high-k-fraction", "0.4"),
    ("--synthetic-high-k-val-size", "16"),
])
def test_k24_rejects_reintroduction_of_high_k_training_or_validation(tmp_path, option, value):
    result = _run(tmp_path, option, value)
    assert result.returncode != 0
    assert f"--small-medium-k24 requires {option} 0" in result.stderr


@pytest.mark.parametrize("other", ["--pilot-kc56", "--keep-swap-highk72", "--keep-state-recovery"])
def test_k24_rejects_conflicting_profiles(tmp_path, other):
    result = _run(tmp_path, other)
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_k24_resume_keeps_this_run_and_zero_high_k_overrides(tmp_path):
    checkpoint_dir = tmp_path / "outputs/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for suffix in (".last.pt", ".proposal.pt"):
        (checkpoint_dir / f"small_medium_test{suffix}").write_bytes(b"dry-run placeholder")
    result = _run(
        tmp_path, "--resume-run", "--proposal-high-k-fraction", "0.0",
        "--joint-high-k-fraction", "0", "--synthetic-high-k-val-size", "0",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = _training(_command(result.stdout, "train_resume"))
    assert args.resume.as_posix().endswith("small_medium_test.last.pt")
    assert args.study_scope == "small_medium_k24"
    assert args.init_checkpoint is None and args.init_full_checkpoint is None
    assert args.proposal_high_k_fraction == args.joint_high_k_fraction == 0
    assert (args.epochs, args.proposal_epochs, args.selector_warmup_epochs) == (128, 64, 4)
    assert args.mse_tolerance == pytest.approx(5e-5)


def test_wrapper_is_thin_and_default_legacy_scan_remains_k56(tmp_path):
    wrapper_source = WRAPPER.read_text(encoding="utf-8")
    assert 'exec bash "$SCRIPT_DIR/run_v16_mse1e-4_3090.sh" --small-medium-k24 "$@"' in wrapper_source
    syntax = subprocess.run([_bash(), "-n", str(WRAPPER)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
    result = _run(tmp_path, wrapper=False)
    assert result.returncode == 0, result.stdout + result.stderr
    args = _training(_command(result.stdout, "train_fresh"))
    assert args.study_scope == "legacy"
    assert (args.max_control_points, args.candidate_knots) == (60, 72)
    assert (args.epochs, args.proposal_epochs, args.selector_warmup_epochs) == (128, 64, 8)
    assert args.mse_tolerance == pytest.approx(1e-4)
    _assert_mse_tolerance(result.stdout, 1e-4)
    assert args.initial_keep_fraction == pytest.approx(30 / 72)
    benchmark = _command(result.stdout, "benchmark_six_methods_plus_verified")
    assert benchmark[benchmark.index("--max-knot-count") + 1] == "56"
