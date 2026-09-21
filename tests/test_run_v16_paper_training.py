from pathlib import Path, PureWindowsPath
import shlex
import subprocess
import sys
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v16_paper_training as entry


def options(*flags):
    return entry.parser().parse_args(["--run-prefix", "paper_r1", *flags])


def argument(command, key):
    return command[command.index(key) + 1]


def test_formal_plan_keeps_two_independent_capacities_and_truthful_native_policy():
    plan = entry.build_plan(options())
    assert plan["run_count"] == 2
    for run in plan["runs"]:
        command = run["command"]
        assert argument(command, "--candidate-knots") == str(run["capacity"])
        assert f"m{run['capacity']}.pt" in argument(command, "--warm-start-checkpoint")
        assert argument(command, "--epochs") == "60"
        assert argument(command, "--proposal-epochs") == "12"
        assert argument(command, "--baseline-protocol") == "native"
        assert "--paper-output" in command and "--native-baselines" in command
        assert "--resize-candidate-warm-start" not in command
        assert argument(command, "--coupled-proposal-steps") == "2"
        assert argument(command, "--coupled-subset-steps") == "2"
        assert argument(command, "--joint-decoder-lr-scale") == "0.25"
        assert argument(command, "--real-samples-per-dataset") == "100"


def test_all_real_records_is_explicit_and_dry_run_does_not_read_weights(tmp_path):
    args = options("--capacities", "16", "--all-real-test-samples")
    plan = entry.build_plan(args)
    assert len(plan["runs"]) == 1
    assert "--all-real-test-samples" in plan["runs"][0]["command"]
    assert entry.main(["--run-prefix", "dry_coupled", "--m16-checkpoint", str(tmp_path / "absent"),
                       "--dry-run"]) == 0


@pytest.mark.parametrize("flags", [["--epochs", "12"], ["--parameter-chord-blend", "nan"],
                                  ["--capacities", "16", "16"], ["--batch-size", "0"],
                                  ["--coupled-subset-steps", "-1"]])
def test_invalid_plans_rejected(flags):
    with pytest.raises(ValueError):
        entry.build_plan(options(*flags))


def _git_bash_argument(value):
    """Bridge Windows host paths for a Linux-shell dry-run, not runner behavior."""
    path = PureWindowsPath(value)
    if len(path.drive) == 2 and path.drive[1] == ":":
        return f"/{path.drive[0].lower()}/" + "/".join(path.parts[1:])
    return value


@pytest.mark.parametrize("path_style", ["raw_windows", "linux_style"])
@pytest.mark.parametrize("capacity,protocol,proposal_steps,subset_steps,blend,all_real", [
    (16, "native", 2, 2, 0., False),
    (32, "native", 2, 2, 0., True),
    (32, "adaptation", 3, 1, .5, False),
])
def test_generated_plan_real_git_bash_expands_valid_final_clis(
    tmp_path, capacity, protocol, proposal_steps, subset_steps, blend, all_real, path_style,
):
    """Exercise Bash arrays/default replacement, then parse real downstream CLIs.

    Only the shell --dry-run runs. Empty sentinel checkpoints prove Python
    training/loading is not executed. Both the raw Windows plan and explicitly
    POSIX-style paths run through the actual Git Bash script.
    """
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if sys.platform != "win32" or not bash.is_file():
        pytest.skip("Windows Git Bash integration test")

    import train_v16 as trainer
    import benchmark_v15_datasets as benchmark
    import visualize_v16_six_methods as visualizer

    initializers = {}
    for size in (16, 32):
        path = tmp_path / f"sentinel checkpoint m{size}.pt"
        path.write_bytes(b"not a torch checkpoint: must never be loaded in dry-run")
        initializers[size] = path
    prefix = "paper_shell_test_" + uuid.uuid4().hex
    flags = ["--run-prefix", prefix, "--capacities", str(capacity),
             "--m16-checkpoint", str(initializers[16]),
             "--m32-checkpoint", str(initializers[32]),
             "--data-root", str(tmp_path / "unread data root"),
             "--bash", str(bash), "--baseline-protocol", protocol,
             "--coupled-proposal-steps", str(proposal_steps),
             "--coupled-subset-steps", str(subset_steps),
             "--parameter-chord-blend", str(blend)]
    if all_real:
        flags.append("--all-real-test-samples")
    plan = entry.build_plan(entry.parser().parse_args(flags))
    run = plan["runs"][0]
    command = run["command"] + ["--dry-run"]
    if path_style == "linux_style":
        command = [str(bash), *map(_git_bash_argument, command[1:])]
    outcome = subprocess.run(command, cwd=entry.ROOT, capture_output=True, text=True,
                             encoding="utf-8", timeout=30, check=False)
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr

    def phase(name):
        marker = f"[{name}]"
        lines = [line for line in outcome.stdout.splitlines() if line.startswith(marker)]
        assert len(lines) == 1, outcome.stdout
        return shlex.split(lines[0][len(marker):])

    train_command = phase("train_fresh")
    assert train_command[1] == "scripts/train_v16.py"
    parsed = trainer.parser().parse_args(train_command[2:])
    assert parsed.candidate_knots == capacity
    assert parsed.max_control_points == min(capacity, 24) + 4
    assert parsed.epochs == 60 and parsed.proposal_epochs == 12
    assert parsed.joint_geometry_calibration_epochs == 8
    assert parsed.train_size == 3000 and parsed.real_val_size == 100
    assert parsed.lr == pytest.approx(2e-5)
    assert parsed.joint_lr == pytest.approx(1e-5)
    assert parsed.joint_proposal_lr_scale == pytest.approx(.1)
    assert parsed.joint_decoder_lr_scale == pytest.approx(.25)
    assert parsed.coupled_proposal_steps == proposal_steps
    assert parsed.coupled_subset_steps == subset_steps
    assert parsed.parameter_chord_blend == pytest.approx(blend)
    assert parsed.proposal_refinement_layers == 2
    assert parsed.selection_refinement_layers == 2
    assert parsed.survivor_refinement_layers == 2
    assert parsed.max_point_error_weight == pytest.approx(.05)
    assert parsed.max_point_error_tolerance == pytest.approx(5e-4)
    assert parsed.mse_tolerance == pytest.approx(5e-5)
    assert parsed.real_fraction == 0 and len(parsed.real_manifest) == 3
    assert parsed.warm_start_checkpoint.name == initializers[capacity].name
    assert parsed.output.name == run["run_name"] + ".pt"
    for key in ("--joint-lr", "--joint-proposal-lr-scale", "--joint-decoder-lr-scale",
                "--coupled-proposal-steps", "--coupled-subset-steps", "--parameter-chord-blend"):
        assert train_command.count(key) == 1

    benchmark_command = phase("benchmark_six_methods")
    measured = benchmark.parser().parse_args(benchmark_command[2:])
    assert measured.baseline_protocol == protocol
    assert measured.published_feasibility_safeguard is False
    assert measured.all_real_test_samples is all_real
    assert measured.samples_per_knot_count == 10
    assert measured.real_samples_per_dataset == 100
    assert measured.max_knot_count == 24 and measured.synthetic_source_max_knots == 24
    assert measured.max_internal_knots == capacity
    assert measured.paper_initial_knots == capacity and measured.liang_dense_knots == capacity
    assert measured.checkpoint == parsed.output

    visual_command = phase("plot_saved_six_method_cases")
    visual = visualizer.parser().parse_args(visual_command[2:])
    assert visual.max_cases_per_dataset == 12
    assert visual.benchmark_dir == measured.output_dir
    assert "[plot_ours_cases]" not in outcome.stdout
    assert "[plot_six_method_real_cases]" not in outcome.stdout
    for kind in ("logs", "comparisons", "figures"):
        assert not (entry.ROOT / "outputs" / kind / run["run_name"]).exists()
    assert not (entry.ROOT / "outputs/checkpoints" / (run["run_name"] + ".pt")).exists()
