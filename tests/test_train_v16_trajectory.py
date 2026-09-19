"""Reachable-teacher options must be explicit and checkpoint/resume safe."""
from copy import deepcopy

import pytest
import torch

from test_train_v16_enhanced import training_command, train_v16


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trajectory_flags():
    return [
        "--teacher-greedy-steps", "3", "--teacher-greedy-max-curves", "1",
        "--teacher-greedy-trajectory-checks", "3",
        "--teacher-geometry-trajectory-targets", "2",
        "--teacher-geometry-distillation-weight", ".2",
    ]


@pytest.mark.parametrize("flags", [
    ["--teacher-greedy-trajectory-checks", "-1"],
    ["--teacher-geometry-trajectory-targets", "-1"],
    ["--teacher-greedy-trajectory-checks", "3"],
    ["--teacher-greedy-steps", "3", "--teacher-geometry-trajectory-targets", "2"],
    ["--teacher-greedy-steps", "3", "--teacher-greedy-trajectory-checks", "3",
     "--teacher-geometry-trajectory-targets", "2"],
])
def test_incomplete_trajectory_configuration_is_rejected(flags):
    with pytest.raises(ValueError):
        train_v16.validate_args(train_v16.parser().parse_args(flags))


def test_trajectory_defaults_preserve_legacy_serialized_training_contract():
    args = train_v16.parser().parse_args([])
    train_v16.validate_args(args)
    old = train_v16.serial_args(args)
    keys = {"teacher_greedy_trajectory_checks", "teacher_geometry_trajectory_targets"}
    assert not keys.intersection(old)
    enhanced = train_v16.parser().parse_args(trajectory_flags())
    train_v16.validate_args(enhanced)
    current = train_v16.serial_args(enhanced)
    assert keys.issubset(train_v16.training_config_changes(current, old, set()))


def test_trajectory_train_and_resume_store_identical_loss_options(tmp_path):
    output = tmp_path / "trajectory.pt"
    args = training_command(output) + trajectory_flags() + [
        "--teacher-prefix-search-steps", "1", "--allow-infeasible-proposals",
    ]
    assert train_v16.main(args) == 0
    last_path = output.with_name("trajectory.last.pt")
    saved = torch.load(last_path, map_location="cpu", weights_only=True)
    for section in ("training_config", "loss_config"):
        assert saved[section]["teacher_greedy_trajectory_checks"] == 3
        assert saved[section]["teacher_geometry_trajectory_targets"] == 2
    assert saved["deployment_config"]["network_forwards"] == 1
    assert saved["deployment_config"]["final_refits"] == 1
    resume = deepcopy(args)
    resume[resume.index("--epochs") + 1] = "3"
    resume.extend(["--resume", str(last_path)])
    assert train_v16.main(resume) == 0
    after = torch.load(last_path, map_location="cpu", weights_only=True)
    assert after["history"][:2] == saved["history"]
    assert after["loss_config"] == saved["loss_config"]
    changed = deepcopy(resume)
    changed[changed.index("--teacher-geometry-trajectory-targets") + 1] = "1"
    with pytest.raises(SystemExit):
        train_v16.main(changed)


@pytest.mark.parametrize("reverse", [False, True])
def test_trajectory_epoch_rates_use_actual_search_and_target_denominators(reverse):
    batches = [
        (32, dict(teacher_greedy_trajectory_checks=8., teacher_greedy_trajectory_pass_rate=.25,
                  teacher_geometry_target_count=4., teacher_geometry_target_k_mean=10.,
                  teacher_geometry_target_pass_rate=.5, teacher_greedy_trajectory_accepted_delta_k=1.)),
        (3, dict(teacher_greedy_trajectory_checks=2., teacher_greedy_trajectory_pass_rate=1.,
                 teacher_geometry_target_count=1., teacher_geometry_target_k_mean=5.,
                 teacher_geometry_target_pass_rate=1., teacher_greedy_trajectory_accepted_delta_k=2.)),
    ]
    total = {}
    for size, metrics in reversed(batches) if reverse else batches:
        train_v16.accumulate_training_metrics(total, metrics, size)
    result = train_v16.finalize_training_metrics(total, 35)
    assert result["teacher_greedy_trajectory_checks"] == 10
    assert result["teacher_greedy_trajectory_pass_rate"] == pytest.approx(.4)
    assert result["teacher_geometry_target_count"] == 5
    assert result["teacher_geometry_target_k_mean"] == pytest.approx(9.)
    assert result["teacher_geometry_target_pass_rate"] == pytest.approx(.6)
    assert result["teacher_greedy_trajectory_accepted_delta_k"] == pytest.approx(38/35)


def test_no_search_epoch_reports_finite_zero_rates():
    total = {}
    metrics = {key: 0. for key in (
        "teacher_greedy_trajectory_checks", "teacher_greedy_trajectory_pass_rate",
        "teacher_geometry_target_count", "teacher_geometry_target_k_mean",
        "teacher_geometry_target_pass_rate",
    )}
    train_v16.accumulate_training_metrics(total, metrics, 8)
    assert train_v16.finalize_training_metrics(total, 8) == metrics
