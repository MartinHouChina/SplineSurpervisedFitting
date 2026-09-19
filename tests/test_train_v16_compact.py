"""Compact training must unlock feasible simplification without changing validation."""
from copy import deepcopy
import json
import math

import pytest
import torch

from test_train_v16 import real_manifest  # noqa: F401
from test_train_v16_enhanced import training_command, train_v16


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def compact_flags():
    return [
        "--simplification-controller", "per_curve", "--feasible-objective",
        "--feasible-fit-margin", ".8", "--feasible-fit-weight", ".02",
        "--teacher-greedy-steps", "2", "--teacher-greedy-max-curves", "1",
        "--teacher-geometry-distillation-weight", ".2", "--count-reserve-alignment",
        "--synthetic-simple-fraction", ".35", "--synthetic-shape-fraction", ".25",
        "--one-shot-safety-knots", "0", "--one-shot-safety-sigma", "0",
        "--final-safety-knots", "0", "--final-safety-sigma", "0",
        "--parameter-trust-enabled", "--complexity-max-scale", "1",
        "--complexity-ramp-epochs", "2", "--safety-anneal-epochs", "2",
    ]


def test_per_curve_controller_progresses_even_if_unseen_validation_source_fails():
    args = train_v16.parser().parse_args(compact_flags())
    train_v16.validate_args(args)
    scale, safety = 0., 1.
    for index in range(4):
        scale, safety = train_v16.update_simplification_controller(
            args, pass_rate=0., complexity_scale=scale, safety_scale=safety,
        )
        assert scale == pytest.approx(min(1., (index+1)*.5))
        assert safety == pytest.approx(max(0., 1-(index+1)*.5))
    assert train_v16.selection_safety(args, safety) == (0., 0)
    args.simplification_controller = "worst_source"
    assert train_v16.update_simplification_controller(
        args, pass_rate=0., complexity_scale=0., safety_scale=1.,
    ) == (0., 1.)


@pytest.mark.parametrize("flags", [
    ["--simplification-controller", "per_curve"],
    ["--feasible-fit-margin", "0"], ["--feasible-fit-margin", "nan"],
    ["--feasible-fit-weight", "1.1"], ["--teacher-greedy-steps", "-1"],
    ["--teacher-greedy-max-curves", "0"],
    ["--teacher-geometry-distillation-weight", ".2"],
    ["--synthetic-simple-fraction", ".8", "--synthetic-shape-fraction", ".3"],
    ["--synthetic-shape-fraction", "nan"],
])
def test_invalid_compact_options_fail_before_training(flags):
    with pytest.raises(ValueError):
        train_v16.validate_args(train_v16.parser().parse_args(flags))


def test_compact_options_are_resume_immutable_but_legacy_defaults_are_omitted():
    old = train_v16.parser().parse_args([])
    train_v16.validate_args(old)
    legacy = train_v16.serial_args(old)
    new = train_v16.parser().parse_args(compact_flags())
    train_v16.validate_args(new)
    current = train_v16.serial_args(new)
    assert not set(legacy) & set(train_v16.ENHANCED_TRAINING_DEFAULTS)
    changes = train_v16.training_config_changes(current, legacy, set())
    assert "simplification_controller" in changes
    assert "synthetic_shape_fraction" in changes
    assert "teacher_greedy_steps" in changes


def test_compact_tiny_training_resume_and_real_validation_isolation(
    tmp_path, real_manifest, monkeypatch,
):
    manifest, records = real_manifest
    output = tmp_path / "compact.pt"
    command = training_command(output) + compact_flags() + [
        "--teacher-prefix-search-steps", "1", "--allow-infeasible-proposals",
        "--real-manifest", str(manifest), "--real-fraction", "0", "--real-val-size", "2",
    ]
    command[command.index("--epochs")+1] = "3"
    command[command.index("--no-certified-minimal-source")] = "--certified-minimal-source"
    command.extend(["--minimality-audit-points", "64"])
    seen_families = []
    original = train_v16.MixedTrainingCurves.__getitem__

    def tracked(self, index):
        item = original(self, index)
        seen_families.append(item["synthetic_family"])
        return item

    monkeypatch.setattr(train_v16.MixedTrainingCurves, "__getitem__", tracked)
    assert train_v16.main(command) == 0
    last_path = tmp_path / "compact.last.pt"
    saved = torch.load(last_path, map_location="cpu", weights_only=True)
    assert saved["real_data_role"] == "validation_only"
    assert saved["synthetic_training_mixture"]["shape_geometry_targets_valid"] is False
    assert saved["training_config"]["simplification_controller"] == "per_curve"
    assert saved["loss_config"]["feasible_objective"] is True
    assert saved["loss_config"]["count_reserve_alignment"] is True
    assert saved["loss_config"]["teacher_greedy_steps"] == 2
    assert saved["loss_config"]["weights"]["teacher_geometry_distillation_weight"] == .2
    joint = [h for h in saved["history"] if h["stage"] == "joint"]
    assert [h["applied_complexity_scale"] for h in joint] == [0., .5]
    assert all(h["applied_selection_safety_knots"] == 0 for h in joint)
    assert all(h["applied_selection_safety_sigma"] == 0 for h in joint)
    assert all(sum(h["training_family_counts"].values()) == 4 for h in saved["history"])
    assert seen_families
    for entry in saved["history"]:
        assert all(math.isfinite(x) for x in entry["train"].values())
        assert set(entry["validation"]["by_source"]) == {"Synthetic", "TestReal"}
    by_id = {record["sample_id"]: record for record in records}
    assert all(by_id[sample]["split"] == "val"
               for sample in saved["validation_real_ids"]["TestReal"])
    resume = deepcopy(command)
    resume[resume.index("--epochs")+1] = "4"
    resume.extend(["--resume", str(last_path)])
    assert train_v16.main(resume) == 0
    after = torch.load(last_path, map_location="cpu", weights_only=True)
    assert after["history"][:3] == saved["history"]
    assert after["loss_config"] == saved["loss_config"]
    assert after["validation_real_ids"] == saved["validation_real_ids"]
    assert after["history"][-1]["applied_complexity_scale"] == 1.
    assert json.loads(output.with_suffix(".history.json").read_text()) == after["history"]
    changed = deepcopy(resume)
    changed[changed.index("--synthetic-shape-fraction")+1] = ".2"
    with pytest.raises(SystemExit):
        train_v16.main(changed)
