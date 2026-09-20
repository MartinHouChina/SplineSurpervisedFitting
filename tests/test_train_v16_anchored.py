"""Short geometry calibration is opt-in, correctly frozen, and resume-safe."""
from copy import deepcopy

import pytest
import torch

from test_train_v16_enhanced import native_payload, tiny_model, training_command, train_v16


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("flags", [
    ["--joint-geometry-calibration-epochs", "-1"],
    ["--epochs", "4", "--proposal-epochs", "2", "--joint-geometry-calibration-epochs", "3"],
    ["--subset-geometry-residual-scale", "nan"],
    ["--subset-geometry-residual-scale", "1.1"],
    ["--teacher-greedy-priority", "compact"],
    ["--teacher-compact-mask-weight", "-1"],
    ["--teacher-compact-mask-weight", ".5"],
])
def test_invalid_anchored_options_fail_before_training(flags):
    with pytest.raises(ValueError):
        train_v16.validate_args(train_v16.parser().parse_args(flags))


def test_anchor_migration_copies_all_weights_and_records_functional_change():
    source = tiny_model()
    target = tiny_model(subset_geometry_mode="anchored", subset_geometry_residual_scale=.15)
    metadata = {}
    copied = train_v16.transfer_all_weights(target, native_payload(source), transfer_metadata=metadata)
    assert len(copied) == len(source.state_dict())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    assert metadata["subset_geometry_transfer"]["changes"]["subset_geometry_mode"] == {
        "source": "legacy", "target": "anchored"}
    assert target.get_config()["subset_geometry_residual_scale"] == .15
    with pytest.raises(ValueError, match="configuration mismatch"):
        train_v16.transfer_all_weights(source, native_payload(target))


def test_calibration_freezes_proposal_without_optimizer_drift_and_unfreezes():
    args = train_v16.parser().parse_args([
        "--epochs", "5", "--proposal-epochs", "1", "--joint-geometry-calibration-epochs", "2",
    ])
    model = tiny_model(parameter_trust_enabled=True)
    optimizer = train_v16.build_optimizer(model, args, stage="joint")
    assert {group["group_name"] for group in optimizer.param_groups} == {"proposal", "selector", "decoder"}
    # Seed AdamW state and stale gradients, then ensure frozen proposal is exact.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.train()
    train_v16.set_proposal_trainability(model, frozen=True)
    train_v16.apply_joint_learning_rate(optimizer, args, epoch=2, end_epoch=5)
    assert not model.encoder.training and not model.parameter_head.training
    assert model.keep_head.training
    for name, parameter in model.named_parameters():
        if name.startswith(train_v16.PROPOSAL_PARAMETER_PREFIXES):
            assert not parameter.requires_grad and parameter.grad is None
        else:
            assert parameter.requires_grad
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith(train_v16.PROPOSAL_PARAMETER_PREFIXES):
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert not torch.equal(model.keep_head.weight, before["keep_head.weight"])
    train_v16.set_proposal_trainability(model, frozen=False)
    train_v16.apply_joint_learning_rate(optimizer, args, epoch=4, end_epoch=5)
    assert model.encoder.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(group["lr"] > 0 for group in optimizer.param_groups)


def test_new_geometry_options_cannot_silently_change_on_resume():
    args = train_v16.parser().parse_args([])
    train_v16.validate_args(args)
    original = train_v16.serial_args(args)
    for key, value in (("subset_geometry_mode", "anchored"),
                       ("subset_geometry_residual_scale", .15),
                       ("joint_geometry_calibration_epochs", 2),
                       ("teacher_greedy_priority", "compact"),
                       ("teacher_compact_mask_weight", .5)):
        changed = {**original, key: value}
        assert train_v16.training_config_changes(changed, original, set()) == [key]
        assert train_v16.training_config_changes(original, changed, set()) == [key]


def test_compact_target_metrics_use_actual_target_denominators():
    total = {}
    train_v16.accumulate_training_metrics(total, {
        "teacher_compact_target_count": 2, "teacher_compact_target_k_mean": 5.,
    }, 32)
    train_v16.accumulate_training_metrics(total, {
        "teacher_compact_target_count": 1, "teacher_compact_target_k_mean": 2.,
    }, 16)
    result = train_v16.finalize_training_metrics(total, 48)
    assert result["teacher_compact_target_count"] == 3
    assert result["teacher_compact_target_k_mean"] == 4.


def test_short_training_freezes_and_resumes_calibration_curriculum(tmp_path):
    output = tmp_path / "anchored.pt"
    command = training_command(output) + [
        "--epochs", "3", "--joint-geometry-calibration-epochs", "1",
        "--subset-geometry-mode", "anchored", "--subset-geometry-residual-scale", ".15",
        "--simplification-controller", "per_curve", "--feasible-objective",
        "--complexity-ramp-epochs", "1", "--safety-anneal-epochs", "1",
        "--teacher-greedy-steps", "1", "--teacher-greedy-max-curves", "1",
        "--teacher-greedy-priority", "compact", "--teacher-greedy-trajectory-checks", "2",
        "--teacher-geometry-trajectory-targets", "1",
        "--teacher-geometry-distillation-weight", ".4", "--teacher-compact-mask-weight", ".5",
    ]
    assert train_v16.main(command) == 0
    last = tmp_path / "anchored.last.pt"
    saved = torch.load(last, map_location="cpu", weights_only=True)
    assert saved["model_config"]["subset_geometry_mode"] == "anchored"
    assert saved["loss_config"]["teacher_greedy_priority"] == "compact"
    assert saved["loss_config"]["weights"]["teacher_compact_mask_weight"] == .5
    joint = [entry for entry in saved["history"] if entry["stage"] == "joint"]
    assert [entry["proposal_frozen"] for entry in joint] == [True, False]
    assert [entry["applied_complexity_scale"] for entry in joint] == [0., 0.]
    assert [entry["next_complexity_scale"] for entry in joint] == [0., 1.]
    assert joint[0]["learning_rates"]["proposal"] == 0
    assert joint[1]["learning_rates"]["proposal"] > 0
    proposal = torch.load(tmp_path / "anchored.proposal.pt", weights_only=True)
    # Epoch 2 is frozen; epoch 3 may move proposal. Saving/resuming must preserve schedule.
    assert proposal["epoch"] == 1
    resume = deepcopy(command) + ["--epochs", "4", "--resume", str(last)]
    assert train_v16.main(resume) == 0
    after = torch.load(last, map_location="cpu", weights_only=True)
    assert after["history"][:3] == saved["history"]
    assert after["history"][-1]["applied_complexity_scale"] == 1.
    changed = resume + ["--subset-geometry-residual-scale", ".2"]
    with pytest.raises(SystemExit):
        train_v16.main(changed)


def test_legacy_best_warmstart_runs_four_epoch_anchor_migration_and_reload(tmp_path):
    source = tmp_path / "legacy_best.pt"
    assert train_v16.main(training_command(source)) == 0
    source_bytes = source.read_bytes()
    output = tmp_path / "migrated.pt"
    command = training_command(output) + [
        "--epochs", "4", "--warm-start-checkpoint", str(source),
        "--subset-geometry-mode", "anchored", "--subset-geometry-residual-scale", ".25",
        "--joint-geometry-calibration-epochs", "2", "--joint-proposal-lr-scale", ".1",
        "--joint-decoder-lr-scale", "1", "--joint-final-lr-ratio", ".25",
    ]
    assert train_v16.main(command) == 0
    last = torch.load(tmp_path / "migrated.last.pt", map_location="cpu", weights_only=True)
    assert source.read_bytes() == source_bytes
    assert [entry["proposal_frozen"] for entry in last["history"]] == [False, True, True, False]
    assert last["initializer_provenance"]["subset_geometry_transfer"]["changes"]["subset_geometry_mode"]["target"] == "anchored"
    model, _, _ = train_v16.build_model_from_checkpoint(last)
    assert model.subset_geometry_mode == "anchored" and model.subset_geometry_residual_scale == .25
    model.eval()
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        result = model.forward_deployment(points)
    assert torch.isfinite(result["params"]).all() and torch.isfinite(result["internal_knots"]).all()
