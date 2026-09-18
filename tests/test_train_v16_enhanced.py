"""Opt-in training enhancements must not alter legacy checkpoint/resume contracts."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_v16  # noqa: E402
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork  # noqa: E402


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(**changes):
    options = dict(point_dim=2, hidden_dim=16, encoder_layers=1,
                   max_internal_knots=8, attention_heads=4, selector_layers=1,
                   one_shot_selection_policy="mass_topk",
                   one_shot_adaptive_threshold=True)
    options.update(changes)
    return V16CandidateSelectionNetwork(**options)


def native_payload(model):
    return dict(
        objective_version=train_v16.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        architecture_revision=train_v16.V16_ADAPTIVE_SELECTION_REVISION,
        simplification_contract=train_v16.V16_SIMPLIFICATION_CONTRACT,
        model_config=model.get_config(), model_state_dict=model.state_dict(),
    )


def training_command(output):
    return ["--epochs", "2", "--proposal-epochs", "1", "--train-size", "4",
            "--val-size", "2", "--batch-size", "2", "--num-points", "24",
            "--candidate-knots", "8", "--hidden-dim", "16", "--encoder-layers", "1",
            "--selector-layers", "1", "--attention-heads", "4", "--min-control-points", "8",
            "--max-control-points", "12", "--policy-samples", "2", "--counterfactual-edits", "1",
            "--no-certified-minimal-source", "--mse-tolerance", "0.001",
            "--proposal-pass-target", "0", "--torch-num-threads", "1", "--device", "cpu",
            "--log-every-batches", "100", "--output", str(output)]


def test_enhanced_defaults_are_omitted_and_legacy_single_group_is_preserved():
    args = train_v16.parser().parse_args([])
    train_v16.validate_args(args)
    config = train_v16.serial_args(args)
    assert not (set(config) & set(train_v16.ENHANCED_TRAINING_DEFAULTS))
    model = tiny_model()
    for stage in ("proposal", "joint"):
        optimizer = train_v16.build_optimizer(model, args, stage=stage)
        assert len(optimizer.param_groups) == 1
        assert "group_name" not in optimizer.param_groups[0]
        assert optimizer.param_groups[0]["lr"] == (args.lr if stage == "proposal" else args.joint_lr)


@pytest.mark.parametrize("option,value", [
    ("--teacher-refinement-steps", "-1"), ("--teacher-refinement-candidates", "0"),
    ("--boundary-ranking-candidates", "0"), ("--boundary-ranking-weight", "nan"),
    ("--joint-proposal-lr-scale", "0"), ("--joint-decoder-lr-scale", "-1"),
    ("--joint-final-lr-ratio", "1.1"), ("--joint-final-lr-ratio", "0"),
])
def test_invalid_enhanced_settings_are_rejected(option, value):
    args = train_v16.parser().parse_args([option, value])
    with pytest.raises(ValueError):
        train_v16.validate_args(args)


def test_differential_lr_groups_partition_every_parameter_once_and_decay():
    args = train_v16.parser().parse_args([
        "--epochs", "6", "--proposal-epochs", "2", "--joint-proposal-lr-scale", "0.25",
        "--joint-decoder-lr-scale", "0.5", "--joint-final-lr-ratio", "0.25",
    ])
    model = tiny_model()
    optimizer = train_v16.build_optimizer(model, args, stage="joint")
    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(by_name) == {"proposal", "selector", "decoder"}
    identities = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(identities) == len(set(identities)) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        expected = ("proposal" if name.startswith(("encoder.", "parameter_head.", "candidate_head."))
                    else "decoder" if name.startswith((
                        "subset_geometry.", "survivor_attention.", "parameter_attention.",
                        "survivor_norm.", "parameter_norm.", "parameter_update.", "relocation_",
                    )) else "selector")
        assert any(value is parameter for value in by_name[expected]["params"])
    assert train_v16.apply_joint_learning_rate(optimizer, args, epoch=3, end_epoch=6) == 1
    for name, scale in {"proposal": 0.25, "selector": 1.0, "decoder": 0.5}.items():
        assert by_name[name]["lr"] == pytest.approx(args.joint_lr * scale)
    for epoch in (6, 9):
        assert train_v16.apply_joint_learning_rate(optimizer, args, epoch=epoch, end_epoch=6) == 0.25
        assert by_name["selector"]["lr"] == pytest.approx(args.joint_lr * 0.25)


def test_config_comparison_rejects_silently_disabling_enhancements():
    args = train_v16.parser().parse_args([])
    current = train_v16.serial_args(args)
    assert train_v16.training_config_changes(current, deepcopy(current), set()) == []
    previous = {**current, "teacher_refinement_steps": 1}
    assert train_v16.training_config_changes(current, previous, set()) == ["teacher_refinement_steps"]


def test_full_transfer_preserves_selector_decoder_and_resets_only_safety():
    source = tiny_model(one_shot_safety_sigma=0.05, one_shot_safety_knots=0)
    target = tiny_model(one_shot_safety_sigma=0.2, one_shot_safety_knots=2)
    with torch.no_grad():
        source.keep_head.weight.fill_(0.37)
        source.parameter_update[-1].weight.fill_(0.41)
    copied = train_v16.transfer_all_weights(target, native_payload(source))
    assert len(copied) == len(source.state_dict())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key])
    assert target.one_shot_safety_sigma == 0.2
    assert target.one_shot_safety_knots == 2


@pytest.mark.parametrize("change", ["capacity", "policy", "tolerance", "objective", "tensor"])
def test_full_transfer_rejects_incompatible_or_corrupt_source(change):
    source, target = tiny_model(), tiny_model()
    payload = native_payload(source)
    if change == "capacity":
        target = tiny_model(max_internal_knots=9)
    elif change == "policy":
        target = tiny_model(one_shot_selection_policy="threshold")
    elif change == "tolerance":
        target = tiny_model(mse_tolerance=5e-5)
    elif change == "objective":
        payload["objective_version"] = "other"
    else:
        payload["model_state_dict"]["keep_head.weight"] = torch.zeros(1, 17)
    with pytest.raises((ValueError, RuntimeError)):
        train_v16.transfer_all_weights(target, payload)


def test_enhanced_training_resumes_grouped_optimizer_and_keeps_decay_horizon(tmp_path):
    output = tmp_path / "enhanced.pt"
    command = training_command(output)
    command[command.index("--epochs") + 1] = "3"
    command.extend([
        "--joint-proposal-lr-scale", "0.25", "--joint-decoder-lr-scale", "0.5",
        "--joint-final-lr-ratio", "0.25", "--teacher-refinement-steps", "1",
        "--teacher-refinement-candidates", "3", "--boundary-ranking-weight", "0.5",
    ])
    assert train_v16.main(command) == 0
    last = tmp_path / "enhanced.last.pt"
    before = torch.load(last, weights_only=True)
    assert len(before["optimizer_state_dict"]["param_groups"]) == 3
    assert before["loss_config"]["teacher_refinement_steps"] == 1
    assert before["loss_config"]["weights"]["boundary_ranking_weight"] == 0.5
    assert before["history"][-1]["learning_rates"]["selector"] == pytest.approx(5e-5 * 0.25)
    command[command.index("--epochs") + 1] = "4"
    command.extend(["--resume", str(last)])
    assert train_v16.main(command) == 0
    after = torch.load(last, weights_only=True)
    assert after["history"][:3] == before["history"]
    assert after["training_schedule"]["joint_end_epoch"] == 3
    assert after["history"][-1]["learning_rates"] == before["history"][-1]["learning_rates"]
    old_step = max(float(state["step"]) for state in before["optimizer_state_dict"]["state"].values())
    new_step = max(float(state["step"]) for state in after["optimizer_state_dict"]["state"].values())
    assert new_step == old_step + 2


def test_legacy_checkpoint_without_any_new_fields_still_resumes(tmp_path):
    output = tmp_path / "legacy.pt"
    command = training_command(output)
    assert train_v16.main(command) == 0
    last = tmp_path / "legacy.last.pt"
    payload = torch.load(last, weights_only=True)
    payload.pop("training_schedule")
    payload.pop("initializer_provenance")
    for key in train_v16.ENHANCED_TRAINING_DEFAULTS:
        payload["training_config"].pop(key, None)
        payload["loss_config"].pop(key, None)
        payload["loss_config"]["weights"].pop(key, None)
    torch.save(payload, last)
    command[command.index("--epochs") + 1] = "3"
    command.extend(["--resume", str(last)])
    assert train_v16.main(command) == 0
    restored = torch.load(last, weights_only=True)
    assert len(restored["optimizer_state_dict"]["param_groups"]) == 1
    assert restored["history"][:2] == payload["history"]


def test_full_warm_start_is_fresh_records_provenance_and_preserves_source(tmp_path):
    source = tmp_path / "source.pt"
    assert train_v16.main(training_command(source)) == 0
    source_bytes = source.read_bytes()
    target = tmp_path / "warm.pt"
    command = training_command(target) + ["--warm-start-checkpoint", str(source)]
    assert train_v16.main(command) == 0
    payload = torch.load(tmp_path / "warm.last.pt", weights_only=True)
    assert [entry["epoch"] for entry in payload["history"]] == [1, 2]
    record = payload["initializer_provenance"]
    assert record["mode"] == "full_model"
    assert record["path"] == str(source.resolve())
    assert record["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert source.read_bytes() == source_bytes
    command = training_command(target)
    command[command.index("--epochs") + 1] = "3"
    command.extend(["--resume", str(tmp_path / "warm.last.pt")])
    assert train_v16.main(command) == 0
    resumed = torch.load(tmp_path / "warm.last.pt", weights_only=True)
    assert resumed["initializer_provenance"] == record


@pytest.mark.parametrize("other", ["--resume", "--init-checkpoint"])
def test_full_warm_start_is_mutually_exclusive(other):
    args = train_v16.parser().parse_args(["--warm-start-checkpoint", "old.pt", other, "x.pt"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_v16.validate_args(args)
