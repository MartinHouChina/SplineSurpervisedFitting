"""Reliable-run trust migration, serialization, grouped optimization and resume."""
from copy import deepcopy
import json
import math

import pytest
import torch

from test_train_v16 import real_manifest  # noqa: F401 -- shared fixture
from test_train_v16_enhanced import tiny_model, native_payload, training_command, train_v16


TRUST_PREFIXES = ("proposal_parameter_trust_head.", "subset_parameter_trust_head.")
RELIABLE_OPTIONS = {
    "parameter_counterfactual_weight": .25,
    "teacher_geometry_candidates": 4,
    "local_fit_weight": .1,
    "proposal_ordered_weight": .5,
}


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reliable_flags():
    flags = ["--parameter-trust-enabled"]
    for name, value in RELIABLE_OPTIONS.items():
        flags.extend(["--" + name.replace("_", "-"), str(value)])
    return flags


@pytest.mark.parametrize("reverse_batches", [False, True])
def test_epoch_trust_extrema_are_global_and_means_weight_unequal_batches(reverse_batches):
    batches = [
        {"proposal_parameter_trust": torch.tensor([.2, .9, .3], dtype=torch.float64),
         "subset_parameter_trust": torch.tensor([.8, .6, .7], dtype=torch.float64),
         "loss": 2.0},
        {"proposal_parameter_trust": torch.tensor([.1], dtype=torch.float64),
         "subset_parameter_trust": torch.tensor([.95], dtype=torch.float64),
         "loss": 8.0},
    ]
    if reverse_batches:
        batches.reverse()
    totals, samples = {}, 0
    for batch in batches:
        metrics = {"loss": batch["loss"]}
        for gate in ("proposal_parameter_trust", "subset_parameter_trust"):
            values = batch[gate]
            metrics.update({f"{gate}_mean": values.mean(),
                            f"{gate}_min": values.min(), f"{gate}_max": values.max()})
        count = len(batch["proposal_parameter_trust"])
        train_v16.accumulate_training_metrics(totals, metrics, count)
        samples += count
    result = train_v16.finalize_training_metrics(totals, samples)
    for gate in ("proposal_parameter_trust", "subset_parameter_trust"):
        combined = torch.cat([batch[gate] for batch in batches])
        assert result[f"{gate}_mean"] == pytest.approx(float(combined.mean()))
        assert result[f"{gate}_min"] == pytest.approx(float(combined.min()))
        assert result[f"{gate}_max"] == pytest.approx(float(combined.max()))
    assert result["loss"] == pytest.approx(3.5)


def test_explicit_old_to_trust_warm_start_copies_every_old_tensor_and_preserves_12_initializers():
    torch.manual_seed(17)
    source = tiny_model()
    target = tiny_model(parameter_trust_enabled=True, parameter_trust_initial=.4)
    initial = {name: value.clone() for name, value in target.state_dict().items()
               if name.startswith(TRUST_PREFIXES)}
    payload = native_payload(source)
    # A truly old checkpoint did not serialize either trust constructor flag.
    payload["model_config"].pop("parameter_trust_enabled")
    payload["model_config"].pop("parameter_trust_initial")
    copied = train_v16.transfer_all_weights(target, payload)
    assert len(initial) == 12
    assert set(copied) == set(source.state_dict())
    assert set(target.state_dict()) - set(copied) == set(initial)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    for name, value in initial.items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    assert target.parameter_trust_enabled
    assert target.get_config()["parameter_trust_initial"] == .4


def test_old_to_old_warm_start_has_exact_state_and_forward_compatibility():
    source, target = tiny_model(), tiny_model()
    payload = native_payload(source)
    payload["model_config"].pop("parameter_trust_enabled")
    payload["model_config"].pop("parameter_trust_initial")
    copied = train_v16.transfer_all_weights(target, payload)
    assert set(copied) == set(target.state_dict()) == set(source.state_dict())
    t = torch.linspace(0, 1, 24)
    observed = torch.stack([t, torch.sin(5 * t)], dim=-1)[None]
    source.eval()
    target.eval()
    with torch.no_grad():
        expected, actual = source(observed), target(observed)
    for key in ("params", "internal_knots", "learned_keep_mask"):
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_warm_start_cannot_silently_disable_parameter_trust():
    source = tiny_model(parameter_trust_enabled=True)
    target = tiny_model()
    with pytest.raises(ValueError, match="configuration mismatch"):
        train_v16.transfer_all_weights(target, native_payload(source))


def test_optimizer_assigns_proposal_and_subset_trust_heads_to_correct_disjoint_groups():
    args = train_v16.parser().parse_args([
        "--parameter-trust-enabled", "--joint-proposal-lr-scale", ".25",
        "--joint-decoder-lr-scale", ".5",
    ])
    model = tiny_model(parameter_trust_enabled=True)
    optimizer = train_v16.build_optimizer(model, args, stage="joint")
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    all_ids = [id(parameter) for group in groups.values() for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        if name.startswith(TRUST_PREFIXES):
            expected = "proposal" if name.startswith(TRUST_PREFIXES[0]) else "decoder"
            assert any(candidate is parameter for candidate in groups[expected]["params"])
    assert groups["proposal"]["lr"] == args.joint_lr * .25
    assert groups["decoder"]["lr"] == args.joint_lr * .5


@pytest.mark.parametrize("name", list(RELIABLE_OPTIONS) + ["parameter_trust_enabled"])
def test_reliable_flags_serialize_and_cannot_be_silently_changed_on_resume(name):
    defaults = train_v16.serial_args(train_v16.parser().parse_args([]))
    args = train_v16.parser().parse_args(reliable_flags())
    train_v16.validate_args(args)
    enabled = train_v16.serial_args(args)
    assert not (set(RELIABLE_OPTIONS) & defaults.keys())
    assert enabled["parameter_trust_enabled"] is True
    assert all(enabled[key] == value for key, value in RELIABLE_OPTIONS.items())
    modified = deepcopy(enabled)
    modified.pop(name)
    assert train_v16.training_config_changes(modified, enabled, set()) == [name]
    assert train_v16.training_config_changes(enabled, modified, set()) == [name]
    assert train_v16.training_config_changes(enabled, deepcopy(enabled), set()) == []


def test_tiny_reliable_training_and_resume_preserve_config_real_validation_only_and_gate_metrics(
    tmp_path, real_manifest, monkeypatch, capsys,
):
    manifest, records = real_manifest
    output = tmp_path / "reliable.pt"
    command = training_command(output) + reliable_flags() + [
        "--joint-proposal-lr-scale", ".25", "--joint-decoder-lr-scale", ".5",
        "--teacher-prefix-search-steps", "1", "--real-manifest", str(manifest),
        "--real-fraction", "0", "--real-val-size", "2",
    ]
    observed_sources = []
    original_getitem = train_v16.MixedTrainingCurves.__getitem__

    def tracked_item(self, index):
        item = original_getitem(self, index)
        observed_sources.append(item["source"])
        return item

    monkeypatch.setattr(train_v16.MixedTrainingCurves, "__getitem__", tracked_item)
    assert train_v16.main(command) == 0
    last_path = tmp_path / "reliable.last.pt"
    before = torch.load(last_path, map_location="cpu", weights_only=True)
    assert observed_sources and set(observed_sources) == {"Synthetic"}
    assert before["real_data_role"] == "validation_only"
    assert before["dataset_type"] == "uncertified_synthetic_open_cubic_bspline"
    selected = before["validation_real_ids"]["TestReal"]
    lookup = {record["sample_id"]: record for record in records}
    assert len(selected) == 2
    assert all(lookup[sample_id]["split"] == "val" for sample_id in selected)
    assert {entry["stage"] for entry in before["history"]} == {"proposal", "joint"}
    assert before["model_config"]["parameter_trust_enabled"] is True
    for key, value in RELIABLE_OPTIONS.items():
        assert before["training_config"][key] == value
        container = before["loss_config"] if key == "teacher_geometry_candidates" else before["loss_config"]["weights"]
        assert container[key] == value
    for entry in before["history"]:
        assert all(math.isfinite(value) for value in entry["train"].values())
        assert 0 < entry["train"]["proposal_parameter_trust_mean"] < 1
        if entry["stage"] == "proposal":
            assert entry["train"]["subset_parameter_trust_mean"] == 0
        else:
            assert 0 < entry["train"]["subset_parameter_trust_mean"] < 1
        by_source = entry["validation"]["by_source"]
        assert set(by_source) == {"Synthetic", "TestReal"}
        for values in by_source.values():
            for prefix in ("proposal_parameter_trust", "subset_parameter_trust"):
                if prefix.startswith("subset") and entry["stage"] == "proposal":
                    continue
                low, mean, high = (values[f"{prefix}_{name}"] for name in ("min", "mean", "max"))
                assert 0 < low <= mean <= high < 1
    # Adam's state proves both new gate groups received finite gradients.
    state_by_id = before["optimizer_state_dict"]["state"]
    trained = train_v16.build_model_from_checkpoint(before)[0]
    args = train_v16.parser().parse_args(command)
    optimizer = train_v16.build_optimizer(trained, args, stage="joint")
    for live, saved in zip(optimizer.param_groups, before["optimizer_state_dict"]["param_groups"]):
        for parameter, identifier in zip(live["params"], saved["params"]):
            name = next(name for name, value in trained.named_parameters() if value is parameter)
            if name.startswith(TRUST_PREFIXES):
                assert identifier in state_by_id
                assert torch.isfinite(state_by_id[identifier]["exp_avg"]).all()
    for prefix in TRUST_PREFIXES:
        tensors = [value for name, value in before["model_state_dict"].items() if name.startswith(prefix)]
        assert len(tensors) == 6 and all(torch.isfinite(value).all() for value in tensors)
    captured = capsys.readouterr().out
    assert "VALIDATION ONLY" in captured and "trust_proposal=" in captured and "trust_subset=" in captured

    resumed_command = deepcopy(command)
    resumed_command[resumed_command.index("--epochs") + 1] = "3"
    resumed_command.extend(["--resume", str(last_path)])
    assert train_v16.main(resumed_command) == 0
    after = torch.load(last_path, map_location="cpu", weights_only=True)
    assert after["history"][:2] == before["history"]
    assert after["loss_config"] == before["loss_config"]
    assert after["validation_real_ids"] == before["validation_real_ids"]
    assert after["training_schedule"] == before["training_schedule"]
    assert set(observed_sources) == {"Synthetic"}
    old_step = max(float(state["step"]) for state in state_by_id.values())
    new_step = max(float(state["step"]) for state in after["optimizer_state_dict"]["state"].values())
    assert new_step == old_step + 2
    assert json.loads(output.with_suffix(".history.json").read_text()) == after["history"]
    invalid = deepcopy(resumed_command)
    invalid[invalid.index("--local-fit-weight") + 1] = ".2"
    with pytest.raises(SystemExit):
        train_v16.main(invalid)
