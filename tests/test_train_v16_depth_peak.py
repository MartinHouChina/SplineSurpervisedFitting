"""Depth/peak-error options survive native warm start, freezing and resume."""
from copy import deepcopy
import json

import pytest
import torch

from test_train_v16_enhanced import native_payload, tiny_model, training_command, train_v16


DEPTHS = dict(proposal_refinement_layers=1, selection_refinement_layers=1,
              survivor_refinement_layers=1)
DEPTH_FLAGS = ["--candidate-refinement-layers", "1", "--selection-refinement-layers", "1",
               "--decoder-refinement-layers", "1"]
PREFIXES = ("proposal_refinement_blocks.", "selection_refinement_blocks.",
            "survivor_refinement_blocks.")


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_depth_cli_configuration_and_peak_units_are_explicit():
    args = train_v16.parser().parse_args(DEPTH_FLAGS + [
        "--max-point-error-weight", ".1", "--max-point-error-tolerance", ".0005",
        "--max-point-error-tail-fraction", ".1", "--synthetic-shape-domain", "industrial",
    ])
    train_v16.validate_args(args)
    config = train_v16.serial_args(args)
    for name, value in DEPTHS.items():
        assert config[name] == value
    assert config["max_point_error_weight"] == .1
    assert config["max_point_error_tolerance"] == .0005
    assert config["max_point_error_tail_fraction"] == .1
    assert config["mse_tolerance"] != config["max_point_error_tolerance"]
    assert config["synthetic_shape_domain"] == "industrial"


@pytest.mark.parametrize("flags", [
    ["--candidate-refinement-layers", "-1"],
    ["--selection-refinement-layers", "-1"],
    ["--decoder-refinement-layers", "-1"],
    ["--max-point-error-weight", ".1"],
    ["--max-point-error-weight", "nan"],
    ["--max-point-error-tolerance", "0"],
    ["--max-point-error-tail-fraction", "0"],
    ["--max-point-error-tail-fraction", "1.1"],
])
def test_invalid_depth_or_peak_options_are_rejected(flags):
    with pytest.raises(ValueError):
        train_v16.validate_args(train_v16.parser().parse_args(flags))


def test_expansion_preserves_every_existing_weight_and_function_at_initialization():
    torch.manual_seed(909)
    shallow = tiny_model(subset_geometry_mode="anchored").eval()
    with torch.no_grad():
        shallow.parameter_update[-1].weight.normal_(std=.15)
        shallow.relocation_update.weight.normal_(std=.15)
    deep = tiny_model(subset_geometry_mode="anchored", **DEPTHS).eval()
    metadata = {}
    copied = train_v16.transfer_all_weights(deep, native_payload(shallow), transfer_metadata=metadata)
    assert set(copied) == set(shallow.state_dict())
    transfer = metadata["refinement_depth_transfer"]
    assert transfer["changes"] == {key: {"source": 0, "target": value}
                                   for key, value in DEPTHS.items()}
    assert transfer["initialized_tensor_names"]
    assert all(name.startswith(PREFIXES) for name in transfer["initialized_tensor_names"])
    points = torch.randn(2, 24, 2)
    old_context, new_context = shallow.encode_candidates(points), deep.encode_candidates(points)
    for mask in (torch.zeros(2, 8, dtype=torch.bool), torch.ones(2, 8, dtype=torch.bool),
                 torch.arange(8).expand(2, -1).remainder(3) == 0):
        old, new = shallow.decode_subset(old_context, mask), deep.decode_subset(new_context, mask)
        for key in old:
            assert torch.equal(old[key], new[key]), key


def test_deep_optimizer_partition_and_proposal_freeze_cover_extra_layers():
    model = tiny_model(parameter_trust_enabled=True, **DEPTHS)
    args = train_v16.parser().parse_args([
        "--epochs", "4", "--proposal-epochs", "1", "--joint-geometry-calibration-epochs", "1",
        "--joint-proposal-lr-scale", ".1", "--joint-decoder-lr-scale", ".5",
    ])
    optimizer = train_v16.build_optimizer(model, args, stage="joint")
    identities = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(identities) == len(set(identities)) == len(list(model.parameters()))
    groups = {id(parameter): group["group_name"] for group in optimizer.param_groups
              for parameter in group["params"]}
    for name, parameter in model.named_parameters():
        if name.startswith(PREFIXES):
            expected = ("proposal" if name.startswith(PREFIXES[0]) else
                        "selector" if name.startswith(PREFIXES[1]) else "decoder")
            assert groups[id(parameter)] == expected
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.train()
    train_v16.set_proposal_trainability(model, frozen=True)
    train_v16.apply_joint_learning_rate(optimizer, args, epoch=2, end_epoch=4)
    assert not model.proposal_refinement_blocks.training
    assert model.selection_refinement_blocks.training and model.survivor_refinement_blocks.training
    for name, parameter in model.named_parameters():
        if name.startswith(train_v16.PROPOSAL_PARAMETER_PREFIXES):
            assert not parameter.requires_grad and parameter.grad is None
        else:
            assert parameter.requires_grad
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith(train_v16.PROPOSAL_PARAMETER_PREFIXES):
            assert torch.equal(before[name], parameter)
    assert not torch.equal(before["selection_refinement_blocks.0.residual_gate"],
                           model.selection_refinement_blocks[0].residual_gate)
    assert not torch.equal(before["survivor_refinement_blocks.0.parameter_gate"],
                           model.survivor_refinement_blocks[0].parameter_gate)
    train_v16.set_proposal_trainability(model, frozen=False)
    assert model.proposal_refinement_blocks.training
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_proposal_only_init_copies_trained_depth_and_trust_heads():
    source = tiny_model(parameter_trust_enabled=True, **DEPTHS)
    target = tiny_model(parameter_trust_enabled=True, **DEPTHS)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.startswith((PREFIXES[0], "proposal_parameter_trust_head.")):
                parameter.fill_(.23)
    selector_before = {name: value.clone() for name, value in target.state_dict().items()
                       if name.startswith(PREFIXES[1:])}
    copied = train_v16.transfer_proposal_weights(target, native_payload(source))
    for name, value in source.state_dict().items():
        if name.startswith((PREFIXES[0], "proposal_parameter_trust_head.")):
            assert name in copied
            assert torch.equal(value, target.state_dict()[name])
    for name, value in selector_before.items():
        assert torch.equal(value, target.state_dict()[name])


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_expansion_does_not_hide_corrupt_unrelated_checkpoint_tensors(corruption):
    payload = native_payload(tiny_model())
    if corruption == "missing":
        del payload["model_state_dict"]["keep_head.weight"]
    elif corruption == "unexpected":
        payload["model_state_dict"]["unrelated_network.weight"] = torch.ones(1)
    else:
        payload["model_state_dict"]["keep_head.weight"] = torch.ones(1, 99)
    with pytest.raises((ValueError, RuntimeError)):
        train_v16.transfer_all_weights(tiny_model(**DEPTHS), payload)


def test_depth_shrink_is_rejected_and_trained_existing_deep_blocks_are_copied():
    source = tiny_model(**DEPTHS)
    with torch.no_grad():
        source.selection_refinement_blocks[0].residual_gate.fill_(.17)
    with pytest.raises(ValueError, match="discard learned refinement"):
        train_v16.transfer_all_weights(tiny_model(), native_payload(source))
    target = tiny_model(**{key: 2 for key in DEPTHS})
    metadata = {}
    train_v16.transfer_all_weights(target, native_payload(source), transfer_metadata=metadata)
    assert target.selection_refinement_blocks[0].residual_gate.item() == pytest.approx(.17)
    assert target.selection_refinement_blocks[1].residual_gate.item() == 0
    assert all(".1." in key for key in metadata["refinement_depth_transfer"]["initialized_tensor_names"])


@pytest.mark.parametrize("key,value", [
    ("proposal_refinement_layers", 1), ("selection_refinement_layers", 1),
    ("survivor_refinement_layers", 1), ("max_point_error_weight", .1),
    ("max_point_error_tolerance", .001), ("max_point_error_tail_fraction", .1),
    ("synthetic_shape_domain", "industrial"),
])
def test_depth_peak_and_shape_domain_are_immutable_on_resume(key, value):
    baseline = train_v16.serial_args(train_v16.parser().parse_args([]))
    changed = {**baseline, key: value}
    assert train_v16.training_config_changes(changed, baseline, set()) == [key]
    assert train_v16.training_config_changes(baseline, changed, set()) == [key]


def test_real_tiny_cpu_training_depth_warmstart_peak_save_reload_and_resume(tmp_path):
    source = tmp_path / "source.pt"
    assert train_v16.main(training_command(source)) == 0
    source_bytes = source.read_bytes()
    output = tmp_path / "deep_peak.pt"
    command = training_command(output) + DEPTH_FLAGS + [
        "--epochs", "3", "--warm-start-checkpoint", str(source),
        "--subset-geometry-mode", "anchored", "--subset-geometry-residual-scale", ".25",
        "--joint-geometry-calibration-epochs", "1", "--joint-proposal-lr-scale", ".1",
        "--max-point-error-weight", ".1", "--max-point-error-tolerance", ".01",
        "--max-point-error-tail-fraction", ".1", "--real-fraction", "0",
        "--synthetic-shape-fraction", ".5", "--synthetic-shape-domain", "industrial",
    ]
    assert train_v16.main(command) == 0
    last_path = tmp_path / "deep_peak.last.pt"
    before = torch.load(last_path, map_location="cpu", weights_only=True)
    assert source.read_bytes() == source_bytes
    assert [row["proposal_frozen"] for row in before["history"]] == [False, True, False]
    for name, value in DEPTHS.items():
        assert before["model_config"][name] == before["training_config"][name] == value
    assert before["loss_config"]["weights"]["max_point_error_weight"] == .1
    assert before["loss_config"]["max_point_error_tolerance"] == .01
    assert before["loss_config"]["max_point_error_tail_fraction"] == .1
    assert before["training_config"]["synthetic_shape_domain"] == "industrial"
    assert before["initializer_provenance"]["refinement_depth_transfer"]["changes"]
    assert len(before["optimizer_state_dict"]["param_groups"]) == 3
    saved_json = json.loads((tmp_path / "deep_peak.history.json").read_text(encoding="utf-8"))
    assert saved_json == before["history"]
    for row in saved_json:
        assert row["train"]["max_point_error_loss"] >= 0
    restored, _, _ = train_v16.build_model_from_checkpoint(before)
    restored.eval()
    with torch.no_grad():
        output_dict = restored(torch.randn(2, 24, 2))
    assert all(torch.isfinite(value).all() for value in output_dict.values())
    # A resumed command drops initialization-only flags; no migration happens
    # a second time, and the exact saved depth/loss/optimizer must be reused.
    resume = deepcopy(command)
    index = resume.index("--warm-start-checkpoint")
    del resume[index:index + 2]
    resume += ["--epochs", "4", "--resume", str(last_path)]
    assert train_v16.main(resume) == 0
    after = torch.load(last_path, map_location="cpu", weights_only=True)
    assert after["history"][:3] == before["history"]
    assert after["initializer_provenance"] == before["initializer_provenance"]
    # The recorded live safety reserve may anneal, while architecture and loss
    # settings must remain unchanged across the resumed optimization step.
    controlled = {"one_shot_safety_sigma", "one_shot_safety_knots"}
    assert {k: v for k, v in after["model_config"].items() if k not in controlled} == {
        k: v for k, v in before["model_config"].items() if k not in controlled}
    for flag, value in (("--candidate-refinement-layers", "2"),
                        ("--max-point-error-tolerance", ".02"),
                        ("--synthetic-shape-domain", "terrain")):
        with pytest.raises(SystemExit):
            train_v16.main(resume + ["--epochs", "5", flag, value])
