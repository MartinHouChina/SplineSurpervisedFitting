from copy import deepcopy

import pytest
import torch

from test_train_v16_enhanced import native_payload, tiny_model, training_command, train_v16


FLAGS = ["--coupled-proposal-steps", "2", "--coupled-subset-steps", "2",
         "--parameter-chord-blend", ".5", "--subset-geometry-mode", "anchored"]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_native_checkpoint_migration_records_depth_and_chord_contract():
    source = tiny_model(subset_geometry_mode="anchored")
    target = tiny_model(subset_geometry_mode="anchored", coupled_proposal_steps=2,
                        coupled_subset_steps=2, parameter_chord_blend=.5)
    metadata = {}
    train_v16.transfer_all_weights(target, native_payload(source), transfer_metadata=metadata)
    assert metadata["subset_geometry_transfer"]["changes"]["parameter_chord_blend"] == {
        "source": 0., "target": .5}
    assert set(metadata["refinement_depth_transfer"]["changes"]) == {
        "coupled_proposal_steps", "coupled_subset_steps"}
    args = train_v16.parser().parse_args(FLAGS + ["--joint-proposal-lr-scale", ".1",
                                               "--joint-decoder-lr-scale", ".25"])
    optimizer = train_v16.build_optimizer(target, args, stage="joint")
    groups = {id(v): g["group_name"] for g in optimizer.param_groups for v in g["params"]}
    for name, value in target.named_parameters():
        if name.startswith("coupled_proposal_blocks."):
            assert groups[id(value)] == "proposal"
        elif name.startswith("coupled_subset_blocks."):
            assert groups[id(value)] == "decoder"
    train_v16.set_proposal_trainability(target, frozen=True)
    assert not any(p.requires_grad for p in target.coupled_proposal_blocks.parameters())
    assert all(p.requires_grad for p in target.coupled_subset_blocks.parameters())


def test_coupled_training_checkpoint_and_resume_are_real_end_to_end(tmp_path):
    path = tmp_path / "coupled.pt"
    command = training_command(path) + FLAGS + [
        "--parameter-trust-enabled", "--parameter-counterfactual-weight", ".1",
        "--max-point-error-weight", ".05", "--max-point-error-tolerance", ".01",
    ]
    assert train_v16.main(command) == 0
    last = path.with_name("coupled.last.pt")
    payload = torch.load(last, map_location="cpu", weights_only=True)
    assert len(payload["history"]) == 2
    assert payload["history"][1]["train"]["deployment_mse"] >= 0
    assert payload["model_config"]["coupled_subset_steps"] == 2
    model, _, _ = train_v16.build_model_from_checkpoint(payload)
    assert torch.isfinite(model.eval()(torch.randn(1, 24, 2))["params"]).all()
    resumed = deepcopy(command) + ["--epochs", "3", "--resume", str(last)]
    assert train_v16.main(resumed) == 0
    after = torch.load(last, map_location="cpu", weights_only=True)
    assert after["history"][:2] == payload["history"]
    with pytest.raises(SystemExit):
        train_v16.main(resumed + ["--epochs", "4", "--parameter-chord-blend", ".75"])


def test_frozen_coupled_proposal_has_no_decoder_dependency():
    net = tiny_model(subset_geometry_mode="anchored", coupled_proposal_steps=2,
                     coupled_subset_steps=2).eval()
    with torch.no_grad():
        for name, value in net.named_parameters():
            if name.startswith("coupled_") and name.endswith("gate"):
                value.fill_(.6)
    train_v16.set_proposal_trainability(net, frozen=True)
    x = torch.randn(2, 24, 2)
    before = net.encode_candidates(x)
    with torch.no_grad():
        for name, value in net.named_parameters():
            if value.requires_grad:
                value.add_(.2)
    after = net.encode_candidates(x)
    for key in ("proposal_params", "proposal_internal_knots"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
