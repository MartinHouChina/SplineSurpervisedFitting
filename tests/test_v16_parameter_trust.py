from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


def _network(**options):
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, **options,
    )


def _points(dtype=torch.float32):
    t = torch.linspace(0, 1, 24, dtype=dtype)
    return torch.stack([
        torch.stack([t, torch.sin(6 * t)], -1),
        torch.stack([t, torch.cos(3 * t)], -1),
    ])


def _activate_parameter_residuals(model):
    with torch.no_grad():
        model.parameter_head.mlp[-1].weight.normal_(std=0.5)
        model.parameter_update[-1].weight.normal_(std=0.3)


def test_disabled_trust_retains_legacy_state_and_parameter_path():
    torch.manual_seed(23)
    model = _network()
    _activate_parameter_residuals(model)
    assert not any("trust_head" in name for name in model.state_dict())
    legacy_config = model.get_config()
    legacy_config.pop("parameter_trust_enabled")
    legacy_config.pop("parameter_trust_initial")
    restored = V16CandidateSelectionNetwork(**legacy_config)
    restored.load_state_dict(model.state_dict(), strict=True)
    output = model(_points())
    other = restored(_points())
    for name in output:
        torch.testing.assert_close(output[name], other[name], rtol=0, atol=0)
    assert torch.equal(output["proposal_params"], output["ungated_proposal_params"])
    assert torch.equal(output["params"], output["ungated_subset_params"])
    assert torch.equal(output["proposal_parameter_trust"], torch.ones(2, 1))
    assert torch.equal(output["subset_parameter_trust"], torch.ones(2, 1))


def test_only_explicit_trust_heads_are_new_checkpoint_parameters():
    torch.manual_seed(29)
    legacy = _network()
    torch.manual_seed(29)
    trusted = _network(parameter_trust_enabled=True)
    for name, tensor in legacy.state_dict().items():
        assert torch.equal(tensor, trusted.state_dict()[name])
    result = trusted.load_state_dict(legacy.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert len(result.missing_keys) == 12
    assert all(name.startswith((
        "proposal_parameter_trust_head.", "subset_parameter_trust_head.",
    )) for name in result.missing_keys)


@pytest.mark.parametrize("kind", ["empty", "mixed", "full"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trust_is_bounded_monotone_and_preserves_selected_geometry(kind, dtype):
    torch.manual_seed(31)
    model = _network(parameter_trust_enabled=True).to(dtype=dtype)
    _activate_parameter_residuals(model)
    context = model.encode_candidates(_points(dtype))
    mask = torch.zeros(2, 6, dtype=torch.bool)
    if kind == "mixed":
        mask[0, [0, 2, 5]] = True
        mask[1, [1, 3]] = True
    elif kind == "full":
        mask[:] = True
    output = model.decode_subset(context, mask)
    for name in ("proposal_parameter_trust", "subset_parameter_trust"):
        assert output[name].shape == (2, 1)
        torch.testing.assert_close(output[name], torch.full((2, 1), .25, dtype=dtype))
    torch.testing.assert_close(
        output["proposal_params"],
        output["chord_params"] + .25 * (
            output["ungated_proposal_params"] - output["chord_params"]
        ),
    )
    torch.testing.assert_close(
        output["params"], output["proposal_params"] + .25 * (
            output["ungated_subset_params"] - output["proposal_params"]
        ),
    )
    assert torch.equal(output["learned_keep_mask"], mask)
    for name in ("chord_params", "proposal_params", "params"):
        values = output[name]
        assert torch.equal(values[:, 0], torch.zeros(2, dtype=dtype))
        assert torch.equal(values[:, -1], torch.ones(2, dtype=dtype))
        assert (values.diff(dim=-1) >= model.min_parameter_gap - 1e-7).all()
    torch.testing.assert_close(output["parameter_gaps"], output["params"].diff(dim=-1))
    expected_warp = model._warp_knots(
        context["proposal_internal_knots"], context["proposal_params"], output["params"],
    )
    torch.testing.assert_close(output["warped_proposal_internal_knots"], expected_warp)
    for row in range(2):
        knots = output["internal_knots"][row, mask[row]]
        boundaries = torch.cat([knots.new_zeros(1), knots, knots.new_ones(1)])
        assert (boundaries.diff() >= model.min_knot_gap - 1e-7).all()


def test_trust_gates_and_original_parameter_heads_receive_gradients():
    torch.manual_seed(37)
    model = _network(parameter_trust_enabled=True)
    _activate_parameter_residuals(model)
    points = _points().requires_grad_()
    context = model.encode_candidates(points)
    mask = torch.tensor([[True, False, True, False, False, True], [True] * 6])
    output = model.decode_subset(context, mask)
    loss = output["params"].square().sum() + output["internal_knots"][mask].square().sum()
    loss.backward()
    for layer in (
        model.proposal_parameter_trust_head[-1], model.subset_parameter_trust_head[-1],
        model.parameter_head.mlp[-1], model.parameter_update[-1], model.relocation_update,
    ):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    assert torch.isfinite(points.grad).all()


@pytest.mark.parametrize("kind", ["identical", "repeated", "tiny_span"])
def test_strict_chord_projection_handles_degenerate_samples(kind):
    model = _network(parameter_trust_enabled=True)
    points = torch.zeros(2, 24, 2)
    if kind != "identical":
        points[:, 12:, 0] = 1.0
    if kind == "tiny_span":
        points[:, 3, 0] = 1e-12
    output = model(points)
    for name in ("chord_params", "proposal_params", "params"):
        assert torch.isfinite(output[name]).all()
        assert (output[name].diff(dim=-1) >= model.min_parameter_gap - 1e-7).all()


def test_trust_roundtrip_remains_one_forward_without_search():
    model = _network(parameter_trust_enabled=True, parameter_trust_initial=.4)
    restored = V16CandidateSelectionNetwork(**model.get_config())
    restored.load_state_dict(model.state_dict(), strict=True)
    with patch.object(restored, "encode_candidates", wraps=restored.encode_candidates) as encode:
        with patch.object(restored, "select_mask", wraps=restored.select_mask) as select:
            with patch.object(restored, "decode_subset", wraps=restored.decode_subset) as decode:
                result = restored(_points())
    assert encode.call_count == select.call_count == decode.call_count == 1
    torch.testing.assert_close(result["params"], model(_points())["params"], rtol=0, atol=0)


@pytest.mark.parametrize("options", [
    {"parameter_trust_enabled": 1}, {"parameter_trust_initial": 0},
    {"parameter_trust_initial": 1}, {"parameter_trust_initial": float("nan")},
    {"parameter_trust_initial": float("inf")}, {"parameter_trust_initial": True},
])
def test_invalid_parameter_trust_configuration(options):
    with pytest.raises(ValueError, match="parameter_trust"):
        _network(**options)
