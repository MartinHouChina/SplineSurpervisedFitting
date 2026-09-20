from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
import torch

from spline_fitting.checkpointing import (
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION, build_model_from_checkpoint,
)
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


def _model(**options):
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, **options,
    ).eval()


def _points(dtype=torch.float32):
    t = torch.linspace(0, 1, 24, dtype=dtype)
    return torch.stack([
        torch.stack([t, (7 * t).sin()], -1),
        torch.stack([t, (4 * t).cos()], -1),
    ])


def _mask(kind):
    mask = torch.zeros(2, 6, dtype=torch.bool)
    if kind == "full":
        mask[:] = True
    elif kind == "mixed":
        mask[0, [0, 1, 5]] = True
        mask[1, [2]] = True
    return mask


def _activate_updates(model, std=0.2):
    with torch.no_grad():
        model.parameter_update[-1].weight.normal_(std=std)
        model.relocation_update.weight.normal_(std=std)


def test_legacy_defaults_and_state_layout_are_unchanged():
    torch.manual_seed(733)
    old = _model()
    _activate_updates(old)
    old_config = old.get_config()
    old_config.pop("subset_geometry_mode")
    old_config.pop("subset_geometry_residual_scale")
    restored = V16CandidateSelectionNetwork(**old_config).eval()
    restored.load_state_dict(old.state_dict(), strict=True)
    explicit = _model(subset_geometry_mode="legacy", subset_geometry_residual_scale=0)
    explicit.load_state_dict(old.state_dict(), strict=True)
    anchor = _model(subset_geometry_mode="anchored")
    anchor.load_state_dict(old.state_dict(), strict=True)
    assert tuple(old.state_dict()) == tuple(anchor.state_dict())
    expected = old(_points())
    for model in (restored, explicit):
        actual = model(_points())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["empty", "mixed", "full"])
@pytest.mark.parametrize("trust", [False, True])
def test_zero_scale_exactly_preserves_proposal_even_with_trained_updates(kind, trust):
    torch.manual_seed(739)
    model = _model(
        subset_geometry_mode="anchored", subset_geometry_residual_scale=0,
        parameter_trust_enabled=trust,
    )
    _activate_updates(model, std=20)
    context = model.encode_candidates(_points())
    output = model.decode_subset(context, _mask(kind))
    assert torch.equal(output["params"], context["proposal_params"])
    assert torch.equal(output["internal_knots"], context["proposal_internal_knots"])
    assert torch.equal(output["warped_proposal_internal_knots"], context["proposal_internal_knots"])
    assert torch.equal(output["parameter_gaps"], context["proposal_params"].diff(dim=-1))


def test_zero_initialized_relocation_has_no_uniform_rank_attraction():
    model = _model(subset_geometry_mode="anchored")
    context = model.encode_candidates(_points())
    mask = torch.tensor([[True, True, False, False, False, False],
                         [False, False, False, False, True, True]])
    output = model.decode_subset(context, mask)
    torch.testing.assert_close(
        output["internal_knots"], context["proposal_internal_knots"], rtol=0, atol=2e-7,
    )
    # Moving survivors toward uniform ranks was mandatory in the legacy path.
    assert output["internal_knots"][0, 1] < 0.4
    assert output["internal_knots"][1, 4] > 0.6


@pytest.mark.parametrize("kind", ["empty", "mixed", "full"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_anchored_large_updates_remain_ordered_bounded_and_warp_consistent(kind, dtype):
    torch.manual_seed(743)
    model = _model(
        subset_geometry_mode="anchored", parameter_trust_enabled=True,
        subset_geometry_residual_scale=.65,
    ).to(dtype=dtype)
    _activate_updates(model, std=50)
    context = model.encode_candidates(_points(dtype))
    mask = _mask(kind)
    output = model.decode_subset(context, mask)
    eps = 2 * torch.finfo(dtype).eps
    for value in output.values():
        assert torch.isfinite(value).all()
    assert torch.equal(output["params"][:, 0], torch.zeros(2, dtype=dtype))
    assert torch.equal(output["params"][:, -1], torch.ones(2, dtype=dtype))
    assert (output["params"].diff(dim=-1) >= model.min_parameter_gap - eps).all()
    torch.testing.assert_close(
        output["warped_proposal_internal_knots"],
        model._warp_knots(context["proposal_internal_knots"], context["proposal_params"], output["params"]),
        rtol=0, atol=eps,
    )
    assert torch.equal(output["internal_knots"][~mask], output["warped_proposal_internal_knots"][~mask])
    for row in range(2):
        chosen = output["internal_knots"][row, mask[row]]
        boundaries = torch.cat([chosen.new_zeros(1), chosen, chosen.new_ones(1)])
        assert (boundaries.diff() >= model.min_knot_gap - eps).all()


def test_parameter_contraction_preserves_knot_gap_instead_of_reprojecting_coordinates():
    model = _model(subset_geometry_mode="anchored", min_knot_gap=.05).double()
    knots = torch.tensor([[.10, .155, .32, .55, .72, .9]], dtype=torch.float64)
    old = torch.linspace(0, 1, 11, dtype=torch.float64).unsqueeze(0)
    # A monotone target t can compress the first two knots below min_knot_gap.
    proposed = torch.tensor([[0., .002, .004, .07, .15, .27, .43, .6, .76, .9, 1.]], dtype=torch.float64)
    mask = torch.ones_like(knots, dtype=torch.bool)
    residual = torch.zeros(1, 6, 2, dtype=torch.float64)
    params, warped, internal, _ = model._anchored_subset_geometry(knots, old, proposed, mask, residual)
    assert not torch.allclose(params, proposed)
    assert not torch.equal(params, old)
    expected = model._warp_knots(knots, old, params)
    torch.testing.assert_close(warped, expected, rtol=0, atol=1e-15)
    assert torch.equal(internal, warped)
    boundaries = torch.cat([knots.new_zeros(1, 1), internal, knots.new_ones(1, 1)], -1)
    assert (boundaries.diff(dim=-1) >= model.min_knot_gap - 1e-15).all()


def test_residual_positions_and_parameters_both_receive_finite_nonzero_gradients():
    torch.manual_seed(751)
    model = _model(subset_geometry_mode="anchored", parameter_trust_enabled=True)
    _activate_updates(model)
    points = _points().requires_grad_()
    mask = _mask("mixed")
    output = model.decode_subset(model.encode_candidates(points), mask)
    loss = output["params"].square().sum() + output["internal_knots"][mask].square().sum()
    loss.backward()
    assert torch.isfinite(points.grad).all()
    for layer in (
        model.parameter_head.mlp[-1], model.parameter_update[-1],
        model.candidate_head.interval_score, model.relocation_update,
        model.subset_parameter_trust_head[-1],
    ):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    assert model.relocation_update.weight.grad[0].abs().sum() > 0
    assert model.relocation_update.weight.grad[1].abs().sum() > 0
    assert model.keep_head.weight.grad is None


def test_selected_only_memory_and_one_forward_are_preserved():
    model = _model(subset_geometry_mode="anchored")
    context = model.encode_candidates(_points())
    mask = _mask("mixed")
    mask[1] = False
    with patch.object(model.survivor_attention, "forward", wraps=model.survivor_attention.forward) as survivor:
        with patch.object(model.parameter_attention, "forward", wraps=model.parameter_attention.forward) as parameter:
            model.decode_subset(context, mask)
    for attention in (survivor, parameter):
        padding = attention.call_args.kwargs["key_padding_mask"]
        assert torch.equal(padding[:, :-1], ~mask)
        assert torch.equal(padding[:, -1], torch.tensor([True, False]))
    with patch.object(model, "encode_candidates", wraps=model.encode_candidates) as encode:
        with patch.object(model, "select_mask", wraps=model.select_mask) as select:
            with patch.object(model, "decode_subset", wraps=model.decode_subset) as decode:
                model.forward_deployment(_points())
    assert encode.call_count == select.call_count == decode.call_count == 1


def test_checkpoint_roundtrip_preserves_anchored_mode_and_live_scale():
    torch.manual_seed(757)
    model = _model(subset_geometry_mode="anchored", subset_geometry_residual_scale=.2)
    _activate_updates(model)
    model.subset_geometry_residual_scale = .4
    assert model.get_config()["subset_geometry_residual_scale"] == .4
    checkpoint = dict(
        objective_version=V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        model_config=model.get_config(), model_state_dict=model.state_dict(),
    )
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    restored, _, _ = build_model_from_checkpoint(torch.load(buffer, weights_only=True))
    restored.eval()
    assert restored.subset_geometry_mode == "anchored"
    assert restored.subset_geometry_residual_scale == .4
    expected, actual = model(_points()), restored(_points())
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("options", [
    {"subset_geometry_mode": "unbounded"},
    {"subset_geometry_residual_scale": -.01},
    {"subset_geometry_residual_scale": 1.01},
    {"subset_geometry_residual_scale": float("nan")},
    {"subset_geometry_residual_scale": float("inf")},
    {"subset_geometry_residual_scale": True},
])
def test_invalid_anchored_geometry_options(options):
    with pytest.raises(ValueError, match="subset_geometry"):
        _model(**options)
