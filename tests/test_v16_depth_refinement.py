from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
import torch

from spline_fitting.checkpointing import (
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION, build_model_from_checkpoint,
)
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


DEPTHS = dict(proposal_refinement_layers=2, selection_refinement_layers=2,
              survivor_refinement_layers=2)
PREFIXES = ("proposal_refinement_blocks.", "selection_refinement_blocks.",
            "survivor_refinement_blocks.")


def _model(**options):
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, subset_geometry_mode="anchored",
        one_shot_selection_policy="mass_topk", one_shot_adaptive_threshold=True,
        **options,
    ).eval()


def _points():
    t = torch.linspace(0, 1, 24)
    return torch.stack([torch.stack([t, (6 * t).sin()], -1),
                        torch.stack([t, (4 * t).cos()], -1)])


def _mask():
    return torch.tensor([[True, False, True, False, False, True],
                         [False, False, False, False, False, False]])


def _activate(model):
    with torch.no_grad():
        model.parameter_update[-1].weight.normal_(std=.2)
        model.relocation_update.weight.normal_(std=.2)
        for name, parameter in model.named_parameters():
            if name.startswith(PREFIXES) and name.endswith("gate"):
                parameter.fill_(.25)


def test_default_depth_adds_no_state_keys_and_old_config_strict_loads():
    model = _model()
    assert not any(name.startswith(PREFIXES) for name in model.state_dict())
    old_config = model.get_config()
    for option in DEPTHS:
        assert old_config.pop(option) == 0
    restored = V16CandidateSelectionNetwork(**old_config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    for name, value in model(_points()).items():
        assert torch.equal(value, restored(_points())[name])


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("kind", ["empty", "mixed", "full"])
def test_new_depth_starts_exactly_at_shallow_function_and_keeps_initial_rng(trust, kind):
    torch.manual_seed(810)
    shallow = _model(parameter_trust_enabled=trust)
    torch.manual_seed(810)
    deep = _model(parameter_trust_enabled=trust, **DEPTHS)
    # Extra allocation happens after historical modules, so initialization is
    # unchanged even before an explicit warm-start weight copy.
    for name, value in shallow.state_dict().items():
        assert torch.equal(value, deep.state_dict()[name])
    with torch.no_grad():
        shallow.parameter_update[-1].weight.normal_(std=.2)
        shallow.relocation_update.weight.normal_(std=.2)
    result = deep.load_state_dict(shallow.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(name.startswith(PREFIXES) for name in result.missing_keys)
    mask = _mask()
    if kind == "empty":
        mask[:] = False
    elif kind == "full":
        mask[:] = True
    old = shallow.decode_subset(shallow.encode_candidates(_points()), mask)
    new = deep.decode_subset(deep.encode_candidates(_points()), mask)
    assert old.keys() == new.keys()
    for name in old:
        torch.testing.assert_close(new[name], old[name], rtol=0, atol=0)


def test_survivor_stack_excludes_removed_kv_and_is_safe_for_empty_mask():
    torch.manual_seed(815)
    model = _model(**DEPTHS)
    _activate(model)
    context = model.encode_candidates(_points())
    mask = _mask()
    calls = []
    hooks = []
    for block in model.survivor_refinement_blocks:
        for attention in (block.token_refinement.self_attention, block.parameter_attention):
            hooks.append(attention.register_forward_pre_hook(
                lambda _module, _args, kwargs: calls.append(kwargs["key_padding_mask"].clone()),
                with_kwargs=True,
            ))
    output = model.decode_subset(context, mask)
    for hook in hooks:
        hook.remove()
    assert len(calls) == 4
    for padding in calls:
        assert torch.equal(padding[:, :-1], ~mask)
        assert torch.equal(padding[:, -1], mask.any(-1))
    changed = dict(context)
    changed["candidate_tokens"] = context["candidate_tokens"].clone()
    changed["candidate_tokens"][~mask] = 1000 * torch.randn_like(context["candidate_tokens"][~mask])
    other = model.decode_subset(changed, mask)
    assert torch.equal(output["params"], other["params"])
    assert torch.equal(output["internal_knots"][mask], other["internal_knots"][mask])
    for value in output.values():
        assert torch.isfinite(value).all()


def test_all_empty_extra_survivor_stack_is_an_identity_even_after_training():
    model = _model(**DEPTHS)
    _activate(model)
    context = model.encode_candidates(_points())
    mask = torch.zeros(2, 6, dtype=torch.bool)
    deep = model.decode_subset(context, mask)
    with torch.no_grad():
        for block in model.survivor_refinement_blocks:
            block.token_refinement.residual_gate.zero_()
            block.parameter_gate.zero_()
    base = model.decode_subset(context, mask)
    for name in deep:
        assert torch.equal(deep[name], base[name])


def test_zero_gates_have_learning_signal_then_attention_and_positions_receive_gradients():
    torch.manual_seed(820)
    model = _model(**DEPTHS)
    with torch.no_grad():
        model.parameter_update[-1].weight.normal_(std=.2)
        model.relocation_update.weight.normal_(std=.2)
    output = model.decode_subset(model.encode_candidates(_points()), _mask())
    loss = (output["params"].square().sum() + output["internal_knots"][_mask()].square().sum()
            + output["keep_probabilities"].square().sum())
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(PREFIXES) and name.endswith("gate"):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    model.zero_grad(set_to_none=True)
    _activate(model)
    points = _points().requires_grad_()
    output = model.decode_subset(model.encode_candidates(points), _mask())
    loss = (output["params"].square().sum() + output["internal_knots"][_mask()].square().sum()
            + output["keep_probabilities"].square().sum())
    loss.backward()
    assert torch.isfinite(points.grad).all()
    for name, parameter in model.named_parameters():
        if name.startswith(PREFIXES):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for block in model.proposal_refinement_blocks:
        assert block.cross_attention.in_proj_weight.grad.abs().sum() > 0
        assert block.position_update.weight.grad.abs().sum() > 0
    for block in model.selection_refinement_blocks:
        assert block.self_attention.in_proj_weight.grad.abs().sum() > 0
    for block in model.survivor_refinement_blocks:
        assert block.token_refinement.cross_attention.in_proj_weight.grad.abs().sum() > 0
        assert block.parameter_attention.in_proj_weight.grad.abs().sum() > 0


def test_deep_geometry_remains_ordered_and_deployment_stays_single_forward():
    torch.manual_seed(825)
    model = _model(**DEPTHS)
    _activate(model)
    with torch.no_grad():
        for block in model.proposal_refinement_blocks:
            block.position_gate.fill_(100)
            block.position_update.weight.normal_(std=100)
        model.parameter_update[-1].weight.normal_(std=100)
        model.relocation_update.weight.normal_(std=100)
    context = model.encode_candidates(_points())
    knots = context["proposal_internal_knots"]
    boundaries = torch.cat([knots.new_zeros(2, 1), knots, knots.new_ones(2, 1)], -1)
    assert (boundaries.diff(dim=-1) >= model.min_knot_gap - 2e-7).all()
    output = model.decode_subset(context, _mask())
    assert (output["params"].diff(dim=-1) >= model.min_parameter_gap - 2e-7).all()
    torch.testing.assert_close(output["warped_proposal_internal_knots"], model._warp_knots(
        knots, context["proposal_params"], output["params"],
    ), rtol=0, atol=2e-7)
    for row, mask in enumerate(_mask()):
        selected = output["internal_knots"][row, mask]
        boundaries = torch.cat([selected.new_zeros(1), selected, selected.new_ones(1)])
        assert (boundaries.diff() >= model.min_knot_gap - 2e-7).all()
    with patch.object(model, "encode_candidates", wraps=model.encode_candidates) as encode:
        with patch.object(model, "select_mask", wraps=model.select_mask) as select:
            with patch.object(model, "decode_subset", wraps=model.decode_subset) as decode:
                model.forward_deployment(_points())
    assert encode.call_count == select.call_count == decode.call_count == 1


def test_deep_checkpoint_roundtrip_restores_depth_and_outputs():
    torch.manual_seed(830)
    model = _model(**DEPTHS)
    _activate(model)
    for name, value in DEPTHS.items():
        assert model.get_config()[name] == value
    stream = BytesIO()
    torch.save(dict(objective_version=V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                    model_config=model.get_config(), model_state_dict=model.state_dict()), stream)
    stream.seek(0)
    restored, _, _ = build_model_from_checkpoint(torch.load(stream, weights_only=True))
    restored.eval()
    for name, value in model(_points()).items():
        assert torch.equal(value, restored(_points())[name])


@pytest.mark.parametrize("name", list(DEPTHS))
@pytest.mark.parametrize("value", [-1, True, .5, "2"])
def test_invalid_extra_depths_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        _model(**{name: value})
