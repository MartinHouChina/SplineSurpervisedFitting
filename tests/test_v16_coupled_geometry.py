"""Actual coordinate feedback, backward compatibility and gradient checks."""
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


def model(**options):
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, attention_heads=2, selector_layers=1,
        max_internal_knots=8, subset_geometry_mode="anchored",
        one_shot_selection_policy="mass_topk", one_shot_adaptive_threshold=True,
        parameter_trust_enabled=True, **options,
    ).eval()


def points():
    t = torch.linspace(0, 1, 32)
    return torch.stack([torch.stack([t, .3 * (7 * t).sin()], -1),
                        torch.stack([t.square(), .2 * (11 * t).cos()], -1)])


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_opt_in_zero_gate_preserves_existing_geometry_and_mask():
    torch.manual_seed(432)
    old = model()
    torch.manual_seed(432)
    new = model(coupled_proposal_steps=2, coupled_subset_steps=2)
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    a, b = old(points()), new(points())
    for key in ("params", "internal_knots", "learned_keep_mask", "keep_probabilities",
                "proposal_params", "proposal_internal_knots"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def activate(net):
    with torch.no_grad():
        for name, value in net.named_parameters():
            if name.startswith("coupled_") and name.endswith("gate"):
                value.fill_(.6)


def test_each_round_receives_updated_numeric_parameters_knots_and_point_features():
    net = model(coupled_proposal_steps=2, coupled_subset_steps=2)
    activate(net)
    inputs = []
    hooks = [block.register_forward_pre_hook(
        lambda _, args: inputs.append(tuple(t.detach().clone() for t in args[:4]))
    ) for block in net.coupled_subset_blocks]
    context = net.encode_candidates(points())
    mask = torch.arange(8).expand(2, -1).remainder(2) == 0
    out = net.decode_subset(context, mask)
    for hook in hooks:
        hook.remove()
    assert len(inputs) == 2
    assert not torch.equal(inputs[0][1], inputs[1][1])  # point evidence
    assert not torch.equal(inputs[0][2][mask], inputs[1][2][mask])
    assert not torch.equal(inputs[0][3], inputs[1][3])
    assert not torch.equal(out["params"], inputs[-1][3])
    assert torch.equal(out["learned_keep_mask"], mask)


@pytest.mark.parametrize("kind", ["full", "mixed", "empty"])
def test_extreme_steps_preserve_order_endpoints_and_finite_gradients(kind):
    net = model(coupled_proposal_steps=2, coupled_subset_steps=3, parameter_chord_blend=.5)
    activate(net)
    with torch.no_grad():
        for name, value in net.named_parameters():
            if name.startswith("coupled_") and ("position_step" in name or "parameter_step" in name):
                value.mul_(200)
    x = points().requires_grad_()
    context = net.encode_candidates(x)
    mask = torch.ones(2, 8, dtype=torch.bool)
    if kind == "mixed":
        mask[:, 1::2] = False
    elif kind == "empty":
        mask[:] = False
    out = net.decode_subset(context, mask)
    t = out["params"]
    assert torch.equal(t[:, 0], torch.zeros(2))
    assert torch.equal(t[:, -1], torch.ones(2))
    assert (t.diff(dim=-1) >= net.min_parameter_gap - 2e-7).all()
    for row, keep in zip(out["internal_knots"], mask):
        u = torch.cat([row.new_zeros(1), row[keep], row.new_ones(1)])
        assert (u.diff() >= net.min_knot_gap - 2e-7).all()
    (t.square().sum() + out["internal_knots"][mask].square().sum()).backward()
    assert torch.isfinite(x.grad).all()


def test_zero_coordinate_gate_gets_initial_parameter_gradient():
    net = model(coupled_proposal_steps=1, coupled_subset_steps=1)
    out = net(points())
    out["params"].square().sum().backward()
    for block in (*net.coupled_proposal_blocks, *net.coupled_subset_blocks):
        assert block.coordinate_gate.grad is not None
        assert torch.isfinite(block.coordinate_gate.grad)
        assert block.coordinate_gate.grad.abs() > 0


def test_dropped_tokens_do_not_reenter_coupled_decoder_memory():
    net = model(coupled_subset_steps=2)
    activate(net)
    context = net.encode_candidates(points())
    mask = torch.arange(8).expand(2, -1).remainder(2) == 0
    baseline = net.decode_subset(context, mask)
    changed = dict(context)
    changed["candidate_tokens"] = context["candidate_tokens"].clone()
    changed["candidate_tokens"][~mask] = 1e4
    other = net.decode_subset(changed, mask)
    torch.testing.assert_close(baseline["params"], other["params"], rtol=0, atol=0)
    torch.testing.assert_close(baseline["internal_knots"][mask], other["internal_knots"][mask], rtol=0, atol=0)


def test_fixed_chord_anchor_is_exact_and_serializable():
    net = model(parameter_chord_blend=1.)
    context = net.encode_candidates(points())
    torch.testing.assert_close(context["proposal_params"], context["chord_params"], rtol=0, atol=1e-8)
    restored = V16CandidateSelectionNetwork(**net.get_config()).eval()
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(net(points())["params"], restored(points())["params"], rtol=0, atol=0)


@pytest.mark.parametrize("options", [dict(coupled_proposal_steps=-1),
                                    dict(coupled_subset_steps=True),
                                    dict(parameter_chord_blend=1.1),
                                    dict(parameter_chord_blend=float("nan"))])
def test_invalid_coupling_configuration_rejected(options):
    with pytest.raises(ValueError):
        model(**options)
