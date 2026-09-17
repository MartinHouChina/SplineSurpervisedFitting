from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spline_fitting.models.candidate_knot_head import CandidateKnotHead


def inputs():
    torch.manual_seed(18)
    return torch.randn(2, 16), torch.randn(2, 40, 16), torch.linspace(0, 1, 40).expand(2, -1)


def head(limit=0.0):
    return CandidateKnotHead(
        hidden_dim=16, num_candidates=12, min_gap=1e-4,
        attention_heads=2, position_parameterization="bounded_anchor_residual",
        global_interval_residual_limit=limit,
    )


def test_zero_warp_preserves_existing_checkpoint_predictions_exactly():
    original, extended = head(), head(1.0)
    mismatch = extended.load_state_dict(original.state_dict(), strict=False)
    assert set(mismatch.missing_keys) == {
        "interval_warp_score.weight", "interval_warp_score.bias",
    }
    assert not mismatch.unexpected_keys
    original.eval()
    extended.eval()
    args = inputs()
    before, after = original(*args), extended(*args)
    for name in ("candidate_knots", "candidate_tokens", "candidate_intervals"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_global_warp_can_leave_anchor_cells_without_losing_order_or_domain():
    model = head(1.0)

    def prescribed_scores(module, args, output):
        values = torch.linspace(-5, 5, 13, device=output.device)
        return values.view(1, 13, 1).expand_as(output)

    hook = model.interval_warp_score.register_forward_hook(prescribed_scores)
    try:
        output = model(*inputs())
    finally:
        hook.remove()
    knots = output["candidate_knots"]
    boundaries = torch.cat([knots.new_zeros(2, 1), knots, knots.new_ones(2, 1)], -1)
    assert bool((boundaries.diff(dim=-1) >= model.min_gap - 1e-6).all())
    half_cell = 0.5 * (1 / 13 - model.min_gap)
    assert float(output["candidate_position_residual"].detach().abs().amax()) > 3 * half_cell
    torch.testing.assert_close(output["candidate_intervals"].sum(-1), torch.ones(2))


def test_warp_receives_finite_geometry_gradient_at_identity():
    model = head(1.0)
    output = model(*inputs())
    target = output["candidate_position_anchors"].square()
    (output["candidate_knots"] - target).square().mean().backward()
    gradient = model.interval_warp_score.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_invalid_warp_limits_fail_early(limit):
    with pytest.raises(ValueError, match="global_interval_residual_limit"):
        head(limit)


def test_teacher_cache_identity_includes_global_warp_configuration():
    from spline_fitting.models.v16_network import V16CandidateSelectionNetwork
    from spline_fitting.training.v16_feasible_teacher import _proposal_fingerprint

    options = dict(hidden_dim=16, encoder_layers=1, max_internal_knots=6, attention_heads=2)
    first = V16CandidateSelectionNetwork(**options, proposal_global_warp_limit=1.0)
    second = V16CandidateSelectionNetwork(**options, proposal_global_warp_limit=2.0)
    second.load_state_dict(first.state_dict())
    assert _proposal_fingerprint(first) != _proposal_fingerprint(second)
