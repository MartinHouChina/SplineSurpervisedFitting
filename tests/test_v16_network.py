from __future__ import annotations

import io
from unittest.mock import patch

import pytest
import torch

from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture
def network():
    torch.manual_seed(19)
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
    )


@pytest.fixture
def points():
    t = torch.linspace(0, 1, 20)
    return torch.stack([torch.stack([t, (3 * t).sin()], -1),
                        torch.stack([t, (5 * t).cos()], -1)])


def test_new_model_default_uses_56_internal_candidates():
    model = V16CandidateSelectionNetwork()
    assert model.max_internal_knots == 56
    assert model.candidate_head.interval_queries.shape[0] == 57
    assert model.get_config()["max_internal_knots"] == 56


@pytest.mark.parametrize("kind", ["empty", "full", "mixed"])
def test_subset_outputs_strict_and_finite(network, points, kind):
    context = network.encode_candidates(points)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    if kind == "full":
        mask[:] = True
    if kind == "mixed":
        mask[0, [0, 1, 5]] = True
        mask[1, [2]] = True
    output = network.decode_subset(context, mask)
    assert torch.equal(output["learned_keep_mask"], mask)
    assert output["internal_knots"].shape == mask.shape
    assert torch.equal(output["predicted_knot_count"], mask.sum(-1))
    for value in output.values():
        assert torch.isfinite(value).all()
    assert torch.equal(output["params"][:, 0], torch.zeros(2))
    assert torch.equal(output["params"][:, -1], torch.ones(2))
    assert (output["params"].diff(dim=-1) >= network.min_parameter_gap - 1e-7).all()
    for row in range(2):
        chosen = output["internal_knots"][row, mask[row]]
        boundaries = torch.cat([torch.zeros(1), chosen, torch.ones(1)])
        assert (boundaries.diff() >= network.min_knot_gap - 1e-7).all()


def test_tolerance_is_conditioning_not_postfit_search(network, points):
    context = network.encode_candidates(points, torch.tensor([1e-5, 1e-4]))
    other = network.encode_candidates(points, 1e-6)
    assert not torch.allclose(context["tolerance_features"], other["tolerance_features"])
    assert not torch.allclose(context["keep_logits"], other["keep_logits"])
    assert torch.equal(network.select_mask(context), context["keep_probabilities"] >= 0.5)
    assert all(value.shape[0] == points.shape[0] for value in context.values())


def test_adaptive_mass_topk_uses_probability_mass_not_fixed_half_threshold(points):
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True,
        one_shot_safety_sigma=0.0,
        one_shot_safety_knots=0,
        min_selected_knots=2,
    )
    probabilities = torch.tensor([
        [0.40, 0.40, 0.40, 0.10, 0.10, 0.10],
        [0.90, 0.80, 0.20, 0.10, 0.05, 0.05],
    ])
    context = {
        "keep_probabilities": probabilities,
        "proposal_internal_knots": torch.linspace(0.1, 0.9, 6).repeat(2, 1),
        "one_shot_requested_count_score": probabilities.sum(-1),
    }
    mask = model.select_mask(context)
    # Both masses are 1.5/2.1, hence ceil -> 2/3.  A literal p>=0.5 rule
    # would instead retain 0/2, which is the v16 collapse being removed.
    assert torch.equal(mask.sum(-1), torch.tensor([2, 3]))
    assert torch.equal(mask[0], torch.tensor([True, True, False, False, False, False]))


def test_count_conditioned_selection_and_teacher_repair_share_coverage_rule():
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True,
        one_shot_coverage_bins=2,
        min_selected_knots=2,
    )
    context = {
        "keep_probabilities": torch.tensor([[0.9, 0.8, 0.7, 0.2, 0.1, 0.05]]),
        "proposal_internal_knots": torch.tensor([[0.1, 0.2, 0.3, 0.7, 0.8, 0.9]]),
    }
    selected = model.select_mask_at_count(context, torch.tensor([2]))
    assert selected.sum() == 2
    assert selected[0, :3].any() and selected[0, 3:].any()

    invalid_teacher = torch.tensor([[True, True, False, False, False, False]])
    repaired = model.constrain_selection_mask(context, invalid_teacher)
    assert repaired[0, :3].any() and repaired[0, 3:].any()
    assert torch.equal(repaired & invalid_teacher, invalid_teacher)


def test_empty_coverage_bins_do_not_erase_an_existing_anchor():
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True,
        one_shot_coverage_bins=4,
    )
    context = {
        "keep_probabilities": torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]),
        "proposal_internal_knots": torch.tensor(
            [[0.02, 0.04, 0.06, 0.08, 0.10, 0.12]]
        ),
    }
    anchors = model._coverage_anchors(context)
    assert anchors.sum() == 1
    assert anchors[0, 0]


def test_count_conditioned_selection_rejects_invalid_count_tensor(points):
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_selection_policy="mass_topk",
    )
    context = model.encode_candidates(points)
    with pytest.raises(ValueError, match="shape"):
        model.select_mask_at_count(context, torch.ones(2, 1))
    with pytest.raises(ValueError, match="tensor"):
        model.select_mask_at_count(context, 2)


def test_adaptive_beta_is_curve_level_and_centers_raw_importance(points):
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True,
    )
    context = model.encode_candidates(points)
    assert context["adaptive_keep_threshold"].shape == (2,)
    torch.testing.assert_close(
        context["centered_keep_importance"].mean(-1), torch.zeros(2), atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(
        context["keep_logits"],
        context["centered_keep_importance"]
        - context["adaptive_keep_threshold"].unsqueeze(-1),
    )


def test_initial_keep_fraction_controls_untrained_adaptive_beta_prior():
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True,
        initial_keep_fraction=0.22,
    )
    expected_beta = torch.log(torch.tensor(0.78 / 0.22))
    torch.testing.assert_close(
        model.adaptive_threshold_head[-1].bias.detach()[0], expected_beta,
    )
    assert model.get_config()["initial_keep_fraction"] == pytest.approx(0.22)


def test_runtime_selection_safety_is_validated_and_serialized():
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_safety_sigma=0.25, one_shot_safety_knots=2,
    )
    model.set_selection_safety(sigma=0.05, knots=0)
    assert model.one_shot_safety_sigma == pytest.approx(0.05)
    assert model.one_shot_safety_knots == 0
    assert model.get_config()["one_shot_safety_sigma"] == pytest.approx(0.05)
    assert model.get_config()["one_shot_safety_knots"] == 0
    with pytest.raises(ValueError, match="sigma"):
        model.set_selection_safety(sigma=-0.1, knots=0)
    with pytest.raises(ValueError, match="knots"):
        model.set_selection_safety(sigma=0.1, knots=True)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf"), [1e-5], [[1e-5], [1e-5]]])
def test_invalid_tolerance_rejected(network, points, tolerance):
    with pytest.raises(ValueError, match="mse_tolerance"):
        network.encode_candidates(points, tolerance)


def test_forward_exactly_one_encode_select_and_decode(network, points):
    with patch.object(network, "encode_candidates", wraps=network.encode_candidates) as encode:
        with patch.object(network, "select_mask", wraps=network.select_mask) as select:
            with patch.object(network, "decode_subset", wraps=network.decode_subset) as decode:
                network.forward_deployment(points)
    assert encode.call_count == select.call_count == decode.call_count == 1


def test_gradients_through_subset_positions_and_parameters(network, points):
    points = points.clone().requires_grad_(True)
    context = network.encode_candidates(points)
    mask = torch.tensor([[True, False, True, False, False, True],
                         [False, True, False, True, False, False]])
    output = network.decode_subset(context, mask)
    loss = output["params"].square().sum() + output["internal_knots"][mask].square().sum()
    loss.backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    for module in [network.candidate_head.interval_score,
                   network.parameter_head.mlp[-1],
                   network.parameter_update[-1], network.relocation_update]:
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    # Correctly, fitting a fixed discrete subset supplies no direct keep-logit gradient.
    assert network.keep_head.weight.grad is None


def test_actual_attention_memory_excludes_discarded_tokens(network, points):
    context = network.encode_candidates(points)
    mask = torch.tensor([[True, False, True, False, False, True], [False] * 6])
    with patch.object(network.survivor_attention, "forward", wraps=network.survivor_attention.forward) as attention:
        network.decode_subset(context, mask)
    padding = attention.call_args.kwargs["key_padding_mask"]
    assert torch.equal(padding[:, :-1], ~mask)
    assert torch.equal(padding[:, -1], torch.tensor([True, False]))


def test_relocation_uses_removed_spans_and_keeps_slot_identity(network, points):
    context = network.encode_candidates(points)
    mask = torch.tensor([[True, True, False, False, False, False],
                         [False, False, False, False, True, True]])
    output = network.decode_subset(context, mask)
    delta = (output["internal_knots"] - context["proposal_internal_knots"]).abs()
    # Some relocation can exceed the original half-cell bound without crossing.
    assert delta[mask].max() > 0.5 / 7
    assert torch.equal(output["internal_knots"][~mask], output["warped_proposal_internal_knots"][~mask])


def test_parameter_update_warps_knots_before_relocation(network, points):
    context = network.encode_candidates(points)
    with torch.no_grad():
        network.parameter_update[-1].weight.normal_(std=0.1)
    mask = torch.ones(2, 6, dtype=torch.bool)
    output = network.decode_subset(context, mask)
    assert not torch.allclose(output["params"], context["proposal_params"])
    expected = network._warp_knots(context["proposal_internal_knots"],
                                   context["proposal_params"], output["params"])
    assert torch.allclose(output["warped_proposal_internal_knots"], expected)


def test_large_relocation_stays_ordered(network, points):
    with torch.no_grad():
        network.relocation_update.weight.normal_(std=50)
    context = network.encode_candidates(points)
    mask = torch.ones(2, 6, dtype=torch.bool)
    output = network.decode_subset(context, mask)
    boundaries = torch.cat([torch.zeros(2, 1), output["internal_knots"], torch.ones(2, 1)], -1)
    assert (boundaries.diff(dim=-1) >= network.min_knot_gap - 1e-7).all()


def test_state_roundtrip_and_no_legacy_modules(network, points):
    buffer = io.BytesIO()
    torch.save({"config": network.get_config(), "state": network.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    restored = V16CandidateSelectionNetwork(**payload["config"])
    restored.load_state_dict(payload["state"], strict=True)
    assert torch.equal(network(points)["internal_knots"], restored(points)["internal_knots"])
    assert not any("pilot" in name or "count_head" in name or "surrogate" in name
                   for name in network.state_dict())


def test_repeated_points_remain_strict(network):
    output = network(torch.zeros(2, 20, 2))
    assert (output["params"].diff(dim=-1) >= network.min_parameter_gap - 1e-7).all()


@pytest.mark.parametrize("options", [
    {"mse_tolerance": 0}, {"min_knot_gap": 0.5}, {"min_parameter_gap": -1},
    {"relocation_blend": float("nan")}, {"hidden_dim": 15},
    {"max_internal_knots": 0}, {"selector_layers": 0}, {"structure_mode": "legacy"},
    {"initial_keep_fraction": 0.0}, {"initial_keep_fraction": 1.0},
])
def test_constructor_validation(options):
    with pytest.raises(ValueError):
        V16CandidateSelectionNetwork(**options)


def test_mask_dtype_and_geometry_validation(network, points):
    context = network.encode_candidates(points)
    with pytest.raises(ValueError, match="boolean"):
        network.decode_subset(context, torch.ones(2, 6))
    with pytest.raises(ValueError, match="shape"):
        network.encode_candidates(points[..., :1])
    with pytest.raises(ValueError, match="dtype"):
        network.encode_candidates(points.double())
    with pytest.raises(ValueError, match="finite"):
        network.encode_candidates(points * float("nan"))


def test_float64_preserves_feasibility(network, points):
    network = network.double()
    output = network(points.double())
    assert output["params"].dtype == torch.float64
    assert (output["params"].diff(dim=-1) >= network.min_parameter_gap - 1e-12).all()
