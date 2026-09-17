from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from spline_fitting.losses.v16_subset_loss import V16SubsetLoss
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


def _network(**options):
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True, **options,
    ).double()


def _points():
    t = torch.linspace(0, 1, 32, dtype=torch.float64)
    return torch.stack([
        torch.stack([t, (11 * t).sin() + 0.1 * (31 * t).cos()], -1),
        torch.stack([t, (8 * t).cos()], -1),
    ])


def _learned_interaction(**options):
    torch.manual_seed(419)
    model = _network(keep_state_interaction=True, **options)
    # A migrated model starts with zero state vectors and identity decoder
    # heads.  Exercise the interaction after those parameters have learned.
    with torch.no_grad():
        model.keep_state_embedding.weight.normal_(std=0.15)
        model.parameter_update[-1].weight.normal_(std=0.05)
        model.relocation_update.weight.normal_(std=0.05)
    return model


def _mask():
    return torch.tensor([
        [True, False, True, False, False, True],
        [False, True, False, True, False, False],
    ])


def test_zero_state_migration_preserves_every_existing_forward_value():
    torch.manual_seed(43)
    legacy = _network()
    enhanced = _network(keep_state_interaction=True)
    incompatible = enhanced.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.missing_keys == ["keep_state_embedding.weight"]
    assert incompatible.unexpected_keys == []
    assert torch.count_nonzero(enhanced.keep_state_embedding.weight) == 0

    old_context = legacy.encode_candidates(_points())
    context = enhanced.encode_candidates(_points())
    for name, value in old_context.items():
        assert torch.equal(value, context[name]), name
    for mask in (_mask(), torch.zeros_like(_mask()), torch.ones_like(_mask())):
        old_output = legacy.decode_subset(old_context, mask)
        output = enhanced.decode_subset(context, mask)
        for name, value in old_output.items():
            assert torch.equal(value, output[name]), name

    old_config = legacy.get_config()
    old_config.pop("keep_state_interaction")
    restored_legacy = V16CandidateSelectionNetwork(**old_config).double()
    restored_legacy.load_state_dict(legacy.state_dict(), strict=True)
    assert restored_legacy.keep_state_interaction is False
    assert not hasattr(restored_legacy, "keep_state_embedding")
    restored = V16CandidateSelectionNetwork(**enhanced.get_config()).double()
    restored.load_state_dict(enhanced.state_dict(), strict=True)
    assert restored.keep_state_interaction is True


def test_preview_embeds_per_candidate_keep_confidence_before_selection_attention():
    model = _learned_interaction()
    legacy = _network()
    shared_weights = {
        name: value for name, value in model.state_dict().items()
        if name != "keep_state_embedding.weight"
    }
    legacy.load_state_dict(shared_weights, strict=True)
    with patch.object(
        legacy.selection_blocks[0], "forward", wraps=legacy.selection_blocks[0].forward,
    ) as baseline_attention:
        legacy.encode_candidates(_points())
    with patch.object(
        model.selection_blocks[0], "forward", wraps=model.selection_blocks[0].forward,
    ) as attention:
        context = model.encode_candidates(_points())
    old_tokens = baseline_attention.call_args.args[0]
    tokens = attention.call_args.args[0]
    p = context["preliminary_keep_probabilities"]
    dropped, retained = model.keep_state_embedding.weight.unbind(0)
    expected = (1 - p).unsqueeze(-1) * dropped + p.unsqueeze(-1) * retained
    torch.testing.assert_close(tokens - old_tokens, expected)
    assert (tokens - old_tokens).abs().sum() > 0
    with patch.object(model, "select_mask", wraps=model.select_mask) as select:
        with patch.object(model, "decode_subset", wraps=model.decode_subset) as decode:
            model.forward_deployment(_points())
    assert select.call_count == decode.call_count == 1


@pytest.mark.parametrize("coupled", [False, True])
def test_keep_interaction_preserves_structure_and_count_beta_routing(coupled):
    model = _learned_interaction(count_structure_coupling=coupled)
    context = model.encode_candidates(_points())
    beta_bias = model.adaptive_threshold_head[-1].bias
    structure_loss = context["structure_keep_logits"].square().sum()
    beta_gradient, embedding_gradient = torch.autograd.grad(
        structure_loss, (beta_bias, model.keep_state_embedding.weight),
        retain_graph=True, allow_unused=True,
    )
    assert beta_gradient is None
    assert embedding_gradient is not None
    assert embedding_gradient.abs().sum() > 0
    count_loss = context["count_calibration_requested_count_score"].sum()
    keep_gradient, embedding_gradient, beta_gradient = torch.autograd.grad(
        count_loss,
        (model.keep_head.weight, model.keep_state_embedding.weight, beta_bias),
        allow_unused=True,
    )
    assert beta_gradient is not None and beta_gradient.abs().sum() > 0
    for gradient in (keep_gradient, embedding_gradient):
        if coupled:
            assert gradient is not None and gradient.abs().sum() > 0
        else:
            assert gradient is None


def test_numerical_fit_reaches_keep_scores_through_continuous_state_features():
    model = _learned_interaction()
    points = _points()
    context = model.encode_candidates(points)
    mask = _mask()
    output = model.decode_subset(context, mask)
    fit_loss = V16SubsetLoss()._fit(
        output["params"], output["internal_knots"], mask, points, model.degree,
    ).sum()
    keep_gradient, state_gradient, beta_gradient = torch.autograd.grad(
        fit_loss,
        (
            model.keep_head.weight, model.keep_state_embedding.weight,
            model.adaptive_threshold_head[-1].bias,
        ),
        allow_unused=True,
    )
    assert torch.isfinite(fit_loss)
    for gradient in (keep_gradient, state_gradient):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert beta_gradient is None
    assert mask.dtype == torch.bool and not mask.requires_grad


def test_decoder_probability_gradient_matches_real_forward_finite_difference():
    model = _learned_interaction()
    points, mask = _points(), _mask()
    context = {
        name: value.detach() for name, value in model.encode_candidates(points).items()
    }
    p = context["structure_keep_probabilities"].clone().requires_grad_()
    context["structure_keep_probabilities"] = p

    def fit(probabilities):
        output = model.decode_subset(
            dict(context, structure_keep_probabilities=probabilities), mask,
        )
        return V16SubsetLoss()._fit(
            output["params"], output["internal_knots"], mask, points, model.degree,
        ).sum()

    gradient = torch.autograd.grad(fit(p), p)[0]
    assert torch.isfinite(gradient).all() and gradient[mask].abs().sum() > 0
    # Unselected probabilities are not surrogate mask gates: after fixing the
    # encoded tokens, they have no influence on the selected-only decoder.
    assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))
    index = gradient.abs().reshape(-1).argmax().item()
    step = torch.zeros_like(p).reshape(-1)
    step[index] = 1e-5
    step = step.reshape_as(p)
    with torch.no_grad():
        numerical = (fit(p + step) - fit(p - step)) / (2e-5)
    torch.testing.assert_close(
        gradient.reshape(-1)[index], numerical, rtol=0.01, atol=1e-9,
    )


def test_state_interaction_keeps_discarded_keys_invisible_and_empty_subsets_finite():
    model = _learned_interaction()
    context = model.encode_candidates(_points())
    mask = _mask()
    mask[1] = False
    with patch.object(
        model.survivor_attention, "forward", wraps=model.survivor_attention.forward,
    ) as attention:
        output = model.decode_subset(context, mask)
    padding = attention.call_args.kwargs["key_padding_mask"]
    assert torch.equal(padding[:, :-1], ~mask)
    assert torch.equal(padding[:, -1], torch.tensor([True, False]))
    changed_tokens = context["candidate_tokens"].clone()
    changed_tokens[~mask] += 1000
    changed = model.decode_subset(dict(context, candidate_tokens=changed_tokens), mask)
    assert torch.equal(changed["params"], output["params"])
    assert torch.equal(changed["internal_knots"][mask], output["internal_knots"][mask])
    for value in output.values():
        assert torch.isfinite(value).all()


def test_keep_state_interaction_requires_boolean():
    with pytest.raises(ValueError, match="keep_state_interaction"):
        _network(keep_state_interaction=1)
