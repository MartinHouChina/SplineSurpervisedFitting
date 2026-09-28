from __future__ import annotations

import copy
from io import BytesIO
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.checkpointing import (
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION, build_model_from_checkpoint,
)
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork
from spline_fitting.training.v16_short_ablation import (
    MODES, ablation_loss, configure_ablation,
)


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(**kwargs):
    torch.manual_seed(913)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, parameter_trust_enabled=True,
        one_shot_selection_policy="mass_topk", one_shot_adaptive_threshold=True,
        proposal_refinement_layers=1, selection_refinement_layers=1,
        survivor_refinement_layers=1, **kwargs,
    ).eval()
    # Mimic nonzero decoder weights in a trained warm-start checkpoint.
    with torch.no_grad():
        model.parameter_update[-1].weight.normal_(std=0.15)
        model.relocation_update.weight.normal_(std=0.15)
        model.proposal_parameter_trust_head[-1].weight.normal_(std=0.1)
    return model


def _points():
    t = torch.linspace(0, 1, 25)
    return torch.stack([
        torch.stack([t, .2 * torch.sin(8 * t)], -1),
        torch.stack([t, .3 * torch.cos(11 * t)], -1),
    ])


@torch.no_grad()
def _teacher(model):
    context = model.encode_candidates(_points(), mse_tolerance=1e-3)
    return {
        "mask": torch.tensor([[True, False, True, False, False, True],
                              [False, True, False, False, True, False]]),
        "params": context["proposal_params"].clone(),
        "knots": context["proposal_internal_knots"].clone(),
        "feasible": torch.tensor([True, True]),
    }


def _loss(model, teacher, mode, **kwargs):
    return ablation_loss(model, _points(), teacher, mode, 1e-3, 4e-3, **kwargs)


@pytest.mark.parametrize("mode", MODES)
def test_optimizer_preserves_every_frozen_weight_and_proposal_geometry(mode):
    model = _model()
    teacher = _teacher(model)
    metadata = configure_ablation(model, mode)
    assert metadata["optimization_uses_eval_mode"]
    named = dict(model.named_parameters())
    assert not named["proposal_parameter_trust_head.1.weight"].requires_grad
    assert not named["tolerance_embedding.0.weight"].requires_grad
    assert bool(metadata["decoder_parameter_names"]) == (mode == "geometry")
    frozen = {name: named[name].clone() for name in metadata["frozen_parameter_names"]}
    before = model.encode_candidates(_points(), mse_tolerance=1e-3)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3, weight_decay=.1,
    )
    model.train()  # The helper must restore eval mode before recomputing features.
    loss, metrics = _loss(model, teacher, mode)
    loss.backward()
    assert torch.isfinite(loss)
    assert not model.training
    assert all(not value.requires_grad and value.ndim == 0 for value in metrics.values())
    assert all(torch.isfinite(value) for value in metrics.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.keep_head.weight.grad is not None
    if mode == "geometry":
        for parameter in (model.parameter_update[-1].weight, model.relocation_update.weight):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    optimizer.step()
    for name, saved in frozen.items():
        assert named[name].grad is None
        assert torch.equal(named[name], saved), name
    after = model.encode_candidates(_points(), mse_tolerance=1e-3)
    for key in ("proposal_params", "proposal_internal_knots", "local_features"):
        assert torch.equal(after[key], before[key]), key


@pytest.mark.parametrize("kind", ["empty", "mixed", "full"])
def test_fixed_keeps_exact_proposal_geometry_for_any_mask(kind):
    model = _model()
    configure_ablation(model, "fixed")
    context = model.encode_candidates(_points())
    mask = _teacher(model)["mask"]
    if kind != "mixed":
        mask.fill_(kind == "full")
    output = model.decode_subset(context, mask)
    for actual, expected in (("params", "proposal_params"),
                             ("internal_knots", "proposal_internal_knots"),
                             ("warped_proposal_internal_knots", "proposal_internal_knots")):
        assert torch.equal(output[actual], context[expected])


@torch.no_grad()
def test_legacy_decoder_preserves_initial_behavior_and_can_move_geometry():
    model = _model()
    teacher = _teacher(model)
    saved_state = {name: value.clone() for name, value in model.state_dict().items()}
    before = model.decode_subset(model.encode_candidates(_points()), teacher["mask"])
    configure_ablation(model, "legacy")
    after = model.decode_subset(model.encode_candidates(_points()), teacher["mask"])
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved_state[name])
    # requires_grad changes can select a different PyTorch attention kernel,
    # producing float32 last-bit differences despite identical computations.
    torch.testing.assert_close(after["params"], before["params"], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(after["internal_knots"], before["internal_knots"], rtol=1e-6, atol=2e-7)
    assert not torch.equal(after["params"], after["proposal_params"])
    assert not torch.equal(after["internal_knots"], after["proposal_internal_knots"])


def test_fixed_and_legacy_have_exact_same_selector_updates_and_no_training_decodes():
    initial = _model()
    teacher = _teacher(initial)
    fixed, legacy = copy.deepcopy(initial), copy.deepcopy(initial)
    for model, mode in ((fixed, "fixed"), (legacy, "legacy")):
        configure_ablation(model, mode)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3,
        )
        with patch.object(model, "decode_subset", wraps=model.decode_subset) as decode:
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = _loss(model, teacher, mode)
                loss.backward()
                optimizer.step()
            assert decode.call_count == 0
    fixed_parameters = dict(fixed.named_parameters())
    for name, parameter in legacy.named_parameters():
        assert torch.equal(parameter, fixed_parameters[name]), name


@pytest.mark.parametrize("mode", MODES)
def test_infeasible_rows_are_excluded_even_with_nan_sentinel_geometry(mode):
    model = _model()
    teacher = _teacher(model)
    teacher["feasible"][1] = False
    configure_ablation(model, mode)
    expected, _ = _loss(model, teacher, mode)
    teacher["mask"][1] = ~teacher["mask"][1]
    teacher["params"][1] = float("nan")
    teacher["knots"][1] = float("nan")
    actual, metrics = _loss(model, teacher, mode)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert metrics["valid_count"] == 1
    assert metrics["feasible_fraction"] == .5
    actual.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("mode", MODES)
def test_all_infeasible_returns_differentiable_zero_without_decoding(mode):
    model = _model()
    teacher = _teacher(model)
    teacher["feasible"][:] = False
    teacher["params"][:] = float("nan")
    teacher["knots"][:] = float("nan")
    configure_ablation(model, mode)
    with patch.object(model, "decode_subset", wraps=model.decode_subset) as decode:
        loss, metrics = _loss(model, teacher, mode)
        assert decode.call_count == 0
    assert loss.requires_grad and loss == 0
    assert all(value == 0 for value in metrics.values())
    loss.backward()
    assert model.keep_head.weight.grad is not None
    assert torch.equal(model.keep_head.weight.grad, torch.zeros_like(model.keep_head.weight))


def test_geometry_refits_both_masks_without_giving_hard_mask_gradients_to_keep_head():
    model = _model()
    teacher = _teacher(model)
    configure_ablation(model, "geometry")
    context = model.encode_candidates(_points(), mse_tolerance=1e-3)
    current_mask = model.select_mask(context)
    assert not torch.equal(current_mask, teacher["mask"])
    with patch.object(model, "decode_subset", wraps=model.decode_subset) as decode:
        loss, metrics = _loss(model, teacher, "geometry")
        assert decode.call_count == 2
        assert torch.equal(decode.call_args_list[0].args[1], teacher["mask"])
        assert torch.equal(decode.call_args_list[1].args[1], current_mask)
    # Remove the selector loss: decoder/refit terms are real differentiable
    # losses but cannot claim to differentiate a hard boolean Top-K operation.
    selector_loss, _ = _loss_for_selector(model, teacher)
    gradients = torch.autograd.grad(
        loss - selector_loss,
        [model.keep_head.weight, model.selection_blocks[0].self_attention.in_proj_weight],
        allow_unused=True,
    )
    assert gradients[0] is None or torch.allclose(gradients[0], torch.zeros_like(gradients[0]), atol=1e-7)
    assert gradients[1] is not None and gradients[1].abs().sum() > 0
    assert metrics["teacher_fit_loss"] > 0 and metrics["current_fit_loss"] > 0


def _loss_for_selector(model, teacher):
    # Same selector objective without reconfiguring the actual geometry arm.
    context = model.encode_candidates(_points(), mse_tolerance=1e-3)
    logits = context["keep_logits"]
    count_target = (teacher["mask"].sum(-1) - .25).clamp_min(0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, teacher["mask"].float())
    count = ((context["one_shot_requested_count_score"] - count_target) / logits.shape[1]).square().mean()
    return bce + .5 * count, context


def test_fixed_checkpoint_strict_roundtrip_keeps_exact_geometry_and_no_new_state_keys():
    model = _model()
    keys = tuple(model.state_dict())
    configure_ablation(model, "fixed")
    assert model.get_config()["subset_geometry_mode"] == "anchored"
    assert model.get_config()["subset_geometry_residual_scale"] == 0.0
    buffer = BytesIO()
    torch.save({
        "objective_version": V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "model_config": model.get_config(), "model_state_dict": model.state_dict(),
    }, buffer)
    buffer.seek(0)
    restored, config, _ = build_model_from_checkpoint(torch.load(buffer, weights_only=False))
    restored.eval()
    assert config["subset_geometry_mode"] == "anchored"
    assert tuple(restored.state_dict()) == keys
    output = restored(_points())
    assert torch.equal(output["params"], output["proposal_params"])
    assert torch.equal(output["internal_knots"], output["proposal_internal_knots"])
    assert torch.equal(output["params"], model(_points())["params"])


def test_fixed_rejects_coupled_proposals_to_protect_cached_teacher():
    model = _model(coupled_proposal_steps=1, subset_geometry_mode="anchored")
    with pytest.raises(ValueError, match="coupled proposal"):
        configure_ablation(model, "fixed")


@pytest.mark.parametrize("field", ["count_weight", "parameter_weight", "knot_weight", "fit_weight"])
def test_loss_rejects_invalid_weights(field):
    model = _model()
    teacher = _teacher(model)
    configure_ablation(model, "fixed")
    with pytest.raises(ValueError, match=field):
        _loss(model, teacher, "fixed", **{field: float("nan")})
