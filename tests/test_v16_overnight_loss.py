from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.losses.deployment_bspline_loss import differentiable_hard_gated_bspline_fit
from spline_fitting.losses.v16_subset_loss import V16SubsetLoss
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def points():
    t = torch.linspace(0, 1, 25)
    return torch.stack([t, 0.2 * torch.sin(8 * t)], dim=-1)[None]


def network():
    torch.manual_seed(71)
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=5,
        attention_heads=2, selector_layers=1, parameter_trust_enabled=True,
    )


@pytest.mark.parametrize("kwargs", [
    {"parameter_counterfactual_weight": -1},
    {"parameter_counterfactual_weight": float("nan")},
    {"local_fit_weight": float("inf")},
    {"local_fit_weight": -1},
    {"proposal_ordered_weight": -1},
    {"teacher_geometry_candidates": -1},
    {"teacher_geometry_candidates": 1.2},
    {"teacher_geometry_candidates": True},
])
def test_new_loss_options_validate(kwargs):
    with pytest.raises(ValueError):
        V16SubsetLoss(**kwargs)


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_disabled_mechanisms_have_exact_zero_metrics_and_no_extra_decodes(stage):
    model = network()
    options = dict(policy_samples=2, counterfactual_edits=0, teacher_prefix_search_steps=1)
    default = V16SubsetLoss(**options)
    explicit = V16SubsetLoss(
        **options, parameter_counterfactual_weight=0,
        teacher_geometry_candidates=0, local_fit_weight=0, proposal_ordered_weight=0,
    )
    with patch.object(model, "decode_subset", wraps=model.decode_subset) as decoder:
        torch.manual_seed(3)
        old_loss, old_metrics = default(model, points(), stage=stage)
        old_calls = decoder.call_count
        decoder.reset_mock()
        torch.manual_seed(3)
        new_loss, new_metrics = explicit(model, points(), stage=stage)
        assert decoder.call_count == old_calls
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    for name in (
        "proposal_ordered_loss", "parameter_counterfactual_loss",
        "parameter_counterfactual_refits", "teacher_geometry_evaluations",
        "teacher_geometry_improved_fraction", "local_fit_loss", "local_max_region_mse",
    ):
        assert new_metrics[name].ndim == 0
        assert new_metrics[name] == old_metrics[name] == 0


def test_refit_exposes_double_euclidean_per_point_residuals_without_changing_mse():
    observed = points().double()
    parameters = torch.linspace(0, 1, observed.shape[1], dtype=torch.float64)[None]
    knots = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    mask = torch.tensor([[True, False]])
    result = differentiable_hard_gated_bspline_fit(
        parameters, observed, knots, mask, smoothness_weight=0, solver_jitter=1e-10,
    )
    reference = refit_bspline_control_points(
        parameters[0], observed[0], knots[0, mask[0]], smoothness_weight=0,
        control_ridge=0, interpolate_endpoints=True,
    )
    residual = result["per_point_squared_error"]
    assert residual.shape == parameters.shape and residual.dtype == torch.float64
    torch.testing.assert_close(residual.mean(-1), result["per_sample_mse"], rtol=0, atol=0)
    torch.testing.assert_close(
        residual[0], (reference.reconstructed_points - observed[0]).square().sum(-1),
        atol=1e-10, rtol=1e-6,
    )
    torch.testing.assert_close(residual[:, [0, -1]], torch.zeros_like(residual[:, [0, -1]]))


def test_counterfactual_warps_relocated_knots_preserves_mask_and_detaches_reference():
    objective = V16SubsetLoss(parameter_counterfactual_weight=1)
    observed = points().double()
    chord = torch.linspace(0, 1, observed.shape[1], dtype=torch.float64)[None].requires_grad_()
    parameters = chord.detach().pow(1.4).requires_grad_()
    knots = torch.tensor([[0.23, 0.65]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False]])
    original_mask = mask.clone()
    tolerance = torch.tensor([1e-4], dtype=torch.float64)
    warped = objective._warp_knots_to_target_parameterization(knots, parameters, chord)
    expected_reference = objective._fit(chord, warped, mask, observed, 3).detach()
    current_mse = (expected_reference * 2 + 1e-5).requires_grad_()
    with patch.object(objective, "_fit", wraps=objective._fit) as refit:
        penalty, reference = objective._parameter_counterfactual(
            parameters, knots, mask, chord, current_mse, observed, 3, tolerance,
        )
    assert refit.call_count == 1
    torch.testing.assert_close(refit.call_args.args[1], warped)
    assert torch.equal(refit.call_args.args[2], original_mask)
    assert torch.equal(mask, original_mask)
    assert not reference.requires_grad
    torch.testing.assert_close(reference, expected_reference)
    grads = torch.autograd.grad(penalty, (current_mse, chord, parameters, knots), allow_unused=True)
    assert grads[0].item() > 0
    assert all(gradient is None for gradient in grads[1:])
    no_worse, _ = objective._parameter_counterfactual(
        parameters, knots, mask, chord, reference * 0.5, observed, 3, tolerance,
    )
    assert no_worse == 0


def test_counterfactual_empty_subset_is_finite_and_keeps_zero_k():
    model = network()
    observed = points()
    context = model.encode_candidates(observed)
    mask = torch.zeros_like(context["proposal_internal_knots"], dtype=torch.bool)
    decoded = model.decode_subset(context, mask)
    objective = V16SubsetLoss(parameter_counterfactual_weight=1)
    mse = objective._fit(decoded["params"], decoded["internal_knots"], mask, observed, 3)
    penalty, reference = objective._parameter_counterfactual(
        decoded["params"], decoded["internal_knots"], mask,
        context["chord_params"], mse, observed, 3, torch.tensor([1e-4]),
    )
    assert torch.isfinite(penalty) and torch.isfinite(reference).all()
    assert mask.sum() == 0


def test_local_regions_are_normalized_and_emphasize_small_high_error_regions():
    objective = V16SubsetLoss(tail_weight=0)
    parameters = torch.linspace(0, 1, 40, dtype=torch.float64)[None]
    tolerance = torch.tensor([1e-4], dtype=torch.float64)
    constant = torch.full_like(parameters, 3e-5, requires_grad=True)
    penalty, maximum = objective._local_fit_penalty(constant, parameters, tolerance)
    torch.testing.assert_close(penalty, objective._fit_penalty(constant.mean(-1), tolerance).mean())
    assert maximum.item() == pytest.approx(3e-5)
    concentrated = torch.zeros_like(parameters)
    concentrated[0, 1:4] = 1e-3
    penalty, maximum = objective._local_fit_penalty(concentrated, parameters, tolerance)
    assert maximum > concentrated.mean()
    assert penalty > 0
    penalty, _ = objective._local_fit_penalty(constant, parameters, tolerance)
    penalty.backward()
    assert torch.isfinite(constant.grad).all() and bool((constant.grad > 0).all())


def test_local_regions_handle_empty_parameter_bins_and_degree_one_samples():
    objective = V16SubsetLoss()
    parameters = torch.tensor([[0., 1.]], dtype=torch.float64)
    penalty, maximum = objective._local_fit_penalty(
        torch.zeros_like(parameters), parameters, torch.tensor([1e-4]),
    )
    assert torch.isfinite(penalty) and maximum == 0


def test_ordered_proposal_supervision_prevents_many_to_one_collapse():
    objective = V16SubsetLoss(proposal_ordered_weight=1)
    proposals = torch.tensor([[0.4, 0.9]], requires_grad=True)
    targets = torch.tensor([[0.39, 0.41]])
    valid = torch.tensor([True])
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    directed, _ = objective._directed_knot_loss(proposals, None, targets, target_mask, valid)
    ordered, _, _ = objective._selected_knot_loss(
        proposals, torch.ones_like(proposals, dtype=torch.bool), targets, target_mask, valid,
    )
    assert ordered > directed * 10
    ordered.backward()
    assert proposals.grad[0, 1] > 0


class FixedGeometryModel:
    """A real refit-only decoder with no learned candidate ranking."""
    def decode_subset(self, context, mask):
        return {"params": context["proposal_params"],
                "internal_knots": context["proposal_internal_knots"],
                "learned_keep_mask": mask}


def geometry_case():
    t = torch.linspace(0, 1, 61, dtype=torch.float64)[None]
    knots = torch.tensor([[0.2, 0.21, 0.65, 0.9]], dtype=torch.float64)
    y = 3 * (t - 0.2).clamp_min(0).pow(3) - 8 * (t - 0.65).clamp_min(0).pow(3)
    observed = torch.stack([t, y], dim=-1)
    context = {"proposal_params": t, "chord_params": t,
               "proposal_internal_knots": knots, "keep_logits": torch.tensor([[4., 3., 2., 1.]])}
    return observed, context


def test_geometry_candidates_are_bounded_rank_independent_and_include_spread_swaps():
    observed, context = geometry_case()
    objective = V16SubsetLoss(teacher_geometry_candidates=4)
    mask = torch.tensor([[True, True, True, False]])
    probes = list(objective._geometry_teacher_masks(FixedGeometryModel(), context, mask, observed, 1))
    context["keep_logits"] = -context["keep_logits"]
    reversed_ranking = list(objective._geometry_teacher_masks(FixedGeometryModel(), context, mask, observed, 1))
    assert len(probes) == 4
    assert all(torch.equal(first, second) for first, second in zip(probes, reversed_ranking))
    assert probes[0].sum() == probes[2].sum() == 2
    assert probes[1].sum() == probes[3].sum() == 3
    assert probes[1][0, 3] and probes[3][0, 3]
    assert not torch.equal(probes[0], probes[2])


def test_geometry_teacher_real_refit_reduces_feasible_count_without_fake_policy_samples():
    observed, context = geometry_case()
    objective = V16SubsetLoss(teacher_geometry_candidates=8)
    model = FixedGeometryModel()
    mask = torch.tensor([[True, True, True, False]])
    initial_mse = objective._decode_mse(model, context, mask, observed, 3)
    best, mse, metrics = objective._geometry_teacher(
        model, context, mask, initial_mse, observed, 3,
        torch.tensor([1e-9], dtype=torch.float64), 0,
    )
    assert best.sum() == 2
    assert mse <= 1e-9
    assert metrics["teacher_geometry_count_reduction"] == 1
    assert 0 < metrics["teacher_geometry_evaluations"] <= 8
    assert not mse.requires_grad


def test_geometry_teacher_does_not_replace_infeasible_incumbent():
    observed, context = geometry_case()
    objective = V16SubsetLoss(teacher_geometry_candidates=4)
    mask = torch.tensor([[False, True, False, True]])
    model = FixedGeometryModel()
    initial_mse = objective._decode_mse(model, context, mask, observed, 3)
    best, mse, metrics = objective._geometry_teacher(
        model, context, mask, initial_mse, observed, 3,
        torch.tensor([1e-30], dtype=torch.float64), 0,
    )
    assert torch.equal(best, mask)
    torch.testing.assert_close(mse, initial_mse)
    assert metrics["teacher_geometry_improved_fraction"] == 0


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_all_new_terms_backpropagate_finite_metrics_and_reuse_network_decodes(stage):
    model = network()
    observed = points()
    options = dict(policy_samples=2, counterfactual_edits=0, teacher_prefix_search_steps=1)
    baseline = V16SubsetLoss(**options)
    objective = V16SubsetLoss(
        **options, parameter_counterfactual_weight=.25, teacher_geometry_candidates=4,
        local_fit_weight=.1, proposal_ordered_weight=.5,
    )
    target_options = dict(
        stage=stage, target_params=torch.linspace(0, 1, observed.shape[1])[None],
        target_internal_knots=torch.tensor([[.25, .6]]),
        target_internal_knot_mask=torch.tensor([[True, True]]),
        target_geometry_valid=torch.tensor([True]),
    )
    with patch.object(model, "decode_subset", wraps=model.decode_subset) as decoder:
        torch.manual_seed(8)
        baseline(model, observed, **target_options)
        base_decodes = decoder.call_count
        decoder.reset_mock()
        torch.manual_seed(8)
        loss, metrics = objective(model, observed, **target_options)
        # Counterfactual/local supervision adds no decode; supervised geometry
        # now reuses the already decoded deployment output. Only teacher probes
        # may add calls, and their number is explicitly bounded and reported.
        assert decoder.call_count <= base_decodes + int(metrics["teacher_geometry_evaluations"])
    loss.backward()
    assert all(value.ndim == 0 and torch.isfinite(value) and not value.requires_grad
               for value in metrics.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert metrics["local_fit_loss"] > 0
    assert metrics["proposal_ordered_loss"] > 0
    assert metrics["parameter_counterfactual_refits"] == (1 if stage == "proposal" else 2)
    if stage == "proposal":
        assert metrics["deployment_parameter_counterfactual_loss"] == 0
        assert metrics["teacher_geometry_evaluations"] == 0
    else:
        assert metrics["subset_best_pass_rate"] >= metrics["deployment_pass_rate"]


@pytest.mark.parametrize("stage", ["proposal", "joint"])
@pytest.mark.parametrize("enabled", [False, True])
def test_trust_statistics_are_scalar_and_stage_safe_without_additional_decodes(stage, enabled):
    model = network()
    if not enabled:
        config = model.get_config()
        config["parameter_trust_enabled"] = False
        model = V16CandidateSelectionNetwork(**config)
    objective = V16SubsetLoss(policy_samples=2, counterfactual_edits=0,
                              teacher_prefix_search_steps=1)
    with patch.object(model, "decode_subset", wraps=model.decode_subset) as decoder:
        _, metrics = objective(model, points(), stage=stage)
        calls = decoder.call_count
    for field in ("proposal_parameter_trust", "subset_parameter_trust"):
        expected = 0 if stage == "proposal" and field.startswith("subset") else (.25 if enabled else 1.)
        for statistic in ("mean", "min", "max"):
            value = metrics[f"{field}_{statistic}"]
            assert value.ndim == 0 and not value.requires_grad
            assert value.item() == pytest.approx(expected)
    # Proposal stage never decodes, including for the absent subset statistics.
    if stage == "proposal":
        assert calls == 0
    else:
        assert calls > 0


def test_missing_legacy_gate_fields_are_zero_safe():
    from test_v16_subset_loss import PolicyOnlyModel, TableSubsetLoss

    objective = TableSubsetLoss(policy_samples=2, counterfactual_edits=0)
    _, metrics = objective(PolicyOnlyModel(), points())
    for field in ("proposal_parameter_trust", "subset_parameter_trust"):
        for statistic in ("mean", "min", "max"):
            assert metrics[f"{field}_{statistic}"] == 0
