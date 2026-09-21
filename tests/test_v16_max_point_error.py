"""Point-error objectives use actual hard B-spline residuals, never plotted MSE."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.losses.v16_subset_loss import V16SubsetLoss
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _case():
    torch.manual_seed(1701)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=5,
        attention_heads=2, selector_layers=1,
    ).eval()
    t = torch.linspace(0, 1, 31)
    points = torch.stack([
        torch.stack([t, .2 * torch.sin(8 * t)], -1),
        torch.stack([t, .1 * torch.cos(11 * t)], -1),
    ])
    return model, points


@pytest.mark.parametrize("kwargs", [
    {"max_point_error_weight": -1},
    {"max_point_error_weight": float("nan")},
    {"max_point_error_weight": .1},
    {"max_point_error_tolerance": 0},
    {"max_point_error_tolerance": -1},
    {"max_point_error_tolerance": float("inf")},
    {"max_point_error_tail_fraction": 0},
    {"max_point_error_tail_fraction": 1.01},
    {"max_point_error_tail_fraction": float("nan")},
])
def test_point_error_settings_reject_invalid_or_implicit_threshold(kwargs):
    with pytest.raises(ValueError):
        V16SubsetLoss(**kwargs)


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_disabled_point_error_preserves_loss_metrics_rng_and_fit_count(stage):
    model, points = _case()
    options = dict(policy_samples=2, counterfactual_edits=0, teacher_prefix_search_steps=1)
    baseline = V16SubsetLoss(**options)
    disabled = V16SubsetLoss(**options, max_point_error_weight=0,
                             max_point_error_tolerance=5e-4, max_point_error_tail_fraction=.5)
    results = []
    for objective in (baseline, disabled):
        with patch.object(objective, "_fit_details", wraps=objective._fit_details) as refit:
            torch.manual_seed(11)
            loss, metrics = objective(model, points, stage=stage)
            results.append((loss, metrics, torch.get_rng_state(), refit.call_count))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    for key in results[0][1]:
        torch.testing.assert_close(results[0][1][key], results[1][1][key], rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])
    assert results[0][3] == results[1][3]
    assert "max_point_error_loss" not in results[1][1]


def test_peak_is_exact_and_tail_distributes_nonzero_gradients():
    objective = V16SubsetLoss(max_point_error_tolerance=5e-4,
                             max_point_error_tail_fraction=.5, tail_weight=0)
    residuals = torch.tensor([[1e-6, 2e-4, 9e-4, 4e-4]], dtype=torch.float64,
                             requires_grad=True)
    loss, peak, tail = objective._max_point_error_penalty(residuals)
    torch.testing.assert_close(peak, residuals.amax(-1), rtol=0, atol=0)
    torch.testing.assert_close(tail, residuals[:, [2, 3]].mean(-1), rtol=0, atol=0)
    expected = .5 * (objective._fit_penalty(peak, peak.new_tensor(5e-4))
                     + objective._fit_penalty(tail, peak.new_tensor(5e-4)))
    torch.testing.assert_close(loss, expected.mean())
    loss.backward()
    assert torch.isfinite(residuals.grad).all()
    assert residuals.grad[0, 2] > residuals.grad[0, 3] > 0
    assert residuals.grad[0, :2].count_nonzero() == 0


def test_zero_and_extreme_point_errors_remain_finite():
    objective = V16SubsetLoss(max_point_error_tolerance=1e-8)
    residuals = torch.tensor([[0., 0., 0.], [1e200, 1e199, 0.]],
                             dtype=torch.float64, requires_grad=True)
    loss, maxima, tail = objective._max_point_error_penalty(residuals)
    assert torch.isfinite(loss) and torch.isfinite(maxima).all() and torch.isfinite(tail).all()
    loss.backward()
    assert torch.isfinite(residuals.grad).all()


@pytest.mark.parametrize("residuals", [
    torch.zeros(3), torch.zeros(0, 3), torch.zeros(1, 0),
    torch.tensor([[-1., 0.]]), torch.tensor([[float("nan")]]),
    torch.tensor([[float("inf")]]), torch.ones(1, 3, dtype=torch.int64),
])
def test_malformed_point_errors_are_rejected(residuals):
    with pytest.raises(ValueError):
        V16SubsetLoss(max_point_error_tolerance=1e-4)._max_point_error_penalty(residuals)


@pytest.mark.parametrize("stage", ["proposal", "joint"])
@pytest.mark.parametrize("selected", [0, 2, 5])
def test_forward_point_error_matches_production_refit_and_preserves_mse_pass(stage, selected):
    model, points = _case()
    options = dict(policy_samples=2, counterfactual_edits=0, teacher_prefix_search_steps=1)
    objective = V16SubsetLoss(**options, max_point_error_weight=.2,
                             max_point_error_tolerance=5e-4)
    baseline = V16SubsetLoss(**options)
    mask = torch.arange(5)[None].expand(2, -1) < selected
    with patch.object(model, "select_mask", return_value=mask):
        torch.manual_seed(8)
        old_loss, old_metrics = baseline(model, points, stage=stage)
        torch.manual_seed(8)
        loss, metrics = objective(model, points, stage=stage)
        with torch.no_grad():
            context = model.encode_candidates(points, mse_tolerance=objective.mse_tolerance)
            dense_mask = torch.ones_like(mask)
            deployed = model.decode_subset(context, mask)
            geometries = [(context["proposal_params"], context["proposal_internal_knots"], dense_mask)]
            geometries.append(geometries[0] if stage == "proposal" else
                              (deployed["params"], deployed["internal_knots"], mask))
            expected_losses = []
            for prefix, (params, knots, keep) in zip(("dense", "deployment"), geometries):
                residuals = []
                for row in range(len(points)):
                    fit = refit_bspline_control_points(
                        params[row].double(), points[row].double(), knots[row, keep[row]].double(),
                        smoothness_weight=0, control_ridge=0, interpolate_endpoints=True,
                    )
                    residuals.append((fit.reconstructed_points - points[row].double()).square().sum(-1))
                residuals = torch.stack(residuals)
                peak = residuals.amax(-1)
                n = math.ceil(points.shape[1] * objective.max_point_error_tail_fraction)
                tail = residuals.topk(n).values.mean(-1)
                expected_losses.append(objective._max_point_error_penalty(residuals)[0])
                for name, expected in (
                    ("max_point_squared_error_mean", peak.mean()),
                    ("max_point_squared_error_max", peak.max()),
                    ("point_tail_squared_error_mean", tail.mean()),
                    ("max_point_error_pass_rate", (peak <= 5e-4).double().mean()),
                ):
                    torch.testing.assert_close(metrics[f"{prefix}_{name}"], expected,
                                               atol=1e-9, rtol=1e-4)
            torch.testing.assert_close(metrics["max_point_error_loss"],
                                       torch.stack(expected_losses).mean(), atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(loss - old_loss, .2 * metrics["max_point_error_loss"],
                               atol=1e-12, rtol=1e-12)
    for name in ("dense_mse", "deployment_mse", "dense_pass_rate", "deployment_pass_rate",
                 "keep_count", "subset_best_count", "subset_best_pass_rate"):
        torch.testing.assert_close(metrics[name], old_metrics[name], atol=0, rtol=0)
    point_term_grads = torch.autograd.grad(
        loss - old_loss, tuple(model.parameters()), allow_unused=True, retain_graph=True,
    )
    nonempty_grads = [gradient for gradient in point_term_grads if gradient is not None]
    assert nonempty_grads and all(torch.isfinite(gradient).all() for gradient in nonempty_grads)
    assert any(bool((gradient.abs() > 1e-10).any()) for gradient in nonempty_grads)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
