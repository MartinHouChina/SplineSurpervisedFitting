from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.losses import (  # noqa: E402
    ParameterFeedbackLoss,
    ParameterFeedbackLossWeights,
)


def _example() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    points = torch.tensor([[[0.0, 0.0], [0.4, 0.2], [1.0, 0.0]]], dtype=torch.float32)
    params = torch.tensor([[0.0, 0.45, 1.0]], requires_grad=True)
    reconstructed = points + 0.01
    output = {
        "params": params,
        "proposal_params": torch.tensor([[0.0, 0.5, 1.0]]),
        "reconstructed_points": reconstructed,
        "parameter_feedback_gap_logit_shift": torch.tensor([[0.1, -0.1]]),
    }
    return output, points


def test_parameter_feedback_loss_is_finite_and_backpropagates() -> None:
    output, points = _example()
    loss_fn = ParameterFeedbackLoss(
        ParameterFeedbackLossWeights(
            fit=0.1,
            threshold_violation=0.1,
            true_parameter=1.0,
            chord_prior=0.1,
            identity=0.1,
        )
    )
    result = loss_fn(
        output,
        points,
        chord_params=torch.tensor([[0.0, 0.4, 1.0]]),
        true_params=torch.tensor([[0.0, 0.42, 1.0]]),
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert output["params"].grad is not None
    assert torch.isfinite(output["params"].grad).all()


def test_parameter_feedback_loss_normalizes_parameter_error() -> None:
    output, points = _example()
    loss_fn = ParameterFeedbackLoss(
        ParameterFeedbackLossWeights(
            fit=0.0,
            threshold_violation=0.0,
            true_parameter=1.0,
            chord_prior=0.0,
            identity=0.0,
        ),
        parameter_error_scale=0.02,
    )
    result = loss_fn(
        output,
        points,
        true_params=torch.tensor([[0.0, 0.43, 1.0]]),
    )
    expected_raw = (output["params"] - torch.tensor([[0.0, 0.43, 1.0]])).square().mean()
    assert result["true_parameter_loss"].item() == pytest.approx(expected_raw.item())
    assert result["loss"].item() == pytest.approx(expected_raw.item() / (0.02**2))
    assert result["true_parameter_supervision_fraction"].item() == 1.0


def test_parameter_feedback_loss_marks_unlabeled_real_curves() -> None:
    output, points = _example()
    result = ParameterFeedbackLoss()(output, points)

    assert result["true_parameter_loss"].item() == 0.0
    assert result["true_parameter_supervision_fraction"].item() == 0.0


def test_parameter_feedback_loss_rejects_invalid_shapes() -> None:
    output, points = _example()
    with pytest.raises(ValueError, match="true_params"):
        ParameterFeedbackLoss()(output, points, true_params=torch.zeros(1, 2))


def test_joint_count_supervises_the_actual_mass_topk_rounding_score() -> None:
    output, points = _example()
    keep_logits = torch.tensor([[0.3, -0.2, 0.8]], requires_grad=True)
    probability = torch.sigmoid(keep_logits)
    uncertainty = (probability * (1.0 - probability)).sum(dim=-1).sqrt()
    requested_score = probability.sum(dim=-1) + 0.25 * uncertainty
    output.update(
        {
            "keep_logits": keep_logits,
            "keep_probability": probability,
            "candidate_mask": torch.ones(1, 3, dtype=torch.bool),
            "one_shot_requested_count_score": requested_score,
        }
    )
    loss_fn = ParameterFeedbackLoss(
        ParameterFeedbackLossWeights(
            fit=0.0,
            threshold_violation=0.0,
            true_parameter=0.0,
            chord_prior=0.0,
            identity=0.0,
            joint_count=1.0,
        )
    )
    result = loss_fn(
        output,
        points,
        teacher_retained_mask=torch.tensor([[True, True, False]]),
        teacher_count=torch.tensor([2]),
    )
    expected = torch.nn.functional.smooth_l1_loss(
        requested_score / 3.0,
        torch.tensor([1.75]) / 3.0,
        beta=0.05,
    )
    torch.testing.assert_close(result["joint_count_loss"], expected)
    result["loss"].backward()
    assert keep_logits.grad is not None
    assert float(keep_logits.grad.abs().sum()) > 0.0


def test_feedback_exact_deployment_loss_updates_parameters_and_kept_knots() -> None:
    parameters = torch.linspace(0.0, 1.0, 32).unsqueeze(0).requires_grad_()
    points = torch.stack(
        [parameters.detach()[0], torch.sin(4.0 * parameters.detach()[0])], dim=-1
    ).unsqueeze(0)
    knots = torch.tensor(
        [[0.2, 0.4, 0.6, 0.8]], dtype=torch.float32, requires_grad=True
    )
    output = {
        "params": parameters,
        "proposal_params": parameters.detach().clone(),
        "reconstructed_points": points.detach().clone(),
        "deployment_internal_knots": knots,
        "learned_keep_mask": torch.tensor([[True, False, True, True]]),
    }
    result = ParameterFeedbackLoss(
        ParameterFeedbackLossWeights(
            fit=0.0,
            threshold_violation=0.0,
            true_parameter=0.0,
            chord_prior=0.0,
            identity=0.0,
            deployment_fit=1.0,
        )
    )(output, points)
    assert result["deployment_fit_loss"] > 0.0
    result["loss"].backward()
    assert parameters.grad is not None
    assert float(parameters.grad.abs().sum()) > 0.0
    assert knots.grad is not None
    assert float(knots.grad[output["learned_keep_mask"]].abs().sum()) > 0.0
