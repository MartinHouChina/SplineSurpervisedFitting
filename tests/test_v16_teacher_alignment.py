"""Regression checks for offline-teacher parameter frames and Keep boundaries."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.losses.v16_subset_loss import V16SubsetLoss


def test_offline_teacher_positions_share_the_frozen_proposal_frame():
    objective = V16SubsetLoss(joint_parameter_warp_gradient_scale=0.0)
    proposal_params = torch.tensor([[0.0, 0.25, 0.50, 0.75, 1.0]])
    decoded_params = torch.tensor(
        [[0.0, 0.10, 0.40, 0.80, 1.0]], requires_grad=True,
    )
    teacher_knots = torch.tensor([[0.375]])
    decoded_knots = objective._warp_knots_to_target_parameterization(
        teacher_knots, proposal_params, decoded_params.detach(),
    ).detach().requires_grad_(True)
    assert (decoded_knots - teacher_knots).abs().item() > 0.1

    aligned = objective._offline_teacher_frame_knots(
        {"internal_knots": decoded_knots, "params": decoded_params},
        proposal_params,
    )
    assert torch.allclose(aligned, teacher_knots, atol=1e-7)
    aligned.sum().backward()
    assert decoded_knots.grad is not None
    assert torch.isfinite(decoded_knots.grad).all()
    assert decoded_knots.grad.abs().sum() > 0
    assert decoded_params.grad is None or torch.count_nonzero(decoded_params.grad) == 0


def test_boundary_ranking_backpropagates_to_hardest_keep_and_delete():
    logits = torch.tensor([[0.0, -2.0, 3.0, 1.0]], requires_grad=True)
    mask = torch.tensor([[True, False, False, True]])
    loss = V16SubsetLoss._keep_boundary_ranking_loss(
        logits, mask, torch.tensor([True]), None, margin=1.0,
    )
    assert loss > 0
    loss.backward()
    assert logits.grad[0, 0] < 0  # gradient descent raises weakest kept slot
    assert logits.grad[0, 2] > 0  # gradient descent lowers strongest deleted slot
    assert logits.grad[0, 1] == 0
    assert logits.grad[0, 3] == 0


def test_boundary_ranking_excludes_feasible_swaps_and_failed_teacher_rows():
    logits = torch.tensor(
        [[0.0, 10.0, 1.0], [0.0, 100.0, 1.0]], requires_grad=True,
    )
    mask = torch.tensor([[True, False, False], [True, False, False]])
    weights = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    loss = V16SubsetLoss._keep_boundary_ranking_loss(
        logits, mask, torch.tensor([True, False]), weights, margin=1.0,
    )
    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(2.0)))
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_boundary_ranking_empty_classes_are_finite_and_zero():
    logits = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
    for mask in (
        torch.tensor([[True, True], [False, False]]),
        torch.tensor([[True, False], [False, True]]),
    ):
        pass_rows = torch.tensor([True, True])
        weights = torch.zeros_like(logits)
        result = V16SubsetLoss._keep_boundary_ranking_loss(
            logits, mask, pass_rows, weights, margin=1.0,
        )
        assert torch.isfinite(result)
        assert result == 0


def test_boundary_ranking_weight_defaults_to_zero_and_rejects_negative():
    assert V16SubsetLoss().keep_boundary_ranking_weight == 0.0
    try:
        V16SubsetLoss(keep_boundary_ranking_weight=-0.1)
    except ValueError as error:
        assert "keep_boundary_ranking_weight" in str(error)
    else:
        raise AssertionError("negative boundary-ranking weight must fail")
