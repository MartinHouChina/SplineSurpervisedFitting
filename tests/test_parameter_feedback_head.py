from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.models import (  # noqa: E402
    ParameterFeedbackHead,
    SplineFittingNetwork,
)
from spline_fitting.models import spline_network as spline_network_module  # noqa: E402


def _strict_parameters(
    batch: int,
    point_count: int,
    min_gap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(batch, point_count - 1)
    free_budget = 1.0 - min_gap * (point_count - 1)
    gaps = min_gap + free_budget * torch.softmax(raw, dim=-1)
    params = torch.cat(
        [torch.zeros(batch, 1), torch.cumsum(gaps, dim=-1)],
        dim=-1,
    )
    params[:, -1] = 1.0
    return params, gaps


def _head_inputs(
    *,
    batch: int = 2,
    point_count: int = 17,
    candidate_count: int = 6,
    hidden_dim: int = 16,
    min_gap: float = 0.01,
) -> dict[str, torch.Tensor]:
    params, gaps = _strict_parameters(batch, point_count, min_gap)
    points = torch.randn(batch, point_count, 2)
    candidates = torch.linspace(0.1, 0.9, candidate_count).expand(batch, -1)
    hard_mask = torch.tensor(
        [[True, False, True, False, True, False]],
    ).expand(batch, -1)
    return {
        "local_features": torch.randn(batch, point_count, hidden_dim),
        "points": points,
        "proposal_params": params,
        "proposal_parameter_gaps": gaps,
        "pilot_reconstructed_points": points + 0.02 * torch.randn_like(points),
        "risk_candidate_positions": candidates,
        "deletion_delta": torch.rand(batch, candidate_count),
        "survivor_tokens": torch.randn(batch, candidate_count, hidden_dim),
        "survivor_positions": candidates,
        "survivor_gate": hard_mask.to(torch.float32),
        "hard_survivor_mask": hard_mask,
        "candidate_mask": torch.ones_like(hard_mask),
    }


def test_zero_initialized_parameter_feedback_is_identity_and_strict() -> None:
    torch.manual_seed(301)
    min_gap = 0.01
    head = ParameterFeedbackHead(
        hidden_dim=16,
        min_gap=min_gap,
        attention_heads=4,
    )
    inputs = _head_inputs(min_gap=min_gap)
    output = head(**inputs, return_attention_weights=True)

    torch.testing.assert_close(
        output["feedback_params"],
        inputs["proposal_params"],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["feedback_parameter_gaps"],
        inputs["proposal_parameter_gaps"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.all(output["feedback_parameter_gaps"] >= min_gap - 1e-7)
    assert torch.all(
        output["feedback_params"][:, 1:] > output["feedback_params"][:, :-1]
    )
    torch.testing.assert_close(output["feedback_params"][:, 0], torch.zeros(2))
    torch.testing.assert_close(output["feedback_params"][:, -1], torch.ones(2))
    assert torch.count_nonzero(output["parameter_feedback_gap_logit_delta"]) == 0


def test_parameter_feedback_remains_finite_with_no_survivors() -> None:
    torch.manual_seed(307)
    head = ParameterFeedbackHead(
        hidden_dim=16,
        min_gap=0.01,
        attention_heads=4,
    )
    inputs = _head_inputs(min_gap=0.01)
    inputs["hard_survivor_mask"] = torch.zeros_like(inputs["hard_survivor_mask"])
    inputs["survivor_gate"] = torch.zeros_like(inputs["survivor_gate"])
    output = head(**inputs, return_attention_weights=True)

    assert torch.isfinite(output["feedback_params"]).all()
    assert torch.isfinite(output["parameter_feedback_interval_tokens"]).all()
    assert output["parameter_feedback_safe_memory_mask"].sum(dim=-1).tolist() == [
        1,
        1,
    ]
    assert torch.count_nonzero(output["parameter_feedback_attention_weights"]) == 0


def test_parameter_feedback_output_projection_receives_gradient() -> None:
    torch.manual_seed(311)
    head = ParameterFeedbackHead(
        hidden_dim=16,
        min_gap=0.01,
        attention_heads=4,
    )
    output = head(**_head_inputs(min_gap=0.01))
    interval_weight = torch.linspace(
        0.1,
        1.0,
        output["feedback_parameter_gaps"].shape[-1],
    )
    loss = (output["feedback_parameter_gaps"] * interval_weight).sum()
    loss.backward()

    assert head.gap_logit_head.weight.grad is not None
    assert float(head.gap_logit_head.weight.grad.abs().sum()) > 0.0
    assert head.chord_blend_weight.grad is not None
    assert float(head.chord_blend_weight.grad.abs()) > 0.0


def test_explicit_chord_blend_is_projected_and_remains_strict() -> None:
    torch.manual_seed(312)
    min_gap = 0.01
    head = ParameterFeedbackHead(
        hidden_dim=16,
        min_gap=min_gap,
        attention_heads=4,
    )
    inputs = _head_inputs(min_gap=min_gap)
    with torch.no_grad():
        head.chord_blend_weight.fill_(0.6)
    output = head(**inputs)

    gap_count = inputs["proposal_parameter_gaps"].shape[-1]
    free_budget = 1.0 - min_gap * gap_count
    proposal_free = (inputs["proposal_parameter_gaps"] - min_gap) / free_budget
    chord = output["parameter_feedback_chord_gaps"]
    expected = min_gap + free_budget * (0.4 * proposal_free + 0.6 * chord)

    torch.testing.assert_close(output["feedback_parameter_gaps"], expected)
    torch.testing.assert_close(
        output["parameter_feedback_chord_blend_weight"],
        torch.full((2,), 0.6),
    )
    assert torch.all(output["feedback_parameter_gaps"] >= min_gap - 1e-7)


def _feedback_network() -> SplineFittingNetwork:
    return SplineFittingNetwork(
        point_dim=2,
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=5,
        structure_mode="candidate_pruning_one_shot",
        structure_attention_heads=4,
        geometry_feature_mode="chord_derivatives",
        one_shot_fixed_proposal_geometry=True,
        one_shot_selection_policy="mass_topk",
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=True,
        parameter_feedback_fusion=True,
    )


def test_network_feedback_preserves_proposal_params_and_adds_no_deployment_solve() -> (
    None
):
    torch.manual_seed(313)
    model = _feedback_network().eval()
    points = torch.randn(2, 24, 2)
    original_solve = spline_network_module.solve_coefficients
    solve_count = 0

    def counted_solve(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        nonlocal solve_count
        solve_count += 1
        return original_solve(*args, **kwargs)

    with mock.patch.object(
        spline_network_module,
        "solve_coefficients",
        side_effect=counted_solve,
    ):
        with torch.no_grad():
            output = model.forward_deployment(points)

    assert solve_count == 1
    assert "reconstructed_points" not in output
    torch.testing.assert_close(output["params"], output["proposal_params"])
    torch.testing.assert_close(
        output["parameter_gaps"], output["proposal_parameter_gaps"]
    )
    torch.testing.assert_close(
        output["internal_knots"],
        output["parameter_feedback_source_internal_knots"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert "parameter_feedback_attention_weights" not in output


class _FixedFeedback(nn.Module):
    def forward(
        self,
        local_features: torch.Tensor,
        points: torch.Tensor,
        proposal_params: torch.Tensor,
        proposal_parameter_gaps: torch.Tensor,
        pilot_reconstructed_points: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del local_features, points, proposal_parameter_gaps, pilot_reconstructed_points
        gap_count = proposal_params.shape[-1] - 1
        weights = torch.arange(
            1,
            gap_count + 1,
            device=proposal_params.device,
            dtype=proposal_params.dtype,
        )
        gaps = weights / weights.sum()
        gaps = gaps.unsqueeze(0).expand(proposal_params.shape[0], -1)
        params = torch.cat(
            [proposal_params.new_zeros(proposal_params.shape[0], 1), gaps.cumsum(-1)],
            dim=-1,
        )
        return {
            "feedback_params": params,
            "feedback_parameter_gaps": gaps,
            "feedback_raw_parameter_gaps": gaps.log(),
        }


def test_training_surrogate_uses_feedback_parameters() -> None:
    torch.manual_seed(317)
    model = _feedback_network()
    model.parameter_feedback_head = _FixedFeedback()
    output = model(torch.randn(1, 18, 2))

    assert not torch.allclose(output["params"], output["proposal_params"])
    torch.testing.assert_close(output["polynomial_basis"][..., 1], output["params"])
    expected_knots = model._warp_parameter_coordinates(
        output["parameter_feedback_source_internal_knots"],
        output["proposal_params"],
        output["params"],
    )
    torch.testing.assert_close(output["internal_knots"], expected_knots)
    torch.testing.assert_close(output["deployment_internal_knots"], expected_knots)
    assert not torch.allclose(
        output["polynomial_basis"][..., 1], output["proposal_params"]
    )
