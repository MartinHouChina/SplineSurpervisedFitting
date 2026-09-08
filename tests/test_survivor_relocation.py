from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.models import SplineFittingNetwork  # noqa: E402
from spline_fitting.models.interactive_pruning_head import (  # noqa: E402
    InteractivePruningHead,
)


def _head(*, relocation: bool) -> InteractivePruningHead:
    return InteractivePruningHead(
        hidden_dim=16,
        residual_feature_dim=1,
        attention_heads=4,
        min_gap=0.01,
        one_shot_adaptive=True,
        one_shot_fixed_proposal_geometry=True,
        one_shot_selection_policy="mass_topk",
        one_shot_joint_position_refinement=True,
        one_shot_survivor_relocation=relocation,
        one_shot_max_position_shift=0.20,
    )


def _inputs(
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    positions = torch.tensor([[0.10, 0.20, 0.30, 0.70, 0.80, 0.90]]).expand(batch, -1)
    return (
        torch.randn(batch, 6, 16),
        positions,
        {
            "coefficient_energy": torch.rand(batch, 6),
            "deletion_delta": torch.rand(batch, 6),
            "residual_features": torch.rand(batch, 6),
        },
    )


def _make_uniform_keep(head: InteractivePruningHead, keep_bias: float) -> None:
    with torch.no_grad():
        head.position_residual_head.weight.zero_()
        head.position_residual_head.bias.zero_()
        head.joint_preliminary_position_head.weight.zero_()
        head.joint_preliminary_position_head.bias.zero_()
        head.joint_final_position_head.weight.zero_()
        head.joint_final_position_head.bias.zero_()
        head.keep_head.weight.zero_()
        head.keep_head.bias.fill_(keep_bias)
        head.adaptive_threshold_head[-1].weight.zero_()
        head.adaptive_threshold_head[-1].bias.zero_()


def test_survivor_relocation_requires_v11_joint_path() -> None:
    with pytest.raises(ValueError, match="survivor relocation requires"):
        InteractivePruningHead(
            hidden_dim=16,
            attention_heads=4,
            one_shot_adaptive=True,
            one_shot_survivor_relocation=True,
        )


def test_v11_state_strictly_initializes_neutral_v12_relocation() -> None:
    torch.manual_seed(211)
    v11 = _head(relocation=False).eval()
    torch.manual_seed(223)
    v12 = _head(relocation=True).eval()
    v12.load_state_dict(v11.state_dict(), strict=True)
    tokens, positions, keyword_inputs = _inputs(batch=2)

    with torch.no_grad():
        expected = v11(tokens, positions, **keyword_inputs)
        actual = v12(tokens, positions, **keyword_inputs)

    torch.testing.assert_close(actual["keep_probability"], expected["keep_probability"])
    assert torch.equal(actual["final_hard_keep_mask"], expected["final_hard_keep_mask"])
    torch.testing.assert_close(
        actual["refined_candidate_positions"],
        expected["refined_candidate_positions"],
    )
    torch.testing.assert_close(
        actual["relocation_position_residual"],
        torch.zeros_like(actual["relocation_position_residual"]),
    )


def test_survivor_relocation_uses_final_subset_and_preserves_order() -> None:
    torch.manual_seed(227)
    head = _head(relocation=True).eval()
    _make_uniform_keep(head, keep_bias=0.0)
    with torch.no_grad():
        head.survivor_relocation_head.weight.zero_()
        head.survivor_relocation_head.bias.fill_(3.0)
    tokens, positions, keyword_inputs = _inputs()

    with torch.no_grad():
        output = head(tokens, positions, **keyword_inputs)

    selected_mask = output["final_hard_keep_mask"]
    assert selected_mask.tolist() == [[True, True, True, False, False, False]]
    assert torch.all(output["relocation_position_residual"][~selected_mask] == 0.0)
    assert torch.any(output["relocation_position_residual"][selected_mask] != 0.0)

    selected_positions = output["refined_candidate_positions"][selected_mask]
    assert torch.all(selected_positions[1:] - selected_positions[:-1] >= 0.01 - 1e-6)
    total_shift = (
        output["refined_candidate_positions"] - output["proposal_candidate_positions"]
    )
    assert torch.all(total_shift.abs() <= 0.20 + 1e-6)

    attention = output["relocation_attention_weights"][0]
    assert torch.all(attention[~selected_mask[0]] == 0.0)
    assert torch.all(attention[:, ~selected_mask[0]] == 0.0)
    # Feature slots are left/right survivor gap, cell span, local coordinate,
    # retained rank, retained count, coverage offset, and the ST keep gate.
    assert output["relocation_relative_features"].shape == (1, 6, 8)
    assert torch.all(
        output["relocation_relative_features"][0, :3, 4]
        == torch.tensor([0.25, 0.50, 0.75])
    )


def test_relocation_is_mask_conditioned_and_keep_receives_position_gradient() -> None:
    torch.manual_seed(229)
    head = _head(relocation=True)
    _make_uniform_keep(head, keep_bias=-0.7)
    with torch.no_grad():
        head.survivor_relocation_head.weight.normal_(std=0.2)
        head.survivor_relocation_head.bias.fill_(0.3)
    tokens, positions, keyword_inputs = _inputs()

    smaller = head(tokens, positions, **keyword_inputs)
    smaller_count = int(smaller["final_hard_keep_mask"].sum())
    smaller_positions = smaller["refined_candidate_positions"].detach().clone()
    head.zero_grad(set_to_none=True)
    smaller["refined_candidate_positions"].square().sum().backward()
    assert head.keep_head.bias.grad is not None
    assert float(head.keep_head.bias.grad.abs().sum()) > 0.0

    _make_uniform_keep(head, keep_bias=0.7)
    with torch.no_grad():
        larger = head(tokens, positions, **keyword_inputs)
    larger_count = int(larger["final_hard_keep_mask"].sum())
    assert smaller_count < larger_count
    assert not torch.allclose(smaller_positions, larger["refined_candidate_positions"])


def test_zero_survivors_are_finite_and_do_not_move() -> None:
    torch.manual_seed(233)
    head = _head(relocation=True).eval()
    _make_uniform_keep(head, keep_bias=-12.0)
    with torch.no_grad():
        head.survivor_relocation_head.weight.normal_(std=1.0)
        head.survivor_relocation_head.bias.fill_(3.0)
    tokens, positions, keyword_inputs = _inputs()

    with torch.no_grad():
        output = head(tokens, positions, **keyword_inputs)

    assert not bool(output["final_hard_keep_mask"].any())
    torch.testing.assert_close(output["refined_candidate_positions"], positions)
    for name in (
        "relocation_tokens",
        "relocation_attention_weights",
        "relocation_position_residual",
        "refined_candidate_positions",
    ):
        assert torch.all(torch.isfinite(output[name])), name


def test_ordered_projection_repairs_crossing_with_shift_and_gap_bounds() -> None:
    head = _head(relocation=True)
    reference = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.70, 0.80, 0.90],
            [0.10, 0.20, 0.30, 0.70, 0.80, 0.90],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [0.80, 0.05, 0.95, 0.10, 0.60, 0.40],
            [0.80, 0.25, 0.05, 0.75, 0.95, 0.40],
        ],
        requires_grad=True,
    )
    active = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, False, True, False, True, False],
        ]
    )

    projected = head._project_ordered_positions(
        target,
        active,
        reference_positions=reference,
        max_shift=0.20,
    )
    for batch_index in range(2):
        selected = projected[batch_index, active[batch_index]]
        assert torch.all(selected[1:] - selected[:-1] >= head.min_gap - 1e-6)
    assert torch.all((projected - reference).abs()[active] <= 0.20 + 1e-6)
    torch.testing.assert_close(projected[~active], target[~active])

    filled = head._fill_inactive_ordered_slots(
        projected,
        active,
        torch.ones_like(active),
    )
    torch.testing.assert_close(filled[active], projected[active])
    assert torch.all(filled[:, 1:] > filled[:, :-1])

    projected.square().sum().backward()
    assert target.grad is not None
    assert torch.isfinite(target.grad).all()


def test_spline_network_accepts_survivor_relocation_and_strict_v11_state() -> None:
    config = {
        "point_dim": 2,
        "hidden_dim": 32,
        "encoder_layers": 1,
        "max_internal_knots": 6,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": "mass_topk",
        "one_shot_joint_position_refinement": True,
    }
    torch.manual_seed(239)
    v11 = SplineFittingNetwork(**config).eval()
    torch.manual_seed(241)
    v12 = SplineFittingNetwork(
        **config,
        one_shot_survivor_relocation=True,
    ).eval()
    v12.load_state_dict(v11.state_dict(), strict=True)
    points = torch.randn(2, 24, 2)

    with torch.no_grad():
        expected = v11(points)
        actual = v12(points)

    torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])
    torch.testing.assert_close(actual["keep_probability"], expected["keep_probability"])
    assert torch.equal(actual["learned_keep_mask"], expected["learned_keep_mask"])
