from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve  # noqa: E402
from spline_fitting.evaluation.hybrid_knot_search import (  # noqa: E402
    hybrid_minimal_knot_search,
    refine_knot_positions,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    build_open_knot_vector,
)


DTYPE = torch.float64


def _sample_curve(
    parameters: torch.Tensor,
    internal_knots: torch.Tensor,
) -> torch.Tensor:
    control_count = int(internal_knots.numel()) + 4
    x = torch.linspace(-0.9, 0.95, control_count, dtype=DTYPE)
    controls = torch.stack(
        [
            x + 0.08 * torch.sin(2.3 * x),
            0.55 * torch.sin(4.7 * x) + 0.18 * x.square() - 0.07 * x,
        ],
        dim=-1,
    )
    return evaluate_bspline_curve(
        parameters,
        controls,
        build_open_knot_vector(internal_knots, degree=3),
        degree=3,
    )


def _search_kwargs() -> dict[str, object]:
    return {
        "degree": 3,
        "smoothness_weight": 0.0,
        "control_ridge": 0.0,
        "beam_width": 2,
        "branch_factor": 2,
        "position_sweeps": 1,
        "position_grid_size": 3,
        "position_restarts": 1,
        "min_gap": 1e-4,
    }


def test_hybrid_is_feasible_and_cannot_use_more_knots_than_greedy() -> None:
    parameters = torch.linspace(0.0, 1.0, 65, dtype=DTYPE)
    source_knots = torch.tensor([0.27, 0.68], dtype=DTYPE)
    proposal = torch.tensor([0.12, 0.27, 0.46, 0.68, 0.86], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)

    result = hybrid_minimal_knot_search(
        parameters,
        points,
        proposal,
        mse_tolerance=1e-12,
        **_search_kwargs(),
    )

    assert result.greedy_threshold_satisfied
    assert result.threshold_satisfied
    assert float(result.fit_mse) <= result.mse_tolerance
    assert result.final_count <= result.greedy_count
    assert result.retained_proposal_mask.dtype == torch.bool
    assert result.retained_proposal_mask.shape == proposal.shape
    assert int(result.retained_proposal_mask.sum()) == result.final_count
    torch.testing.assert_close(
        torch.nonzero(result.retained_proposal_mask).flatten(),
        result.retained_proposal_indices,
    )
    assert result.visited_state_count > 0
    assert result.refit_count >= result.visited_state_count
    assert result.levels_explored
    assert result.position_refined_counts
    assert result.final_source


def test_coordinate_refinement_is_legal_and_never_increases_mse() -> None:
    parameters = torch.linspace(0.0, 1.0, 81, dtype=DTYPE)
    source_knots = torch.tensor([0.35], dtype=DTYPE)
    initial_knots = torch.tensor([0.20], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)

    result = refine_knot_positions(
        parameters,
        points,
        initial_knots,
        smoothness_weight=0.0,
        min_gap=1e-4,
        sweeps=2,
        grid_size=5,
        restarts=2,
    )

    assert result.final_mse <= result.initial_mse
    assert result.final_mse < result.initial_mse
    refined = result.final_fit.internal_knots
    assert torch.all(refined > 0.0)
    assert torch.all(refined < 1.0)
    assert result.refit_count > 1
    assert result.sweeps_completed == 4


def test_infeasible_learned_start_falls_back_to_permanent_greedy_incumbent() -> None:
    parameters = torch.linspace(0.0, 1.0, 65, dtype=DTYPE)
    source_knots = torch.tensor([0.29, 0.73], dtype=DTYPE)
    proposal = torch.tensor([0.14, 0.29, 0.51, 0.73, 0.88], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)
    learned_mask = torch.zeros_like(proposal, dtype=torch.bool)

    result = hybrid_minimal_knot_search(
        parameters,
        points,
        proposal,
        deployment_knots=proposal.clone(),
        learned_mask=learned_mask,
        mse_tolerance=1e-12,
        **_search_kwargs(),
    )

    assert "learned" in result.start_sources
    assert result.greedy_threshold_satisfied
    assert result.threshold_satisfied
    assert result.final_count <= result.greedy_count
    assert result.final_count > 0
    assert float(result.fit_mse) <= 1e-12


def test_learned_mask_positions_are_refined_before_structure_is_rejected() -> None:
    parameters = torch.linspace(0.0, 1.0, 81, dtype=DTYPE)
    source_knots = torch.tensor([0.50], dtype=DTYPE)
    proposal = torch.tensor([0.10, 0.20, 0.80, 0.90], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)
    learned_mask = torch.tensor([False, True, False, False])

    result = hybrid_minimal_knot_search(
        parameters,
        points,
        proposal,
        deployment_knots=proposal.clone(),
        learned_mask=learned_mask,
        mse_tolerance=1e-12,
        smoothness_weight=0.0,
        beam_width=2,
        branch_factor=0,
        position_sweeps=2,
        position_grid_size=5,
        position_restarts=2,
        min_gap=1e-4,
    )

    assert "learned_refined" in result.start_sources
    assert result.threshold_satisfied
    assert result.final_count == 1
    torch.testing.assert_close(
        result.final_internal_knots,
        source_knots,
        atol=1e-12,
        rtol=0.0,
    )
    assert not torch.equal(
        result.final_internal_knots,
        result.retained_proposal_knots,
    )
    assert result.mean_absolute_position_shift > 0.19
    assert result.max_absolute_position_shift > 0.19


def test_deployment_order_is_validated_only_for_learned_survivors() -> None:
    parameters = torch.linspace(0.0, 1.0, 65, dtype=DTYPE)
    source_knots = torch.tensor([0.15, 0.55], dtype=DTYPE)
    proposal = torch.tensor([0.10, 0.30, 0.50, 0.70], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)
    learned_mask = torch.tensor([True, False, True, False])

    result = hybrid_minimal_knot_search(
        parameters,
        points,
        proposal,
        deployment_knots=torch.tensor([0.15, 0.95, 0.55, 0.25], dtype=DTYPE),
        learned_mask=learned_mask,
        mse_tolerance=1e-12,
        **_search_kwargs(),
    )

    assert "learned" in result.start_sources

    with pytest.raises(ValueError, match="non-decreasing"):
        hybrid_minimal_knot_search(
            parameters,
            points,
            proposal,
            deployment_knots=torch.tensor([0.55, 0.95, 0.15, 0.25], dtype=DTYPE),
            learned_mask=learned_mask,
            mse_tolerance=1e-12,
            **_search_kwargs(),
        )


def test_search_is_deterministic() -> None:
    parameters = torch.linspace(0.0, 1.0, 41, dtype=DTYPE)
    source_knots = torch.tensor([0.41], dtype=DTYPE)
    proposal = torch.tensor([0.18, 0.41, 0.77], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)

    first = hybrid_minimal_knot_search(
        parameters,
        points,
        proposal,
        mse_tolerance=1e-12,
        **_search_kwargs(),
    )
    second = hybrid_minimal_knot_search(
        parameters,
        points,
        proposal,
        mse_tolerance=1e-12,
        **_search_kwargs(),
    )

    torch.testing.assert_close(first.final_internal_knots, second.final_internal_knots)
    torch.testing.assert_close(first.fit_mse, second.fit_mse)
    torch.testing.assert_close(
        first.retained_proposal_indices,
        second.retained_proposal_indices,
    )
    assert first.levels_explored == second.levels_explored
    assert first.position_refined_counts == second.position_refined_counts
    assert first.refit_count == second.refit_count
    assert first.visited_state_count == second.visited_state_count


def test_zero_knot_proposal_is_a_valid_complete_search() -> None:
    parameters = torch.linspace(0.0, 1.0, 33, dtype=DTYPE)
    points = torch.stack(
        [parameters, 0.1 + 0.2 * parameters - 0.4 * parameters**2 + parameters**3],
        dim=-1,
    )
    proposal = torch.empty(0, dtype=DTYPE)

    result = hybrid_minimal_knot_search(
        parameters,
        points,
        proposal,
        mse_tolerance=1e-20,
        **_search_kwargs(),
    )

    assert result.threshold_satisfied
    assert result.greedy_threshold_satisfied
    assert result.final_count == result.greedy_count == 0
    assert result.final_internal_knots.numel() == 0
    assert result.retained_proposal_mask.shape == proposal.shape
    assert result.retained_proposal_indices.numel() == 0
    assert result.levels_explored == ()
