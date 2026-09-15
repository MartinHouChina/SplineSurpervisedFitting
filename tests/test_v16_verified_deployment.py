from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import bspline_basis_matrix
from spline_fitting.evaluation.knot_diagnostics import (
    build_open_knot_vector,
    warp_internal_knots_to_parameterization,
)
from spline_fitting.evaluation.v16_verified_deployment import (
    verify_v16_one_shot_subset,
)


def _sample_curve() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = torch.linspace(0, 1, 41, dtype=torch.float64)
    true_knots = torch.tensor([0.25, 0.75], dtype=torch.float64)
    control = torch.tensor(
        [[0, 0], [0.14, 0.8], [0.35, -0.5], [0.55, 0.9],
         [0.76, -0.7], [1.0, 0]],
        dtype=torch.float64,
    )
    points = bspline_basis_matrix(
        parameters, build_open_knot_vector(true_knots, 3), 3, 6
    ) @ control
    return parameters, true_knots, points


def _output(
    decoded_params: torch.Tensor,
    proposal_params: torch.Tensor,
    proposal_knots: torch.Tensor,
    *,
    mask: tuple[bool, ...],
) -> dict[str, torch.Tensor]:
    warped = warp_internal_knots_to_parameterization(
        proposal_knots, proposal_params, decoded_params
    )
    return {
        "params": decoded_params.unsqueeze(0),
        "proposal_params": proposal_params.unsqueeze(0),
        "proposal_internal_knots": proposal_knots.unsqueeze(0),
        "warped_proposal_internal_knots": warped.unsqueeze(0),
        "internal_knots": warped.unsqueeze(0),
        "learned_keep_mask": torch.tensor([mask], dtype=torch.bool),
        "keep_probabilities": torch.tensor(
            [[0.95, 0.05]], dtype=torch.float64
        ),
    }


def test_v16_verifier_preserves_raw_result_and_adds_candidate_only_on_failure() -> None:
    params, knots, points = _sample_curve()
    output = _output(params, params, knots, mask=(True, False))

    result = verify_v16_one_shot_subset(
        output, points, mse_tolerance=1e-10, network_only_ms=0.4,
        proposal_domain_fallback=False,
    )

    assert not result.raw_pass
    assert result.verified_pass
    assert result.status == "repaired_decoded_domain"
    assert result.raw_k == 1
    assert result.verified_k == 2
    assert result.raw_mse > 1e-10
    assert result.verified_mse <= 1e-10
    assert result.network_only_ms == 0.4
    assert result.additional_standard_refit_count >= 1
    assert result.proposal_domain_fallback_attempted is False
    assert result.diagnostics()["globally_minimum_k_certified"] is False
    assert result.method_labels == (
        "ours_one_shot", "ours_mse_verified_numerical_repair"
    )


def test_v16_verifier_explicitly_switches_to_proposal_domain_if_decoded_fails() -> None:
    params, knots, points = _sample_curve()
    decoded_params = params.square()
    output = _output(decoded_params, params, knots, mask=(True, False))

    result = verify_v16_one_shot_subset(
        output, points, mse_tolerance=1e-10, proposal_domain_fallback=True,
    )

    assert not result.raw_pass
    assert result.verified_pass
    assert result.status == "repaired_proposal_domain"
    assert result.verified_parameterization == "frozen_proposal"
    assert result.proposal_domain_fallback_attempted
    assert result.decoded_full_candidate_mse is not None
    assert result.decoded_full_candidate_mse > 1e-10
    torch.testing.assert_close(result.verified_parameters, params)
    assert result.verified_k == 2
    assert result.verified_mse <= 1e-10
    assert result.direct_standard_refit_count >= 3


def test_v16_verifier_reports_infeasible_complete_candidate_budget() -> None:
    params = torch.linspace(0, 1, 41, dtype=torch.float64)
    points = torch.stack([params, torch.sin(12 * math.pi * params)], dim=-1)
    knots = torch.tensor([0.25, 0.75], dtype=torch.float64)
    output = _output(params, params, knots, mask=(True, False))

    result = verify_v16_one_shot_subset(
        output, points, mse_tolerance=1e-10, proposal_domain_fallback=True,
    )

    assert not result.raw_pass
    assert not result.verified_pass
    assert result.status == "candidate_budget_infeasible"
    assert result.decoded_full_candidate_mse is not None
    assert result.proposal_full_candidate_mse is not None
    assert result.decoded_full_candidate_mse > 1e-10
    assert result.proposal_full_candidate_mse > 1e-10
    assert result.verified_mse > 1e-10
    assert result.diagnostics()["verified_pass"] is False


def test_v16_verifier_keeps_already_feasible_one_shot_mask() -> None:
    params, knots, points = _sample_curve()
    output = _output(params, params, knots, mask=(True, True))

    result = verify_v16_one_shot_subset(
        output, points, mse_tolerance=1e-10,
    )

    assert result.raw_pass and result.verified_pass
    assert result.status == "raw_feasible"
    assert result.raw_k == result.verified_k == 2
    assert result.direct_standard_refit_count == 1
    assert result.additional_standard_refit_count == 0
    assert result.decoded_full_candidate_mse is None
    assert not result.proposal_domain_fallback_attempted


def test_v16_verifier_marks_optional_compaction_as_numerical_work() -> None:
    params, knots, points = _sample_curve()
    output = _output(params, params, knots, mask=(True, True))

    result = verify_v16_one_shot_subset(
        output, points, mse_tolerance=1.0, compact=True,
    )

    assert result.raw_pass and result.verified_pass
    assert result.verified_k < result.raw_k
    assert result.status == "numerically_compacted"
    assert result.fit_evaluation_count > result.direct_standard_refit_count


def test_v16_verifier_checks_unrooted_tolerance_and_source_inputs() -> None:
    params, knots, points = _sample_curve()
    output = _output(params, params, knots, mask=(True, False))
    with pytest.raises(ValueError, match="mse_tolerance"):
        verify_v16_one_shot_subset(output, points, mse_tolerance=0)
    with pytest.raises(ValueError, match="network_only_ms"):
        verify_v16_one_shot_subset(
            output, points, mse_tolerance=1e-5, network_only_ms=-1,
        )
    with pytest.raises(ValueError, match="sample_index"):
        verify_v16_one_shot_subset(
            output, points, mse_tolerance=1e-5, sample_index=1,
        )
    with pytest.raises(ValueError, match="points"):
        verify_v16_one_shot_subset(
            output, points[0], mse_tolerance=1e-5,
        )
