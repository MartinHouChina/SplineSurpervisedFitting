from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve  # noqa: E402
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    build_open_knot_vector,
    warp_internal_knots_to_parameterization,
)
from spline_fitting.evaluation.verified_knot_repair import (  # noqa: E402
    verified_confidence_repair,
)


DTYPE = torch.float64
REPAIR_MODULE = importlib.import_module(
    "spline_fitting.evaluation.verified_knot_repair"
)


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


def _repair_inputs() -> tuple[torch.Tensor, ...]:
    parameters = torch.linspace(0.0, 1.0, 81, dtype=DTYPE)
    source_knots = torch.tensor([0.27, 0.68], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)
    proposal = torch.tensor([0.10, 0.27, 0.46, 0.68, 0.88], dtype=DTYPE)
    # Unselected slots deliberately make the complete deployment vector
    # unordered.  Only the selected subset is authoritative in v12.
    deployment = torch.tensor([0.72, 0.27, 0.18, 0.68, 0.35], dtype=DTYPE)
    mask = torch.tensor([False, True, False, False, False])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.05], dtype=DTYPE)
    return parameters, points, proposal, deployment, mask, scores


def test_verified_repair_returns_learned_fit_without_fallback_when_feasible() -> None:
    parameters = torch.linspace(0.0, 1.0, 41, dtype=DTYPE)
    points = torch.stack(
        [parameters, 0.2 - 0.3 * parameters + 0.4 * parameters**3], dim=-1
    )
    proposal = torch.tensor([0.2, 0.5, 0.8], dtype=DTYPE)
    deployment = torch.tensor([0.7, 0.5, 0.1], dtype=DTYPE)
    mask = torch.zeros(3, dtype=torch.bool)
    scores = torch.tensor([0.7, 0.2, 0.1], dtype=DTYPE)

    result = verified_confidence_repair(
        parameters,
        points,
        proposal,
        deployment,
        mask,
        scores,
        fit_tolerance_rms=1e-10,
        smoothness_weight=0.0,
        compact=False,
    )

    assert result.threshold_satisfied
    assert result.learned_threshold_satisfied
    assert not result.fallback_used
    assert result.final_source == "learned_verified"
    assert result.final_count == 0
    assert result.prefix_counts_evaluated == ()
    assert result.direct_refit_count == result.fit_evaluation_count == 1


def test_compact_simplifies_an_already_feasible_learned_subset() -> None:
    parameters, points, proposal, _, _, scores = _repair_inputs()
    mask = torch.ones(proposal.numel(), dtype=torch.bool)

    result = verified_confidence_repair(
        parameters,
        points,
        proposal,
        proposal,
        mask,
        scores,
        fit_tolerance_rms=1e-8,
        smoothness_weight=0.0,
        compact=True,
    )

    assert result.learned_threshold_satisfied
    assert result.threshold_satisfied
    assert result.cleanup_used
    assert not result.fallback_used
    assert result.final_source == "learned_verified_compact"
    assert result.final_count < int(mask.sum())
    assert int(result.retained_proposal_mask.sum()) == result.final_count
    assert result.fit_evaluation_count > result.direct_refit_count


def test_confidence_add_back_repairs_an_infeasible_learned_subset() -> None:
    parameters, points, proposal, deployment, mask, scores = _repair_inputs()

    result = verified_confidence_repair(
        parameters,
        points,
        proposal,
        deployment,
        mask,
        scores,
        fit_tolerance_rms=1e-8,
        smoothness_weight=0.0,
        compact=False,
    )

    assert result.fallback_used
    assert not result.learned_threshold_satisfied
    assert result.threshold_satisfied
    assert result.final_source == "confidence_add_back"
    assert result.final_count == 2
    torch.testing.assert_close(
        result.final_fit.internal_knots,
        torch.tensor([0.27, 0.68], dtype=DTYPE),
    )
    assert result.prefix_counts_evaluated[0] == 1
    assert proposal.numel() in result.prefix_counts_evaluated
    assert result.direct_refit_count == 1 + len(result.prefix_counts_evaluated)


def test_compact_repair_preserves_feasibility_and_never_adds_knots() -> None:
    parameters, points, proposal, deployment, mask, scores = _repair_inputs()

    fast = verified_confidence_repair(
        parameters,
        points,
        proposal,
        deployment,
        mask,
        scores,
        fit_tolerance_rms=1e-8,
        smoothness_weight=0.0,
        compact=False,
    )
    compact = verified_confidence_repair(
        parameters,
        points,
        proposal,
        deployment,
        mask,
        scores,
        fit_tolerance_rms=1e-8,
        smoothness_weight=0.0,
        compact=True,
    )

    assert fast.threshold_satisfied and compact.threshold_satisfied
    assert compact.cleanup_used
    assert compact.final_source.endswith("_compact")
    assert compact.final_count <= fast.final_count
    assert compact.fit_evaluation_count > fast.fit_evaluation_count


def test_minimum_count_is_enforced_even_when_learned_fit_passes() -> None:
    parameters = torch.linspace(0.0, 1.0, 41, dtype=DTYPE)
    points = torch.stack([parameters, parameters.square()], dim=-1)
    proposal = torch.tensor([0.2, 0.5, 0.8], dtype=DTYPE)
    mask = torch.zeros(3, dtype=torch.bool)
    scores = torch.tensor([0.2, 0.9, 0.4], dtype=DTYPE)

    result = verified_confidence_repair(
        parameters,
        points,
        proposal,
        proposal,
        mask,
        scores,
        fit_tolerance_rms=1.0,
        min_internal_knots=2,
        smoothness_weight=0.0,
        compact=True,
    )

    assert result.threshold_satisfied
    assert result.fallback_used
    assert result.final_count == 2


def test_invalid_selected_deployment_order_is_rejected() -> None:
    parameters, points, proposal, deployment, mask, scores = _repair_inputs()
    mask = torch.tensor([True, True, False, False, False])

    with pytest.raises(ValueError, match="selected deployment knots"):
        verified_confidence_repair(
            parameters,
            points,
            proposal,
            deployment,
            mask,
            scores,
            fit_tolerance_rms=0.005,
        )


def test_residual_fallback_inserts_sample_knot_only_after_full_proposal_fails() -> None:
    parameters = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)
    source_knots = torch.tensor([0.35], dtype=DTYPE)
    points = _sample_curve(parameters, source_knots)
    proposal = torch.tensor([0.10, 0.90], dtype=DTYPE)
    mask = torch.zeros(2, dtype=torch.bool)
    scores = torch.tensor([0.8, 0.2], dtype=DTYPE)

    # The complete proposal misses the RMS bound, whereas inserting the
    # highest-residual legal sample parameter makes the exact refit feasible.
    full_fit = refit_bspline_control_points(
        parameters,
        points,
        proposal,
        smoothness_weight=0.0,
    )
    squared_residual = (full_fit.reconstructed_points - points).square().sum(dim=-1)
    expected_inserted = parameters[int(torch.argmax(squared_residual).item())]
    tolerance = 0.037
    assert float(full_fit.fit_rmse) > tolerance
    assert 0.0 < float(expected_inserted) < 1.0
    assert torch.all((proposal - expected_inserted).abs() > 1e-3)

    result = verified_confidence_repair(
        parameters,
        points,
        proposal,
        proposal,
        mask,
        scores,
        fit_tolerance_rms=tolerance,
        smoothness_weight=0.0,
        compact=False,
        hard_fallback=False,
        residual_fallback=True,
        max_residual_insertions=1,
        residual_min_gap=1e-3,
    )

    assert result.threshold_satisfied
    assert result.residual_fallback_used
    assert result.residual_fallback_refit_count == 1
    torch.testing.assert_close(
        result.inserted_internal_knots,
        expected_inserted.reshape(1),
    )
    assert result.proposal_count == proposal.numel()
    assert result.deployment_candidate_count == proposal.numel() + 1
    assert result.deployment_retained_mask.dtype == torch.bool
    assert result.deployment_retained_mask.shape == (proposal.numel() + 1,)
    torch.testing.assert_close(
        result.deployment_retained_mask[: proposal.numel()],
        result.retained_proposal_mask,
    )
    assert bool(result.deployment_retained_mask[-1])
    assert int(result.deployment_retained_mask.sum()) == result.final_count
    torch.testing.assert_close(
        result.final_fit.internal_knots,
        torch.sort(torch.cat([proposal, expected_inserted.reshape(1)])).values,
    )


def test_residual_fallback_skips_peak_that_violates_minimum_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = torch.tensor([0.0, 0.19, 0.21, 0.50, 1.0], dtype=DTYPE)
    points = torch.zeros((parameters.numel(), 2), dtype=DTYPE)
    proposal = torch.tensor([0.20], dtype=DTYPE)
    mask = torch.zeros(1, dtype=torch.bool)
    scores = torch.zeros(1, dtype=DTYPE)
    refitted_knots: list[torch.Tensor] = []

    def fake_refit(
        _parameters: torch.Tensor,
        _points: torch.Tensor,
        internal_knots: torch.Tensor,
        **_kwargs: object,
    ) -> SimpleNamespace:
        knots = internal_knots.detach().clone()
        refitted_knots.append(knots)
        contains_legal_peak = bool(
            knots.numel()
            and torch.any(torch.isclose(knots, torch.tensor(0.50, dtype=DTYPE)))
        )
        reconstructed = torch.zeros_like(points)
        if not contains_legal_peak:
            # t=0.21 has the largest residual but is too close to proposal 0.20.
            # The second peak at t=0.50 is legal and must be inserted instead.
            reconstructed[2, 0] = 10.0
            reconstructed[3, 0] = 5.0
        mse = torch.tensor(0.0 if contains_legal_peak else 1.0, dtype=DTYPE)
        return SimpleNamespace(
            internal_knots=knots,
            reconstructed_points=reconstructed,
            fit_mse=mse,
            fit_rmse=torch.sqrt(mse),
        )

    monkeypatch.setattr(REPAIR_MODULE, "refit_bspline_control_points", fake_refit)
    result = verified_confidence_repair(
        parameters,
        points,
        proposal,
        proposal,
        mask,
        scores,
        fit_tolerance_rms=0.5,
        compact=False,
        hard_fallback=False,
        residual_fallback=True,
        max_residual_insertions=1,
        residual_min_gap=0.05,
    )

    assert result.threshold_satisfied
    assert result.residual_fallback_used
    torch.testing.assert_close(
        result.inserted_internal_knots,
        torch.tensor([0.50], dtype=DTYPE),
    )
    assert not torch.any(
        torch.isclose(
            result.inserted_internal_knots,
            torch.tensor(0.21, dtype=DTYPE),
        )
    )
    full_proposal_call = next(
        index
        for index, knots in enumerate(refitted_knots)
        if torch.equal(knots, proposal)
    )
    insertion_call = next(
        index for index, knots in enumerate(refitted_knots) if knots.numel() == 2
    )
    assert full_proposal_call < insertion_call
    assert result.proposal_count == 1
    assert result.deployment_candidate_count == 2
    assert result.deployment_retained_mask.tolist() == [True, True]


def test_residual_fallback_stops_cleanly_when_no_legal_sample_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = torch.tensor([0.0, 0.49, 0.50, 0.51, 1.0], dtype=DTYPE)
    points = torch.zeros((parameters.numel(), 2), dtype=DTYPE)
    proposal = torch.tensor([0.50], dtype=DTYPE)
    mask = torch.ones(1, dtype=torch.bool)
    scores = torch.ones(1, dtype=DTYPE)

    def always_infeasible(
        _parameters: torch.Tensor,
        _points: torch.Tensor,
        internal_knots: torch.Tensor,
        **_kwargs: object,
    ) -> SimpleNamespace:
        reconstructed = torch.ones_like(points)
        mse = torch.tensor(1.0, dtype=DTYPE)
        return SimpleNamespace(
            internal_knots=internal_knots.detach().clone(),
            reconstructed_points=reconstructed,
            fit_mse=mse,
            fit_rmse=torch.sqrt(mse),
        )

    monkeypatch.setattr(
        REPAIR_MODULE,
        "refit_bspline_control_points",
        always_infeasible,
    )
    result = verified_confidence_repair(
        parameters,
        points,
        proposal,
        proposal,
        mask,
        scores,
        fit_tolerance_rms=0.5,
        compact=False,
        hard_fallback=False,
        residual_fallback=True,
        max_residual_insertions=4,
        residual_min_gap=0.10,
    )

    assert not result.threshold_satisfied
    assert not result.residual_fallback_used
    assert result.residual_fallback_refit_count == 0
    assert result.inserted_internal_knots.numel() == 0
    assert result.proposal_count == result.deployment_candidate_count == 1
    assert result.deployment_retained_mask.shape == (1,)
    assert int(result.deployment_retained_mask.sum()) == result.final_count


def test_chord_parameterization_fallback_is_exact_and_preserves_domain() -> None:
    chord = torch.linspace(0.0, 1.0, 121, dtype=DTYPE)
    predicted = chord.square()
    source_knots = torch.tensor([0.28, 0.63], dtype=DTYPE)
    points = _sample_curve(chord, source_knots)
    proposal = warp_internal_knots_to_parameterization(
        source_knots,
        chord,
        predicted,
    )
    mask = torch.ones(proposal.numel(), dtype=torch.bool)
    scores = torch.ones_like(proposal)
    network_fit = refit_bspline_control_points(
        predicted,
        points,
        proposal,
        smoothness_weight=0.0,
    )
    chord_fit = refit_bspline_control_points(
        chord,
        points,
        source_knots,
        smoothness_weight=0.0,
    )
    assert float(network_fit.fit_rmse) > 1e-3
    assert float(chord_fit.fit_rmse) < 1e-10

    result = verified_confidence_repair(
        predicted,
        points,
        proposal,
        proposal,
        mask,
        scores,
        fit_tolerance_rms=0.5 * float(network_fit.fit_rmse),
        smoothness_weight=0.0,
        compact=False,
        hard_fallback=False,
        residual_fallback=False,
        alternate_parameters=chord,
        parameterization_policy="chord-fallback",
    )

    assert result.threshold_satisfied
    assert not result.learned_threshold_satisfied
    assert result.parameterization_policy == "chord-fallback"
    assert result.learned_parameterization == "network_predicted"
    assert result.final_parameterization == "chord_length"
    assert result.parameterization_fallback_attempted
    assert result.parameterization_fallback_used
    assert result.parameterization_fallback_full_threshold_satisfied
    assert result.final_source.startswith("chord_fallback_")
    torch.testing.assert_close(result.final_parameters, chord)
    torch.testing.assert_close(
        result.final_fit.internal_knots,
        source_knots,
        atol=1e-12,
        rtol=1e-12,
    )


def test_chord_policy_requires_monotone_complete_alternate_parameterization() -> None:
    parameters, points, proposal, deployment, mask, scores = _repair_inputs()
    invalid_chord = parameters.clone()
    invalid_chord[10] = invalid_chord[9] - 0.01

    with pytest.raises(ValueError, match="non-decreasing"):
        verified_confidence_repair(
            parameters,
            points,
            proposal,
            deployment,
            mask,
            scores,
            fit_tolerance_rms=0.005,
            alternate_parameters=invalid_chord,
            parameterization_policy="chord",
        )
