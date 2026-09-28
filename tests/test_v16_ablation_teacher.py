from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork
from spline_fitting.training import v16_ablation_teacher as teacher


def _geometry(dtype=torch.float32):
    params = torch.linspace(0, 1, 65, dtype=dtype).pow(1.3).unsqueeze(0)
    points = torch.stack((params, 0.1 * (2 * torch.pi * params).sin()), dim=-1)
    knots = torch.linspace(0.08, 0.92, 10, dtype=dtype).unsqueeze(0)
    return points, params, knots


def test_teacher_compacts_with_dual_bounds_and_preserves_exact_geometry():
    points, params, knots = _geometry()
    original = [tensor.clone() for tensor in (points, params, knots)]
    record = teacher.build_teacher_from_geometry(points, params, knots)[0]
    assert record["feasible"] and record["dense_pass"]
    assert 0 < record["count"] < knots.shape[-1]
    assert record["mse"] <= 5e-5 and record["maxse"] <= 5e-4
    assert record["steps"] == knots.shape[-1] - record["count"]
    assert record["refits"] >= 1 + record["steps"]
    assert record["mask"].dtype == torch.bool
    assert torch.equal(record["params"], params[0])
    assert torch.equal(record["knots"], knots[0])
    assert record["params"].dtype == params.dtype
    fit = refit_bspline_control_points(
        record["params"].double(), points[0].double(),
        record["knots"][record["mask"]].double(), smoothness_weight=0.0,
    )
    squared = (fit.reconstructed_points - points[0].double()).square().sum(-1)
    assert float(squared.mean()) == pytest.approx(record["mse"], rel=1e-8)
    assert float(squared.max()) == pytest.approx(record["maxse"], rel=1e-8)
    for value, before in zip((points, params, knots), original):
        assert torch.equal(value, before)
    record["params"].zero_()
    record["knots"].zero_()
    assert torch.equal(params, original[1]) and torch.equal(knots, original[2])


def test_dense_infeasible_never_invents_allkeep_positive_labels():
    points, params, knots = _geometry()
    points[0, 31, 1] = 100.0
    record = teacher.build_teacher_from_geometry(points, params, knots)[0]
    assert not record["feasible"] and not record["dense_pass"]
    assert not bool(record["mask"].any()) and record["count"] == 0
    assert record["steps"] == 0 and record["refits"] == 1
    assert record["termination"] == "dense_infeasible"
    assert record["dense_maxse"] > 5e-4


def test_anchor_and_minimum_count_constraints_are_preserved():
    points, params, knots = _geometry()
    points = torch.stack((params, params), dim=-1)
    anchors = torch.zeros_like(knots, dtype=torch.bool)
    anchors[0, [0, 4, 9]] = True
    record = teacher.build_teacher_from_geometry(
        points, params, knots, anchors=anchors, min_selected_knots=4,
    )[0]
    assert record["count"] == 4 and record["termination"] == "minimum_count"
    assert bool(record["mask"][anchors[0]].all())
    only_anchors = teacher.build_teacher_from_geometry(
        points, params, knots, anchors=anchors,
    )[0]
    assert torch.equal(only_anchors["mask"], anchors[0])
    assert only_anchors["termination"] == "anchors_only"


def test_budget_counts_accepted_deletions():
    points, params, knots = _geometry()
    record = teacher.build_teacher_from_geometry(points, params, knots, max_deletions=1)[0]
    assert record["steps"] == 1 and record["count"] == 9
    assert record["termination"] == "deletion_budget"


def test_peak_rejection_scans_later_candidates_and_exactly_checks_batch_failures():
    points, params, knots = _geometry()
    calls = iter([(1e-6, 1e-5), (1e-6, 1e-3), (2e-6, 2e-5)])
    with patch.object(teacher, "_refit_errors", side_effect=lambda *args: next(calls)), patch.object(
        teacher, "single_knot_deletion_mse_batch", return_value=torch.ones(1, 10),
    ):
        record = teacher.build_teacher_from_geometry(points, params, knots, max_deletions=1)[0]
    # Batch scores claim every deletion is infeasible; production verification
    # rejects the first on peak error, but accepts the next under both bounds.
    assert record["mask"][0] and not record["mask"][1]
    assert record["feasible"] and record["refits"] == 3


def test_numerical_and_invalid_rows_are_explicit_failures_without_losing_denominator():
    points, params, knots = _geometry()
    points = points.repeat(2, 1, 1)
    points[0, 1, 0] = float("nan")
    records = teacher.build_teacher_from_geometry(points, params.repeat(2, 1), knots.repeat(2, 1))
    assert len(records) == 2 and not records[0]["feasible"] and records[1]["feasible"]
    assert records[0]["termination"] == "invalid_geometry"
    with patch.object(teacher, "_refit_errors", side_effect=RuntimeError("lstsq failed")):
        record = teacher.build_teacher_from_geometry(points[1:], params, knots)[0]
    assert not record["feasible"] and record["termination"] == "dense_refit_failed"
    assert record["refit_failures"] == 1


def test_fixed_seed_teacher_is_reproducible_and_preserves_model_modes():
    torch.manual_seed(123)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_coverage_bins=2, min_selected_knots=2,
    )
    points, _, _ = _geometry()
    model.train()
    model.encoder.eval()
    modes = [module.training for module in model.modules()]
    kwargs = dict(mse_tolerance=1.0, max_squared_error_tolerance=1.0)
    with patch.object(model, "decode_subset", side_effect=AssertionError("teacher must not decode")):
        first = teacher.build_fixed_teacher(model, points, **kwargs)[0]
        torch.manual_seed(123)
        second = teacher.build_fixed_teacher(model, points, **kwargs)[0]
    assert [module.training for module in model.modules()] == modes
    for key in ("mask", "params", "knots", "anchors"):
        assert torch.equal(first[key], second[key])
    assert first["feasible"] and first["count"] == 2
    assert all(parameter.grad is None for parameter in model.parameters())
    assert bool(first["mask"][first["anchors"]].all())


def test_structural_misuse_is_not_silently_converted_to_failed_labels():
    points, params, knots = _geometry()
    with pytest.raises(ValueError, match="at most 32"):
        teacher.build_teacher_from_geometry(points, params, torch.linspace(0.01, 0.99, 33)[None])
    with pytest.raises(ValueError, match="positive and finite"):
        teacher.build_teacher_from_geometry(points, params, knots, mse_tolerance=float("nan"))
    with pytest.raises(ValueError, match="max_deletions"):
        teacher.build_teacher_from_geometry(points, params, knots, max_deletions=-1)
