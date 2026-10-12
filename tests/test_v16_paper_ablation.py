from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_v16_paper_ablation import _forward_variant, summarize
from spline_fitting.models.geometry_encoder import GeometryEncoder
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


def test_feature_ablation_preserves_layout_and_zeros_only_requested_components():
    encoder = GeometryEncoder(point_dim=2, hidden_dim=16, num_layers=1)
    t = torch.linspace(0, 1, 12)
    points = torch.stack([t, torch.sin(4 * t)], dim=-1).unsqueeze(0)

    encoder.set_ablation_feature_components("all")
    complete = encoder._geometric_features(points)
    encoder.set_ablation_feature_components("coordinates_first")
    first_only = encoder._geometric_features(points)
    encoder.set_ablation_feature_components("coordinates")
    coordinates = encoder._geometric_features(points)

    assert complete.shape == first_only.shape == coordinates.shape == (1, 12, 6)
    torch.testing.assert_close(complete[..., :2], points)
    torch.testing.assert_close(first_only[..., :4], complete[..., :4])
    assert torch.count_nonzero(first_only[..., 4:]) == 0
    torch.testing.assert_close(coordinates[..., :2], points)
    assert torch.count_nonzero(coordinates[..., 2:]) == 0


def test_feature_ablation_rejects_unknown_mode():
    encoder = GeometryEncoder(point_dim=2, hidden_dim=16, num_layers=1)
    with pytest.raises(ValueError, match="feature ablation mode"):
        encoder.set_ablation_feature_components("curvature_only")


def test_module_interventions_keep_expected_tensor_contracts():
    torch.manual_seed(7)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16,
        encoder_layers=1,
        max_internal_knots=6,
        attention_heads=2,
        selector_layers=1,
        one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True,
        min_selected_knots=2,
    ).eval()
    t = torch.linspace(0, 1, 20)
    points = torch.stack([t, torch.sin(5 * t)], dim=-1).unsqueeze(0)

    params, knots, mask = _forward_variant(
        model, points, 5e-5, "module", "all_candidates"
    )
    assert params.shape == (1, 20)
    assert knots.shape == mask.shape == (1, 6)
    assert mask.all()

    raw_params, raw_knots, raw_mask = _forward_variant(
        model, points, 5e-5, "module", "no_survivor_decoder"
    )
    assert raw_params.shape == params.shape
    assert raw_knots.shape == raw_mask.shape == mask.shape
    assert torch.isfinite(raw_params).all() and torch.isfinite(raw_knots).all()


def test_summary_keeps_failed_curves_in_pass_rate_denominator():
    base = {"group": "module", "arm": "full", "label": "Complete network", "dataset": "Synthetic"}
    rows = [
        {**base, "status": "ok", "joint_pass": True, "k": 5, "mse": 1e-5,
         "max_squared_error": 1e-4, "network_ms": 1.0, "total_ms": 2.0},
        {**base, "status": "failed", "joint_pass": False},
    ]
    record = next(
        row for row in summarize(rows, seed=1)
        if row["group"] == "module" and row["arm"] == "full" and row["dataset"] == "All"
    )
    assert record["n"] == 2
    assert record["n_finite"] == 1
    assert record["failures"] == 1
    assert record["dual_pass_rate"] == pytest.approx(0.5)
