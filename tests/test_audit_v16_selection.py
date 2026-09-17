from __future__ import annotations

# ruff: noqa: E402

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_v16_selection", ROOT / "scripts/audit_v16_selection.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

from spline_fitting.checkpointing import V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork
from spline_fitting.training.one_shot_teacher import build_one_shot_teacher_batch, save_one_shot_teacher_cache


def _model():
    torch.manual_seed(37)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1,
        one_shot_adaptive_threshold=True, one_shot_selection_policy="mass_topk",
        initial_keep_fraction=0.5, min_selected_knots=2,
    ).eval()
    with torch.no_grad():
        model.parameter_update[-1].weight.normal_(std=0.03)
        model.relocation_update.weight.normal_(std=0.03)
    return model


def _points():
    t = torch.linspace(0, 1, 24)
    return torch.stack([t, (5 * t).sin()], -1)


def test_native_audit_reuses_one_mask_and_standard_cpu_double_refits():
    model = _model()
    with patch.object(model, "select_mask", wraps=model.select_mask) as select:
        with patch.object(model, "decode_subset", wraps=model.decode_subset) as decode:
            with patch.object(
                MODULE, "refit_bspline_control_points", wraps=MODULE.refit_bspline_control_points,
            ) as refit:
                result = MODULE.audit_model(
                    model, _points(), device=torch.device("cpu"), tolerance=1e-4,
                )
    assert select.call_count == decode.call_count == 1
    assert refit.call_count == 5
    for call in refit.call_args_list:
        assert all(value.device.type == "cpu" and value.dtype == torch.float64 for value in call.args)
        assert call.kwargs["smoothness_weight"] == call.kwargs["control_ridge"] == 0
        assert call.kwargs["interpolate_endpoints"] is True
    selected_count = len(result["selected_slots"])
    for name in ("free_mask_original", "free_mask_parameter_only", "free_mask_knots_only", "free_mask_decoded"):
        assert result[name]["k"] == selected_count
        assert result[name]["geometry"]["valid"]
        assert result[name]["mse"] is not None
    assert result["teacher"] is None and result["teacher_count_topk"] is None
    assert result["refinement_change"]["mse_change"] == pytest.approx(
        result["free_mask_decoded"]["mse"] - result["free_mask_original"]["mse"]
    )


def test_motion_removes_parameter_domain_change():
    original_params = torch.tensor([0., 0.2, 0.6, 1.], dtype=torch.float64)
    decoded_params = torch.tensor([0., 0.3, 0.8, 1.], dtype=torch.float64)
    original_knots = original_params[1:3]
    decoded_knots = decoded_params[1:3]
    assert (decoded_knots - original_knots).abs().mean() > 0.1
    motion = MODULE._refinement_motion(original_params, original_knots, decoded_params, decoded_knots)
    assert motion["mean_abs"] == pytest.approx(0., abs=1e-15)
    assert motion["max_abs"] == pytest.approx(0., abs=1e-15)
    assert motion["domain"] == "normalized_observation_index"


def test_invalid_knots_are_reported_without_sorting_or_repair():
    parameters = torch.linspace(0, 1, 10, dtype=torch.float64)
    knots = torch.tensor([0.7, 0.3], dtype=torch.float64)
    with patch.object(MODULE, "refit_bspline_control_points") as solver:
        result = MODULE._fit(parameters, torch.zeros(10, 2), knots, degree=3, tolerance=1e-4)
    assert solver.call_count == 0
    assert not result["geometry"]["knots_strict"]
    assert result["mse"] is None and not result["pass"]


def test_explicit_teacher_compares_teacher_mask_and_same_count_learned_ranking():
    model = _model()
    result = MODULE.audit_model(
        model, _points(), device=torch.device("cpu"), tolerance=0.05, build_greedy_teacher=True,
    )
    teacher = result["teacher"]
    count_topk = result["teacher_count_topk"]
    assert teacher is not None and count_topk is not None
    assert teacher["matches_training_anchor_counterfactual_teacher"] is False
    assert teacher["original"]["k"] == teacher["decoded"]["k"]
    assert count_topk["requested_count"] == teacher["original"]["k"]
    assert count_topk["original"]["k"] == count_topk["decoded"]["k"] == teacher["original"]["k"]
    assert teacher["original"]["pass"]


def test_teacher_training_cache_is_validated_but_never_reused_as_heldout_labels(tmp_path):
    model = _model()
    points = _points().unsqueeze(0)
    with torch.inference_mode():
        context = model.encode_candidates(points)
    config = MODULE.OneShotTeacherConfig(
        error_tolerance=0.1, min_internal_knots=2, smoothness_weight=0,
        proposal_fingerprint=MODULE._proposal_fingerprint(model), dataset_fingerprint="fixture_training_data",
    )
    batch = build_one_shot_teacher_batch(
        context["proposal_params"].double(), points.double(), context["proposal_internal_knots"].double(),
        sample_indices=[7], config=config,
    )
    path = tmp_path / "teacher.pt"
    save_one_shot_teacher_cache(path, batch)
    statistics = MODULE.teacher_cache_statistics(path, model, tolerance=1e-4)
    assert statistics["proposal_fingerprint_matches_new_model"]
    assert statistics["used_as_diagnostic_case_labels"] is False
    assert statistics["sample_count"] == 1
    assert statistics["candidate_count"] == 6
    assert statistics["count_mean"] == int(batch.teacher_count[0])
    with torch.no_grad():
        model.candidate_head.interval_score.weight.add_(0.01)
    mismatch = MODULE.teacher_cache_statistics(path, model, tolerance=1e-4)
    assert not mismatch["proposal_fingerprint_matches_new_model"]


def test_small_real_checkpoint_cli_is_paired_and_outputs_stratified_metrics(tmp_path):
    model = _model()
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        "model_config": model.get_config(), "model_state_dict": model.state_dict(),
        "dataset_config": {"num_points": 24, "certified_minimal_source": False},
    }, checkpoint)
    output = tmp_path / "audit.json"
    arguments = [
        "--old-checkpoint", str(checkpoint), "--new-checkpoint", str(checkpoint),
        "--output-json", str(output), "--device", "cpu", "--source-k", "4",
        "--samples-per-k", "1", "--torch-num-threads", "1",
    ]
    assert MODULE.main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["old"] == row["new"]
    assert row["synthetic_seed"] == 9_100_000_000 + 4 * 100_003
    assert len(row["points_sha256"]) == 64
    assert report["summary"]["by_source_k"]["4"]["n"] == 1
    assert report["summary"]["overall"]["new_minus_old_decoded"]["mse_change_mean"] == 0
    assert report["summary"]["high_k_ge_40"]["n"] == 0
    with pytest.raises(FileExistsError):
        MODULE.main(arguments)
