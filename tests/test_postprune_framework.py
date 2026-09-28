"""Architecture/diagram contracts for the selected K32 historical checkpoint."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("postprune_framework", ROOT / "scripts/plot_postprune_framework.py")
framework = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(framework)


@pytest.fixture
def contract_files(tmp_path, monkeypatch):
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"selected-checkpoint")
    data = {"model_config": dict(framework.REQUIRED), "epoch": 14,
            "objective_version": "candidate_selection_counterfactual_bspline_v16"}
    monkeypatch.setattr(framework.torch, "load", lambda *args, **kwargs: data)
    report = {
        "metadata": {
            "checkpoint_sha256": framework.sha256(checkpoint), "max_internal_knots": 32,
            "model_capacity": 32, "num_points": 192, "mse_tolerance": 5e-5,
            "max_squared_error_tolerance": 5e-4,
            "ours_post_pruning": {"enabled": True, "stage": "after_common_insertion",
                                  "min_internal_knots": 0, "max_deletions": 32},
            "endpoint_convention": {"ours": True}, "repair_protocol": "common insertion",
        }
    }
    report_path = tmp_path / "comparison.json"
    report_path.write_text(json.dumps(report), encoding="utf8")
    return checkpoint, report_path, data, report


def test_legacy_defaults_not_new_coupled_decoder(contract_files):
    checkpoint, report_path, _, _ = contract_files
    result = framework.load_contract(checkpoint, report_path)
    assert result["model_config_effective"]["subset_geometry_mode"] == "legacy"
    assert result["model_config_effective"]["coupled_subset_steps"] == 0
    assert result["architecture"]["candidate_queries"] == 33
    assert result["architecture"]["candidate_capacity"] == 32
    assert result["architecture"]["decoder_iterations"] == 1
    assert "surviving knot coordinates" in result["postprocessing"]["fixed_during_numerical_stages"]


@pytest.mark.parametrize("field,value", [("subset_geometry_mode", "anchored"),
                                        ("coupled_subset_steps", 2),
                                        ("max_internal_knots", 64),
                                        ("one_shot_selection_policy", "threshold")])
def test_reject_architectural_mismatch(contract_files, field, value):
    checkpoint, report_path, data, _ = contract_files
    data["model_config"][field] = value
    with pytest.raises(ValueError, match=field):
        framework.load_contract(checkpoint, report_path)


def test_reject_unmatched_checkpoint(contract_files):
    checkpoint, report_path, _, _ = contract_files
    checkpoint.write_bytes(b"other-weights")
    with pytest.raises(ValueError, match="not the benchmark checkpoint"):
        framework.load_contract(checkpoint, report_path)


@pytest.mark.parametrize("change", ["disabled", "wrong_order", "wrong_tolerance"])
def test_reject_wrong_numerical_protocol(contract_files, change):
    checkpoint, report_path, _, report = contract_files
    if change == "disabled":
        report["metadata"]["ours_post_pruning"]["enabled"] = False
    elif change == "wrong_order":
        report["metadata"]["ours_post_pruning"]["stage"] = "before_insertion"
    else:
        report["metadata"]["mse_tolerance"] = 1e-4
    report_path.write_text(json.dumps(report), encoding="utf8")
    with pytest.raises(ValueError):
        framework.load_contract(checkpoint, report_path)


def test_render_retains_auditable_metadata(contract_files, tmp_path):
    checkpoint, report_path, _, _ = contract_files
    metadata = framework.load_contract(checkpoint, report_path)
    output = tmp_path / "figures"
    result = framework.make_figure(metadata, output, dpi=50)
    assert (output / "framework.png").stat().st_size > 5000
    saved = json.loads((output / "framework_metadata.json").read_text(encoding="utf8"))
    assert saved["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert saved["postprocessing"]["ours_pruning"]["enabled"] is True
    assert result["pixel_size"] == [1035, 490]
