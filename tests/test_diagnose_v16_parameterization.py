from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from spline_fitting.checkpointing import V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_v16_parameterization.py"
    spec = importlib.util.spec_from_file_location("parameterization_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warp_transports_all_knots_without_overwriting_outer_knots(diagnostic):
    old = torch.tensor([0., .25, 1.], dtype=torch.float64)
    new = torch.tensor([0., .5, 1.], dtype=torch.float64)
    knots = torch.tensor([.1, .5, .9], dtype=torch.float64)
    actual = diagnostic.warp_knots(knots, old, new)
    torch.testing.assert_close(actual, torch.tensor([.2, 2 / 3, 14 / 15], dtype=torch.float64))
    torch.testing.assert_close(diagnostic.warp_knots(knots[:1], old, new), actual[:1])
    assert diagnostic.warp_knots(knots[:0], old, new).numel() == 0
    torch.testing.assert_close(diagnostic.warp_knots(knots, old, old), knots)
    with pytest.raises(ValueError, match="strictly increasing"):
        diagnostic.warp_knots(knots, torch.tensor([0., 0., 1.]), new)


def test_chord_is_not_assumed_better_and_both_arms_share_count(diagnostic):
    t = torch.linspace(0, 1, 80, dtype=torch.float64)
    points = torch.stack([t, t ** 3], -1)
    knots = torch.tensor([.3, .65], dtype=torch.float64)
    result = diagnostic.compare_geometry(points, t, knots, degree=3, tolerance=5e-5)
    assert result["learned"]["mse"] < 1e-20
    assert result["chord_warped"]["mse"] > result["learned"]["mse"]
    assert result["learned"]["k"] == result["chord_warped"]["k"] == 2
    assert result["chord_warped_knots"] != result["learned_knots"]
    assert not result["chord_mse_better"]


def test_two_checkpoints_same_cases_and_read_only_outputs(diagnostic, tmp_path):
    previous = torch.get_num_threads()
    try:
        torch.manual_seed(12)
        model = V16CandidateSelectionNetwork(hidden_dim=16, encoder_layers=1,
            selector_layers=1, attention_heads=2, max_internal_knots=4, mse_tolerance=5e-5)
        checkpoints = [tmp_path / "a.pt", tmp_path / "b.pt"]
        for path in checkpoints:
            torch.save(dict(objective_version=V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                model_config=model.get_config(), model_state_dict=model.state_dict(), epoch=3), path)
        before = [p.read_bytes() for p in checkpoints]
        t = torch.linspace(0, 1, 24)
        cases = tmp_path / "cases.json"
        cases.write_text(json.dumps(dict(records=[dict(dataset="line", sample_id=str(i),
            network_input_points_normalized=torch.stack([t, .2 * t], -1).tolist()) for i in range(3)])),
            encoding="utf-8")
        output = tmp_path / "diagnostic.json"
        args = ["--checkpoint", str(checkpoints[0]), "--checkpoint", str(checkpoints[1]),
                "--cases-json", str(cases), "--samples-per-dataset", "2", "--output", str(output)]
        assert diagnostic.main(args) == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["role"].startswith("diagnostic_only")
        assert len(report["records"]) == 4
        assert [r["sample_content_sha256"] for r in report["records"][:2]] == [
            r["sample_content_sha256"] for r in report["records"][2:]]
        for row in report["records"]:
            assert sum(row["learned_keep_mask"]) == row["final"]["learned"]["k"]
            for geometry in ("final", "dense"):
                assert row[geometry]["learned"]["k"] == row[geometry]["chord_warped"]["k"]
        assert before == [p.read_bytes() for p in checkpoints]
        original_output = output.read_bytes()
        with pytest.raises(SystemExit):
            diagnostic.main(args)
        assert output.read_bytes() == original_output
    finally:
        torch.set_num_threads(previous)


def test_case_hash_mismatch_fails_closed(diagnostic, tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(dict(records=[dict(dataset="line", sample_id="0",
        network_input_points_normalized=[[0., 0.], [1., 1.]])])), encoding="utf-8")
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(dict(measurements=[dict(dataset="line", sample_id="0", method="ours")],
        metadata=dict(sample_content_sha256=["not the correct hash"], datasets=[]))), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        diagnostic.load_cases(cases, comparison, 8)
    result = diagnostic.load_cases(cases, comparison, 8, allow_content_mismatch=True)
    assert not result[0]["remote_content_hash_matches"]
    assert result[0]["remote_content_sha256"] == "not the correct hash"
