from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from spline_fitting.checkpointing import V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_v16_compact_decoder.py"
    spec = importlib.util.spec_from_file_location("audit_compact", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_uses_saved_inputs_and_does_not_mutate_checkpoint(audit, tmp_path):
    torch.manual_seed(3)
    previous = torch.get_num_threads()
    try:
        model = V16CandidateSelectionNetwork(
            hidden_dim=16, encoder_layers=1, selector_layers=1,
            attention_heads=2, max_internal_knots=4, mse_tolerance=5e-5,
        )
        checkpoint = tmp_path / "model.pt"
        torch.save(dict(objective_version=V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                        model_config=model.get_config(), model_state_dict=model.state_dict(),
                        epoch=3), checkpoint)
        original = checkpoint.read_bytes()
        t = torch.linspace(0, 1, 24)
        points = torch.stack([t, t * .2], -1).tolist()
        cases = tmp_path / "cases.json"
        cases.write_text(json.dumps(dict(records=[
            dict(dataset="line", sample_id=str(i), network_input_points_normalized=points)
            for i in range(3)
        ])), encoding="utf-8")
        report = tmp_path / "audit.json"
        args = ["--checkpoint", str(checkpoint), "--cases-json", str(cases),
                "--output", str(report), "--samples-per-dataset", "2"]
        assert audit.main(args) == 0
        result = json.loads(report.read_text(encoding="utf-8"))
        assert len(result["records"]) == 2
        assert result["role"].startswith("diagnostic_only")
        for row in result["records"]:
            assert row["fixed_geometry_k"] <= row["deployment_k"]
            assert row["deleted_knots"] == row["deployment_k"] - row["fixed_geometry_k"]
            assert row["redecoded_mse"] >= 0
        assert checkpoint.read_bytes() == original
        before = report.read_bytes()
        with pytest.raises(SystemExit):
            audit.main(args)
        assert report.read_bytes() == before
        with pytest.raises(SystemExit):
            audit.main(["--checkpoint", str(checkpoint), "--cases-json", str(cases),
                        "--dataset", "missing"])
    finally:
        torch.set_num_threads(previous)
