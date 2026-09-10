from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fit_v16_point_cloud as entry
from spline_fitting.checkpointing import V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_reference_mapping_uses_original_chord_not_resampled_polyline():
    class ParameterCurve:
        def evaluate(self, parameters):
            return torch.stack([parameters, torch.zeros_like(parameters)], -1)

    source = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 3]], dtype=torch.float64)
    params = torch.tensor([0, 0.2, 1], dtype=torch.float64)
    metrics = entry.original_reference_metrics(
        ParameterCurve(), params, source, torch.zeros(2, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    expected = torch.tensor([0, 0.1, 0.2, 1], dtype=torch.float64)
    torch.testing.assert_close(metrics["parameters"], expected)
    expected_points = torch.stack([expected, torch.zeros_like(expected)], -1)
    assert metrics["normalized_mse"] == pytest.approx(float((expected_points - source).square().sum(-1).mean()))
    assert metrics["source_units_mse"] == metrics["normalized_mse"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rejects_nonpositive_or_nonfinite_tolerance(value):
    with pytest.raises(ValueError, match="finite and positive"):
        entry.resolve_mse_tolerance({"model_config": {}}, value)


def test_tolerance_reads_saved_model_and_explicit_override():
    checkpoint = {"model_config": {"mse_tolerance": 1e-5}}
    assert entry.resolve_mse_tolerance(checkpoint, None) == 1e-5
    assert entry.resolve_mse_tolerance(checkpoint, 3e-5) == 3e-5


def test_missing_point_path_is_clear_cli_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        entry.main(["--point-cloud", str(tmp_path / "missing.csv")])
    assert error.value.code == 2
    assert "point-cloud file does not exist" in capsys.readouterr().err


@pytest.mark.parametrize("dimension", [2, 3])
def test_tiny_v16_checkpoint_exports_one_refit_with_correct_units(
    tmp_path, monkeypatch, dimension,
):
    torch.manual_seed(17)
    model = V16CandidateSelectionNetwork(
        point_dim=dimension, hidden_dim=16, encoder_layers=1, attention_heads=4,
        selector_layers=1, max_internal_knots=3,
    )
    checkpoint_path = tmp_path / "untrained_test_v16.pt"
    torch.save({
        "objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        "model_config": model.get_config(), "model_state_dict": model.state_dict(),
        "dataset_config": {"num_points": 24},
    }, checkpoint_path)
    parameter = np.linspace(0, 1, 37)
    coordinates = [10 + 5 * parameter, -2 + np.sin(3 * parameter)]
    if dimension == 3:
        coordinates.append(8 + parameter ** 2)
    source = np.stack(coordinates, axis=-1)
    input_path = tmp_path / "source.npy"
    np.save(input_path, source)
    output_dir = tmp_path / "fit"
    calls = {"fit": 0, "forward": 0}
    original_refit = entry.refit_bspline_control_points
    original_forward = V16CandidateSelectionNetwork.forward_deployment

    def counted_refit(*args, **kwargs):
        calls["fit"] += 1
        assert args[0].dtype == args[1].dtype == args[2].dtype == torch.float64
        assert args[0].device.type == "cpu"
        assert kwargs["smoothness_weight"] == kwargs["control_ridge"] == 0
        assert kwargs["interpolate_endpoints"]
        return original_refit(*args, **kwargs)

    def counted_forward(self, *args, **kwargs):
        calls["forward"] += 1
        assert kwargs["mse_tolerance"] == 1e-5
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(entry, "refit_bspline_control_points", counted_refit)
    monkeypatch.setattr(V16CandidateSelectionNetwork, "forward_deployment", counted_forward)
    base_command = ["--checkpoint", str(checkpoint_path), "--point-cloud", str(input_path),
                    "--output-dir", str(output_dir), "--device", "cpu", "--mse-tolerance", "1e-5",
                    "--network-warmups", "1", "--network-repeats", "2"]
    with pytest.raises(SystemExit) as error:
        entry.main(base_command)
    assert error.value.code == 2
    assert not output_dir.exists()
    command = [*base_command, "--allow-unqualified-diagnostic"]
    result = entry.main(command)
    assert calls == {"fit": 1, "forward": 4}
    assert result["final_refits"] == result["timing"]["total_refits_including_timing"] == 1
    assert result["original_point_count"] == 37
    assert result["network_point_count"] == 24
    assert result["point_dimension"] == dimension
    assert len(result["network_input_parameters"]) == 24
    assert len(result["original_reference_parameters"]) == 37
    assert len(result["internal_knots"]) == result["internal_knot_count"]
    vector = result["full_knot_vector"]
    assert vector[:4] == [0.0] * 4 and vector[-4:] == [1.0] * 4
    controls = np.asarray(result["control_points_source_units"])
    np.testing.assert_allclose(controls[[0, -1]], source[[0, -1]], atol=1e-6)
    scale = result["normalization"]["scale"]
    assert result["input_source_units_mse"] == pytest.approx(result["input_normalized_mse"] * scale ** 2)
    assert result["original_reference_source_units_mse"] == pytest.approx(
        result["original_reference_normalized_mse"] * scale ** 2
    )
    assert result["input_threshold_satisfied"] == (result["input_normalized_mse"] <= 1e-5)
    assert result["timing"]["deployment_ms"] > 0
    assert result["timing"]["network_ms"] > 0
    assert result["diagnostic_not_final"]
    assert not result["checkpoint_qualification"]["formal_reporting_eligible"]
    assert (output_dir / "fit.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    raw_report = (output_dir / "report.json").read_text(encoding="utf-8")
    assert json.loads(raw_report) == result
    assert "NaN" not in raw_report and "Infinity" not in raw_report
    before = (output_dir / "report.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        entry.main(command)
    assert error.value.code == 2
    assert (output_dir / "report.json").read_bytes() == before
    assert calls == {"fit": 1, "forward": 4}


def test_dimension_mismatch_fails_before_inference(tmp_path, capsys):
    model = V16CandidateSelectionNetwork(hidden_dim=16, encoder_layers=1,
                                         selector_layers=1, max_internal_knots=3)
    checkpoint_path = tmp_path / "v16.pt"
    torch.save({"objective_version": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
                "model_config": model.get_config(), "model_state_dict": model.state_dict()},
               checkpoint_path)
    point_path = tmp_path / "3d.npy"
    np.save(point_path, np.arange(15).reshape(5, 3))
    with pytest.raises(SystemExit) as error:
        entry.main(["--checkpoint", str(checkpoint_path), "--point-cloud", str(point_path),
                    "--output-dir", str(tmp_path / "output"), "--device", "cpu",
                    "--allow-unqualified-diagnostic"])
    assert error.value.code == 2
    assert "expects point dimension 2, got 3" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
