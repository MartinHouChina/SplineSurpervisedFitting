from __future__ import annotations

import copy
import json
import struct
import sys
from pathlib import Path

import pytest
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_v15_dataset_benchmark as plotting  # noqa: E402


def test_version_labels_follow_objective_without_relabeling_unknown_models():
    assert plotting.model_version({
        "objective_version": "candidate_pruning_deployment_aligned_feedback_v15",
    }) == "v15"
    assert plotting.model_version({
        "objective_version": "candidate_selection_counterfactual_bspline_v16",
    }) == "v16"
    assert plotting.model_version({}) == "v15"
    with pytest.raises(ValueError, match="Unsupported"):
        plotting.model_version({"objective_version": "unknown"})


@pytest.fixture
def report() -> dict:
    """Small paired report, including smoke-era metadata without num_points."""
    rows = []
    for dataset in ("Synthetic", "UJI"):
        for j, method in enumerate(plotting.METHODS):
            rows.append({
                "dataset": dataset,
                "method": method,
                "n": 2,
                "failed": 0,
                "mse_mean": 1e-5 * (j + 1),
                "fit_pass_rate": 1.0 if j < 2 else 0.5,
                "reference_mse_mean": 2e-5 * (j + 1) if dataset == "UJI" else None,
                "reference_pass_rate": 0.5 if dataset == "UJI" else None,
                "total_ms_mean": 10.0 * (j + 1),
                "network_ms_mean": 3.0 if method == "ours" else None,
            })
    return {
        "metadata": {
            "mse_tolerance": 2.5e-5,
            "datasets": [{"dataset": "Synthetic", "config": {"num_points": 32}}],
            "hardware": {"device": "cpu", "threads": 1},
        },
        "summary": rows,
    }


def _write_report(tmp_path: Path, report: dict) -> Path:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_read_report_preserves_measurements_and_nullable_failed_mse(tmp_path, report) -> None:
    report["summary"][1].update(failed=1, mse_mean=None, fit_pass_rate=0.0, total_ms_mean=8.5)
    report["summary"][2]["mse_mean"] = 0.0
    parsed = plotting.read_report(_write_report(tmp_path, report))

    assert parsed == report
    assert parsed["summary"][1]["mse_mean"] is None
    assert parsed["summary"][1]["total_ms_mean"] == 8.5
    assert plotting._num_points(parsed["metadata"]) == "32"


@pytest.mark.parametrize("tolerance", [None, 0.0, -1e-5, float("nan"), float("inf")])
def test_read_report_rejects_invalid_threshold(tmp_path, report, tolerance) -> None:
    report["metadata"]["mse_tolerance"] = tolerance
    with pytest.raises(ValueError, match="mse_tolerance"):
        plotting.read_report(_write_report(tmp_path, report))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mse_mean", -1e-5),
        ("reference_mse_mean", float("nan")),
        ("total_ms_mean", float("inf")),
        ("network_ms_mean", -1.0),
        ("fit_pass_rate", 1.1),
        ("reference_pass_rate", -0.1),
    ],
)
def test_read_report_rejects_invalid_metrics(tmp_path, report, field, value) -> None:
    report["summary"][0][field] = value
    with pytest.raises(ValueError, match=field):
        plotting.read_report(_write_report(tmp_path, report))


def test_read_report_rejects_empty_duplicate_and_unknown_method_rows(tmp_path, report) -> None:
    empty = copy.deepcopy(report)
    empty["summary"] = []
    with pytest.raises(ValueError, match="no completed summary"):
        plotting.read_report(_write_report(tmp_path, empty))

    duplicate = copy.deepcopy(report)
    duplicate["summary"].append(copy.deepcopy(duplicate["summary"][0]))
    with pytest.raises(ValueError, match="Duplicate summary"):
        plotting.read_report(_write_report(tmp_path, duplicate))

    unknown = copy.deepcopy(report)
    unknown["summary"][0]["method"] = "unregistered_method"
    with pytest.raises(ValueError, match="Unsupported method"):
        plotting.read_report(_write_report(tmp_path, unknown))


def test_render_writes_both_pngs_without_changing_measured_values(tmp_path, report) -> None:
    before = copy.deepcopy(report)
    outputs = [
        plotting.render_report(report, tmp_path, dpi=40, reference=False),
        plotting.render_report(report, tmp_path, dpi=40, reference=True),
    ]

    assert [path.name for path in outputs] == [
        "comparison_three_metrics.png", "comparison_reference_metrics.png",
    ]
    for path in outputs:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        assert width >= 500 and height >= 200
    assert report == before


def test_reference_plot_is_skipped_when_reference_metrics_are_absent(tmp_path, report) -> None:
    report["summary"] = [row for row in report["summary"] if row["dataset"] == "Synthetic"]

    assert plotting.render_report(report, tmp_path, reference=True) is None
    assert not (tmp_path / "comparison_reference_metrics.png").exists()


def test_diagnostic_report_is_watermarked(tmp_path, report, monkeypatch) -> None:
    report["metadata"]["diagnostic_not_final"] = True
    labels = []
    original = Figure.text

    def capture(self, x, y, text, *args, **kwargs):
        labels.append(text)
        return original(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Figure, "text", capture)
    plotting.render_report(report, tmp_path, dpi=40)
    assert any("DIAGNOSTIC NOT FINAL" in text for text in labels)
