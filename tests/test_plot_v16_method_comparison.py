from __future__ import annotations

import copy
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_v16_method_comparison as plotting  # noqa: E402


@pytest.fixture
def report() -> dict:
    summary = []
    measurements = []
    for dataset in ("Synthetic", "UJI"):
        for index, method in enumerate(plotting.PUBLISHED_METHODS):
            summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": 1,
                    "failed": 0,
                    "mse_mean": 8e-6 * (index + 1),
                    "fit_pass_rate": 1.0 if index < 3 else 0.0,
                    "reference_mse_mean": (
                        9e-6 * (index + 1) if dataset == "UJI" else None
                    ),
                    "reference_pass_rate": (
                        1.0 if dataset == "UJI" and index < 2 else (
                            0.0 if dataset == "UJI" else None
                        )
                    ),
                    "final_k_mean": float(10 + index),
                    "canonical_k_mean": 8.5 if dataset == "Synthetic" else None,
                    "total_ms_mean": float(2 + 4 * index),
                    "network_ms_mean": 1.25 if method == "ours" else None,
                }
            )
            measurements.append(
                {
                    "dataset": dataset,
                    "sample_id": f"{dataset.lower()}-0",
                    "method": method,
                    "status": "ok",
                }
            )
    metadata = {
        "objective_version": "candidate_selection_counterfactual_bspline_v16",
        "checkpoint": "outputs/checkpoints/formal-v16.pt",
        "checkpoint_sha256": "0" * 64,
        "diagnostic_not_final": False,
        "mse_tolerance": 2.5e-5,
        "num_points": 192,
        "hardware": {"device": "cuda", "gpu": "test GPU", "threads": 1},
        "datasets": [],
    }
    metadata["fingerprint"] = plotting._metadata_fingerprint(metadata)
    return {
        "metadata": metadata,
        "summary": summary,
        "measurements": measurements,
    }


def test_formal_validation_rejects_relabelled_partial_or_tampered_reports(report) -> None:
    plotting.validate_v16_benchmark(report)

    v15 = copy.deepcopy(report)
    v15["metadata"]["objective_version"] = (
        "candidate_pruning_deployment_aligned_feedback_v15"
    )
    v15["metadata"]["fingerprint"] = plotting._metadata_fingerprint(v15["metadata"])
    with pytest.raises(ValueError, match="native v16"):
        plotting.validate_v16_benchmark(v15)

    partial = copy.deepcopy(report)
    missing_method = plotting.PUBLISHED_METHODS[-1]
    partial["summary"] = [
        row for row in partial["summary"] if row["method"] != missing_method
    ]
    with pytest.raises(ValueError, match="incomplete"):
        plotting.validate_v16_benchmark(partial)

    tampered = copy.deepcopy(report)
    tampered["metadata"]["mse_tolerance"] = 1e-3
    with pytest.raises(ValueError, match="fingerprint"):
        plotting.validate_v16_benchmark(tampered)


def test_unqualified_report_requires_explicit_diagnostic_opt_in(report) -> None:
    report["metadata"]["diagnostic_not_final"] = True
    report["metadata"]["fingerprint"] = plotting._metadata_fingerprint(
        report["metadata"]
    )
    with pytest.raises(ValueError, match="unqualified"):
        plotting.validate_v16_benchmark(report)
    plotting.validate_v16_benchmark(
        report,
        allow_unqualified_diagnostic=True,
    )


def test_four_metric_renderer_writes_input_and_reference_pngs_without_mutation(
    tmp_path: Path,
    report: dict,
) -> None:
    before = copy.deepcopy(report)
    outputs = [
        plotting.render_comparison(report, tmp_path, dpi=40, reference=False),
        plotting.render_comparison(report, tmp_path, dpi=40, reference=True),
    ]

    assert [path.name for path in outputs] == [
        "v16_published_methods_input.png",
        "v16_published_methods_reference.png",
    ]
    for path in outputs:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        assert width >= 500
        assert height >= 300
    assert report == before


def test_only_synthetic_canonical_k_is_used_as_reference(report: dict) -> None:
    index = {
        (row["dataset"], row["method"]): row for row in report["summary"]
    }
    assert plotting._synthetic_canonical_k(
        index,
        plotting.PUBLISHED_METHODS,
    ) == pytest.approx(8.5)

    index[("Synthetic", plotting.PUBLISHED_METHODS[-1])]["canonical_k_mean"] = 9.0
    with pytest.raises(ValueError, match="inconsistent"):
        plotting._synthetic_canonical_k(index, plotting.PUBLISHED_METHODS)


def test_industrial_ticks_disclose_generation_not_measurement():
    assert "procedural, not measured" in plotting._dataset_tick_label("IndustrialOffset", {})
    metadata = {"datasets": [{"dataset": "CustomOffset", "source_kind": "procedural_cad_offset"}]}
    assert "procedural, not measured" in plotting._dataset_tick_label("CustomOffset", metadata)
    assert plotting._dataset_tick_label("UJI", {}) == plotting.DATASET_LABELS.get("UJI", "UJI")


def test_five_source_comparison_renders_with_honest_industrial_label(tmp_path, report, monkeypatch):
    from matplotlib.figure import Figure

    for name in ("NaturalEarth", "USGS", "IndustrialOffset"):
        report["summary"].extend({**row, "dataset": name} for row in list(report["summary"])
                                 if row["dataset"] == "UJI")
    figures = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = plotting.render_comparison(report, tmp_path, dpi=60)
    assert output.is_file()
    tick_labels = [tick.get_text() for axis in figures[0].axes for tick in axis.get_xticklabels()]
    assert any("IndustrialOffset" in label and "not measured" in label for label in tick_labels)
    assert {row["dataset"] for row in report["summary"]} == {
        "Synthetic", "UJI", "NaturalEarth", "USGS", "IndustrialOffset"}


def test_reference_renderer_skips_report_without_reference_rows(
    tmp_path: Path,
    report: dict,
) -> None:
    report["summary"] = [
        row for row in report["summary"] if row["dataset"] == "Synthetic"
    ]

    assert plotting.render_comparison(report, tmp_path, reference=True) is None
    assert not (tmp_path / "v16_published_methods_reference.png").exists()


def _add_peak_measurements(report):
    """Known test values intentionally unlike MSE to catch incorrect aliases."""
    for row, record in zip(report["summary"], report["measurements"]):
        peak = row["mse_mean"] * 7.0
        reference_peak = peak * 1.4 if row["dataset"] == "UJI" else None
        record.update(max_squared_error=peak, reference_max_squared_error=reference_peak,
                      has_reference=row["dataset"] == "UJI")
        for suffix in ("mean", "p95", "max"):
            row[f"max_squared_error_{suffix}"] = peak
            row[f"reference_max_squared_error_{suffix}"] = reference_peak


def test_peak_metric_validator_audits_source_observations_not_mse(report):
    _add_peak_measurements(report)
    plotting.validate_v16_benchmark(report)
    report["summary"][0]["max_squared_error_max"] = report["summary"][0]["mse_mean"]
    with pytest.raises(ValueError, match="max_squared_error_max differs"):
        plotting.validate_v16_benchmark(report)


def test_peak_metric_validator_rejects_invented_historical_maximum(report):
    report["summary"][0]["max_squared_error_max"] = report["summary"][0]["mse_mean"]
    with pytest.raises(ValueError, match="max_squared_error_max differs"):
        plotting.validate_v16_benchmark(report)


def test_peak_metric_validator_requires_complete_coverage_for_dataset_maximum(report):
    _add_peak_measurements(report)
    # Add a second paired curve without a newly measured pointwise maximum.
    # The first curve's finite maximum must not stand in for the entire dataset.
    extra = [{**row, "sample_id": row["sample_id"] + "-missing",
              "max_squared_error": None, "reference_max_squared_error": None}
             for row in report["measurements"]]
    report["measurements"].extend(extra)
    for row in report["summary"]:
        row["n"] = 2
        for prefix in ("max_squared_error", "reference_max_squared_error"):
            for suffix in ("mean", "p95", "max"):
                row[f"{prefix}_{suffix}"] = None
    plotting.validate_v16_benchmark(report)
    report["summary"][0]["max_squared_error_max"] = report["measurements"][0]["max_squared_error"]
    with pytest.raises(ValueError, match="max_squared_error_max differs"):
        plotting.validate_v16_benchmark(report)


def test_cli_emits_four_and_five_metric_views_by_default(report, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(plotting, "read_report", lambda _: report)
    monkeypatch.setattr(plotting, "render_comparison", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["plot_v16_method_comparison.py", "--input", str(tmp_path / "comparison.json")])
    plotting.main()
    assert [(call["reference"], call.get("include_max_error", False)) for call in calls] == [
        (False, False), (False, True), (True, False), (True, True)]


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("has_peak", [False, True])
def test_five_metric_renderer_uses_recorded_maxima_or_na(tmp_path, report, monkeypatch,
                                                       reference, has_peak):
    from matplotlib.figure import Figure

    if has_peak:
        _add_peak_measurements(report)
    before = copy.deepcopy(report)
    figures = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = plotting.render_comparison(report, tmp_path, dpi=50, reference=reference,
                                        include_max_error=True)
    suffix = "reference" if reference else "input"
    assert output.name == f"v16_published_methods_five_metrics_{suffix}.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    peak_axis = figures[0].axes[4]
    assert "Worst observed point" in peak_axis.get_title()
    explanation = " ".join(text.get_text() for text in figures[0].axes[5].texts)
    assert "not the largest curve-level MSE" in explanation
    assert "pass criterion" in explanation
    if has_peak:
        key = "reference_max_squared_error_max" if reference else "max_squared_error_max"
        datasets = plotting._ordered_datasets(report["summary"], reference=reference)
        expected = [next(row[key] for row in report["summary"]
                         if row["method"] == method and row["dataset"] == dataset)
                    for method in plotting.PUBLISHED_METHODS for dataset in datasets]
        assert [patch.get_height() for patch in peak_axis.patches] == pytest.approx(expected)
        assert not any(text.get_text() == "N/A" for text in peak_axis.texts)
    else:
        assert not peak_axis.patches
        assert all(text.get_text() == "N/A" for text in peak_axis.texts)
        assert peak_axis.texts
    assert report == before
