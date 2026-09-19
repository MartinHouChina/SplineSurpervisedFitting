"""Reporting/CLI fixtures only; these invented timings are not benchmark evidence."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import torch
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_v15_datasets as benchmark
import plot_v15_dataset_benchmark as plotting15
import plot_v16_method_comparison as plotting16
import visualize_v16_real_deployments as cases
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points


def benchmark_parser():
    return benchmark.parser(default_checkpoint=Path("fixture.pt"), default_output_dir=Path("fixture-output"))


@pytest.mark.parametrize("enabled", [True, False])
def test_benchmark_and_case_cli_forward_identical_safeguard_option(enabled):
    flag = "--published-feasibility-safeguard" if enabled else "--no-published-feasibility-safeguard"
    for parser in (benchmark_parser(), cases.parser()):
        assert parser.parse_args([]).published_feasibility_safeguard is True
        args = parser.parse_args([flag, "--max-internal-knots", "64", "--paper-initial-knots", "64"])
        for warmup in (False, True):
            kwargs = benchmark.published_baseline_kwargs(args, degree=3, warmup=warmup)
            assert kwargs["published_feasibility_safeguard"] is enabled
            assert kwargs["degree"] == 3


def test_protocol_fingerprint_changes_for_flag_and_evaluation_code(monkeypatch):
    args = benchmark_parser().parse_args([])
    protocol = benchmark.published_baseline_protocol(args)
    fingerprint = plotting16._metadata_fingerprint({"published_baseline_protocol": protocol})
    for name in ("published_baselines.py", "sparse_knot_paper.py", "dung_direct_knot.py", "luo_linf_de.py"):
        assert any(path.endswith("/" + name) or path.endswith("\\" + name)
                   for path in protocol["evaluation_code_sha256"])
    args.published_feasibility_safeguard = False
    disabled = benchmark.published_baseline_protocol(args)
    assert disabled["kang_long_cluster_correction_always_enabled"] is True
    assert plotting16._metadata_fingerprint({"published_baseline_protocol": disabled}) != fingerprint
    args.published_feasibility_safeguard = True
    monkeypatch.setattr(benchmark, "sha256_file", lambda path: "changed source hash")
    changed = benchmark.published_baseline_protocol(args)
    assert plotting16._metadata_fingerprint({"published_baseline_protocol": changed}) != fingerprint


def make_report(enabled=True):
    args = benchmark_parser().parse_args([])
    args.published_feasibility_safeguard = enabled
    metadata = {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "checkpoint": "report-fixture.pt", "epoch": 1,
        "mse_tolerance": 5e-5, "num_points": 96, "hardware": {"device": "cpu"},
        "published_baseline_protocol": benchmark.published_baseline_protocol(args),
        "configuration": {"published_feasibility_safeguard": enabled},
    }
    metadata["fingerprint"] = plotting16._metadata_fingerprint(metadata)
    rows = []
    for method in plotting16.PUBLISHED_METHODS:
        diagnostics = {}
        if method in benchmark.SAFEGUARDED_PUBLISHED_METHODS:
            diagnostics = {
                "comparison_feasibility_native_k": 4,
                "comparison_feasibility_native_mse": 7e-4 if enabled else 4e-5,
                "comparison_feasibility_safeguard_enabled": enabled,
                "comparison_feasibility_safeguard_used": enabled,
                "comparison_feasibility_refit_count": 54 if enabled else 0,
                "comparison_feasibility_final_source": "residual_guided_augmentation" if enabled else "native",
                "comparison_feasibility_status": "repaired_feasible" if enabled else "disabled",
                "comparison_feasibility_capacity": 64,
            }
        rows.append({
            "dataset": "Synthetic", "sample_id": "fixture-0", "method": method,
            "status": "ok", "mse": 4e-5, "fit_pass": True,
            "reference_mse": None, "reference_pass": None, "final_k": 22 if enabled else 4,
            "total_ms": 7.0, "network_ms": 1.0 if method == "ours" else None,
            "diagnostics": diagnostics,
        })
    return {"metadata": metadata, "summary": benchmark.summarize(rows), "measurements": rows}


@pytest.mark.parametrize("enabled", [True, False])
def test_native_final_report_audit_and_csv_are_explicit(tmp_path, enabled):
    report = make_report(enabled)
    before = copy.deepcopy(report)
    benchmark.write_reports(tmp_path, report["metadata"], report["measurements"])
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"]
    audit = saved["native_baseline_summary"]
    assert len(audit) == 3
    assert all(row["native_k"] == 4 and row["final_k"] == (22 if enabled else 4) for row in audit)
    assert all(row["extra_refit_count"] == (54 if enabled else 0) for row in audit)
    assert all(row["complete_ms"] == 7.0 for row in audit)
    with (tmp_path / "native_baseline_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 3 and csv_rows[0]["native_mse"]
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "native_baseline_summary.csv" in markdown
    assert "[threshold-safe adaptation]" in markdown if enabled else "[threshold-safe adaptation]" not in markdown
    assert report == before


def test_missing_native_diagnostics_stay_missing_in_audit():
    row = make_report()["measurements"][-1]
    row["diagnostics"] = {}
    audit = benchmark.native_baseline_audit([row])[0]
    assert audit["native_k"] is None and audit["native_mse"] is None
    assert audit["extra_refit_count"] is None and audit["safeguard_enabled"] is None


def test_historical_reports_are_not_relabelled_and_contradictions_rejected():
    assert plotting15.published_method_labels({}, plotting16.SHORT_LABELS) == plotting16.SHORT_LABELS
    assert "does not record" in plotting15.published_protocol_caption({})
    with pytest.raises(ValueError, match="mismatch"):
        plotting15.published_safeguard_mode({
            "published_baseline_protocol": {"feasibility_safeguard_enabled": True},
            "configuration": {"published_feasibility_safeguard": False},
        })


@pytest.mark.parametrize("renderer", [plotting15.render_report, plotting16.render_comparison])
def test_figures_label_safe_adaptations_and_keep_network_timing_separate(tmp_path, monkeypatch, renderer):
    report = make_report()
    before = copy.deepcopy(report)
    captured = []
    original = Figure.savefig
    def save(figure, *args, **kwargs):
        captured.append(figure)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", save)
    target = renderer(report, tmp_path, dpi=40)
    assert target.is_file()
    figure = captured[0]
    legend = " ".join(text.get_text() for entry in figure.legends for text in entry.get_texts())
    assert legend.count("[threshold-safe]") == 3
    text = " ".join(item.get_text() for item in figure.texts)
    assert "threshold-safe adaptations" in text
    assert "extra repair refits included in total time" in text
    assert "network only" in text
    assert report == before


def test_case_plot_shows_native_and_final_fit_and_refit_budget(tmp_path, monkeypatch):
    parameters = torch.linspace(0, 1, 24, dtype=torch.float64)
    points = torch.stack((parameters, 0.1 * parameters.square()), dim=-1)
    fit = refit_bspline_control_points(parameters, points, torch.empty(0, dtype=torch.float64),
                                     degree=3, smoothness_weight=0.0, control_ridge=0.0,
                                     interpolate_endpoints=True)
    case = {"dataset": "FIXTURE", "sample_id": "0", "points": points, "reference": points}
    report = make_report()
    result = dict(report["measurements"][-1], fit=fit, parameters=parameters,
                  published_feasibility_safeguard=True, error=None)
    figures = []
    original = Figure.savefig
    def save(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", save)
    cases.plot_case(tmp_path / "case.png", case=case, results=[result], tolerance=5e-5, diagnostic=True, dpi=50)
    title = figures[0].axes[0].get_title()
    assert "threshold-safe adaptation" in title
    assert "native K=4" in title and "extra refits=54" in title
    assert "K=22" in title and "MSE=4.00e-05" in title
    assert "not paper-original" in figures[0]._suptitle.get_text()
