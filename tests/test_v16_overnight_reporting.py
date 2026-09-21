"""Report/orchestration regression tests; mocked timings are never benchmark data."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_v15_datasets as benchmark  # noqa: E402
import plot_v16_method_comparison as plotting  # noqa: E402


def test_published_method_set_matches_renderer_and_preserves_all_default():
    args = benchmark.parser().parse_args(["--checkpoint", "model.pt", "--output-dir", "out"])
    assert args.method_set == "all"
    assert len(benchmark.METHODS) == 8
    assert benchmark.PUBLISHED_METHODS == plotting.PUBLISHED_METHODS
    assert len(benchmark.PUBLISHED_METHODS) == 6


def test_canonical_truth_and_timing_include_failed_cases_not_just_successes():
    good = dict(dataset="Synthetic", sample_id="low-k", method="ours", status="ok",
                mse=1e-6, fit_pass=True, reference_mse=None, reference_pass=None,
                canonical_k=4, final_k=5, total_ms=2.0, network_ms=1.0)
    failed = dict(good, sample_id="high-k", status="failed", mse=None, fit_pass=False,
                  canonical_k=16, final_k=None, total_ms=10.0, network_ms=None)
    summary = benchmark.summarize([good, failed])[0]
    assert summary["canonical_k_mean"] == 10.0
    assert summary["canonical_count_mae"] == 1.0
    assert summary["final_k_mean"] == 5.0
    assert summary["total_ms_mean"] == 6.0
    assert summary["fit_pass_rate"] == 0.5
    assert summary["failed"] == 1


@pytest.fixture
def paired_report():
    rows = []
    for dataset in plotting.DATASET_ORDER:
        for sample_index in range(2):
            for method in benchmark.PUBLISHED_METHODS:
                real = dataset != "Synthetic"
                failed = method == benchmark.PUBLISHED_METHODS[-1]
                passed = not failed and sample_index == 0
                rows.append(dict(
                    dataset=dataset, sample_id=f"{dataset}-{sample_index}", method=method,
                    status="failed" if failed else "ok", has_reference=real,
                    mse=None if failed else (1e-6 if passed else 1e-3), fit_pass=passed,
                    reference_mse=None if failed or not real else (2e-6 if passed else 2e-3),
                    reference_pass=passed if real and not failed else None,
                    canonical_k=8 + sample_index * 8 if not real else None,
                    final_k=None if failed else 5 + sample_index,
                    total_ms=10.0 if failed else 2.0 + sample_index,
                    network_ms=0.5 if method == "ours" else None,
                ))
    metadata = dict(
        objective_version=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        checkpoint="TEST_FIXTURE_NOT_A_MEASURED_CHECKPOINT.pt", epoch=1,
        mse_tolerance=2.5e-5, num_points=24, hardware={"device": "cpu"},
        diagnostic_not_final=True, configuration={"force_diagnostic": True},
        checkpoint_qualification={"formal_reporting_eligible": True},
        diagnostic_reasons=["Forced diagnostic: quick or reduced benchmark protocol"],
        datasets=[dict(dataset=name, selected_count=2) for name in plotting.DATASET_ORDER],
    )
    metadata["fingerprint"] = plotting._metadata_fingerprint(metadata)
    return dict(metadata=metadata, measurements=rows, summary=benchmark.summarize(rows))


def test_six_methods_four_datasets_preserve_entire_failed_method_and_render(
    tmp_path, paired_report, monkeypatch,
):
    plotting.validate_v16_benchmark(paired_report, allow_unqualified_diagnostic=True)
    before = copy.deepcopy(paired_report)
    captured = []
    original_savefig = Figure.savefig

    def capture(self, *args, **kwargs):
        captured.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    for reference in (False, True):
        path = plotting.render_comparison(paired_report, tmp_path, dpi=50, reference=reference)
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paired_report == before
    for fig, dataset_count in zip(captured, (4, 3)):
        assert len(fig.axes) == 4
        assert len(fig.axes[1].patches) == 6 * dataset_count
        assert len(fig.axes[3].patches) == 6 * dataset_count
        assert len(fig.axes[0].patches) == 5 * dataset_count
        assert len(fig.axes[2].patches) == 5 * dataset_count
        assert any("REDUCED BENCHMARK PROTOCOL" in text.get_text() for text in fig.texts)
        assert any("N/A" in text.get_text() for text in fig.axes[0].texts)
    # Canonical reference K must remain visible even above all retained-K bars.
    assert captured[0].axes[2].get_ylim()[1] > 12.0


def test_all_failed_reference_method_stays_in_markdown_table(tmp_path, paired_report):
    benchmark.write_reports(tmp_path, paired_report["metadata"], paired_report["measurements"])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | — |" in text
    assert "quick or reduced benchmark protocol" in text
    assert "UNQUALIFIED CHECKPOINT" not in text


def test_plot_rejects_dropped_failure_even_when_summary_matches_remaining_rows(paired_report):
    report = copy.deepcopy(paired_report)
    report["measurements"] = [row for row in report["measurements"] if not (
        row["dataset"] == "Synthetic" and row["sample_id"] == "Synthetic-1"
        and row["method"] == benchmark.PUBLISHED_METHODS[-1]
    )]
    report["summary"] = benchmark.summarize(report["measurements"])
    with pytest.raises(ValueError, match="Incomplete selected cases"):
        plotting.validate_v16_benchmark(report, allow_unqualified_diagnostic=True)


def test_plot_rejects_unpaired_ids_and_fabricated_pass_denominator(paired_report):
    report = copy.deepcopy(paired_report)
    report["measurements"][0]["sample_id"] = "different-case"
    with pytest.raises(ValueError, match="Unpaired"):
        plotting.validate_v16_benchmark(report, allow_unqualified_diagnostic=True)
    report = copy.deepcopy(paired_report)
    report["summary"][0]["fit_pass_rate"] = 1.0
    with pytest.raises(ValueError, match="fit_pass_rate"):
        plotting.validate_v16_benchmark(report, allow_unqualified_diagnostic=True)


def test_plot_rejects_dropped_declared_dataset(paired_report):
    for key in ("measurements", "summary"):
        paired_report[key] = [row for row in paired_report[key] if row["dataset"] != "USGS"]
    with pytest.raises(ValueError, match="incomplete"):
        plotting.validate_v16_benchmark(paired_report, allow_unqualified_diagnostic=True)


@pytest.mark.parametrize("method_set", ["published", "all"])
def test_benchmark_executes_selected_methods_and_records_diagnostic_protocol(
    tmp_path, monkeypatch, method_set,
):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"TEST FIXTURE")
    model = SimpleNamespace(degree=3)
    model.to = lambda device: model
    model.eval = lambda: model
    config = {"max_internal_knots": 16, "structure_mode": "candidate_pruning_one_shot"}
    monkeypatch.setattr(benchmark.torch, "load", lambda *a, **k: dict(
        objective_version=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        model_config=config, epoch=1,
    ))
    qualification_calls = []

    def qualify(*args, **kwargs):
        qualification_calls.append(kwargs)
        return {"formal_reporting_eligible": True, "reasons": []}, False

    monkeypatch.setattr(benchmark, "validate_checkpoint_for_benchmark", qualify)
    monkeypatch.setattr(benchmark, "build_model_from_checkpoint", lambda cp: (model, config, None))
    case = dict(dataset="Synthetic", sample_id="case-0", group_id="g0", source_k=8,
                canonical_k=8, points=torch.zeros(24, 2), reference=None)
    monkeypatch.setattr(benchmark, "prepare_cases", lambda *a: (
        [case], [{"dataset": "Synthetic", "selected_count": 1}],
    ))
    warmups, measured = [], []
    monkeypatch.setattr(benchmark, "run_published_baseline", lambda method, *a, **k: warmups.append(method))
    # The K32 error contract evaluates both MSE and peak error from fitted
    # points, rather than trusting fit_mse alone. Keep this orchestration
    # fixture consistent with its declared residual (0.001, 0).
    fit = SimpleNamespace(
        fit_mse=1e-6, internal_knots=torch.tensor([0.5]),
        evaluate=lambda parameters: torch.stack([
            torch.full_like(parameters, 0.001), torch.zeros_like(parameters),
        ], dim=-1),
    )
    monkeypatch.setattr(benchmark, "measure_ours", lambda *a, **k: (
        fit, torch.linspace(0, 1, 24), 3.0, 1.0, {},
    ))

    def measure(method, *args, **kwargs):
        measured.append(method)
        return fit, torch.linspace(0, 1, 24), 4.0, None, {}

    monkeypatch.setattr(benchmark, "measure_numerical_baseline", measure)
    output = tmp_path / method_set
    benchmark.main([
        "--checkpoint", str(checkpoint), "--output-dir", str(output),
        "--method-set", method_set, "--force-diagnostic", "--device", "cpu",
    ], expected_objective=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION)
    methods = benchmark.PUBLISHED_METHODS if method_set == "published" else benchmark.METHODS
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert warmups == list(methods[1:])
    assert measured == list(methods[1:])
    assert [row["method"] for row in report["measurements"]] == list(methods)
    assert all(row["mse"] == pytest.approx(1e-6) for row in report["measurements"])
    assert all(row["max_squared_error"] == pytest.approx(1e-6)
               for row in report["measurements"])
    assert report["metadata"]["methods"] == list(methods)
    assert set(report["metadata"]["method_labels"]) == set(methods)
    assert report["metadata"]["diagnostic_not_final"]
    assert qualification_calls[0]["allow_unqualified_diagnostic"] is False
    plotting.validate_v16_benchmark(report, allow_unqualified_diagnostic=True)

    # A watermark override never permits a different network structure.
    config["structure_mode"] = "not_the_historical_one_shot_model"
    rejected_output = tmp_path / "structurally_rejected"
    with pytest.raises(SystemExit) as error:
        benchmark.main([
            "--checkpoint", str(checkpoint), "--output-dir", str(rejected_output),
            "--force-diagnostic", "--device", "cpu",
        ], expected_objective=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION)
    assert error.value.code == 2
    assert not rejected_output.exists()
