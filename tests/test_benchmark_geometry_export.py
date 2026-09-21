"""Small synthetic fixtures validate exports; fixture timings are not research results."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_v15_datasets as benchmark  # noqa: E402
import visualize_v16_six_methods as plotter  # noqa: E402
import plot_v16_method_comparison as aggregate_plotter  # noqa: E402
from benchmark_geometry import load_geometry_artifact, write_case_geometry, write_method_geometry  # noqa: E402
from spline_fitting.data.real_world import write_curve_manifest  # noqa: E402


@pytest.fixture
def measured_case():
    parameters = torch.linspace(0, 1, 24, dtype=torch.float64)
    points = torch.stack((parameters, .2 * torch.sin(2 * torch.pi * parameters)), dim=-1)
    fit = benchmark.refit_bspline_control_points(parameters, points, torch.tensor([.4, .6], dtype=torch.float64),
                                               smoothness_weight=0., control_ridge=0., interpolate_endpoints=True)
    case = {"dataset": "TEST_FIXTURE", "dataset_label": "TEST FIXTURE (not research data)",
            "sample_id": "curve", "group_id": "group", "points": points, "reference": points,
            "reference_grid": parameters, "reference_original": points * 7 + torch.tensor([5., -2.]),
            "center": torch.tensor([5., -2.]), "scale": torch.tensor(7.), "split": "test",
            "source_k": None, "canonical_k": None, "source_kind": "unit_test_fixture"}
    row = {"dataset": case["dataset"], "sample_id": case["sample_id"], "method": "ours", "status": "ok",
           **benchmark.fit_error_metrics(fit, parameters, case), "final_k": 2, "fit_pass": True,
           "total_ms": 2., "network_ms": 1., "diagnostics": {"test_fixture": True}}
    return case, row, fit, parameters


def test_export_is_lossless_reconstructible_and_preserves_coordinate_transform(tmp_path, measured_case):
    case, row, fit, parameters = measured_case
    source = write_case_geometry(tmp_path, case, fingerprint="test")
    result = write_method_geometry(tmp_path, case, row, fit, parameters,
                                   case_artifact=source, fingerprint="test", dense_points=64)
    document, arrays = load_geometry_artifact(tmp_path, result)
    source_document, source_arrays = load_geometry_artifact(tmp_path, source)
    assert document["measurement"] == row
    assert document["errors_original"]["mse"] == pytest.approx(row["mse"] * 49)
    assert source_document["normalization"]["scale"] == 7
    np.testing.assert_array_equal(arrays["control_points_normalized"], fit.control_points.numpy())
    np.testing.assert_array_equal(arrays["sample_parameters"], parameters.numpy())
    np.testing.assert_array_equal(source_arrays["reference_points_original"], case["reference_original"].numpy())
    np.testing.assert_allclose(arrays["control_points_original"], fit.control_points.numpy() * 7 + [5, -2])
    np.testing.assert_allclose(arrays["input_squared_residuals_original"], arrays["input_squared_residuals_normalized"] * 49)
    assert arrays["input_squared_residuals_normalized"].mean() == pytest.approx(row["mse"])
    assert arrays["input_squared_residuals_normalized"].max() == pytest.approx(row["max_squared_error"])
    assert arrays["reference_squared_residuals_normalized"].max() == pytest.approx(row["reference_max_squared_error"])
    from spline_fitting.data.synthetic import bspline_basis_matrix
    basis = bspline_basis_matrix(torch.from_numpy(arrays["dense_parameters"]), torch.from_numpy(arrays["full_knot_vector"]),
                                 document["spline"]["degree"], num_control_points=len(arrays["control_points_normalized"]))
    np.testing.assert_allclose(basis.numpy() @ arrays["control_points_normalized"], arrays["dense_curve_normalized"])


@pytest.mark.parametrize("status", ["failed", "unavailable"])
def test_unsolved_case_has_source_but_no_invented_spline_or_time(tmp_path, measured_case, status):
    case, _, _, _ = measured_case
    source = write_case_geometry(tmp_path, case, fingerprint="test")
    row = {"dataset": case["dataset"], "sample_id": case["sample_id"], "method": "not_run", "status": status,
           "mse": None, "max_squared_error": None, "total_ms": None, "network_ms": None}
    artifact = write_method_geometry(tmp_path, case, row, None, None, case_artifact=source,
                                     fingerprint="test", dense_points=64)
    document, arrays = load_geometry_artifact(tmp_path, artifact)
    assert document["spline"] is None and not arrays
    assert document["measurement"] == row
    assert load_geometry_artifact(tmp_path, document["case_artifact"])[1]["input_points_normalized"].shape == (24, 2)


def test_corrupt_geometry_rejected(tmp_path, measured_case):
    case, _, _, _ = measured_case
    source = write_case_geometry(tmp_path, case, fingerprint="test")
    path = tmp_path / source["path"]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_geometry_artifact(tmp_path, source)


@pytest.mark.parametrize("all_test", [False, True])
def test_sampling_uses_only_test_split_and_all_test_opt_in(tmp_path, monkeypatch, all_test):
    records = []
    for i in range(7):
        points = np.array([[0., 0.], [.4, i + .1], [1., 1.]], dtype=np.float64)
        filename = tmp_path / f"points_{i}.npy"
        np.save(filename, points)
        records.append({"sample_id": f"curve_{i}", "group_id": f"group_{i}", "split": "test" if i < 5 else "train",
                        "source_dataset": "test_fixture", "points_path": str(filename), "num_points": 3,
                        "point_dim": 2, "has_knot_labels": False, "metadata": {}})
    manifest = write_curve_manifest(records, tmp_path / "manifest.jsonl")
    args = benchmark.parser().parse_args(["--checkpoint", "unused", "--output-dir", "unused", "--skip-synthetic",
                                         "--manifest", f"Fixture={manifest}", "--real-samples-per-dataset", "2"])
    args.all_real_test_samples = all_test
    monkeypatch.setattr(benchmark, "_dataset_config_from_checkpoint", lambda *a: {"num_points": 12, "point_dim": 2})
    cases, sources = benchmark.prepare_cases(args, {}, {})
    assert len(cases) == (5 if all_test else 2)
    assert all(case["split"] == "test" and case["sample_id"] not in {"curve_5", "curve_6"} for case in cases)
    assert sources[0]["available_test_curves"] == 5
    assert sources[0]["all_test_samples"] is all_test
    again, _ = benchmark.prepare_cases(args, {}, {})
    assert [case["sample_id"] for case in cases] == [case["sample_id"] for case in again]


def test_native_benchmark_exports_unavailable_and_plotter_uses_saved_results_only(tmp_path, monkeypatch, measured_case):
    case, _, fit, parameters = measured_case
    checkpoint_path = tmp_path / "unit-test-only.pt"
    checkpoint_path.write_bytes(b"MOCK CHECKPOINT FOR TESTING ONLY")
    config = {"max_internal_knots": 8, "degree": 3, "structure_mode": "candidate_pruning_one_shot"}
    checkpoint = {"objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION, "model_config": config}
    model = SimpleNamespace(degree=3)
    model.to = lambda *a: model
    model.eval = lambda: model
    monkeypatch.setattr(benchmark.torch, "load", lambda *a, **k: checkpoint)
    monkeypatch.setattr(benchmark, "build_model_from_checkpoint", lambda *a: (model, config, None))
    monkeypatch.setattr(benchmark, "validate_checkpoint_for_benchmark", lambda *a, **k: (None, False))
    monkeypatch.setattr(benchmark, "prepare_cases", lambda *a: ([case, dict(case, sample_id="curve2")], []))
    calls = []
    monkeypatch.setattr(benchmark, "measure_ours", lambda *a, **k: (calls.append("ours") or fit, parameters, 2., 1., {"test_fixture": True}))

    def forbidden(*args, **kwargs):
        raise AssertionError("unverified adaptation must not run in strict native mode")

    monkeypatch.setattr(benchmark, "measure_numerical_baseline", forbidden)
    monkeypatch.setattr(benchmark, "run_published_baseline", forbidden)
    directory = tmp_path / "benchmark"
    argv = ["--checkpoint", str(checkpoint_path), "--output-dir", str(directory), "--method-set", "published",
            "--baseline-protocol", "native", "--device", "cpu", "--torch-num-threads", "1"]
    benchmark.main(argv, expected_objective=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION)
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    assert len(report["measurements"]) == 12 and calls == ["ours", "ours"]
    assert report["metadata"]["diagnostic_not_final"]
    assert report["metadata"]["native_comparison_complete"] is False
    assert report["metadata"]["published_baseline_protocol"]["feasibility_safeguard_enabled"] is False
    assert len([row for row in report["measurements"] if row["status"] == "unavailable"]) == 10
    assert all(row["geometry_artifact"] for row in report["measurements"])
    unavailable = [row for row in report["summary"] if row["method"] != "ours"]
    assert all(row["fit_pass_rate"] is None and row["total_ms_mean"] is None and row["unavailable"] == 2
               and row["failed"] == 0 and row["evaluated_count"] == 0 for row in unavailable)
    assert all(row["fit_pass"] is None for row in report["measurements"] if row["status"] == "unavailable")
    aggregate_plotter.validate_v16_benchmark(report, allow_unqualified_diagnostic=True)
    aggregate_image = aggregate_plotter.render_comparison(report, tmp_path / "aggregate", dpi=50)
    assert aggregate_image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(benchmark, "measure_ours", forbidden)
    benchmark.main([*argv, "--resume"], expected_objective=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION)
    figure_manifest = plotter.main(["--benchmark-dir", str(directory), "--output-dir", str(tmp_path / "figures"),
                                    "--max-cases-per-dataset", "1", "--dpi", "50"])
    assert figure_manifest["reran_methods"] is False
    assert figure_manifest["evaluated_cases"] == 2
    assert figure_manifest["requested_measurements"] == 12
    assert figure_manifest["evaluated_measurements"] == 2
    assert figure_manifest["unavailable_measurements"] == 10
    assert figure_manifest["diagnostic_not_final"] is True
    assert figure_manifest["diagnostic_reasons"] == report["metadata"]["diagnostic_reasons"]
    assert figure_manifest["published_baseline_protocol"] == report["metadata"]["published_baseline_protocol"]
    assert figure_manifest["watermark_rendered"] is False
    assert len(figure_manifest["case_figures"]) == 1
    for item in figure_manifest["case_figures"] + figure_manifest["summary_figures"]:
        assert Path(item["image"]["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    original_report = (directory / "comparison.json").read_text(encoding="utf-8")
    for field, replacement in (
        ("published_baseline_protocol", {"baseline_protocol": "adaptation"}),
        ("method_labels", {"ours": "Changed historical identity"}),
        ("diagnostic_not_final", False),
    ):
        modified_report = json.loads(original_report)
        modified_report["metadata"][field] = replacement
        (directory / "comparison.json").write_text(json.dumps(modified_report), encoding="utf-8")
        with pytest.raises(ValueError, match="fingerprint does not match"):
            plotter.main(["--benchmark-dir", str(directory), "--output-dir", str(tmp_path / "tampered")])
        assert not (tmp_path / "tampered").exists()
    (directory / "comparison.json").write_text(original_report, encoding="utf-8")


@pytest.mark.parametrize("diagnostic", [False, True])
def test_saved_figures_have_no_diagnostic_watermarks_and_keep_measured_errors(
    tmp_path, monkeypatch, measured_case, diagnostic,
):
    case, row, fit, parameters = measured_case
    source = write_case_geometry(tmp_path, case, fingerprint="fixture")
    artifact = write_method_geometry(tmp_path, case, row, fit, parameters,
                                     case_artifact=source, fingerprint="fixture", dense_points=64)
    _, arrays = load_geometry_artifact(tmp_path, artifact)
    _, source_arrays = load_geometry_artifact(tmp_path, source)
    metadata = {"methods": ["ours"], "method_labels": {"ours": "Ours"},
                "diagnostic_not_final": diagnostic,
                "diagnostic_reasons": ["REDUCED BENCHMARK PROTOCOL"],
                "published_baseline_protocol": {"baseline_protocol": "adaptation",
                                               "feasibility_safeguard_enabled": False}}
    captured = []

    def capture(figure, path, dpi):
        captured.append(figure)
        plotter.plt.close(figure)
        return {"path": str(path)}

    monkeypatch.setattr(plotter, "_save_png", capture)
    plotter._plot_case(case, source_arrays, {"ours": (row, arrays)}, metadata,
                       tmp_path / "case.png", 50)
    plotter._plot_dataset_summary(case["dataset"], [row], metadata,
                                  tmp_path / "summary.png", 50)
    for figure in captured:
        texts = [item.get_text() for item in figure.texts]
        texts.extend(item.get_text() for axis in figure.axes for item in axis.texts)
        assert not any("DIAGNOSTIC" in text or "REDUCED BENCHMARK" in text for text in texts)
        assert not any(item.get_rotation() for item in figure.texts)
        assert any("Repository adaptations" in text for text in texts)
    assert f"MSE={row['mse']:.3e}" in captured[0].axes[0].get_title()
    assert "max SE=" in captured[0].axes[0].get_title()


def test_plotter_refuses_to_invent_historical_geometry(tmp_path):
    (tmp_path / "comparison.json").write_text(json.dumps({"metadata": {}, "measurements": [{"method": "ours"}]}))
    with pytest.raises(ValueError, match="cannot reconstruct legacy fits"):
        plotter.main(["--benchmark-dir", str(tmp_path), "--output-dir", str(tmp_path / "figures")])


def test_unavailable_is_not_an_algorithm_failure_or_a_zero_percent_pass_rate():
    base = {"dataset": "Fixture", "method": "ours", "mse": None, "reference_mse": None,
            "fit_pass": False, "reference_pass": None, "total_ms": 9., "network_ms": None, "final_k": None}
    failed = {**base, "status": "failed"}
    unavailable = {**base, "status": "unavailable", "fit_pass": None, "total_ms": None}
    summary = benchmark.summarize([failed, unavailable])[0]
    assert summary["n"] == 2 and summary["evaluated_count"] == 1
    assert summary["failed"] == 1 and summary["unavailable"] == 1
    assert summary["fit_pass_rate"] == 0 and summary["total_ms_mean"] == 9
    summary = benchmark.summarize([unavailable])[0]
    assert summary["fit_pass_rate"] is None and summary["total_ms_mean"] is None
