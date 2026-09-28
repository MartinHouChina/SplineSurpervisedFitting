"""Unit fixtures for historical model screening; values are not experiment results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark_historical_dual_error as benchmark
import select_historical_checkpoint as selection
from benchmark_geometry import file_sha256, load_geometry_artifact, write_case_geometry, write_method_geometry
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points


def test_selection_counts_failed_cases_and_uses_source_macro():
    rows = [dict(dataset="A", status="ok", joint_pass=True, fit_pass=True, k=4, mse=1e-6),
            dict(dataset="A", status="failed", joint_pass=False, fit_pass=False),
            dict(dataset="B", status="ok", joint_pass=False, fit_pass=True, k=7, mse=2e-5)]
    summary = selection.selection_summary(rows)
    assert summary["failures"] == 1
    assert summary["pooled_dual_pass"] == pytest.approx(1 / 3)
    assert summary["macro_dual_pass"] == .25
    assert summary["worst_source_dual_pass"] == 0
    assert summary["mean_k"] == 5.5
    assert summary["per_source"]["A"]["n"] == 2


def test_ranking_ignores_repaired_and_test_results():
    base = dict(macro_dual_pass=.5, worst_source_dual_pass=.25, failures=0,
                mean_k=15., mean_mse=3e-5)
    better = dict(path="raw_better.pt", summary={**base, "macro_dual_pass": .75},
                  repair_pass=0, test_pass=0)
    worse = dict(path="raw_worse.pt", summary=base, repair_pass=1, test_pass=1)
    assert selection.rank_key(better) < selection.rank_key(worse)


def test_raw_fit_uses_authoritative_mask_not_probability_threshold(monkeypatch):
    t = torch.linspace(0, 1, 24)
    points = torch.stack((t, t.square()), -1)
    output = dict(params=t.unsqueeze(0), internal_knots=torch.tensor([[.2, .4, .6]]),
                  deployment_internal_knots=torch.tensor([[.25, .45, .65]]),
                  keep_probability=torch.tensor([[.99, .01, .99]]),
                  learned_keep_mask=torch.tensor([[False, True, False]]))
    monkeypatch.setattr(selection, "network_forward", lambda *a: output)
    fit, parameters = selection.raw_network_fit(SimpleNamespace(degree=3), {}, points, "cpu", 5e-5)
    assert fit.internal_knots.tolist() == pytest.approx([.45])
    assert parameters.dtype == torch.float64
    torch.testing.assert_close(fit.control_points[0], points[0].double())
    torch.testing.assert_close(fit.control_points[-1], points[-1].double())


def test_mean_and_peak_constraints_are_independent(monkeypatch):
    monkeypatch.setattr(benchmark, "fit_error_metrics", lambda *a: dict(
        mse=4e-5, max_squared_error=6e-4, reference_mse=None, reference_max_squared_error=None))
    fit = SimpleNamespace(internal_knots=torch.tensor([.2, .7]))
    row = benchmark.row_from_fit(dict(dataset="fixture", sample_id="curve", reference=None),
                                 "ours", fit, None, 7., 5e-5, 5e-4)
    assert row["fit_pass"] and not row["joint_pass"]
    assert row["max_squared_error"] == 6e-4  # Never square-root the peak.
    failed = benchmark.failure_row(dict(dataset="fixture", sample_id="curve", reference=None),
                                   "ours", "fixture failure", 9.)
    assert not failed["fit_pass"] and not failed["joint_pass"]
    assert failed["mse"] is None and failed["total_ms"] == 9.


@pytest.mark.parametrize("repair_fails", [False, True])
@pytest.mark.parametrize("post_prune, pruning_failure", [
    (False, None), (True, None), (True, "prune"), (True, "metrics"),
])
def test_fresh_benchmark_exports_raw_and_repair_with_original_endpoint_modes(
        tmp_path, monkeypatch, repair_fails, post_prune, pruning_failure):
    t = torch.linspace(0, 1, 24, dtype=torch.float64)
    points = torch.stack((t, .1 * torch.sin(2 * torch.pi * t)), -1)
    fit = refit_bspline_control_points(t, points, torch.tensor([.4, .6], dtype=torch.float64),
                                      smoothness_weight=0., control_ridge=0., interpolate_endpoints=True)
    case = dict(dataset="FIXTURE", sample_id="test_curve", group_id="test_group", split="test",
                points=points, reference=None, reference_grid=None, center=None, scale=None,
                input_points_sha256=hashlib.sha256(points.numpy().tobytes()).hexdigest())
    checkpoint_path = tmp_path / "fixture.pt"
    checkpoint_path.write_bytes(b"MOCK CHECKPOINT, NOT TRAINED WEIGHTS")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(dict(
        protocol=dict(mse_tolerance=5e-5, max_squared_error_tolerance=5e-4,
                      max_internal_knots=32, panel=[]),
        winner=dict(path=str(checkpoint_path), sha256=file_sha256(checkpoint_path), epoch=1, capacity=32))), encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "comparison.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(benchmark, "source_cases", lambda *a: (
        {"metadata": {"configuration": {"paper_admm_iterations": 17}}}, [case]))
    model = SimpleNamespace(degree=3)
    model.to = lambda *a: model
    model.eval = lambda: model
    monkeypatch.setattr(benchmark.torch, "load", lambda *a, **k: {"objective_version": "fixture"})
    monkeypatch.setattr(benchmark, "build_model_from_checkpoint", lambda *a: (model, {}, None))
    monkeypatch.setattr(benchmark, "network_forward", lambda *a: {})
    monkeypatch.setattr(benchmark, "raw_network_fit", lambda *a: (fit, t))
    monkeypatch.setattr(benchmark, "PUBLISHED_METHODS", ("ours", "dung_direct_knot_2017_adaptation"))
    baseline_calls = []
    def baseline(method, observed, **kwargs):
        baseline_calls.append(kwargs)
        return SimpleNamespace(fit=fit, parameters=t, diagnostics={"fixture": True})
    monkeypatch.setattr(benchmark, "run_published_baseline", baseline)
    repair_calls = []
    def repair(parameters, observed, initial_fit, **kwargs):
        repair_calls.append(kwargs)
        if repair_fails:
            raise RuntimeError("fixture repair failure")
        return SimpleNamespace(initial_fit=initial_fit, final_fit=initial_fit, elapsed_repair_ms=2.5,
                               inserted_knots=(), refit_count=0, termination="fixture_no_change")
    monkeypatch.setattr(benchmark, "repair_knots_to_dual_tolerance", repair)
    pruning_calls = []
    different_fit = refit_bspline_control_points(
        t, points, torch.empty(0, dtype=torch.float64),
        smoothness_weight=0., control_ridge=0., interpolate_endpoints=True)
    def prune(parameters, observed, initial_fit, **kwargs):
        pruning_calls.append(kwargs)
        if pruning_failure == "prune":
            raise RuntimeError("fixture pruning failure")
        candidate = different_fit if pruning_failure == "metrics" else initial_fit
        return SimpleNamespace(initial_fit=initial_fit, final_fit=candidate,
                               elapsed_pruning_ms=1.25, removed_knots=(),
                               after_pass=True, termination="fixture_no_deletion")
    monkeypatch.setattr(benchmark, "prune_knots_to_dual_tolerance", prune)
    original_row_from_fit = benchmark.row_from_fit
    def row_from_fit(case, method, candidate, *args):
        if candidate is different_fit:
            raise ValueError("fixture post-pruning metrics failure")
        return original_row_from_fit(case, method, candidate, *args)
    monkeypatch.setattr(benchmark, "row_from_fit", row_from_fit)
    exported_fits = {}
    original_write_geometry = benchmark.write_method_geometry
    def write_geometry(directory, case, row, exported_fit, *args, **kwargs):
        exported_fits[(Path(directory).name, row["method"])] = exported_fit
        return original_write_geometry(directory, case, row, exported_fit, *args, **kwargs)
    monkeypatch.setattr(benchmark, "write_method_geometry", write_geometry)
    destination = tmp_path / "fresh"
    monkeypatch.setattr(sys, "argv", ["benchmark_historical_dual_error.py", "--selection", str(selection_path),
        "--source-benchmark", str(source_dir), "--output-dir", str(destination), "--device", "cpu",
        "--network-repeats", "1"] + (["--post-prune-ours"] if post_prune else []))
    assert benchmark.main() == 0
    report = json.loads((destination / "comparison.json").read_text(encoding="utf-8"))
    assert len(report["raw_measurements"]) == len(report["measurements"]) == 2
    assert baseline_calls[0]["paper_admm_iterations"] == 17
    assert baseline_calls[0]["published_feasibility_safeguard"] is False
    assert baseline_calls[0]["max_internal_knots"] == 32
    assert [call["interpolate_endpoints"] for call in repair_calls] == [True, False]
    assert len(pruning_calls) == int(post_prune)
    if post_prune:
        assert len(report["pre_pruning_measurements"]) == 2
        assert report["metadata"]["ours_post_pruning"]["applies_to"] == "ours_only"
    for raw, final in zip(report["raw_measurements"], report["measurements"]):
        assert final["total_ms"] == pytest.approx(raw["total_ms"] + final["repair_ms"] + final.get("pruning_ms", 0))
        if post_prune:
            if final["method"] == "ours" and pruning_failure is not None:
                assert final["pruning_ms"] > 0
                assert final["pruning_termination"] == "pruning_failed_previous_fit_retained"
                assert "fixture" in final["pruning_error"]
                assert final["removed_knots_count"] == 0
                assert "pruning_diagnostics" not in final
                assert final["mse"] == raw["mse"] and final["knots"] == raw["knots"]
                assert exported_fits[("fresh", "ours")] is fit
                assert exported_fits[("before_pruning", "ours")] is fit
                assert final["total_ms"] == pytest.approx(final["pre_pruning_ms"] + final["pruning_ms"])
            else:
                assert final["pruning_ms"] == (1.25 if final["method"] == "ours" else 0.)
        if repair_fails:
            assert final["repair_ms"] >= 0
            assert final["repair_termination"] == "wrapper_failed_original_retained"
            assert final["mse"] == raw["mse"]
            assert final["knots"] == raw["knots"]
        else:
            assert final["repair_ms"] == 2.5
        assert final["raw_ms"] == raw["total_ms"]
        assert raw["geometry_artifact"]["path"].startswith("raw/")
        for row in (raw, final):
            document, arrays = load_geometry_artifact(destination, row["geometry_artifact"])
            assert document["timing_scope"] == report["metadata"]["timing_protocol"]
            _, source = load_geometry_artifact(destination, document["case_artifact"])
            np.testing.assert_array_equal(source["input_points_normalized"], points.numpy())
            np.testing.assert_array_equal(arrays["internal_knots"], fit.internal_knots.numpy())
            np.testing.assert_array_equal(arrays["control_points_normalized"], fit.control_points.numpy())
            assert arrays["input_squared_residuals_normalized"].mean() == pytest.approx(row["mse"])
            assert arrays["input_squared_residuals_normalized"].max() == pytest.approx(row["max_squared_error"])


def test_reused_source_inputs_are_hash_verified_and_unchanged(tmp_path, monkeypatch):
    import plot_local_model_paper_panels as panels
    monkeypatch.setattr(panels, "validate_v16_benchmark", lambda *a, **k: None)
    t = torch.linspace(0, 1, 24, dtype=torch.float64)
    points = torch.stack((t, t.square()), -1)
    fit = refit_bspline_control_points(t, points, torch.empty(0, dtype=torch.float64),
                                      smoothness_weight=0., control_ridge=0., interpolate_endpoints=True)
    case = dict(dataset="FIXTURE", sample_id="curve", group_id="group", points=points,
                reference=None, center=None, scale=None, split="test")
    case_artifact = write_case_geometry(tmp_path, case, fingerprint="fixture")
    row = benchmark.row_from_fit(case, "ours", fit, t, 1., 5e-5, 5e-4)
    row["geometry_artifact"] = write_method_geometry(tmp_path, case, row, fit, t,
        case_artifact=case_artifact, fingerprint="fixture", dense_points=31)
    report = dict(metadata=dict(fingerprint="fixture", methods=["ours"]), measurements=[row])
    (tmp_path / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
    _, loaded = benchmark.source_cases(tmp_path)
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0]["points"].numpy(), points.numpy())
    assert loaded[0]["input_points_sha256"] == hashlib.sha256(points.numpy().tobytes()).hexdigest()
    document, _ = load_geometry_artifact(tmp_path, case_artifact)
    arrays_path = (tmp_path / case_artifact["path"]).parent / document["arrays_file"]
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tampering")
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark.source_cases(tmp_path)
