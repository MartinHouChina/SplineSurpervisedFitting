"""Tiny deterministic fixtures, not reported experiment results."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import postprune_saved_comparison as pruning
from benchmark_geometry import file_sha256, load_geometry_artifact, write_case_geometry, write_method_geometry
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points


def _fixture(tmp_path, *, failed_baseline=False, solver_options=None):
    source = tmp_path / "source"
    source.mkdir()
    parameters = torch.linspace(0, 1, 32, dtype=torch.float64)
    points = torch.stack((parameters, parameters.square()), -1)
    case = dict(dataset="FIXTURE", sample_id="polynomial", group_id="group", points=points,
                reference=points.clone(), reference_grid=parameters.clone(),
                center=torch.tensor([2., 3.]), scale=torch.tensor(5.))
    case["input_points_sha256"] = hashlib.sha256(points.numpy().tobytes()).hexdigest()
    solver_options = solver_options or dict(interpolate_endpoints=True, smoothness_weight=0., control_ridge=0.)
    case_artifact = write_case_geometry(source, case, fingerprint="fixture")
    fit = refit_bspline_control_points(parameters, points, torch.tensor([.2, .4, .6, .8], dtype=torch.float64),
                                      **solver_options)
    metadata = dict(methods=["ours", "baseline"], mse_tolerance=5e-5,
                    max_squared_error_tolerance=5e-4, max_internal_knots=32,
                    endpoint_convention={"ours": True, "baseline": False},
                    fingerprint="fixture", raw_protocol="original raw", repair_protocol="common insertion",
                    timing_protocol="historical timing", cases=[{k: case[k] for k in
                    ("dataset", "sample_id", "group_id", "input_points_sha256")}])
    report = dict(metadata=metadata, raw_measurements=[], measurements=[])
    for method in metadata["methods"]:
        failed = method == "baseline" and failed_baseline
        metrics = {k: None for k in ("mse", "max_squared_error", "reference_mse", "reference_max_squared_error")} if failed else pruning.fit_error_metrics(fit, parameters, case)
        row = dict(dataset=case["dataset"], sample_id=case["sample_id"], group_id="group", method=method,
                   status="failed" if failed else "ok", input_points_sha256=case["input_points_sha256"],
                   final_k=None if failed else 4, knots=None if failed else fit.internal_knots.tolist(),
                   total_ms=10., network_ms=1. if method == "ours" else None,
                   joint_pass=not failed, fit_pass=not failed, **metrics)
        raw = copy.deepcopy(row)
        artifact = write_method_geometry(source / "raw", case, raw, None if failed else fit, parameters,
                                         case_artifact=case_artifact, fingerprint="fixture", dense_points=41)
        raw["geometry_artifact"] = dict(artifact, path="raw/" + artifact["path"])
        row.update(raw_mse=raw["mse"], raw_max_squared_error=raw["max_squared_error"],
                   raw_final_k=raw["final_k"], raw_ms=raw["total_ms"], repair_ms=2., total_ms=12.,
                   added_knots_count=0, repair_diagnostics={"unchanged": True, "solver_options": solver_options})
        row["geometry_artifact"] = write_method_geometry(source, case, row, None if failed else fit, parameters,
                                                        case_artifact=case_artifact, fingerprint="fixture", dense_points=41)
        report["raw_measurements"].append(raw)
        report["measurements"].append(row)
    path = source / "comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, report, fit, parameters


@pytest.mark.parametrize("failed_baseline", [False, True])
def test_saved_pruning_preserves_all_cases_baselines_raw_and_original_geometry(tmp_path, failed_baseline):
    source_path, before, original_fit, parameters = _fixture(tmp_path, failed_baseline=failed_baseline)
    source_hash = file_sha256(source_path)
    destination = tmp_path / "pruned"
    report = pruning.postprune_report(source_path, destination, torch_num_threads=1)
    assert file_sha256(source_path) == source_hash
    assert file_sha256(destination / "source_snapshot/comparison.json") == source_hash
    assert report["raw_measurements"] == before["raw_measurements"]
    assert report["measurements"][1] == before["measurements"][1]
    assert len(report["measurements"]) == len(before["measurements"]) == 2
    assert report["metadata"]["raw_protocol"] == before["metadata"]["raw_protocol"]
    assert report["metadata"]["repair_protocol"] == before["metadata"]["repair_protocol"]
    assert "Composite timing" in report["metadata"]["timing_protocol"]
    assert report["metadata"]["ours_post_pruning"]["complete"]
    assert pruning.protocol_fingerprint(report["metadata"]) == report["metadata"]["fingerprint"]
    ours = report["measurements"][0]
    assert ours["final_k"] == 0
    assert ours["removed_knots_count"] == 4
    assert ours["joint_pass"] and ours["fit_pass"] and ours["reference_joint_pass"]
    assert ours["pre_pruning_k"] == 4
    assert ours["total_ms"] == pytest.approx(ours["pre_pruning_ms"] + ours["pruning_ms"])
    assert ours["repair_diagnostics"] == before["measurements"][0]["repair_diagnostics"]
    for row in report["raw_measurements"] + report["measurements"] + report["pre_pruning_measurements"]:
        pruning._geometry(destination, row)
    document, arrays = load_geometry_artifact(destination, ours["geometry_artifact"])
    np.testing.assert_array_equal(arrays["sample_parameters"], parameters.numpy())
    assert document["timing_scope"] == report["metadata"]["timing_protocol"]
    assert arrays["input_squared_residuals_normalized"].mean() == pytest.approx(ours["mse"])
    assert arrays["reference_squared_residuals_normalized"].max() == pytest.approx(ours["reference_max_squared_error"])
    archived = report["pre_pruning_measurements"][0]
    _, old_arrays = load_geometry_artifact(destination, archived["geometry_artifact"])
    np.testing.assert_array_equal(old_arrays["control_points_normalized"], original_fit.control_points.numpy())
    assert archived["geometry_artifact"]["sha256"] == before["measurements"][0]["geometry_artifact"]["sha256"]


def test_zero_budget_retains_exact_saved_control_points_and_knots(tmp_path):
    source_path, original, fit, _ = _fixture(tmp_path)
    destination = tmp_path / "zero"
    report = pruning.postprune_report(source_path, destination, max_deletions=0, torch_num_threads=1)
    row = report["measurements"][0]
    _, arrays = load_geometry_artifact(destination, row["geometry_artifact"])
    np.testing.assert_array_equal(arrays["control_points_normalized"], fit.control_points.numpy())
    np.testing.assert_array_equal(arrays["internal_knots"], fit.internal_knots.numpy())
    assert row["mse"] == original["measurements"][0]["mse"]
    assert row["removed_knots_count"] == 0


def test_hash_corruption_is_rejected_before_creating_output(tmp_path):
    source_path, report, _, _ = _fixture(tmp_path)
    row = report["measurements"][1]
    path = source_path.parent / row["geometry_artifact"]["path"]
    path.write_bytes(path.read_bytes() + b"tampered")
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="hash mismatch"):
        pruning.postprune_report(source_path, output)
    assert not output.exists()


def test_source_and_existing_destinations_are_never_overwritten(tmp_path):
    source_path, _, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="separate directory"):
        pruning.postprune_report(source_path, source_path.parent / "inside")
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        pruning.postprune_report(source_path, destination)


def test_restored_metrics_do_not_trust_self_consistent_cached_json(tmp_path):
    source_path, report, _, _ = _fixture(tmp_path)
    row = report["measurements"][0]
    document, arrays = load_geometry_artifact(source_path.parent, row["geometry_artifact"])
    case, source, _ = pruning._geometry(source_path.parent, row)
    fit, parameters = pruning.restore_saved_fit(document, arrays, torch.from_numpy(source["input_points_normalized"]))
    altered = dict(row, mse=row["mse"] + .1)
    with pytest.raises(ValueError, match="reconstructed spline"):
        pruning._check_metrics(altered, pruning.fit_error_metrics(fit, parameters, pruning._case_from_arrays(case, source)))


def test_active_output_excludes_stale_derived_artifacts_but_snapshot_keeps_them(tmp_path):
    source_path, _, _, _ = _fixture(tmp_path)
    for relative in ("old_plot.png", "summary.csv", "geometry/stale_summary.json", "raw/stale_contact_sheet.png"):
        path = source_path.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"STALE DERIVED FILE - MUST NOT APPEAR IN NEW RESULTS")
    output = tmp_path / "pruned"
    pruning.postprune_report(source_path, output, max_deletions=0, torch_num_threads=1)
    for relative in ("old_plot.png", "summary.csv", "geometry/stale_summary.json", "raw/stale_contact_sheet.png"):
        assert not (output / relative).exists()
        assert (output / "source_snapshot" / relative).read_bytes() == (source_path.parent / relative).read_bytes()


def test_fingerprint_is_reproducible_and_independent_of_old_hash_or_completion():
    metadata = dict(fingerprint="old copied hash", source="fixture",
                    ours_post_pruning=dict(complete=False, acquisition_utc="fixture timestamp", source_fingerprint="source hash"))
    expected = pruning.protocol_fingerprint(metadata)
    metadata["fingerprint"] = "new stored hash"
    metadata["ours_post_pruning"]["complete"] = True
    assert pruning.protocol_fingerprint(metadata) == expected
    metadata["ours_post_pruning"]["source_fingerprint"] = "different source hash"
    assert pruning.protocol_fingerprint(metadata) != expected


def test_source_regularization_is_replayed_exactly(tmp_path, monkeypatch):
    options = dict(interpolate_endpoints=True, smoothness_weight=1e-7, control_ridge=2e-8)
    source_path, _, _, _ = _fixture(tmp_path, solver_options=options)
    original = pruning.prune_knots_to_dual_tolerance
    observed = []

    def tracked(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pruning, "prune_knots_to_dual_tolerance", tracked)
    report = pruning.postprune_report(source_path, tmp_path / "pruned", max_deletions=0, torch_num_threads=1)
    assert len(observed) == 1
    for name, expected in options.items():
        assert observed[0][name] == expected
        assert report["measurements"][0]["pruning_diagnostics"]["solver_options"][name] == expected


@pytest.mark.parametrize("options", [None, {}, {"interpolate_endpoints": True},
    dict(interpolate_endpoints=False, smoothness_weight=0., control_ridge=0.),
    dict(interpolate_endpoints=True, smoothness_weight=-1., control_ridge=0.),
    dict(interpolate_endpoints=True, smoothness_weight=0., control_ridge=float("nan"))])
def test_unsupported_source_solver_contract_is_rejected_before_output_creation(tmp_path, options):
    source_path, report, _, _ = _fixture(tmp_path)
    report["measurements"][0]["repair_diagnostics"]["solver_options"] = options
    source_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="solver"):
        pruning.postprune_report(source_path, output, torch_num_threads=1)
    assert not output.exists()
