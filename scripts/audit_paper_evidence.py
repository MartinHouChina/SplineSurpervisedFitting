"""Independent saved-geometry audit for the paired paper evidence experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from benchmark_ours_thresholds import write_json
from benchmark_geometry import file_sha256, load_geometry_artifact
from plot_dual_error_comparison import _geometry
from plot_paper_evidence import validate


def audit(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    validate(report)
    meta = report["metadata"]
    for name in ("checkpoint", "source_report"):
        if file_sha256(Path(meta[name])) != meta[name + "_sha256"]:
            raise ValueError(f"Changed {name}")
    inputs, by_job, parameterizations = {}, {}, {}
    record_paths = set()
    checked, failures = 0, 0
    for i, row in enumerate(report["measurements"], 1):
        key = row["dataset"], str(row["sample_id"])
        artifact_parts = Path(row["geometry_artifact"]["path"]).parts
        record_paths.add(path.parent.joinpath(*artifact_parts[:2], "record.json"))
        case, original, fitted = _geometry(path.parent, row)
        points = original["input_points_normalized"]
        if row["status"] == "ok":
            parameterizations[(*key, row["method"], row["mse_tolerance"], row["stage"])] = fitted["sample_parameters"]
        if key in inputs:
            np.testing.assert_array_equal(inputs[key], points)
        inputs[key] = points
        job = (*key, row["method"], row["mse_tolerance"])
        by_job.setdefault(job, {})[row["stage"]] = row
        if row["status"] != "ok":
            assert not row["joint_pass"]
            failures += 1
            continue
        document, _ = load_geometry_artifact(path.parent, row["geometry_artifact"])
        predicted = BSpline(fitted["full_knot_vector"], fitted["control_points_normalized"], document["spline"]["degree"])(fitted["sample_parameters"])
        squared = ((predicted - points) ** 2).sum(-1)
        np.testing.assert_allclose([squared.mean(), squared.max()], [row["mse"], row["max_squared_error"]], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(predicted, fitted["fitted_input_normalized"], rtol=1e-8, atol=1e-10)
        assert len(fitted["internal_knots"]) == row["final_k"] <= meta["max_internal_knots"]
        assert row["joint_pass"] == (row["mse"] <= row["mse_tolerance"] and row["max_squared_error"] <= row["max_squared_error_tolerance"])
        np.testing.assert_allclose(row["total_ms"], sum(row["timing_components"].values()), rtol=1e-12)
        checked += 1
        if i % 300 == 0:
            print(f"Geometry audit {i}/{len(report['measurements'])}", flush=True)
    for stages in by_job.values():
        raw, repaired, pruned = [stages[s] for s in ("raw", "repaired", "pruned")]
        if all(r["status"] == "ok" for r in stages.values()):
            assert repaired["final_k"] >= raw["final_k"]
            assert pruned["final_k"] <= repaired["final_k"]
            assert not repaired["joint_pass"] or pruned["joint_pass"]
            assert raw["total_ms"] <= repaired["total_ms"] <= pruned["total_ms"]
    for (source, sample, method, epsilon, stage), parameters in parameterizations.items():
        raw_key = (source, sample, method, epsilon, "raw")
        np.testing.assert_array_equal(parameters, parameterizations[raw_key])
        if method in {"uniform_shared_params", "all_candidates_shared_params", "uniform_matched_count_shared_params"}:
            np.testing.assert_array_equal(parameters, parameterizations[(source, sample, "ours", epsilon, "raw")])
    stage_errors = []
    for record_path in sorted(record_paths):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for stage in ("repair", "pruning"):
            detail = record.get("details", {}).get(stage, {})
            if detail.get("error"):
                stage_errors.append(dict(record=str(record_path), stage=stage, error=detail["error"]))
    return dict(complete=True, requested_rows=len(report["measurements"]), checked_finite_geometries=checked,
                failed_rows=failures, identical_input_cases=len(inputs), checked_jobs=len(by_job),
                postprocessing_execution_errors=stage_errors,
                report_sha256=file_sha256(path),
                audit_script_sha256=file_sha256(Path(__file__)),
                checks="Complete identities; source/checkpoint and geometry hashes; identical paired inputs; independent SciPy curve evaluation; true squared errors; caps/pass flags; timing sums; additive repair and feasibility-preserving deletion; unchanged postprocessing parameters and exact D/E paired parameterizations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = audit(args.report)
    name = "geometry_audit.json" if args.report.stem == "evidence" else f"{args.report.stem}_audit.json"
    write_json(args.report.parent / name, result)
    print(json.dumps(result, indent=2))
