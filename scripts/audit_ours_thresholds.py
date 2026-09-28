"""Independently verify exported threshold-sweep curves and measurement values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from benchmark_ours_thresholds import summarize, write_json
from benchmark_geometry import load_geometry_artifact
from plot_dual_error_comparison import _geometry


def audit(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError("Refusing to certify an incomplete sweep")
    summary = summarize(report)
    count = len(report["measurements"])
    if any(len(report[key]) != count for key in ("raw_measurements", "pre_pruning_measurements")):
        raise ValueError("Stage measurement counts differ")
    common = {}
    checked = 0
    for row, raw, before in zip(report["measurements"], report["raw_measurements"], report["pre_pruning_measurements"]):
        identity = lambda r: (r["dataset"], str(r["sample_id"]), r["mse_tolerance"])
        if identity(row) != identity(raw) or identity(row) != identity(before):
            raise ValueError("Stage identities differ")
        case, inputs, fit = _geometry(path.parent, row)
        points = inputs["input_points_normalized"]
        key = identity(row)[:2]
        if key in common and not np.array_equal(common[key], points):
            raise ValueError("Inputs changed across thresholds")
        common[key] = points
        if row["status"] != "ok":
            if row["joint_pass"]:
                raise ValueError("Failed execution recorded as a pass")
            continue
        document, _ = load_geometry_artifact(path.parent, row["geometry_artifact"])
        degree = document["spline"]["degree"]
        evaluated = BSpline(fit["full_knot_vector"], fit["control_points_normalized"], degree)(fit["sample_parameters"])
        squared = np.sum((evaluated - points) ** 2, axis=-1)
        np.testing.assert_allclose(evaluated, fit["fitted_input_normalized"], atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(squared, fit["input_squared_residuals_normalized"], atol=1e-12, rtol=1e-8)
        np.testing.assert_allclose([squared.mean(), squared.max()], [row["mse"], row["max_squared_error"]], atol=1e-12, rtol=1e-8)
        k = len(fit["internal_knots"])
        if k != row["final_k"] or k > report["metadata"]["max_internal_knots"]:
            raise ValueError("Internal-knot count disagrees with measured geometry/cap")
        # Use the recorded solver values for exact boundary comparisons after
        # independently checking them above (different evaluators round slightly).
        passed = row["mse"] <= row["mse_tolerance"] and row["max_squared_error"] <= row["max_squared_error_tolerance"]
        if passed != row["joint_pass"]:
            raise ValueError("Pass flag disagrees with both error bounds")
        if before["joint_pass"] and not passed:
            raise ValueError("Pruning invalidated a feasible fit")
        if k > before["final_k"]:
            raise ValueError("Pruning increased the knot count")
        np.testing.assert_allclose(row["total_ms"], row["raw_ms"] + row["repair_ms"] + row["pruning_ms"], rtol=1e-12)
        checked += 1
    return dict(complete=True, requested_measurements=count, independently_checked_finite_fits=checked,
                identical_input_cases=len(common), summary_groups=len(summary),
                verification="JSON/NPZ hashes; identical inputs; independent SciPy B-spline evaluation; squared errors; knot caps; stage identities; pass flags; pruning feasibility; timing sums")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = audit(args.report)
    write_json(args.report.parent / "geometry_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
