"""Verify and prune saved measured fits without retraining or rerunning methods.

The before state is the existing common-error-repaired fit. The default applies
the SAME post-pruning to all methods. Old method times and newly measured prune
times are separate; their sum is NOT a fresh end-to-end timing measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_geometry import file_sha256, write_case_geometry, write_method_geometry
from plot_dual_error_comparison import _geometry, pair_measurements
from spline_fitting.data.synthetic import bspline_basis_matrix
from spline_fitting.data.point_cloud_io import interpolate_parameters_by_chord
from spline_fitting.evaluation.bspline_inference import BSplineLeastSquaresFit
from spline_fitting.evaluation.dual_error_knot_pruning import prune_knots_with_dual_tolerance


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def identity(row):
    return row["dataset"], str(row["sample_id"]), row["method"]


def metrics(fit, parameters, case):
    squared = (fit.evaluate(parameters) - case["points"]).square().sum(-1)
    result = dict(mse=float(squared.mean()), max_squared_error=float(squared.max()),
                  reference_mse=None, reference_max_squared_error=None)
    if not all(math.isfinite(result[k]) for k in ("mse", "max_squared_error")):
        raise ValueError("Nonfinite actual fit errors")
    if case.get("reference") is not None:
        grid = torch.linspace(0, 1, parameters.numel(), dtype=torch.float64)
        reference_t = interpolate_parameters_by_chord(grid, parameters, case["reference_grid"])
        squared = (fit.evaluate(reference_t)-case["reference"]).square().sum(-1)
        result.update(reference_mse=float(squared.mean()), reference_max_squared_error=float(squared.max()))
    return result


def restore_measurement(directory, row):
    """Restore the SAVED control polygon, not a new least-squares solution."""
    document, source, arrays = _geometry(directory, row)
    case = {k: v for k, v in document.items() if k not in ("array_shapes", "arrays_file", "arrays_sha256", "normalization")}
    def tensor(value):
        return torch.as_tensor(value).double().cpu()
    case.update(points=tensor(source["input_points_normalized"]),
                reference=tensor(source["reference_points_normalized"]) if "reference_points_normalized" in source else None,
                reference_grid=tensor(source["reference_chord_parameters"]) if "reference_chord_parameters" in source else None,
                center=tensor(source["normalization_center"]) if "normalization_center" in source else None,
                scale=tensor(source["normalization_scale"]) if "normalization_scale" in source else None)
    for name in ("source_internal_knots", "canonical_internal_knots", "source_parameters"):
        if name in source:
            case[name] = tensor(source[name])
    case["input_points_sha256"] = hashlib.sha256(case["points"].numpy().tobytes()).hexdigest()
    if row.get("input_points_sha256") and row["input_points_sha256"] != case["input_points_sha256"]:
        raise ValueError("Recorded input point hash differs from actual geometry")
    if row["status"] != "ok":
        return case, None, None
    params, knots = tensor(arrays["sample_parameters"]), tensor(arrays["internal_knots"])
    full, controls = tensor(arrays["full_knot_vector"]), tensor(arrays["control_points_normalized"])
    degree = full.numel()-controls.shape[0]-1
    basis = bspline_basis_matrix(params, full, degree, controls.shape[0])
    prediction = basis @ controls
    squared = (prediction-case["points"]).square().sum(-1)
    mse, total = squared.mean(), squared.sum()
    smooth = controls.diff(n=2, dim=0).square().sum()
    fit = BSplineLeastSquaresFit(
        degree=degree, internal_knots=knots, knot_vector=full, basis_matrix=basis,
        control_points=controls, reconstructed_points=prediction, fit_mse=mse,
        fit_rmse=mse.sqrt(), coordinate_mse=mse/case["points"].shape[1],
        coordinate_rmse=(mse/case["points"].shape[1]).sqrt(), data_squared_error=total,
        smoothness_squared=smooth, control_squared=controls.square().sum(),
        augmented_objective=total, solver_rank=None,
    )
    measured = metrics(fit, params, case)
    for key in ("mse", "max_squared_error", "reference_mse", "reference_max_squared_error"):
        if measured[key] is not None and row.get(key) is not None and not math.isclose(measured[key], row[key], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(f"Stored {key} disagrees with actual saved geometry: {identity(row)}")
    if knots.numel() != row["final_k"]:
        raise ValueError("Stored knot count disagrees with geometry")
    return case, fit, params


def clean_row(row):
    return {k: v for k, v in row.items() if k not in ("geometry_artifact", "diagnostics", "repair_diagnostics")}


def set_metrics(row, case, fit, params, metadata):
    row.update(metrics(fit, params, case))
    row.update(final_k=int(fit.internal_knots.numel()), knots=fit.internal_knots.tolist(),
               fit_pass=row["mse"] <= metadata["mse_tolerance"],
               joint_pass=row["mse"] <= metadata["mse_tolerance"] and row["max_squared_error"] <= metadata["max_squared_error_tolerance"],
               reference_pass=row["reference_mse"] <= metadata["mse_tolerance"] if row["reference_mse"] is not None else None,
               reference_joint_pass=row["reference_mse"] <= metadata["mse_tolerance"] and row["reference_max_squared_error"] <= metadata["max_squared_error_tolerance"] if row["reference_mse"] is not None else None)


def verify_pruning(before, after, before_fit, after_fit):
    if before_fit is None:
        return
    if after["final_k"] > before["final_k"]:
        raise ValueError("Pruning increased knot count")
    if before["joint_pass"] and not after["joint_pass"]:
        raise ValueError("Post-pruning broke a previously satisfied constraint")
    original = Counter(before_fit.internal_knots.tolist())
    final = Counter(after_fit.internal_knots.tolist())
    if any(n > original[k] for k, n in final.items()):
        raise ValueError("Pruning inserted or moved a survivor knot")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/comparisons/historical_best_dual_error_20260922/comparison.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["all"], help="all (fair six-method control, default) or explicit method IDs such as ours")
    parser.add_argument("--max-deletions", type=int, default=32)
    parser.add_argument("--min-internal-knots", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_deletions < 0 or args.min_internal_knots < 0 or args.torch_num_threads < 1:
        parser.error("invalid negative budget or nonpositive thread count")
    if args.output_dir.exists() and not args.resume:
        parser.error("output directory already exists; choose a new name or --resume")
    if args.output_dir.resolve() == args.report.resolve().parent:
        parser.error("post-pruning must not overwrite its source benchmark")
    source = json.loads(args.report.read_text(encoding="utf-8"))
    pairs = pair_measurements(source)
    methods = source["metadata"]["methods"]
    selected = methods if args.methods == ["all"] else args.methods
    if len(set(selected)) != len(selected) or not set(selected) <= set(methods):
        parser.error("unknown or duplicate method IDs")
    cap = source["metadata"]["max_internal_knots"]
    if args.min_internal_knots > cap:
        parser.error("minimum exceeds original experiment capacity")
    torch.set_num_threads(args.torch_num_threads)
    code = [Path(__file__), ROOT / "src/spline_fitting/evaluation/dual_error_knot_pruning.py",
            ROOT / "src/spline_fitting/evaluation/dual_error_knot_repair.py",
            ROOT / "src/spline_fitting/evaluation/bspline_inference.py"]
    metadata = dict(schema_version="saved_measured_fit_post_pruning_v1",
                    methods=methods, post_prune_methods=selected,
                    checkpoint=source["metadata"].get("checkpoint"),
                    source_report_sha256=file_sha256(args.report),
                    source_report=str(args.report.resolve()),
                    mse_tolerance=source["metadata"]["mse_tolerance"],
                    max_squared_error_tolerance=source["metadata"]["max_squared_error_tolerance"],
                    max_internal_knots=cap, min_internal_knots=args.min_internal_knots,
                    max_deletions=args.max_deletions,
                    endpoint_convention=source["metadata"]["endpoint_convention"],
                    cases=source["metadata"]["cases"],
                    code_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): file_sha256(p) for p in code},
                    torch_threads=args.torch_num_threads,
                    before_protocol="Saved exact common dual-error insertion output, not rerun; all source curves/method failures retained",
                    pruning_protocol="Fixed t and survivor U; full single-deletion exact scan; unregularized control refit; accept only both input constraints; no network anchors, no inserted knots, no relocation",
                    timing_protocol="prune_ms is freshly measured CPU post-processing only. source_total_ms is copied historical base+insertion time. total_ms is their arithmetic sum, NOT a freshly measured end-to-end latency. No new network timing was acquired.",
                    pass_definition="Normalized input-sample MSE and maximum single-point squared Euclidean error; no continuous/reference error guarantee",
                    limitation="Hybrid post-processed results, not improved neural learning or native literature algorithms. Greedy is locally single-deletion terminal only when an exhaustive successful final scan certifies it; no global minimality claim.")
    metadata["fingerprint"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    output = args.output_dir / "comparison.json"
    report = dict(metadata=metadata, before_measurements=[], measurements=[], network_measurements=[])
    if args.resume:
        if not output.exists():
            parser.error("resume requires an existing comparison.json")
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["metadata"] != metadata:
            parser.error("resume protocol/code/source differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = {identity(r) for r in report["measurements"]}
    for index, (network_row, source_row) in enumerate(pairs, 1):
        if identity(source_row) in completed:
            continue
        case, initial, params = restore_measurement(args.report.parent, source_row)
        case_artifact = write_case_geometry(args.output_dir, case, fingerprint=metadata["fingerprint"])
        before = clean_row(source_row)
        before.update(prune_ms=0., removed_knots_count=0, source_total_ms=source_row["total_ms"])
        if initial is not None:
            set_metrics(before, case, initial, params, metadata)
        final = dict(before)
        final_fit = initial
        final.update(prune_termination="not_requested" if source_row["method"] not in selected else "no_valid_source_fit")
        if initial is not None and source_row["method"] in selected:
            started = time.perf_counter()
            try:
                result = prune_knots_with_dual_tolerance(
                    params, case["points"], initial,
                    mse_tolerance=metadata["mse_tolerance"], max_squared_error_tolerance=metadata["max_squared_error_tolerance"],
                    max_internal_knots=cap, min_internal_knots=args.min_internal_knots, max_deletions=args.max_deletions,
                    interpolate_endpoints=metadata["endpoint_convention"][source_row["method"]],
                    smoothness_weight=0., control_ridge=0.,
                )
                final_fit = result.final_fit
                final.update(prune_ms=result.elapsed_prune_ms, removed_knots_count=len(result.removed_knots),
                             prune_refit_count=result.refit_count, prune_termination=result.termination,
                             pruning_diagnostics={k: v for k, v in vars(result).items() if k not in ("initial_fit", "final_fit")})
            except (ValueError, RuntimeError) as error:
                final.update(prune_ms=(time.perf_counter()-started)*1000,
                             prune_termination="post_pruning_error_original_retained", post_pruning_error=str(error))
            set_metrics(final, case, final_fit, params, metadata)
        final.update(before_mse=before["mse"], before_max_squared_error=before["max_squared_error"],
                     before_final_k=before["final_k"], before_joint_pass=before["joint_pass"],
                     total_ms=before["source_total_ms"]+final["prune_ms"],
                     total_time_role="historical_source_time_plus_fresh_pruning_not_retimed_end_to_end")
        verify_pruning(before, final, initial, final_fit)
        for phase, row, fit in (("before", before, initial), ("", final, final_fit)):
            artifact = write_method_geometry(args.output_dir / phase, case, row, fit, params,
                                             case_artifact=case_artifact, fingerprint=metadata["fingerprint"], dense_points=801,
                                             timing_scope=metadata["timing_protocol"], pass_criterion=metadata["pass_definition"])
            if phase:
                artifact["path"] = phase + "/" + artifact["path"]
            row["geometry_artifact"] = artifact
        if network_row["method"] == "ours":
            raw_case, raw_fit, raw_params = restore_measurement(args.report.parent, network_row)
            if not torch.equal(raw_case["points"], case["points"]):
                raise ValueError("Original network and repaired source inputs differ")
            raw = clean_row(network_row)
            raw["geometry_artifact"] = write_method_geometry(args.output_dir / "network", case, raw, raw_fit, raw_params,
                                                             case_artifact=case_artifact, fingerprint=metadata["fingerprint"], dense_points=801,
                                                             timing_scope="Historical original-network timing; no new timing acquisition", pass_criterion=metadata["pass_definition"])
            raw["geometry_artifact"]["path"] = "network/" + raw["geometry_artifact"]["path"]
            report["network_measurements"].append(raw)
        report["before_measurements"].append(before)
        report["measurements"].append(final)
        write_json(output, report)
        print(f"[{index}/{len(pairs)}] {case['dataset']}/{case['sample_id']} {source_row['method']}: "
              f"K {before['final_k']}->{final['final_k']} dual {before['joint_pass']}->{final['joint_pass']} "
              f"prune={final['prune_ms']:.1f}ms", flush=True)
    print(f"Saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
