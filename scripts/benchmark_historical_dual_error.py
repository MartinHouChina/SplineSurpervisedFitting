"""Fresh six-method runs on saved identical test inputs, with disclosed common knot insertion.

Model choice must come from a separate validation screen. Published adapters
are unchanged; raw and repaired measured geometry are both preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_geometry import file_sha256, load_geometry_artifact, write_case_geometry, write_method_geometry
from benchmark_v15_datasets import PUBLISHED_METHODS, parser as baseline_parser, published_baseline_kwargs, fit_error_metrics
from plot_local_model_paper_panels import _load_all_geometry, choose_comparison_cases
from select_historical_checkpoint import network_forward, raw_network_fit, write_json
from spline_fitting.checkpointing import build_model_from_checkpoint
from spline_fitting.evaluation.dual_error_knot_repair import repair_knots_to_dual_tolerance
from spline_fitting.evaluation.dual_error_knot_pruning import prune_knots_to_dual_tolerance
from spline_fitting.evaluation.published_baselines import run_published_baseline
from spline_fitting.evaluation.timing import synchronize_device


def source_cases(directory):
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    grouped, stored, _ = _load_all_geometry(directory, report)
    cases = []
    for key in grouped:
        document, arrays = stored[key]
        case = {k: v for k, v in document.items() if k not in {"array_shapes", "arrays_file", "arrays_sha256", "normalization"}}
        case.update(points=torch.from_numpy(arrays["input_points_normalized"]).double(),
                    reference=torch.from_numpy(arrays["reference_points_normalized"]).double() if "reference_points_normalized" in arrays else None,
                    reference_grid=torch.from_numpy(arrays["reference_chord_parameters"]).double() if "reference_chord_parameters" in arrays else None,
                    center=torch.from_numpy(arrays["normalization_center"]).double() if "normalization_center" in arrays else None,
                    scale=torch.as_tensor(arrays["normalization_scale"], dtype=torch.float64) if "normalization_scale" in arrays else None)
        for name in ("source_internal_knots", "canonical_internal_knots", "source_parameters"):
            if name in arrays:
                case[name] = torch.from_numpy(arrays[name]).double()
        case["input_points_sha256"] = hashlib.sha256(case["points"].numpy().tobytes()).hexdigest()
        cases.append(case)
    return report, cases


def row_from_fit(case, method, fit, parameters, elapsed, tolerance, peak_tolerance):
    metrics = fit_error_metrics(fit, parameters, case)
    return {**{k: case.get(k) for k in ("dataset", "sample_id", "group_id", "source_k", "canonical_k", "source_kind", "source_note", "split", "input_points_sha256")},
            "method": method, "status": "ok", "has_reference": case.get("reference") is not None,
            **metrics, "fit_pass": metrics["mse"] <= tolerance,
            "joint_pass": metrics["mse"] <= tolerance and metrics["max_squared_error"] <= peak_tolerance,
            "reference_pass": metrics["reference_mse"] <= tolerance if metrics["reference_mse"] is not None else None,
            "reference_joint_pass": metrics["reference_mse"] <= tolerance and metrics["reference_max_squared_error"] <= peak_tolerance if metrics["reference_mse"] is not None else None,
            "final_k": int(fit.internal_knots.numel()), "knots": fit.internal_knots.tolist(),
            "total_ms": elapsed, "network_ms": None, "error": None}


def failure_row(case, method, error, elapsed):
    return {**{k: case.get(k) for k in ("dataset", "sample_id", "group_id", "source_k", "canonical_k", "source_kind", "source_note", "split", "input_points_sha256")},
            "method": method, "status": "failed", "has_reference": case.get("reference") is not None,
            "mse": None, "max_squared_error": None, "reference_mse": None, "reference_max_squared_error": None,
            "fit_pass": False, "joint_pass": False, "reference_pass": False if case.get("reference") is not None else None,
            "reference_joint_pass": False if case.get("reference") is not None else None,
            "final_k": None, "knots": None, "total_ms": elapsed, "network_ms": None, "error": str(error)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=ROOT / "outputs/diagnostics/historical_confirm_validation_20260922.json")
    p.add_argument("--source-benchmark", type=Path, default=ROOT / "outputs/comparisons/local_models_paper_20260922_m32")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/comparisons/historical_best_dual_error_20260922")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mse-tolerance", type=float, default=5e-5)
    p.add_argument("--max-squared-error-tolerance", type=float, default=5e-4)
    p.add_argument("--max-internal-knots", type=int, default=32)
    p.add_argument("--smoke", action="store_true", help="One fixed random test case per source; never for model selection")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--network-repeats", type=int, default=5)
    p.add_argument("--post-prune-ours", action="store_true", help="After common feasibility repair, check and remove redundant Ours knots by exact dual-constrained refits")
    p.add_argument("--post-prune-min-knots", type=int, default=0)
    p.add_argument("--post-prune-max-deletions", type=int, default=32)
    args = p.parse_args()
    if args.network_repeats < 1:
        p.error("network-repeats must be positive")
    if not 0 <= args.post_prune_min_knots <= args.max_internal_knots or args.post_prune_max_deletions < 0:
        p.error("Invalid post-pruning minimum or deletion budget")
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    device = torch.device(args.device)
    screen = json.loads(args.selection.read_text(encoding="utf-8"))
    for field in ("mse_tolerance", "max_squared_error_tolerance", "max_internal_knots"):
        if screen["protocol"][field] != getattr(args, field):
            p.error(f"Selection protocol differs: {field}")
    winner = screen["winner"]
    checkpoint_path = ROOT / winner["path"]
    if file_sha256(checkpoint_path) != winner["sha256"]:
        p.error("Selected checkpoint contents changed")
    old_report, cases = source_cases(args.source_benchmark)
    validation_keys = {(r["dataset"], str(r["sample_id"])) for r in screen["protocol"]["panel"]}
    validation_groups = {(r["dataset"], str(r["group_id"])) for r in screen["protocol"]["panel"]}
    validation_hashes = {r["points_sha256"] for r in screen["protocol"]["panel"]}
    if any((c["dataset"], str(c["sample_id"])) in validation_keys for c in cases):
        p.error("Model selection and final test samples overlap")
    if any((c["dataset"], str(c["group_id"])) in validation_groups or c["input_points_sha256"] in validation_hashes for c in cases):
        p.error("Model selection and test groups/content overlap")
    if args.smoke:
        selected = set(choose_comparison_cases({(c["dataset"], c["sample_id"]): {} for c in cases}, 20260922).values())
        cases = [c for c in cases if (c["dataset"], c["sample_id"]) in selected]
    options = baseline_parser(default_checkpoint=checkpoint_path, default_output_dir=args.output_dir).parse_args([])
    # Replay the established baseline budgets; no legacy MSE-only safeguard.
    for name, value in old_report["metadata"]["configuration"].items():
        if hasattr(options, name):
            setattr(options, name, value)
    options.mse_tolerance = args.mse_tolerance
    options.max_internal_knots = options.paper_initial_knots = options.liang_dense_knots = args.max_internal_knots
    options.published_feasibility_safeguard = False
    options.baseline_protocol = "adaptation"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    model.to(device).eval()
    baseline_options = published_baseline_kwargs(options, degree=model.degree)
    protocol = dict(schema_version="historical_selection_common_dual_insertion_v1",
                    checkpoint=str(checkpoint_path), checkpoint_sha256=winner["sha256"], epoch=winner["epoch"],
                    model_capacity=winner["capacity"], objective_version=checkpoint.get("objective_version"),
                    selection_report=str(args.selection), selection_report_sha256=file_sha256(args.selection),
                    source_benchmark=str(args.source_benchmark), source_benchmark_sha256=file_sha256(args.source_benchmark / "comparison.json"),
                    methods=list(PUBLISHED_METHODS), num_points=192,
                    mse_tolerance=args.mse_tolerance, max_squared_error_tolerance=args.max_squared_error_tolerance,
                    max_internal_knots=args.max_internal_knots, baseline_options=baseline_options,
                    raw_protocol="unchanged repository adaptations; historical network mask + endpoint constrained unregularized LS; no repair in raw rows",
                    repair_protocol="common additive residual insertion for all six methods; fixed original t and U; no replacement/deletion/relocation; original endpoint convention",
                    endpoint_convention={m: m not in {"dung_direct_knot_2017_adaptation", "kang_sparse_2015_adaptation"} for m in PUBLISHED_METHODS},
                    pass_definition="MSE and maximum point squared Euclidean error both pass on the 192 normalized input points; not a continuous or Hausdorff certificate",
                    reference_definition="Untouched original reference points are evaluated separately after parameter interpolation; never used for insertion or selection",
                    timing_protocol="fresh single raw run and fresh common insertion, total_ms=raw_ms+repair_ms; metric/export/plot time excluded; network-only independently median of repeated synchronized forwards",
                    smoke=args.smoke, hardware=dict(device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    torch_threads=4),
                    cases=[{k: c.get(k) for k in ("dataset", "sample_id", "group_id", "source_k", "input_points_sha256")} for c in cases],
                    code_sha256={str(path.relative_to(ROOT)): file_sha256(path) for path in
                                 (Path(__file__), ROOT / "src/spline_fitting/evaluation/dual_error_knot_repair.py")})
    if args.post_prune_ours:
        protocol["ours_post_pruning"] = dict(
            enabled=True, stage="after_common_insertion", min_internal_knots=args.post_prune_min_knots,
            max_deletions=args.post_prune_max_deletions,
            operation="exact single-knot deletion scans; fixed t and surviving U; refit control points; both input error bounds must pass",
            applies_to="ours_only", label="Ours", part_of_measured_ours_deployment=True,
            no_global_minimum_guarantee=True,
        )
        protocol["timing_protocol"] = (
            "fresh single raw run + fresh common insertion + fresh Ours-only post-pruning; "
            "total_ms=raw_ms+repair_ms+pruning_ms; all numerical postprocessing included; "
            "metric/export/plot time excluded; network-only independently median of repeated synchronized forwards"
        )
        path = ROOT / "src/spline_fitting/evaluation/dual_error_knot_pruning.py"
        protocol["code_sha256"][str(path.relative_to(ROOT))] = file_sha256(path)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["fingerprint"] = fingerprint
    destination = args.output_dir
    output = destination / "comparison.json"
    report = dict(metadata=protocol, raw_measurements=[], measurements=[])
    if args.post_prune_ours:
        report["pre_pruning_measurements"] = []
    if output.exists():
        if not args.resume:
            p.error("Output exists; select a new output directory or --resume")
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["metadata"] != protocol:
            p.error("Resume protocol changed")
    destination.mkdir(parents=True, exist_ok=True)
    done = {(r["dataset"], str(r["sample_id"]), r["method"]) for r in report["measurements"]}
    print(f"Selected {checkpoint_path.name}; {len(cases)} paired test curves, six raw + six repaired results each", flush=True)
    # Warm up neural kernels without touching labels or selecting a model.
    for _ in range(3):
        network_forward(model, checkpoint, cases[0]["points"], device, args.mse_tolerance)
    synchronize_device(device)
    for index, case in enumerate(cases, 1):
        artifact = write_case_geometry(destination, case, fingerprint=fingerprint)
        for method in PUBLISHED_METHODS:
            key = (case["dataset"], str(case["sample_id"]), method)
            if key in done:
                continue
            fit = parameters = None
            net_ms = None
            raw_details = {}
            if method == "ours":
                timings = []
                try:
                    for _ in range(args.network_repeats):
                        synchronize_device(device)
                        start = time.perf_counter()
                        network_forward(model, checkpoint, case["points"], device, args.mse_tolerance)
                        synchronize_device(device)
                        timings.append((time.perf_counter()-start)*1000)
                except (ValueError, RuntimeError, TypeError, KeyError) as error:
                    raw_details["network_timing_error"] = str(error)
                net_ms = float(np.median(timings)) if timings else None
                raw_details["network_repeat_ms"] = timings
            start = time.perf_counter()
            try:
                if method == "ours":
                    fit, parameters = raw_network_fit(model, checkpoint, case["points"], device, args.mse_tolerance)
                    synchronize_device(device)
                else:
                    value = run_published_baseline(method, case["points"], **baseline_options)
                    fit, parameters = value.fit, value.parameters
                    raw_details = value.diagnostics
                raw_ms = (time.perf_counter()-start)*1000
                if fit.internal_knots.numel() > args.max_internal_knots:
                    raise ValueError("Raw method exceeded shared internal-knot budget")
                raw = row_from_fit(case, method, fit, parameters, raw_ms, args.mse_tolerance, args.max_squared_error_tolerance)
                raw["diagnostics"] = raw_details
            except (ValueError, RuntimeError, TypeError, KeyError) as error:
                raw_ms = (time.perf_counter()-start)*1000
                raw = failure_row(case, method, f"{type(error).__name__}: {error}", raw_ms)
            raw["network_ms"] = net_ms
            repair_details = None
            final_fit = fit
            if raw["status"] == "ok":
                repair_start = time.perf_counter()
                try:
                    repaired = repair_knots_to_dual_tolerance(parameters.double().cpu(), case["points"], fit,
                        mse_tolerance=args.mse_tolerance, max_squared_error_tolerance=args.max_squared_error_tolerance,
                        max_internal_knots=args.max_internal_knots, interpolate_endpoints=protocol["endpoint_convention"][method])
                    final_fit = repaired.final_fit
                    repair_details = {k: v for k, v in vars(repaired).items() if k not in {"initial_fit", "final_fit"}}
                    final = row_from_fit(case, method, final_fit, parameters, raw_ms+repaired.elapsed_repair_ms,
                                         args.mse_tolerance, args.max_squared_error_tolerance)
                    final.update(repair_ms=repaired.elapsed_repair_ms, added_knots_count=len(repaired.inserted_knots),
                                 repair_refit_count=repaired.refit_count, repair_termination=repaired.termination)
                except (ValueError, RuntimeError, TypeError) as error:
                    failed_repair_ms = (time.perf_counter()-repair_start)*1000
                    final_fit = fit
                    final = {**raw, "repair_error": f"{type(error).__name__}: {error}", "repair_ms": failed_repair_ms,
                             "total_ms": raw_ms+failed_repair_ms,
                             "repair_termination": "wrapper_failed_original_retained", "added_knots_count": 0}
            else:
                final = {**raw, "repair_ms": 0.0, "repair_termination": "raw_algorithm_failed_no_fit_to_repair", "added_knots_count": 0}
            final.update(raw_mse=raw["mse"], raw_max_squared_error=raw["max_squared_error"], raw_final_k=raw["final_k"],
                         raw_ms=raw_ms, network_ms=net_ms, repair_diagnostics=repair_details)
            if args.post_prune_ours:
                before = dict(final)
                before_artifact = write_method_geometry(
                    destination / "before_pruning", case, before, final_fit, parameters,
                    case_artifact=artifact, fingerprint=fingerprint, dense_points=801,
                    timing_scope="fresh raw run plus common insertion, BEFORE the added Ours post-pruning",
                    pass_criterion=protocol["pass_definition"],
                )
                before_artifact["path"] = "before_pruning/" + before_artifact["path"]
                before["geometry_artifact"] = before_artifact
                report["pre_pruning_measurements"].append(before)
                final.update(pruning_ms=0., removed_knots_count=0,
                             pruning_termination="not_applied_to_baseline" if method != "ours" else "no_valid_initial_fit")
                if method == "ours" and final["status"] == "ok":
                    prune_started = time.perf_counter()
                    try:
                        pruned = prune_knots_to_dual_tolerance(
                            parameters.double().cpu(), case["points"], final_fit,
                            mse_tolerance=args.mse_tolerance,
                            max_squared_error_tolerance=args.max_squared_error_tolerance,
                            max_internal_knots=args.max_internal_knots,
                            min_internal_knots=args.post_prune_min_knots,
                            max_deletions=args.post_prune_max_deletions,
                            interpolate_endpoints=protocol["endpoint_convention"][method],
                        )
                        if before["joint_pass"] and not pruned.after_pass:
                            raise RuntimeError("Post-pruning violated a previously satisfied error bound")
                        # Build the complete candidate row before committing either
                        # metrics or geometry. A metric/diagnostic failure must
                        # retain exactly the same previous fit as its saved row.
                        candidate_fit = pruned.final_fit
                        candidate_row = {
                            **final,
                            **row_from_fit(case, method, candidate_fit, parameters,
                                           before["total_ms"]+pruned.elapsed_pruning_ms,
                                           args.mse_tolerance, args.max_squared_error_tolerance),
                            "pruning_ms": pruned.elapsed_pruning_ms,
                            "removed_knots_count": len(pruned.removed_knots),
                            "pruning_termination": pruned.termination,
                            "pruning_diagnostics": {k: v for k, v in vars(pruned).items()
                                                    if k not in {"initial_fit", "final_fit"}},
                        }
                        final = candidate_row
                        final_fit = candidate_fit
                    except (ValueError, RuntimeError, TypeError) as error:
                        elapsed_pruning_ms = (time.perf_counter()-prune_started)*1000
                        final.update(pruning_ms=elapsed_pruning_ms,
                                     total_ms=before["total_ms"]+elapsed_pruning_ms,
                                     pruning_termination="pruning_failed_previous_fit_retained",
                                     pruning_error=f"{type(error).__name__}: {error}")
                    final.update(network_ms=net_ms, pre_pruning_mse=before["mse"],
                                 pre_pruning_max_squared_error=before["max_squared_error"],
                                 pre_pruning_k=before["final_k"], pre_pruning_ms=before["total_ms"])
            raw_artifact = write_method_geometry(destination / "raw", case, raw, fit, parameters,
                                                 case_artifact=artifact, fingerprint=fingerprint, dense_points=801,
                                                 timing_scope=protocol["timing_protocol"], pass_criterion=protocol["pass_definition"])
            raw_artifact["path"] = "raw/" + raw_artifact["path"]
            raw["geometry_artifact"] = raw_artifact
            final["geometry_artifact"] = write_method_geometry(destination, case, final, final_fit, parameters,
                                         case_artifact=artifact, fingerprint=fingerprint, dense_points=801,
                                         timing_scope=protocol["timing_protocol"], pass_criterion=protocol["pass_definition"])
            report["raw_measurements"].append(raw)
            report["measurements"].append(final)
            write_json(output, report)
            print(f"[{index}/{len(cases)}] {case['dataset']} {method}: K {raw['final_k']}->{final['final_k']} dual {raw['joint_pass']}->{final['joint_pass']} MSE={final['mse']} MaxSE={final['max_squared_error']}", flush=True)
    print(f"Saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
