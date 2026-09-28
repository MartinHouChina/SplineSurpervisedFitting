"""Paired inference interventions, multi-tolerance baselines and stage timings.

No training and no hidden model selection. Original adapters are unchanged.
All methods receive the SAME disclosed additive repair and deletion algorithms;
raw and repaired-only outputs remain separately available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from benchmark_ours_thresholds import DEFAULT_REPORT, load_cases, write_json
from benchmark_geometry import file_sha256, write_case_geometry, write_method_geometry
from benchmark_historical_dual_error import row_from_fit, failure_row
from benchmark_v15_datasets import PUBLISHED_METHODS
from select_historical_checkpoint import network_forward
from visualize_batch_comparison import deployment_geometries
from spline_fitting.checkpointing import build_model_from_checkpoint
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.dual_error_knot_repair import repair_knots_to_dual_tolerance
from spline_fitting.evaluation.dual_error_knot_pruning import prune_knots_to_dual_tolerance
from spline_fitting.evaluation.gradient_knot_pruning import chord_length_parameters
from spline_fitting.evaluation.published_baselines import run_published_baseline
from spline_fitting.evaluation.timing import synchronize_device

VARIANTS = ("uniform_shared_params", "all_candidates_shared_params", "uniform_chord")
STAGES = ("raw", "repaired", "pruned")
COMPONENTS = ("network_ms", "transfer_ms", "initial_refit_ms", "initializer_ms", "baseline_ms", "repair_ms", "pruning_ms")
UNCONSTRAINED = {"dung_direct_knot_2017_adaptation", "kang_sparse_2015_adaptation"}


def timed(function, device=None):
    if device is not None:
        synchronize_device(device)
    start = time.perf_counter()
    value = function()
    if device is not None:
        synchronize_device(device)
    return value, (time.perf_counter() - start) * 1000


def baseline_options(source, tolerance, capacity):
    options = dict(source["metadata"]["baseline_options"])
    options.update(mse_tolerance=tolerance, max_internal_knots=capacity,
                   paper_initial_knots=capacity, liang_dense_knots=capacity,
                   published_feasibility_safeguard=False, baseline_protocol="adaptation")
    return options


def initial_network(model, checkpoint, case, device, epsilon, repeats):
    output, network_ms = timed(lambda: network_forward(model, checkpoint, case["points"], device, epsilon), device)
    def extract():
        _, knots, mask = deployment_geometries(output)
        return (output["params"][0].detach().cpu().double(), knots.detach().cpu().double(), mask.detach().cpu())
    (params, all_knots, mask), transfer_ms = timed(extract)
    fit, fit_ms = timed(lambda: refit(params, case["points"], all_knots[mask], model.degree))
    repeated = [network_ms]
    for _ in range(repeats - 1):
        _, duration = timed(lambda: network_forward(model, checkpoint, case["points"], device, epsilon), device)
        repeated.append(duration)
    components = dict(network_ms=network_ms, transfer_ms=transfer_ms, initial_refit_ms=fit_ms)
    shared = dict(params=params, all_knots=all_knots, network_ms=network_ms, transfer_ms=transfer_ms)
    return fit, params, components, dict(network_repeat_ms=repeated, network_median_ms=float(np.median(repeated))), shared


def refit(params, points, knots, degree):
    return refit_bspline_control_points(params, points, knots.sort().values, degree=degree,
                                       smoothness_weight=0., control_ridge=0., interpolate_endpoints=True)


def variant_initial(method, case, shared, capacity, degree):
    components = {}
    start = time.perf_counter()
    if method == "uniform_chord":
        params = chord_length_parameters(case["points"])
    else:
        params = shared["params"]
        # Reuse the SAME forward for paired interventions; charge its full
        # observed cost to each variant rather than treating parameters as free.
        components.update(network_ms=shared["network_ms"], transfer_ms=shared["transfer_ms"])
    if method == "all_candidates_shared_params":
        knots = shared["all_knots"]
    else:
        knots = torch.linspace(0, 1, capacity + 2, dtype=torch.float64)[1:-1]
    components["initializer_ms"] = (time.perf_counter() - start) * 1000
    fit, duration = timed(lambda: refit(params, case["points"], knots, degree))
    components["initial_refit_ms"] = duration
    return fit, params, components


def run_chain(case, method, fit, params, components, epsilon, peak_ratio, capacity):
    """Return the three actual measured splines, charging stages cumulatively."""
    components = {k: float(components.get(k, 0.)) for k in COMPONENTS}
    peak = epsilon * peak_ratio
    snapshots, details = [], {}
    endpoint = method not in UNCONSTRAINED
    def snapshot(stage, current):
        if current.internal_knots.numel() > capacity:
            raise ValueError("Method exceeds the common internal-knot cap")
        row = row_from_fit(case, method, current, params, sum(components.values()), epsilon, peak)
        row.update(stage=stage, mse_tolerance=epsilon, max_squared_error_tolerance=peak,
                   timing_components=dict(components), repair_refit_count=details.get("repair", {}).get("refit_count", 0),
                   pruning_refit_count=details.get("pruning", {}).get("refit_count", 0))
        snapshots.append((row, current))
    snapshot("raw", fit)
    for stage, function, name in (("repaired", repair_knots_to_dual_tolerance, "repair"),
                                  ("pruned", prune_knots_to_dual_tolerance, "pruning")):
        start = time.perf_counter()
        try:
            kwargs = {"max_deletions": capacity} if name == "pruning" else {}
            result = function(params, case["points"], fit, mse_tolerance=epsilon,
                              max_squared_error_tolerance=peak, max_internal_knots=capacity,
                              interpolate_endpoints=endpoint, **kwargs)
            duration = getattr(result, f"elapsed_{name}_ms")
            if name == "pruning" and result.before_pass and not result.after_pass:
                raise RuntimeError("Pruning invalidated a feasible curve")
            fit = result.final_fit
            details[name] = {k: v for k, v in vars(result).items() if k not in {"initial_fit", "final_fit"}}
        except (ValueError, RuntimeError, TypeError) as error:
            duration = (time.perf_counter() - start) * 1000
            details[name] = dict(error=str(error), termination="failed_previous_fit_retained")
        components[f"{name}_ms"] = duration
        snapshot(stage, fit)
    return snapshots, details


def job_id(case, method, epsilon):
    return hashlib.sha256(json.dumps([case["dataset"], str(case["sample_id"]), method, epsilon]).encode()).hexdigest()[:24]


def expected_jobs(meta):
    return {(c["dataset"], str(c["sample_id"]), method, float(eps))
            for c in meta["cases"] for eps in meta["mse_tolerances"]
            for method in (list(PUBLISHED_METHODS) + (list(VARIANTS) if eps == meta["ablation_tolerance"] else []))}


def assemble(directory, meta, complete=False):
    rows = []
    for path in sorted((directory / "runs").glob("*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["fingerprint"] != meta["fingerprint"]:
            raise ValueError("Mixed experiment fingerprints")
        rows.extend(record["measurements"])
    expected = {(*key, stage) for key in expected_jobs(meta) for stage in STAGES}
    actual = [(r["dataset"], str(r["sample_id"]), r["method"], float(r["mse_tolerance"]), r["stage"]) for r in rows]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected):
        raise ValueError("Duplicate or unexpected result identities")
    if complete and set(actual) != expected:
        raise ValueError("Missing requested measurements")
    report = dict(metadata=meta, complete=complete, expected_measurements=len(expected), measurements=rows)
    write_json(directory / "evidence.json", report)
    return report


def save_job(directory, case, method, epsilon, snapshots, params, details, meta):
    destination = directory / "runs" / job_id(case, method, epsilon)
    artifact = write_case_geometry(directory, case, fingerprint=meta["fingerprint"])
    rows = []
    for row, fit in snapshots:
        folder = destination / row["stage"]
        geometry = write_method_geometry(folder, case, row, fit, params, case_artifact=artifact,
                    fingerprint=meta["fingerprint"], dense_points=801, timing_scope=meta["timing_protocol"],
                    pass_criterion=f"MSE <= epsilon and maximum squared point error <= {meta['peak_ratio']:g} epsilon")
        geometry["path"] = (folder.relative_to(directory) / geometry["path"]).as_posix()
        row["geometry_artifact"] = geometry
        rows.append(row)
    write_json(destination / "record.json", dict(fingerprint=meta["fingerprint"], measurements=rows, details=details))


def failed_snapshots(case, method, epsilon, ratio, error, elapsed):
    rows = []
    for stage in STAGES:
        row = failure_row(case, method, str(error), elapsed)
        row.update(stage=stage, mse_tolerance=epsilon, max_squared_error_tolerance=epsilon * ratio,
                   timing_components={k: None for k in COMPONENTS}, repair_refit_count=None, pruning_refit_count=None)
        rows.append((row, None))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/candidate_selection_v16_mse5e-5_k64.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mse-tolerances", type=float, nargs="+", default=[1e-5, 2e-5, 5e-5, 1e-4])
    p.add_argument("--ablation-tolerance", type=float, default=5e-5)
    p.add_argument("--peak-ratio", type=float, default=10.)
    p.add_argument("--max-internal-knots", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--network-repeats", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="First saved case from each source; excluded from full results")
    args = p.parse_args(argv)
    epsilons = sorted(args.mse_tolerances)
    if (len(set(epsilons)) != len(epsilons) or args.ablation_tolerance not in epsilons
            or any(not math.isfinite(v) or v <= 0 for v in [*epsilons, args.peak_ratio])
            or min(args.network_repeats, args.torch_threads, args.max_internal_knots) < 1):
        p.error("Invalid tolerances, capacity, thread/repeat counts, or missing ablation tolerance")
    source, cases = load_cases(args.source_report)
    if args.smoke:
        first = {}
        for c in cases:
            first.setdefault(c["dataset"], c)
        cases = list(first.values())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_config"]["max_internal_knots"] != args.max_internal_knots:
        p.error("The fixed checkpoint capacity must equal the common cap")
    model, _, _ = build_model_from_checkpoint(checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(20260923)
    code_paths = [Path(__file__), ROOT / "scripts/benchmark_geometry.py", ROOT / "scripts/benchmark_historical_dual_error.py",
                  ROOT / "scripts/select_historical_checkpoint.py", ROOT / "src/spline_fitting/models/v16_network.py"]
    code_paths += list((ROOT / "src/spline_fitting/evaluation").glob("*.py"))
    meta = dict(schema_version="paired_paper_evidence_v1", checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=file_sha256(args.checkpoint), checkpoint_epoch=checkpoint.get("epoch"),
                source_report=str(args.source_report.resolve()), source_report_sha256=file_sha256(args.source_report),
                max_internal_knots=args.max_internal_knots, mse_tolerances=epsilons, peak_ratio=args.peak_ratio,
                ablation_tolerance=args.ablation_tolerance, methods=list(PUBLISHED_METHODS), variants=list(VARIANTS),
                baseline_options=baseline_options(source, args.ablation_tolerance, args.max_internal_knots),
                endpoint_convention={m: m not in UNCONSTRAINED for m in [*PUBLISHED_METHODS, *VARIANTS]},
                comparisons="Unchanged repository literature adaptations, plus common additive dual-error repair and common dual-error deletion for ALL methods; each stage separately saved. Not authors' original code.",
                ablation="Inference-time interventions, not retrained ablations. Uniform_shared_params and all_candidates_shared_params freeze the SAME deployed parameters. All-candidates bypasses only the output mask, not selector-conditioned decoding; all 64 final-frame candidate positions retained. Shared forward cost charged to each paired intervention. Uniform_chord is an additional network-free control.",
                timing_protocol="Synchronized forward including H2D; separate D2H/extraction and first LS; baseline algorithm time is indivisible. Shared repair/prune elapsed includes all trial refits. Total=sum of measured components; metrics/export/plot excluded. One full numerical run per sample, three network forward timings (median diagnostic only). No parallel timed workers. CPU float64 numerical stages; GPU network. Normalization outside timing is identical for all.",
                refit_counts="Only common postprocessing refits are comparable and counted; internal baseline solves are not reported as zero.",
                hardware=dict(device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                              torch_threads=args.torch_threads, torch_version=str(torch.__version__)),
                network_repeats=args.network_repeats, smoke=args.smoke,
                cases=[{k: c.get(k) for k in ("dataset", "sample_id", "group_id", "input_points_sha256")} for c in cases],
                code_sha256={str(x.relative_to(ROOT)): file_sha256(x) for x in code_paths})
    meta["fingerprint"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    destination = args.output_dir
    protocol = destination / "protocol.json"
    if protocol.exists():
        if not args.resume or json.loads(protocol.read_text(encoding="utf-8")) != meta:
            p.error("Existing protocol differs or --resume was not supplied")
    elif destination.exists() and any(destination.iterdir()):
        p.error("Use a new/empty output directory")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(protocol, meta)
    for _ in range(3):
        network_forward(model, checkpoint, cases[0]["points"], device, args.ablation_tolerance)
    total_jobs = len(expected_jobs(meta))
    done = len(list((destination / "runs").glob("*/record.json")))
    def exists(c, m, eps):
        return (destination / "runs" / job_id(c, m, eps) / "record.json").exists()
    def commit(c, m, eps, snapshots, params, details):
        nonlocal done
        save_job(destination, c, m, eps, snapshots, params, details, meta)
        done += 1
        last = snapshots[-1][0]
        print(f"[{done}/{total_jobs}] eps={eps:g} {c['dataset']} {m} K={last['final_k']} pass={last['joint_pass']} MSE={last['mse']} total={last['total_ms']:.1f} ms", flush=True)
        if done % 30 == 0:
            assemble(destination, meta)
    # Complete all paired inference interventions before the expensive adapters.
    ordered = [args.ablation_tolerance] + [v for v in epsilons if v != args.ablation_tolerance]
    for eps in ordered:
        for method in PUBLISHED_METHODS:
            options = baseline_options(source, eps, args.max_internal_knots)
            for case in cases:
                variants = VARIANTS if method == "ours" and eps == args.ablation_tolerance else ()
                if exists(case, method, eps) and all(exists(case, v, eps) for v in variants):
                    continue
                start = time.perf_counter()
                fit = params = shared = None
                try:
                    if method == "ours":
                        fit, params, components, details, shared = initial_network(model, checkpoint, case, device, eps, args.network_repeats)
                    else:
                        result, duration = timed(lambda: run_published_baseline(method, case["points"], **options))
                        fit, params = result.fit, result.parameters
                        components, details = dict(baseline_ms=duration), dict(initial=result.diagnostics)
                    snapshots, post = run_chain(case, method, fit, params, components, eps, args.peak_ratio, args.max_internal_knots)
                    details.update(post)
                except (ValueError, RuntimeError, TypeError, KeyError) as error:
                    snapshots = failed_snapshots(case, method, eps, args.peak_ratio, error, (time.perf_counter() - start) * 1000)
                    details = dict(error=str(error))
                if not exists(case, method, eps):
                    commit(case, method, eps, snapshots, params, details)
                for variant in variants:
                    if exists(case, variant, eps):
                        continue
                    start = time.perf_counter()
                    variant_params = None
                    try:
                        if shared is None and variant != "uniform_chord":
                            raise RuntimeError("Shared network forward failed")
                        variant_fit, variant_params, components = variant_initial(variant, case, shared, args.max_internal_knots, model.degree)
                        vs, vd = run_chain(case, variant, variant_fit, variant_params, components, eps, args.peak_ratio, args.max_internal_knots)
                    except (ValueError, RuntimeError, TypeError, KeyError) as error:
                        vs = failed_snapshots(case, variant, eps, args.peak_ratio, error, (time.perf_counter() - start) * 1000)
                        vd = dict(error=str(error))
                    commit(case, variant, eps, vs, variant_params, vd)
            assemble(destination, meta)
    report = assemble(destination, meta, complete=True)
    from plot_paper_evidence import render
    render(report, destination)
    print(f"COMPLETE: {destination / 'evidence.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
