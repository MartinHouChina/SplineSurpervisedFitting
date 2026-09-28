"""Fresh fixed-checkpoint inference across tolerances on identical saved test curves."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from benchmark_geometry import file_sha256, write_case_geometry, write_method_geometry
from benchmark_historical_dual_error import row_from_fit, failure_row
from plot_dual_error_comparison import _geometry, pair_measurements, SOURCES
from postprune_saved_comparison import _case_from_arrays
from select_historical_checkpoint import raw_network_fit, network_forward, write_json as _write_json
from spline_fitting.checkpointing import build_model_from_checkpoint
from spline_fitting.evaluation.dual_error_knot_repair import repair_knots_to_dual_tolerance
from spline_fitting.evaluation.dual_error_knot_pruning import prune_knots_to_dual_tolerance
from spline_fitting.evaluation.timing import synchronize_device

DEFAULT_REPORT = ROOT / "outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json"
TOLERANCES = (1e-5, 2.5e-5, 5e-5, 1e-4)


def write_json(path, payload):
    """Retry a briefly locked Windows atomic replacement; preserve prior results."""
    for attempt in range(6):
        try:
            _write_json(path, payload)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1 * (attempt + 1))


def load_cases(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pairs = pair_measurements(report)
    cases = []
    for _, row in pairs:
        if row["method"] != "ours":
            continue
        document, arrays, _ = _geometry(report_path.parent, row)
        case = _case_from_arrays(document, arrays)
        if case["input_points_sha256"] != row["input_points_sha256"]:
            raise ValueError("Input point bytes disagree with the original comparison")
        cases.append(case)
    keys = {(c["dataset"], str(c["sample_id"])) for c in cases}
    expected = {(c["dataset"], str(c["sample_id"])) for c in report["metadata"]["cases"]}
    if keys != expected or len(keys) != len(cases):
        raise ValueError("Sweep must include every original paired test case exactly once")
    return report, cases


def run_case(model, checkpoint, case, device, epsilon, peak, capacity):
    """Execute a new network prediction, new insertion and new deletion each time."""
    synchronize_device(device)
    started = time.perf_counter()
    fit = parameters = None
    try:
        fit, parameters = raw_network_fit(model, checkpoint, case["points"], device, epsilon)
        synchronize_device(device)
        raw_ms = (time.perf_counter() - started) * 1000
        if fit.internal_knots.numel() > capacity:
            raise ValueError("Network prediction exceeded the shared knot cap")
        raw = row_from_fit(case, "ours", fit, parameters, raw_ms, epsilon, peak)
    except (RuntimeError, ValueError, TypeError) as error:
        row = failure_row(case, "ours", str(error), (time.perf_counter() - started) * 1000)
        row.update(mse_tolerance=epsilon, max_squared_error_tolerance=peak)
        return row, dict(row), dict(row), None, parameters
    current = dict(raw)
    details = {}
    elapsed = raw_ms
    for stage, function in (("repair", repair_knots_to_dual_tolerance), ("pruning", prune_knots_to_dual_tolerance)):
        stage_started = time.perf_counter()
        try:
            options = {"max_deletions": capacity} if stage == "pruning" else {}
            result = function(parameters.double().cpu(), case["points"], fit,
                              mse_tolerance=epsilon, max_squared_error_tolerance=peak,
                              max_internal_knots=capacity, interpolate_endpoints=True, **options)
            duration = getattr(result, "elapsed_repair_ms" if stage == "repair" else "elapsed_pruning_ms")
            candidate = row_from_fit(case, "ours", result.final_fit, parameters, elapsed + duration, epsilon, peak)
            if stage == "pruning" and current["joint_pass"] and not candidate["joint_pass"]:
                raise RuntimeError("Pruning must not invalidate an already feasible fit")
            details[stage] = {k: v for k, v in vars(result).items() if k not in {"initial_fit", "final_fit"}}
            elapsed += duration
            current, fit = candidate, result.final_fit
        except (RuntimeError, ValueError, TypeError) as error:
            duration = (time.perf_counter() - stage_started) * 1000
            elapsed += duration
            current = {**current, "total_ms": elapsed}
            details[stage] = dict(error=str(error), termination="failed_previous_fit_retained")
        details[stage + "_ms"] = duration
        if stage == "repair":
            before = dict(current)
    current.update(raw_ms=raw_ms, repair_ms=details["repair_ms"], pruning_ms=details["pruning_ms"],
                   diagnostics=details, mse_tolerance=epsilon, max_squared_error_tolerance=peak)
    for row in (raw, before):
        row.update(mse_tolerance=epsilon, max_squared_error_tolerance=peak)
    return current, raw, before, fit, parameters


def summarize(report):
    meta, rows = report["metadata"], report["measurements"]
    expected = {(float(t), c["dataset"], str(c["sample_id"])) for t in meta["mse_tolerances"] for c in meta["cases"]}
    actual = [(float(r["mse_tolerance"]), r["dataset"], str(r["sample_id"])) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated threshold/case measurements")
    groups = ["All"] + [s for s in SOURCES if any(c["dataset"] == s for c in meta["cases"])]
    out = []
    for group in groups:
        for tolerance in meta["mse_tolerances"]:
            selected = [r for r in rows if r["mse_tolerance"] == tolerance and (group == "All" or r["dataset"] == group)]
            valid = [r for r in selected if r["status"] == "ok"]
            record = dict(dataset=group, mse_tolerance=tolerance, max_squared_error_tolerance=tolerance * meta["peak_ratio"],
                          n=len(selected), n_finite=len(valid),
                          mse_pass_count=sum(bool(r["fit_pass"]) for r in selected),
                          dual_pass_count=sum(bool(r["joint_pass"]) for r in selected))
            record["dual_pass_percent"] = 100 * record["dual_pass_count"] / record["n"]
            for name in ("mse", "max_squared_error", "final_k"):
                values = [r[name] for r in valid]
                record[name + "_mean"] = float(np.mean(values)) if values else None
                record[name + "_max"] = float(np.max(values)) if values else None
            record["total_ms_mean"] = float(np.mean([r["total_ms"] for r in selected]))
            out.append(record)
    return out


def render(report, directory):
    rows = summarize(report)
    overall = [r for r in rows if r["dataset"] == "All"]
    x = np.array([r["mse_tolerance"] for r in overall])
    labels = [f"{v:.2g}" for v in x]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    for ax in axes.flat:
        ax.set_xscale("linear" if len(x) > 5 else "log")
        ax.set_xticks(x, labels)
        if len(x) > 5:
            ax.tick_params(axis="x", labelrotation=45, labelsize=9)
        ax.set_xlabel("Requested MSE tolerance")
        ax.grid(alpha=.22)
    axes[0, 0].plot(x, [r["mse_mean"] for r in overall], "o-", color="#00877a", label="Mean MSE (all finite fits)")
    axes[0, 0].plot(x, x, "--", color="#777777", label="MSE bound")
    axes[0, 0].set(title="(a) Mean fitting error", ylabel="Squared Euclidean error", yscale="log")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].plot(x, [r["max_squared_error_mean"] for r in overall], "o-", color="#00877a", label="Mean per-curve MaxSE")
    axes[0, 1].plot(x, [r["max_squared_error_max"] for r in overall], "s-", color="#bd6544", label="Worst MaxSE")
    axes[0, 1].plot(x, x * report["metadata"]["peak_ratio"], "--", color="#777777", label="MaxSE bound")
    axes[0, 1].set(title="(b) Maximum point error", ylabel="Squared Euclidean error", yscale="log")
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].plot(x, [r["dual_pass_percent"] for r in overall], "o-", color="#00877a")
    axes[1, 0].set(title="(c) Both error bounds satisfied", ylabel="Pass rate (%)", ylim=(0, 105))
    axes[1, 1].plot(x, [r["final_k_mean"] for r in overall], "o-", color="#00877a")
    axes[1, 1].set(title="(d) Representation complexity", ylabel="Mean internal-knot count", ylim=(0, report["metadata"]["max_internal_knots"] + 1))
    fig.suptitle(f"Ours | K cap = {report['metadata']['max_internal_knots']} | {len(x)} tolerances | {overall[0]['n']} identical curves", fontsize=16)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.text(.5, .018, f"Fixed weights and inputs; fresh deployment at each tolerance. MaxSE bound = {report['metadata']['peak_ratio']:g} × MSE bound.", ha="center", fontsize=10)
    fig.savefig(directory / "ours_threshold_sweep.png", dpi=300)
    plt.close(fig)
    # Separate source-wise table keeps the main plot legible.
    lines = ["# Ours multi-tolerance test", "", "Fixed weights; each tolerance reruns the network, standard refit, insertion and deletion. All requested cases remain in the pass denominator. Errors/counts include every finite returned fit, not only passes.", "",
             "MSE and MaxSE are squared Euclidean errors on normalized input points, never square-rooted. MaxSE bound is the MSE bound multiplied by the recorded peak_ratio. No continuous-curve/Hausdorff guarantee.", "",
             "This is a fixed-checkpoint deployment sensitivity study, not separately trained models. No model selection uses these test results. Checkpoint identity and capacity are recorded in the JSON metadata.", ""]
    for group in dict.fromkeys(r["dataset"] for r in rows):
        lines += [f"## {group}", "", "| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["dataset"] != group:
                continue
            fmt = lambda value: "—" if value is None else f"{value:.6g}"
            fields = [fmt(r["mse_tolerance"]), fmt(r["max_squared_error_tolerance"]), f"{r['n_finite']}/{r['n']}", f"{r['dual_pass_count']}/{r['n']} ({r['dual_pass_percent']:.1f}%)"]
            fields += [fmt(r[k]) for k in ("mse_mean", "max_squared_error_mean", "max_squared_error_max", "final_k_mean", "total_ms_mean")]
            lines.append("| " + " | ".join(fields) + " |")
        lines.append("")
    (directory / "threshold_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(directory / "threshold_summary.json", dict(metadata=report["metadata"], summary=rows))
    return rows


def resolve_capacity(model_config, requested, source_capacity, changed_checkpoint):
    trained = int(model_config["max_internal_knots"])
    capacity = requested if requested is not None else (trained if changed_checkpoint else source_capacity)
    if capacity < 1 or capacity > trained:
        raise ValueError(f"Knot cap must be between 1 and the trained capacity {trained}")
    return capacity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-checkpoint-change", action="store_true",
                        help="Explicitly reuse only source test inputs with a different trained checkpoint")
    parser.add_argument("--max-internal-knots", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mse-tolerances", type=float, nargs="+", default=TOLERANCES)
    parser.add_argument("--peak-ratio", type=float, default=10.)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)
    values = sorted(args.mse_tolerances)
    if len(set(values)) != len(values) or len(values) < 2 or any(not math.isfinite(v) or v <= 0 for v in [*values, args.peak_ratio]) or args.torch_threads < 1:
        parser.error("At least two distinct positive finite tolerances and positive thread count/peak ratio are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty")
    source, cases = load_cases(args.source_report)
    path = args.checkpoint or Path(source["metadata"]["checkpoint"])
    changed = file_sha256(path) != source["metadata"]["checkpoint_sha256"]
    if changed and not args.allow_checkpoint_change:
        parser.error("Checkpoint hash differs; explicitly use --allow-checkpoint-change to reuse test inputs only")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    try:
        capacity = resolve_capacity(checkpoint["model_config"], args.max_internal_knots,
                                    source["metadata"]["max_internal_knots"], changed)
    except ValueError as error:
        parser.error(str(error))
    model, _, _ = build_model_from_checkpoint(checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(20260923)
    meta = dict(schema_version="ours_fixed_checkpoint_threshold_sweep_v2", checkpoint=str(path.resolve()), checkpoint_sha256=file_sha256(path),
                source_report=str(args.source_report.resolve()), source_report_sha256=file_sha256(args.source_report),
                mse_tolerances=values, peak_ratio=args.peak_ratio, max_internal_knots=capacity,
                trained_capacity=checkpoint["model_config"]["max_internal_knots"],
                checkpoint_epoch=checkpoint.get("epoch"), model_config=checkpoint["model_config"],
                checkpoint_training_tolerance=checkpoint["model_config"].get("mse_tolerance"),
                source_checkpoint_sha256=source["metadata"]["checkpoint_sha256"],
                explicit_checkpoint_change=changed,
                training="No training or model reselection; identical fixed weights at all tolerances",
                execution="Each pair reruns threshold-conditioned network + endpoint-constrained standard LS + fresh dual-bound insertion + fresh redundant-knot deletion",
                timing="raw_ms+repair_ms+pruning_ms, synchronized GPU; metric/export/plot overhead excluded; exploratory single-run timings",
                hardware=dict(device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None, torch_threads=args.torch_threads),
                cases=[{k: c.get(k) for k in ("dataset", "sample_id", "group_id", "input_points_sha256")} for c in cases],
                code_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in (Path(__file__), ROOT / "src/spline_fitting/evaluation/dual_error_knot_repair.py", ROOT / "src/spline_fitting/evaluation/dual_error_knot_pruning.py")})
    fingerprint = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    meta["fingerprint"] = fingerprint
    report = dict(metadata=meta, complete=False, measurements=[], raw_measurements=[], pre_pruning_measurements=[])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tier, epsilon in enumerate(values):
        peak = epsilon * args.peak_ratio
        destination = args.output_dir / f"tier_{tier + 1}"
        for _ in range(3):
            network_forward(model, checkpoint, cases[0]["points"], device, epsilon)
        for i, case in enumerate(cases, 1):
            final, raw, before, fit, parameters = run_case(model, checkpoint, case, device, epsilon, peak, meta["max_internal_knots"])
            artifact = write_case_geometry(args.output_dir, case, fingerprint=fingerprint)
            geometry = write_method_geometry(destination, case, final, fit, parameters, case_artifact=artifact, fingerprint=fingerprint,
                                             dense_points=801, timing_scope=meta["timing"], pass_criterion="Both per-tier squared observation-error bounds")
            geometry["path"] = f"tier_{tier + 1}/" + geometry["path"]
            final["geometry_artifact"] = geometry
            report["measurements"].append(final)
            report["raw_measurements"].append(raw)
            report["pre_pruning_measurements"].append(before)
            write_json(args.output_dir / "threshold_sweep.json", report)
            print(f"Tier {tier + 1}/{len(values)} MSE<={epsilon:.2g} [{i}/{len(cases)}] {case['dataset']} K={final['final_k']} pass={final['joint_pass']} MSE={final['mse']}", flush=True)
    summarize(report)  # Completion is asserted before figures are exported.
    report["complete"] = True
    write_json(args.output_dir / "threshold_sweep.json", report)
    render(report, args.output_dir)
    print(f"Saved completed {len(values)}-tolerance measurements and PNG: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
