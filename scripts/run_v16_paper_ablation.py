"""Paired feature and module ablations for the selected V16 checkpoint.

The experiment changes one inference component at a time, keeps the trained
weights and fixed test curves unchanged, and performs exactly one standard
B-spline refit.  It intentionally excludes residual insertion and verified
post-pruning so that numerical search cannot hide a network intervention.
These are shared-weight inference ablations, not claims about separately
retrained architectures.
"""
from __future__ import annotations

import argparse
import csv
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

from benchmark_geometry import file_sha256
from benchmark_ours_thresholds import DEFAULT_REPORT, load_cases
from spline_fitting.checkpointing import build_model_from_checkpoint
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.timing import synchronize_device


FEATURE_ARMS = (
    ("coordinates", "Coordinates only"),
    ("coordinates_first", r"Coordinates + first derivative"),
    ("full", r"Coordinates + first + second derivatives"),
)
MODULE_ARMS = (
    ("full", "Complete network"),
    ("fixed_threshold", "Fixed 0.5 KeepMask"),
    ("no_survivor_decoder", "No survivor-conditioned decoder"),
    ("all_candidates", "No candidate selection"),
    ("chord_parameters", "Chord parameters after selection"),
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sync(device: torch.device) -> None:
    synchronize_device(device)


def _chord_params(points: torch.Tensor, model) -> torch.Tensor:
    lengths = (points[:, 1:] - points[:, :-1]).norm(dim=-1)
    lengths = lengths.clamp_min(torch.finfo(points.dtype).eps)
    chord = torch.cat(
        [lengths.new_zeros(lengths.shape[0], 1),
         (lengths / lengths.sum(-1, keepdim=True)).cumsum(-1)],
        dim=-1,
    )
    chord = torch.cat([chord[:, :-1], chord.new_ones(chord.shape[0], 1)], -1)
    return model._strict_chord_reference(chord)


def _forward_variant(model, batch, tolerance: float, group: str, arm: str):
    feature_mode = {
        "coordinates": "coordinates",
        "coordinates_first": "coordinates_first",
    }.get(arm, "all") if group == "feature" else "all"
    model.encoder.set_ablation_feature_components(feature_mode)
    context = model.encode_candidates(batch, mse_tolerance=tolerance)

    if group == "module" and arm == "fixed_threshold":
        mask = context["keep_probabilities"] >= 0.5
    elif group == "module" and arm == "all_candidates":
        mask = torch.ones_like(context["keep_probabilities"], dtype=torch.bool)
    else:
        mask = model.select_mask(context)

    if group == "module" and arm == "no_survivor_decoder":
        params = context["proposal_params"]
        knots = context["proposal_internal_knots"]
    else:
        decoded = model.decode_subset(context, mask)
        params = decoded["params"]
        knots = decoded["internal_knots"]

    if group == "module" and arm == "chord_parameters":
        chord = context["chord_params"]
        knots = model._warp_knots(knots, params, chord)
        params = chord
    return params, knots, mask


def run_one(model, case, device, tolerance, peak_tolerance, group, arm, repeats):
    points = case["points"].double().cpu()
    dtype = next(model.parameters()).dtype
    batch = case["points"].unsqueeze(0).to(device=device, dtype=dtype)
    durations = []
    output = None
    # Warm-up is deliberately excluded and repeated for every intervention.
    with torch.inference_mode():
        _forward_variant(model, batch, tolerance, group, arm)
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            output = _forward_variant(model, batch, tolerance, group, arm)
            _sync(device)
            durations.append((time.perf_counter() - started) * 1000)
    assert output is not None
    params, knots, mask = output
    parameters = params[0].detach().cpu().double()
    selected = knots[0, mask[0]].detach().cpu().double().sort().values
    started = time.perf_counter()
    fit = refit_bspline_control_points(
        parameters,
        points,
        selected,
        degree=model.degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )
    refit_ms = (time.perf_counter() - started) * 1000
    squared = (fit.reconstructed_points - points).square().sum(-1)
    mse = float(squared.mean())
    peak = float(squared.max())
    return {
        "status": "ok",
        "k": int(selected.numel()),
        "mse": mse,
        "max_squared_error": peak,
        "mse_pass": mse <= tolerance,
        "joint_pass": mse <= tolerance and peak <= peak_tolerance,
        "network_ms": float(np.median(durations)),
        "network_repeat_ms": durations,
        "refit_ms": refit_ms,
        "total_ms": float(np.median(durations)) + refit_ms,
    }


def _percentile_interval(values, statistic, rng, samples=2000):
    values = np.asarray(values)
    if not len(values):
        return None, None
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = values[rng.integers(0, len(values), len(values))]
        draws[index] = statistic(selected)
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def summarize(rows, seed=230923):
    output = []
    sources = ["All", *dict.fromkeys(row["dataset"] for row in rows)]
    arms = [("feature", *arm) for arm in FEATURE_ARMS]
    arms += [("module", *arm) for arm in MODULE_ARMS]
    for group, arm, label in arms:
        for source in sources:
            selected = [row for row in rows if row["group"] == group and row["arm"] == arm
                        and (source == "All" or row["dataset"] == source)]
            valid = [row for row in selected if row["status"] == "ok"]
            rng = np.random.default_rng(seed + sum(map(ord, group + arm + source)))
            record = {
                "group": group,
                "arm": arm,
                "label": label,
                "dataset": source,
                "n": len(selected),
                "n_finite": len(valid),
                "failures": len(selected) - len(valid),
                "dual_pass_rate": sum(row.get("joint_pass", False) for row in selected) / max(len(selected), 1),
            }
            metrics = {
                "mean_k": ("k", np.mean),
                "mean_mse": ("mse", np.mean),
                "mean_max_squared_error": ("max_squared_error", np.mean),
                "worst_max_squared_error": ("max_squared_error", np.max),
                "mean_network_ms": ("network_ms", np.mean),
                "mean_total_ms": ("total_ms", np.mean),
            }
            pass_values = [float(row.get("joint_pass", False)) for row in selected]
            lo, hi = _percentile_interval(pass_values, np.mean, rng)
            record["dual_pass_rate_ci95_low"], record["dual_pass_rate_ci95_high"] = lo, hi
            for name, (field, statistic) in metrics.items():
                values = [row[field] for row in valid]
                record[name] = float(statistic(values)) if values else None
                lo, hi = _percentile_interval(values, statistic, rng)
                record[name + "_ci95_low"], record[name + "_ci95_high"] = lo, hi
            output.append(record)
    return output


def _errorbar(ax, values, lows, highs, *, color):
    x = np.arange(len(values))
    y = np.asarray(values, dtype=float)
    low = np.asarray(lows, dtype=float)
    high = np.asarray(highs, dtype=float)
    error = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])
    ax.errorbar(x, y, yerr=error, fmt="o", color=color, capsize=3, markersize=5)
    return x


def render_group(summary, group, arms, destination, tolerance, peak_tolerance):
    rows = [row for row in summary if row["group"] == group and row["dataset"] == "All"]
    order = {arm: index for index, (arm, _) in enumerate(arms)}
    rows.sort(key=lambda row: order[row["arm"]])
    labels = [row["label"] for row in rows]
    fields = (
        ("dual_pass_rate", "(a) Dual-constraint feasibility", "Pass rate (%)", 100.0, False, None),
        ("mean_k", "(b) Representation complexity", "Mean internal knots", 1.0, False, None),
        ("mean_mse", "(c) Mean fitting error", "Mean squared Euclidean error", 1.0, True, tolerance),
        ("mean_max_squared_error", "(d) Local error peaks", "Mean per-curve MaxSE", 1.0, True, peak_tolerance),
        ("worst_max_squared_error", "(e) Worst observed local error", "Worst MaxSE", 1.0, True, peak_tolerance),
        ("mean_network_ms", "(f) Learned inference", "Mean network time (ms)", 1.0, False, None),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6), constrained_layout=True)
    color = "#147d74" if group == "feature" else "#355f9d"
    for ax, (field, title, ylabel, factor, logarithmic, threshold) in zip(axes.flat, fields):
        values = [row[field] * factor if row[field] is not None else np.nan for row in rows]
        lows = [row[field + "_ci95_low"] * factor if row[field + "_ci95_low"] is not None else np.nan for row in rows]
        highs = [row[field + "_ci95_high"] * factor if row[field + "_ci95_high"] is not None else np.nan for row in rows]
        x = _errorbar(ax, values, lows, highs, color=color)
        ax.set_xticks(x, labels, rotation=16, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.22)
        if logarithmic:
            ax.set_yscale("log")
        if threshold is not None:
            ax.axhline(threshold, color=".35", linestyle="--", linewidth=1, label="constraint")
            ax.legend(frameon=False, fontsize=9)
        if field == "dual_pass_rate":
            ax.set_ylim(0, 105)
    heading = "Input-feature ablation" if group == "feature" else "Network-module ablation"
    fig.suptitle(
        f"{heading} | identical weights and curves | raw network + one B-spline refit\n"
        f"MSE <= {tolerance:g}, MaxSE <= {peak_tolerance:g}; bars show paired curve-bootstrap 95% intervals",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_tables(summary, directory):
    overall = [row for row in summary if row["dataset"] == "All"]
    fields = ["group", "arm", "label", "dataset", "n", "n_finite", "failures",
              "dual_pass_rate", "mean_k", "mean_mse", "mean_max_squared_error",
              "worst_max_squared_error", "mean_network_ms", "mean_total_ms"]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Paired feature and module ablation", "",
             "All rows use the same checkpoint and fixed curves. Removed feature channels are zeroed without changing the parameter count. Module arms intervene at inference. No residual insertion or verified post-pruning is included.", "",
             "| Group | Variant | Dual pass | Mean K | Mean MSE | Mean MaxSE | Worst MaxSE | Network ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in overall:
        lines.append(
            f"| {row['group']} | {row['label']} | {100*row['dual_pass_rate']:.1f}% | "
            f"{row['mean_k']:.2f} | {row['mean_mse']:.3e} | "
            f"{row['mean_max_squared_error']:.3e} | {row['worst_max_squared_error']:.3e} | "
            f"{row['mean_network_ms']:.2f} |"
        )
    (directory / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex = [r"\begin{table*}[t]", r"\centering", r"\caption{Shared-weight feature and module ablations on the fixed test cohort. No numerical repair or post-pruning is applied.}", r"\label{tab:paired-ablation}", r"\begin{tabular}{llrrrrrr}", r"\toprule", r"Group & Variant & Pass (\%) & Mean $K$ & Mean MSE & Mean MaxSE & Worst MaxSE & Network (ms)\\", r"\midrule"]
    for row in overall:
        label = row["label"].replace("+", r"$+$").replace("0.5", r"$0.5$")
        latex.append(
            f"{row['group'].capitalize()} & {label} & {100*row['dual_pass_rate']:.1f} & "
            f"{row['mean_k']:.2f} & {row['mean_mse']:.3e} & "
            f"{row['mean_max_squared_error']:.3e} & {row['worst_max_squared_error']:.3e} & "
            f"{row['mean_network_ms']:.2f}\\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (directory / "ablation_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/overnight_stable_k32_3090_r1.pt")
    parser.add_argument("--source-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mse-tolerance", type=float, default=5e-5)
    parser.add_argument("--max-squared-error-tolerance", type=float, default=5e-4)
    parser.add_argument("--network-repeats", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-cases", type=int, help="Diagnostic subset only; omitted for the paper run")
    parser.add_argument(
        "--paper-figure-dir",
        type=Path,
        help="Also export publication-ready ablation plots to this directory",
    )
    args = parser.parse_args(argv)
    if any(not path.is_file() for path in (args.checkpoint, args.source_report)):
        parser.error("checkpoint and source report must exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new or empty")
    if (args.network_repeats < 1 or args.torch_threads < 1
            or args.max_cases is not None and args.max_cases < 1):
        parser.error("repeat/thread/case counts must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.mse_tolerance, args.max_squared_error_tolerance)):
        parser.error("error tolerances must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, config, _ = build_model_from_checkpoint(checkpoint)
    if config["max_internal_knots"] != 32:
        raise ValueError("The paper ablation is pinned to the selected K32 checkpoint")
    _, cases = load_cases(args.source_report)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a diagnostic run")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(230923)
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": "v16_paired_feature_module_ablation_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_config": config,
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": file_sha256(args.source_report),
        "mse_tolerance": args.mse_tolerance,
        "max_squared_error_tolerance": args.max_squared_error_tolerance,
        "network_repeats": args.network_repeats,
        "cases": [{key: case.get(key) for key in ("dataset", "sample_id", "group_id", "input_points_sha256")} for case in cases],
        "feature_arms": list(FEATURE_ARMS),
        "module_arms": list(MODULE_ARMS),
        "scope": "Shared pretrained weights; raw network prediction and one endpoint-constrained cubic B-spline refit; no insertion or post-pruning.",
        "error_definition": "Squared Euclidean errors on normalized sampled observations; no square root and no continuous Hausdorff guarantee.",
        "hardware": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch_threads": args.torch_threads, "torch_version": str(torch.__version__)},
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in (
            Path(__file__), ROOT / "src/spline_fitting/models/geometry_encoder.py",
            ROOT / "src/spline_fitting/models/v16_network.py",
        )},
    }
    atomic_json(args.output_dir / "protocol.json", protocol)
    rows = []
    arms = [("feature", *arm) for arm in FEATURE_ARMS]
    arms += [("module", *arm) for arm in MODULE_ARMS]
    total = len(cases) * len(arms)
    completed = 0
    for group, arm, label in arms:
        for case in cases:
            base = {key: case.get(key) for key in (
                "dataset", "dataset_label", "sample_id", "group_id", "split",
                "source_kind", "source_k", "canonical_k", "input_points_sha256",
            )}
            row = {**base, "group": group, "arm": arm, "label": label}
            try:
                row.update(run_one(
                    model, case, device, args.mse_tolerance,
                    args.max_squared_error_tolerance, group, arm,
                    args.network_repeats,
                ))
            except (RuntimeError, ValueError, TypeError) as error:
                row.update(status="failed", mse_pass=False, joint_pass=False,
                           error=f"{type(error).__name__}: {error}")
            rows.append(row)
            completed += 1
            if completed == 1 or completed % 20 == 0 or completed == total:
                print(f"[{completed}/{total}] {group}/{arm} {case['dataset']} {case['sample_id']}", flush=True)
    model.encoder.set_ablation_feature_components("all")
    with (args.output_dir / "measurements.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = summarize(rows)
    atomic_json(args.output_dir / "summary.json", {"protocol": protocol, "summary": summary})
    render_group(summary, "feature", FEATURE_ARMS, args.output_dir / "feature_ablation",
                 args.mse_tolerance, args.max_squared_error_tolerance)
    render_group(summary, "module", MODULE_ARMS, args.output_dir / "module_ablation",
                 args.mse_tolerance, args.max_squared_error_tolerance)
    if args.paper_figure_dir is not None:
        args.paper_figure_dir.mkdir(parents=True, exist_ok=True)
        render_group(summary, "feature", FEATURE_ARMS,
                     args.paper_figure_dir / "ablation_features",
                     args.mse_tolerance, args.max_squared_error_tolerance)
        render_group(summary, "module", MODULE_ARMS,
                     args.paper_figure_dir / "ablation_modules",
                     args.mse_tolerance, args.max_squared_error_tolerance)
    write_tables(summary, args.output_dir)
    print(f"Completed: {args.output_dir / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
