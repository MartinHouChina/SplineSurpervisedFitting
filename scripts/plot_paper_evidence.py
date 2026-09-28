"""Plots and full-denominator summaries of the paired paper evidence benchmark."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_ours_thresholds import write_json
from benchmark_geometry import file_sha256
from plot_dual_error_comparison import SOURCES, PAPER_LABELS

COLORS = ("#00877a", "#d95f02", "#7570b3", "#c04472", "#3b78ab", "#7e8529")
MARKERS = ("o", "s", "^", "D", "v", "P")
ABLATIONS = (("ours", "raw", "A: Network\n+ first refit"),
             ("ours", "repaired", "B: A + repair"),
             ("ours", "pruned", "C: B + pruning"),
             ("uniform_shared_params", "pruned", "D: Uniform K64\nshared learned t"),
             ("all_candidates_shared_params", "pruned", "E: All candidates\nshared learned t"),
             ("uniform_chord", "pruned", "F: Uniform K64\nchord t"))
COMPONENT_LABELS = {"network_ms": "Network (GPU)", "transfer_ms": "D2H / extraction",
                    "initial_refit_ms": "First LS refit", "initializer_ms": "Initialization",
                    "baseline_ms": "Numerical method (CPU)", "repair_ms": "Common insertion repair",
                    "pruning_ms": "Common deletion", "failed_ms": "Unclassified failed execution"}


def validate(report):
    if not report.get("complete"):
        raise ValueError("Refusing to plot an incomplete experiment as final evidence")
    meta = report["metadata"]
    expected = {(c["dataset"], str(c["sample_id"]), method, float(eps), stage)
                for c in meta["cases"] for eps in meta["mse_tolerances"]
                for method in meta["methods"] + (meta["variants"] if eps == meta["ablation_tolerance"] else [])
                for stage in ("raw", "repaired", "pruned")}
    actual = [(r["dataset"], str(r["sample_id"]), r["method"], float(r["mse_tolerance"]), r["stage"])
              for r in report["measurements"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing or duplicate stage/case/method/threshold records")


def summarize(report):
    validate(report)
    groups = defaultdict(list)
    for row in report["measurements"]:
        for source in ("All", row["dataset"]):
            groups[(source, row["method"], row["stage"], row["mse_tolerance"])].append(row)
    result = []
    for (source, method, stage, epsilon), rows in sorted(groups.items()):
        finite = [r for r in rows if r["status"] == "ok"]
        item = dict(dataset=source, method=method, stage=stage, mse_tolerance=epsilon, n=len(rows), n_finite=len(finite),
                    pass_count=sum(bool(r["joint_pass"]) for r in rows),
                    pass_percent=100 * sum(bool(r["joint_pass"]) for r in rows) / len(rows))
        for field in ("mse", "max_squared_error", "final_k", "repair_refit_count", "pruning_refit_count"):
            values = [r[field] for r in finite if r.get(field) is not None]
            item[field + "_mean"] = float(np.mean(values)) if values else None
            item[field + "_max"] = float(np.max(values)) if values else None
        timings = [r["total_ms"] for r in rows]
        item.update(total_ms_mean=float(np.mean(timings)), total_ms_median=float(np.median(timings)),
                    total_ms_p95=float(np.percentile(timings, 95)))
        passed_timings = [r["total_ms"] for r in rows if r["joint_pass"]]
        item["total_ms_pass_mean"] = float(np.mean(passed_timings)) if passed_timings else None
        item["timing_components_mean"] = {}
        for field in COMPONENT_LABELS:
            if field == "failed_ms":
                values = [r["total_ms"] if r["status"] != "ok" else 0. for r in rows]
            else:
                values = [r["timing_components"].get(field, 0.) if r["status"] == "ok" else 0. for r in rows]
            item["timing_components_mean"][field] = float(np.mean(values))
        result.append(item)
    return result


def lookup(summary, source, method, stage, epsilon):
    return next(r for r in summary if (r["dataset"], r["method"], r["stage"], r["mse_tolerance"]) == (source, method, stage, epsilon))


def paired_success_summary(report):
    """Conditional paired comparisons, supplementary to all-case pass rates."""
    validate(report)
    output = []
    sources = ["All"] + sorted({c["dataset"] for c in report["metadata"]["cases"]})
    for source in sources:
        for epsilon in report["metadata"]["mse_tolerances"]:
            rows = [r for r in report["measurements"] if r["stage"] == "pruned"
                    and r["mse_tolerance"] == epsilon and (source == "All" or r["dataset"] == source)]
            ours = {(r["dataset"], str(r["sample_id"])): r for r in rows if r["method"] == "ours"}
            methods = sorted({r["method"] for r in rows if r["method"] != "ours"})
            for method in methods:
                other = {(r["dataset"], str(r["sample_id"])): r for r in rows if r["method"] == method}
                if set(ours) != set(other):
                    raise ValueError("Paired comparison requires identical attempted cases")
                pairs = [(a, other[key]) for key, a in ours.items() if a["joint_pass"] and other[key]["joint_pass"]]
                item = dict(dataset=source, mse_tolerance=epsilon, comparator=method, n_all=len(ours),
                            ours_pass=sum(r["joint_pass"] for r in ours.values()),
                            other_pass=sum(r["joint_pass"] for r in other.values()), both_pass=len(pairs),
                            ours_fewer=sum(a["final_k"] < b["final_k"] for a, b in pairs),
                            same_count=sum(a["final_k"] == b["final_k"] for a, b in pairs),
                            ours_more=sum(a["final_k"] > b["final_k"] for a, b in pairs))
                for field in ("final_k", "mse", "max_squared_error", "total_ms"):
                    item["ours_" + field + "_mean"] = float(np.mean([a[field] for a, _ in pairs])) if pairs else None
                    item["other_" + field + "_mean"] = float(np.mean([b[field] for _, b in pairs])) if pairs else None
                output.append(item)
    return output


def save(figure, path):
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def ablation_plot(summary, meta, directory):
    eps = meta["ablation_tolerance"]
    rows = [lookup(summary, "All", m, s, eps) for m, s, _ in ABLATIONS]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = [label for _, _, label in ABLATIONS]
    fields = (("pass_percent", "(a) Both error bounds satisfied", "Pass rate (%)", False),
              ("final_k_mean", "(b) Representation complexity", "Mean internal-knot count", False),
              ("mse_mean", "(c) Fitting error (all finite outputs)", "Mean squared Euclidean error", True),
              ("total_ms_mean", "(d) Full pipeline time", "Mean wall time per curve (ms)", True))
    for ax, (field, title, ylabel, logarithmic) in zip(axes.flat, fields):
        values = [r[field] if r[field] is not None else np.nan for r in rows]
        bars = ax.bar(range(len(rows)), values, color=COLORS, width=.65)
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if logarithmic:
            ax.set_yscale("log")
            positive = [v for v in values if np.isfinite(v) and v > 0]
            if positive:
                ax.set_ylim(min(positive) * .5, max(positive) * 1.35)
        elif field == "final_k_mean":
            positive = [v for v in values if np.isfinite(v) and v > 0]
            if positive:
                ax.set_ylim(0, max(positive) * 1.16)
        if field == "mse_mean":
            ax.axhline(eps, color=".4", linestyle="--", linewidth=1)
        if field == "pass_percent":
            ax.set_ylim(0, 112)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(f"{value:.1f}" if not field.startswith("mse") else f"{value:.2e}",
                            (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=9)
    fig.suptitle(f"Inference-time ablation | {rows[0]['n']} identical curves | MSE limit {eps:g}, MaxSE limit {eps * meta['peak_ratio']:g}", fontsize=16)
    fig.tight_layout(rect=(0, .055, 1, .955))
    fig.text(.5, .018, "D/E freeze C's parameterization and include its network cost; F is network-free. No retraining. All misses retained.", ha="center", fontsize=10)
    save(fig, directory / "ablation_stages.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    repairs = np.array([r["repair_refit_count_mean"] or 0 for r in rows])
    pruning = np.array([r["pruning_refit_count_mean"] or 0 for r in rows])
    ax.bar(range(6), repairs, label="Common insertion trial refits", color="#749caf")
    ax.bar(range(6), pruning, bottom=repairs, label="Common deletion trial refits", color="#c59b53")
    ax.set_xticks(range(6), labels, fontsize=9)
    ax.set(ylabel="Mean postprocessing refit count", title="Numerical search work after initialization (excludes initial/network/baseline internal solves)")
    ax.legend()
    fig.tight_layout()
    save(fig, directory / "ablation_postprocessing_refits.png")


def comparison_stage(method, protocol):
    if protocol == "raw":
        return "raw"
    if protocol == "historical":
        return "pruned" if method == "ours" else "repaired"
    return "pruned"


def tradeoff_plot(summary, meta, directory, protocol="common"):
    epsilons = meta["mse_tolerances"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, method in enumerate(meta["methods"]):
        rows = [lookup(summary, "All", method, comparison_stage(method, protocol), e) for e in epsilons]
        values = lambda field: [r[field] if r[field] is not None else np.nan for r in rows]
        options = dict(color=COLORS[j], marker=MARKERS[j], label=PAPER_LABELS[method], linewidth=1.5)
        axes[0, 0].plot(values("final_k_mean"), values("mse_mean"), **options)
        axes[0, 1].plot(values("final_k_mean"), values("max_squared_error_mean"), **options)
        axes[1, 0].plot(epsilons, values("pass_percent"), **options)
        axes[1, 1].plot(epsilons, values("final_k_mean"), **options)
    for ax, title, ylabel in ((axes[0, 0], "(a) Error vs complexity", "Mean MSE"),
                             (axes[0, 1], "(b) Peak error vs complexity", "Mean per-curve MaxSE")):
        ax.set(xlabel="Mean internal-knot count", ylabel=ylabel, title=title, yscale="log")
    for ax in axes[1]:
        ax.set_xscale("log")
        ax.set_xticks(epsilons, [f"{e:.0e}" for e in epsilons])
        ax.set_xlabel("Requested MSE tolerance")
    axes[1, 0].set(title="(c) Both error bounds satisfied", ylabel="Pass rate (%)", ylim=(0, 105))
    axes[1, 1].set(title="(d) Complexity vs tolerance", ylabel="Mean internal-knot count")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    labels = {"common": "All methods + common insertion + common deletion", "historical": "Ours + insertion/deletion; baselines + insertion only", "raw": "Raw methods, without common postprocessing"}
    fig.suptitle(f"Six-method trade-off | K cap {meta['max_internal_knots']} | {len(meta['cases'])} curves\n{labels[protocol]}", fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, .045, 1, .91))
    save(fig, directory / f"six_method_tradeoff_{protocol}.png")
    if protocol != "common":
        return
    sources = [s for s in SOURCES if any(c["dataset"] == s for c in meta["cases"])]
    fig, axes = plt.subplots(len(sources), 3, figsize=(15, 3.4 * len(sources)), squeeze=False)
    for i, source in enumerate(sources):
        for j, method in enumerate(meta["methods"]):
            rows = [lookup(summary, source, method, "pruned", e) for e in epsilons]
            vals = lambda field: [r[field] if r[field] is not None else np.nan for r in rows]
            opts = dict(color=COLORS[j], marker=MARKERS[j], label=PAPER_LABELS[method], linewidth=1.2, markersize=4)
            axes[i, 0].plot(vals("final_k_mean"), vals("mse_mean"), **opts)
            axes[i, 1].plot(vals("final_k_mean"), vals("max_squared_error_mean"), **opts)
            axes[i, 2].plot(epsilons, vals("pass_percent"), **opts)
        for j in range(2):
            axes[i, j].set(yscale="log", xlabel="Mean internal K", ylabel="Mean MSE" if j == 0 else "Mean MaxSE")
        axes[i, 2].set(xscale="log", ylim=(0, 105), xlabel="MSE tolerance", ylabel="Dual pass (%)")
        axes[i, 2].set_xticks(epsilons, [f"{e:.0e}" for e in epsilons])
        for ax in axes[i]:
            ax.grid(alpha=.2)
            ax.set_title(source)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Per-source comparison | all methods with common insertion and deletion", fontsize=16)
    fig.tight_layout(rect=(0, .035, 1, .97))
    save(fig, directory / "six_method_tradeoff_by_source.png")


def timing_plot(summary, meta, directory):
    eps = meta["ablation_tolerance"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    methods = meta["methods"]
    sources = ["All"] + [s for s in SOURCES if any(c["dataset"] == s for c in meta["cases"])]
    groups = ([lookup(summary, "All", m, "pruned", eps) for m in methods],
              [lookup(summary, s, "ours", "pruned", eps) for s in sources])
    palette = ("#00877a", "#b8d8d2", "#597dbe", "#afcaea", "#888888", "#d99547", "#a868ad", "#333333")
    for ax, rows, labels in zip(axes, groups, ([PAPER_LABELS[m] for m in methods], sources)):
        bottom = np.zeros(len(rows))
        for (field, label), color in zip(COMPONENT_LABELS.items(), palette):
            values = np.array([r["timing_components_mean"][field] if r["timing_components_mean"][field] is not None else np.nan for r in rows])
            ax.bar(range(len(rows)), values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set_xticks(range(len(rows)), [f"{label}\npass={row['pass_percent']:.1f}%" for label, row in zip(labels, rows)])
        ax.set_ylim(0, max(float(np.nanmax(bottom)), 1.) * 1.15)
        for index, value in enumerate(bottom):
            if np.isfinite(value):
                ax.annotate(f"{value:.0f}", (index, value), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=9)
        ax.set_ylabel("Mean per-curve wall time (ms)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title("(a) Six methods: full pipeline with common insertion and deletion")
    axes[1].set_title("(b) Ours: stage breakdown by data source")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Time decomposition | MSE limit {eps:g} | network: {meta['hardware']['gpu'] or 'CPU'}; numerical stages: CPU", fontsize=14)
    fig.tight_layout(rect=(0, .08, 1, .95))
    save(fig, directory / "time_decomposition.png")


def render(report, directory):
    directory = Path(directory)
    summary = summarize(report)
    meta = report["metadata"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    ablation_plot(summary, meta, directory)
    for protocol in ("common", "historical", "raw"):
        tradeoff_plot(summary, meta, directory, protocol)
    timing_plot(summary, meta, directory)
    write_json(directory / "summary.json", dict(metadata=meta, summary=summary))
    write_json(directory / "paired_success_summary.json", dict(
        scope="Conditional on BOTH methods passing on the same inputs; use together with all-case pass rates, not as their replacement.",
        comparisons=paired_success_summary(report)))
    lines = ["# Paired paper evidence", "", "All requested samples remain in pass-rate denominators. Error/count means include all finite outputs, including misses. Literature labels refer to repository adaptations, not author code. Common repair and pruning are separate experimental wrappers.", "",
             "Primary comparison: all methods receive the same repair/deletion rules and internal-knot cap. Historical comparison: only Ours receives deletion; other methods receive insertion only. Raw comparison: no added wrappers. Endpoint conventions inherited from each adapter are explicitly recorded in protocol.json.", "",
             "Ablations are inference interventions, not separately retrained networks. D/E keep the exact same learned parameters as C and are charged its network forward; E overrides only the final mask. F uses chord parameters without a network. Counts of solver calls below refer ONLY to common postprocessing.", ""]
    for source in ["All"] + list(SOURCES):
        subset = [r for r in summary if r["dataset"] == source]
        if not subset:
            continue
        lines += [f"## {source}", "", "| Method | Stage | MSE limit | Finite/all | Dual pass | Mean K | Mean MSE | Mean MaxSE | Worst MaxSE | All mean / pass mean / median / P95 ms |", "|---|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda v: "unavailable" if v is None else f"{v:.6g}"
        for r in subset:
            lines.append("| " + " | ".join([r["method"], r["stage"], fmt(r["mse_tolerance"]), f"{r['n_finite']}/{r['n']}",
                         f"{r['pass_count']}/{r['n']} ({r['pass_percent']:.1f}%)"] + [fmt(r[k]) for k in ("final_k_mean", "mse_mean", "max_squared_error_mean", "max_squared_error_max")] +
                         [" / ".join(fmt(r[k]) for k in ("total_ms_mean", "total_ms_pass_mean", "total_ms_median", "total_ms_p95"))]) + " |")
        lines.append("")
    (directory / "results.md").write_text("\n".join(lines), encoding="utf-8")
    provenance = dict(plot_script_sha256=file_sha256(Path(__file__)),
                      images={p.name: file_sha256(p) for p in sorted(directory.glob("*.png"))})
    if (directory / "evidence.json").exists():
        provenance["report_sha256"] = file_sha256(directory / "evidence.json")
    write_json(directory / "plot_provenance.json", provenance)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    render(json.loads(args.report.read_text(encoding="utf-8")), args.report.parent)
