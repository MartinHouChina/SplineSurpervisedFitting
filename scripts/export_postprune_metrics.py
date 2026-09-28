"""Publish four metric groups from every final measured case, never a showcase subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from plot_dual_error_comparison import _statistics, pair_measurements, SOURCES

NAMES = {
    "ours": "Ours", "park_dominant_point_2007_adaptation": "Park",
    "liang_feature_iki_2017_adaptation": "Liang", "dung_direct_knot_2017_adaptation": "Dung",
    "kang_sparse_2015_adaptation": "Kang", "luo_linf_de_2022_adaptation": "Luo",
}
SOURCE_NAMES = {"All": "All test curves", "Synthetic": "Synthetic",
                "UJI": "UJI handwriting", "NaturalEarth": "Natural Earth coastline",
                "USGS": "USGS contours", "IndustrialOffset": "Industrial offsets"}


def summary(report):
    """Every declared case stays in pass-rate denominators, including fit failures."""
    pairs = pair_measurements(report)
    final = [b for _, b in pairs]
    methods = report["metadata"]["methods"]
    unknown = set(methods) - set(NAMES)
    if unknown:
        raise ValueError(f"Unrecognized method labels: {sorted(unknown)}")
    observed = {r["dataset"] for r in final}
    sources = ["All"] + [s for s in SOURCES if s in observed] + sorted(observed - set(SOURCES))
    result = []
    for source in sources:
        for method in methods:
            rows = [r for r in final if r["method"] == method and (source == "All" or r["dataset"] == source)]
            if rows:
                result.append(dict(dataset=source, method=method, label=NAMES[method],
                                   **_statistics(rows, report["metadata"])))
    return result


def number(value, spec):
    return "—" if value is None or not math.isfinite(value) else format(value, spec)


def table_cells(row):
    return [row["label"], f"{row['n_finite']}/{row['n_requested']}",
            number(row["final_k_mean"], ".2f"), number(row["dual_pass_percent"], ".1f"),
            number(row["mse_mean"], ".3e"), number(row["max_squared_error_max"], ".3e"),
            number(row["total_ms_mean"], ".2f")]


def _draw_table(ax, rows, title, font=11):
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    widths = np.array([.14, .12, .14, .14, .16, .17, .13])
    edges = np.r_[0, widths.cumsum()]
    ax.text(0, 1.015, title, ha="left", va="bottom", fontsize=font + 2.5, weight="bold", color="#17324b")
    top, group_h, label_h = .97, .11, .12
    group_y = top - group_h
    groups = [(0, 1, "Method"), (1, 2, "Fits"), (2, 3, "Complexity"),
              (3, 4, "Feasibility"), (4, 6, "Fit error"), (6, 7, "Runtime")]
    for lo, hi, text in groups:
        ax.add_patch(Rectangle((edges[lo], group_y), edges[hi] - edges[lo], group_h,
                               facecolor="#244b69", edgecolor="white", linewidth=1))
        ax.text((edges[lo] + edges[hi]) / 2, group_y + group_h / 2, text,
                ha="center", va="center", fontsize=font, color="white", weight="bold")
    labels = ["", "Valid / all", "Mean internal K", "Dual pass (%)", "Mean MSE", "Max. squared error", "Mean time (ms)"]
    label_y = group_y - label_h
    for col, text in enumerate(labels):
        ax.add_patch(Rectangle((edges[col], label_y), widths[col], label_h,
                               facecolor="#eaf0f5", edgecolor="white", linewidth=.8))
        ax.text((edges[col] + edges[col + 1]) / 2, label_y + label_h / 2, text,
                ha="center", va="center", fontsize=font - 1, color="#17324b")
    row_h = (label_y - .03) / len(rows)
    for i, row in enumerate(rows):
        bottom = label_y - (i + 1) * row_h
        face = "#e6f2ee" if row["method"] == "ours" else ("#f6f8fa" if i % 2 else "white")
        ax.add_patch(Rectangle((0, bottom), 1, row_h, facecolor=face, edgecolor="white", linewidth=1))
        for col, value in enumerate(table_cells(row)):
            ax.text((edges[col] + edges[col + 1]) / 2, bottom + row_h / 2, value,
                    ha="center", va="center", fontsize=font,
                    weight="bold" if row["method"] == "ours" else "normal", color="#182f43")
    ax.plot([0, 1], [.03, .03], color="#a6b5c1", linewidth=.8)


def render_tables(rows, directory, dpi):
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    overall = [r for r in rows if r["dataset"] == "All"]
    n = overall[0]["n_requested"]
    fig, ax = plt.subplots(figsize=(14, 4.1))
    _draw_table(ax, overall, f"Six-method comparison | {n} test curves", font=11.2)
    fig.subplots_adjust(left=.025, right=.98, top=.89, bottom=.035)
    fig.savefig(directory / "core_metrics.png", dpi=dpi)
    plt.close(fig)
    sources = list(dict.fromkeys(r["dataset"] for r in rows if r["dataset"] != "All"))
    fig, axes = plt.subplots(len(sources), 1, figsize=(14, 3.45 * len(sources)), squeeze=False)
    for ax, source in zip(axes[:, 0], sources):
        group = [r for r in rows if r["dataset"] == source]
        _draw_table(ax, group, f"{SOURCE_NAMES.get(source, source)} | n = {group[0]['n_requested']}", font=10.5)
    fig.subplots_adjust(left=.025, right=.98, top=.975, bottom=.02, hspace=.28)
    fig.savefig(directory / "core_metrics_by_dataset.png", dpi=dpi)
    plt.close(fig)


def definitions(metadata):
    return {
        "four_groups": ["complexity", "feasibility", "fit_error", "runtime"],
        "complexity": "Mean final internal-knot count across finite returned fits; endpoint repetitions excluded.",
        "feasibility": f"Both MSE <= {metadata['mse_tolerance']} and maximum point squared Euclidean error <= {metadata['max_squared_error_tolerance']}; denominator includes every requested case and algorithm failure.",
        "mean_mse": "Arithmetic mean of per-curve mean squared Euclidean error on normalized input points; all finite returned fits, including tolerance misses. No square root or coordinate-dimension division.",
        "max_squared_error": "Maximum single-point squared Euclidean error across all input points of all finite returned curves, not mean peak error and not a continuous/Hausdorff error certificate.",
        "runtime": "Arithmetic mean of measured total_ms across all requested cases; includes original method, common insertion, and any Ours post-pruning. Network-only time is stored separately, never substituted.",
        "valid_all": "Finite returned fits / requested cases; a finite fit may still fail either tolerance.",
        "population": "All report measurements, not only selected examples or passing curves. Industrial offsets are procedurally generated industrial-model curves, not field measurements.",
        "protocol": "Five literature repository adaptations retain their existing common insertion wrapper; new redundant-knot post-pruning is Ours-only. Detail is recorded off-figure for the manuscript.",
        "timing_scope": metadata["timing_protocol"],
        "timing_caution": "Local exploratory timings on the recorded hardware; single raw measurements and CPU/GPU differences do not establish a controlled speedup claim.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output_names = ("core_metrics.png", "core_metrics_by_dataset.png", "core_metrics.json", "core_metrics.md")
    if any((args.output_dir / name).exists() for name in output_names):
        parser.error("Metric artifacts already exist; choose a new output directory")
    if args.dpi < 72:
        parser.error("dpi must be >= 72")
    original = args.report.read_bytes()
    report = json.loads(original.decode("utf-8"))
    rows = summary(report)
    notes = definitions(report["metadata"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_tables(rows, args.output_dir, args.dpi)
    document = dict(source_report=str(args.report.resolve()), source_report_sha256=hashlib.sha256(original).hexdigest(),
                    checkpoint=report["metadata"].get("checkpoint"),
                    checkpoint_sha256=report["metadata"].get("checkpoint_sha256"),
                    definitions=notes, summary=rows)
    (args.output_dir / "core_metrics.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Final six-method comparison", "", *[f"- **{key}**: {value}" for key, value in notes.items()], ""]
    for source in dict.fromkeys(r["dataset"] for r in rows):
        lines += [f"## {SOURCE_NAMES.get(source, source)}", "", "| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |", "|---|---:|---:|---:|---:|---:|---:|"]
        lines += ["| " + " | ".join(table_cells(r)) + " |" for r in rows if r["dataset"] == source]
        lines += [""]
    (args.output_dir / "core_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    if args.report.read_bytes() != original:
        raise RuntimeError("Source report changed during rendering")
    print(f"Saved four metric groups (MSE and MaxSE both included): {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
