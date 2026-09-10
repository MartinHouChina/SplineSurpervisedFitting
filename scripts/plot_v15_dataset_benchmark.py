"""Render measured v15/v16 dataset-benchmark results as standalone PNG figures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter
import numpy as np


METHODS = (
    "ours",
    "park_dominant_point_2007_adaptation",
    "liang_feature_iki_2017_adaptation",
    "dung_direct_knot_2017_adaptation",
    "kang_sparse_2015_adaptation",
    "luo_linf_de_2022_adaptation",
    "yeh_feature_cdf_2020",
    "uniform_gradient_pruning",
)
LABELS = {
    "ours": "Ours v15",
    "park_dominant_point_2007_adaptation": "Park & Lee 2007 (DOM adaptation)",
    "liang_feature_iki_2017_adaptation": "Liang et al. 2017 (feature-IKI adaptation)",
    "dung_direct_knot_2017_adaptation": "Dung & Tjahjowidodo 2017 (serial adaptation)",
    "kang_sparse_2015_adaptation": "Kang-inspired ADMM (diagnostic)",
    "luo_linf_de_2022_adaptation": "Luo et al. 2022 ($l_{\\infty,1}$ + DE adaptation)",
    "yeh_feature_cdf_2020": "Yeh 2020 (feature-CDF adaptation)",
    "uniform_gradient_pruning": "Uniform greedy + gradient",
}
COLORS = {
    "ours": "#168875",
    "park_dominant_point_2007_adaptation": "#3565a8",
    "liang_feature_iki_2017_adaptation": "#42a5c6",
    "dung_direct_knot_2017_adaptation": "#dc8a31",
    "kang_sparse_2015_adaptation": "#7854ac",
    "luo_linf_de_2022_adaptation": "#b65391",
    "yeh_feature_cdf_2020": "#3275ad",
    "uniform_gradient_pruning": "#ce5151",
}
DATASET_ORDER = ("Synthetic", "UJI", "NaturalEarth", "USGS")
DATASET_LABELS = {"NaturalEarth": "Natural Earth"}


def model_version(metadata: dict) -> str:
    versions = {
        "candidate_pruning_deployment_aligned_feedback_v15": "v15",
        "candidate_selection_counterfactual_bspline_v16": "v16",
        "candidate_selection_supervised_bspline_v16": "v16",
    }
    objective = metadata.get("objective_version")
    if objective in versions:
        return versions[objective]
    # Historical smoke reports predate explicit objective metadata. Keep their
    # v15 label, but never mislabel an explicit unknown architecture as v15.
    if objective is None:
        return str(metadata.get("model_version", "v15"))
    raise ValueError(f"Unsupported benchmark objective: {objective}")


def read_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata, summary = report.get("metadata", {}), report.get("summary", [])
    tolerance = metadata.get("mse_tolerance")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("metadata.mse_tolerance must be finite and positive")
    if not isinstance(summary, list) or not summary:
        raise ValueError("The report has no completed summary rows")
    pairs = set()
    for row in summary:
        pair = (row["dataset"], row["method"])
        if pair in pairs:
            raise ValueError(f"Duplicate summary row: {pair}")
        pairs.add(pair)
        if row["method"] not in METHODS:
            raise ValueError(f"Unsupported method: {row['method']}")
        for key in ("mse_mean", "reference_mse_mean", "total_ms_mean", "network_ms_mean"):
            value = row.get(key)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {key} for {pair}: {value}")
        for key in ("fit_pass_rate", "reference_pass_rate"):
            value = row.get(key)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid {key} for {pair}: {value}")
    return report


def _num_points(metadata: dict) -> str:
    value = metadata.get("num_points")
    if value is None:
        for source in metadata.get("datasets", []):
            value = source.get("config", {}).get("num_points")
            if value is not None:
                break
    return str(value) if value is not None else "checkpoint-configured"


def _datasets(rows: list[dict], reference: bool) -> list[str]:
    selected = {
        row["dataset"] for row in rows
        if not reference or row.get("reference_pass_rate") is not None
    }
    return [name for name in DATASET_ORDER if name in selected] + sorted(selected.difference(DATASET_ORDER))


def _network_caption(index: dict, datasets: list[str]) -> str:
    entries = []
    for name in datasets:
        value = index.get((name, "ours"), {}).get("network_ms_mean")
        if value is not None:
            entries.append(f"{DATASET_LABELS.get(name, name)} {value:.2f}")
    return (
        "Ours network only (separate measurement, ms): " + "; ".join(entries) + "."
        if entries else "Ours network-only timing is unavailable in this report."
    )


def render_report(report: dict, output_dir: Path, *, dpi: int = 220, reference: bool = False) -> Path | None:
    """Plot summary values without rescaling reported MSE or hiding failures."""
    metadata, rows = report["metadata"], report["summary"]
    version = model_version(metadata)
    labels = {**LABELS, "ours": f"Ours {version}"}
    datasets = _datasets(rows, reference)
    if not datasets:
        return None
    index = {(row["dataset"], row["method"]): row for row in rows}
    methods = [method for method in METHODS if any((name, method) in index for name in datasets)]
    tolerance = metadata["mse_tolerance"]
    diagnostic = bool(metadata.get("diagnostic_not_final", False))
    mse_key = "reference_mse_mean" if reference else "mse_mean"
    pass_key = "reference_pass_rate" if reference else "fit_pass_rate"
    metric_keys = (mse_key, pass_key, "total_ms_mean")
    x = np.arange(len(datasets), dtype=float)
    width = 0.78 / len(methods)

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 3, figsize=(21.5, 8.2))
        fig.subplots_adjust(left=0.052, right=0.992, bottom=0.27, top=0.70, wspace=0.25)
        kind = "Original-reference evaluation" if reference else "Input-point fitting evaluation"
        fig.suptitle(f"{version} versus numerical baselines | {kind}", y=0.974, fontsize=19)
        subtitle = (
            f"Normalized Euclidean MSE (no square root); threshold = {tolerance:.2e}; "
            f"{_num_points(metadata)} input points per curve"
        )
        fig.text(0.5, 0.914, subtitle, ha="center", fontsize=11.5, color="#444444")
        if diagnostic:
            fig.text(
                0.5, 0.5, "DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT",
                ha="center", va="center", rotation=24, fontsize=30,
                color="crimson", alpha=0.20, weight="bold", zorder=100,
            )
        legend = [Patch(facecolor=COLORS[method], label=labels[method]) for method in methods]
        fig.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.875),
            ncol=min(4, len(methods)),
            frameon=False,
            fontsize=9.8,
        )

        tick_labels = []
        for name in datasets:
            counts = sorted({int(index[(name, method)]["n"]) for method in methods if (name, method) in index})
            sample_count = str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
            tick_labels.append(f"{DATASET_LABELS.get(name, name)}\n$n={sample_count}$")

        for panel, (ax, key) in enumerate(zip(axes, metric_keys)):
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#dfe3e8", linewidth=0.8)
            ax.set_xticks(x, tick_labels)
            ax.tick_params(axis="x", length=0, pad=10)
            ax.set_xlim(-0.55, len(datasets) - 0.45)
            positives = [float(index[(name, method)][key]) for name in datasets for method in methods
                         if (name, method) in index and index[(name, method)].get(key) is not None
                         and index[(name, method)][key] > 0]
            if panel != 1:
                positives += [tolerance] if panel == 0 else []
                floor = min(positives) / 2.0 if positives else 1e-12
                ceiling = max(positives) * 2.2 if positives else 1.0
                ax.set_yscale("log")
                ax.set_ylim(floor, max(ceiling, floor * 10.0))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                floor = 0.0
                ax.set_ylim(0, 112)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

            for j, method in enumerate(methods):
                offsets = x + (j - (len(methods) - 1) / 2) * width
                for name, xpos in zip(datasets, offsets):
                    value = index.get((name, method), {}).get(key)
                    if value is None:
                        ax.text(xpos, 0.035, "N/A", transform=ax.get_xaxis_transform(),
                                ha="center", va="bottom", rotation=90, fontsize=8, color="#777777")
                        continue
                    value = float(value) * (100.0 if panel == 1 else 1.0)
                    if value == 0 and panel != 1:
                        # A log axis cannot show zero; disclose an exact zero
                        # at its lower bound instead of substituting a value.
                        ax.scatter([xpos], [floor * 1.05], marker="v", s=32, color=COLORS[method])
                        ax.annotate("0", (xpos, floor * 1.05), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
                    else:
                        ax.bar(xpos, value, width=width * 0.94, color=COLORS[method], edgecolor="white", linewidth=0.7)
                    if panel == 1:
                        ax.text(xpos, value + 2.0, f"{value:.0f}", ha="center", va="bottom", fontsize=6.7, rotation=90)

        axes[0].set_title("(a) Final mean MSE", pad=13)
        axes[0].set_ylabel("Mean squared Euclidean error")
        axes[0].axhline(tolerance, color="#30363d", linestyle="--", linewidth=1.3)
        axes[0].text(0.98, tolerance * 1.12, "MSE threshold", transform=axes[0].get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=9, color="#444444")
        axes[1].set_title("(b) Threshold-satisfied curves", pad=13)
        axes[1].set_ylabel("Pass rate")
        axes[2].set_title("(c) Complete algorithm time", pad=13)
        axes[2].set_ylabel("Mean time per curve (ms)")

        fig.text(0.065, 0.163, "Timing bars include each method's full deployment/search and final refit; all methods start from the same normalized input points.", fontsize=10.3)
        fig.text(0.065, 0.126, _network_caption(index, datasets), fontsize=10.3, color="#137c6b")
        hardware = metadata.get("hardware", {})
        device = hardware.get("gpu") if hardware.get("device") == "cuda" else "CPU"
        device = device or hardware.get("device", "unspecified")
        fig.text(0.065, 0.089, f"Ours network: {device}; numerical methods and common refit: CPU float64. Numerical threads: {hardware.get('threads', 'unspecified')}.", fontsize=10.1, color="#555555")
        failures = sum(int(index[(name, method)].get("failed", 0)) for name in datasets for method in methods if (name, method) in index)
        detail = (
            "Reference points are evaluated after fitting the resampled input; they are not an independent ground-truth spline."
            if reference else "Bars show measured means; pass rates count failed runs as failures. No confidence intervals are inferred from these samples."
        )
        if failures:
            detail += f" Failed runs: {failures}; MSE means exclude failures; timing includes failed attempts."
        fig.text(0.065, 0.052, detail, fontsize=9.5, color="#555555")
        if any(method.endswith("_adaptation") for method in methods):
            fig.text(
                0.065,
                0.018,
                "Published-method bars are disclosed repository adaptations under one shared MSE/refit protocol; they do not claim bit-exact reproduction of the authors' software.",
                fontsize=9.3,
                color="#765097",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / ("comparison_reference_metrics.png" if reference else "comparison_three_metrics.png")
        fig.savefig(target, dpi=dpi)
        plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="comparison.json from benchmark_v15_datasets.py or benchmark_v16_datasets.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the input report's directory")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--reference", action=argparse.BooleanOptionalAction, default=True,
                        help="Also render original-reference metrics when available")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    report = read_report(args.input)
    output_dir = args.output_dir or args.input.parent
    for reference in (False, True) if args.reference else (False,):
        result = render_report(report, output_dir, dpi=args.dpi, reference=reference)
        if result is not None:
            print(result)


if __name__ == "__main__":
    main()
