"""Render publication-ready v16-versus-published-method comparison PNGs.

The script is intentionally a pure report renderer: it never runs a method and
never substitutes, rescales, clips, or fabricates benchmark measurements.  Its
input must be the ``comparison.json`` written by ``benchmark_v16_datasets.py``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter
import numpy as np

from plot_v15_dataset_benchmark import (
    COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    METHODS,
    _diagnostic_label,
    _network_caption,
    _num_points,
    model_version,
    published_method_labels,
    published_protocol_caption,
    read_report,
)


PUBLISHED_METHODS = (
    "ours",
    "park_dominant_point_2007_adaptation",
    "liang_feature_iki_2017_adaptation",
    "dung_direct_knot_2017_adaptation",
    "kang_sparse_2015_adaptation",
    "luo_linf_de_2022_adaptation",
)
SHORT_LABELS = {
    "ours": "Ours v16",
    "park_dominant_point_2007_adaptation": "Park & Lee (2007)",
    "liang_feature_iki_2017_adaptation": "Liang et al. (2017)",
    "dung_direct_knot_2017_adaptation": "Dung & Tjahjowidodo (2017)",
    "kang_sparse_2015_adaptation": "Kang et al. (2015)",
    "luo_linf_de_2022_adaptation": "Luo et al. (2022)",
    "yeh_feature_cdf_2020": "Yeh et al. (2020)",
    "uniform_gradient_pruning": "Uniform greedy + gradient",
}
HATCHES = ("", "//", "\\\\", "xx", "..", "--", "++", "oo")


def _metadata_fingerprint(metadata: dict) -> str:
    payload = copy.deepcopy(metadata)
    payload.pop("fingerprint", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_v16_benchmark(
    report: dict,
    *,
    methods: tuple[str, ...] = PUBLISHED_METHODS,
    allow_unqualified_diagnostic: bool = False,
) -> None:
    """Reject stale, partial, relabelled, or hand-assembled comparison reports."""
    metadata = report["metadata"]
    if model_version(metadata) != "v16":
        raise ValueError("The publication plot requires a native v16 benchmark report")
    if metadata.get("diagnostic_not_final") and not allow_unqualified_diagnostic:
        raise ValueError(
            "The report uses an unqualified checkpoint or diagnostic protocol; pass "
            "--allow-unqualified-diagnostic only for visibly watermarked debugging"
        )
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("The benchmark metadata has no valid experiment fingerprint")
    if fingerprint != _metadata_fingerprint(metadata):
        raise ValueError("The benchmark experiment fingerprint does not match its metadata")

    measurements = report.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError("A formal benchmark report must contain measured per-curve records")
    identities = [
        (row.get("dataset"), row.get("sample_id"), row.get("method"))
        for row in measurements
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("The benchmark contains duplicate per-curve measurements")
    measured_counts = Counter((dataset, method) for dataset, _, method in identities)
    measured_groups = defaultdict(list)
    for row in measurements:
        measured_groups[(row["dataset"], row["method"])].append(row)

    summary = report["summary"]
    summary_index = {(row["dataset"], row["method"]): row for row in summary}
    if len(summary_index) != len(summary):
        raise ValueError("The benchmark contains duplicate summary rows")
    declared = {row["dataset"]: row for row in metadata.get("datasets", [])}
    datasets = sorted(
        {row["dataset"] for row in summary}
        | {row["dataset"] for row in measurements}
        | set(declared)
    )
    missing = [
        (dataset, method)
        for dataset in datasets
        for method in methods
        if (dataset, method) not in summary_index
    ]
    if missing:
        raise ValueError(f"The benchmark is incomplete; missing summary rows: {missing}")
    for dataset in datasets:
        paired_ids = None
        for method in methods:
            row = summary_index[(dataset, method)]
            count = measured_counts[(dataset, method)]
            if count == 0 or int(row["n"]) != count:
                raise ValueError(
                    f"Summary count does not match measurements for {(dataset, method)}"
                )
            expected_count = declared.get(dataset, {}).get("selected_count")
            if expected_count is not None and count != expected_count:
                raise ValueError(f"Incomplete selected cases for {(dataset, method)}")
            values = measured_groups[(dataset, method)]
            sample_ids = {value["sample_id"] for value in values}
            if paired_ids is not None and sample_ids != paired_ids:
                raise ValueError(f"Unpaired sample identities for {(dataset, method)}")
            paired_ids = sample_ids
            valid = [value for value in values if value["status"] == "ok"]
            if row.get("failed") != count - len(valid):
                raise ValueError(f"Failed count does not match measurements for {(dataset, method)}")
            # Audit the means and pass-rate denominator against source records;
            # a renderer must never silently drop failures or fabricate bars.
            if all("fit_pass" in value for value in values):
                expected_pass = sum(bool(value["fit_pass"]) for value in valid) / count
                if not math.isclose(row["fit_pass_rate"], expected_pass, abs_tol=1e-12):
                    raise ValueError(f"fit_pass_rate excludes failures or differs from measurements for {(dataset, method)}")
            if all("reference_pass" in value for value in values):
                has_reference = any(value.get("has_reference", value.get("reference_mse") is not None) for value in values)
                expected_reference_pass = (
                    sum(bool(value["reference_pass"]) for value in valid) / count
                    if has_reference else None
                )
                actual_reference_pass = row.get("reference_pass_rate")
                if (actual_reference_pass is None) != (expected_reference_pass is None) or (
                    actual_reference_pass is not None
                    and not math.isclose(actual_reference_pass, expected_reference_pass, abs_tol=1e-12)
                ):
                    raise ValueError(f"reference_pass_rate excludes failures or differs from measurements for {(dataset, method)}")
            for field, summary_key, population in (
                ("mse", "mse_mean", valid),
                ("reference_mse", "reference_mse_mean", valid),
                ("final_k", "final_k_mean", valid),
                ("total_ms", "total_ms_mean", values),
                ("canonical_k", "canonical_k_mean", values),
            ):
                # Early native reports may omit optional record fields. When
                # present, nullable values remain missing rather than zeros.
                if not all(field in value for value in values):
                    continue
                observed = [value[field] for value in population if value[field] is not None]
                expected_value = statistics.fmean(observed) if observed else None
                actual_value = row.get(summary_key)
                if (actual_value is None) != (expected_value is None) or (
                    actual_value is not None
                    and not math.isclose(actual_value, expected_value, rel_tol=1e-10, abs_tol=1e-12)
                ):
                    raise ValueError(f"{summary_key} differs from measurements for {(dataset, method)}")
            final_k = row.get("final_k_mean")
            if final_k is not None and (
                not isinstance(final_k, (int, float))
                or not math.isfinite(final_k)
                or final_k < 0
            ):
                raise ValueError(f"Invalid final_k_mean for {(dataset, method)}: {final_k}")


def _ordered_datasets(rows: list[dict], *, reference: bool) -> list[str]:
    names = {
        row["dataset"]
        for row in rows
        if not reference or row.get("reference_pass_rate") is not None
    }
    return [name for name in DATASET_ORDER if name in names] + sorted(
        names.difference(DATASET_ORDER)
    )


def _synthetic_canonical_k(
    index: dict[tuple[str, str], dict],
    methods: tuple[str, ...],
) -> float | None:
    values = [
        float(index[("Synthetic", method)]["canonical_k_mean"])
        for method in methods
        if ("Synthetic", method) in index
        and index[("Synthetic", method)].get("canonical_k_mean") is not None
    ]
    if not values:
        return None
    if max(values) - min(values) > 1e-9:
        raise ValueError(
            "Synthetic canonical_k_mean is inconsistent across paired methods"
        )
    return values[0]


def _plot_value(
    ax,
    *,
    value: float | None,
    xpos: float,
    width: float,
    method: str,
    hatch: str,
    log_axis: bool,
    floor: float,
) -> None:
    if value is None:
        ax.text(
            xpos,
            0.025,
            "N/A",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.5,
            color="#777777",
        )
        return
    if value == 0 and log_axis:
        ax.scatter([xpos], [floor * 1.05], marker="v", s=27, color=COLORS[method])
        ax.annotate(
            "0",
            (xpos, floor * 1.05),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
        return
    ax.bar(
        xpos,
        value,
        width=width * 0.94,
        color=COLORS[method],
        edgecolor="white",
        linewidth=0.65,
        hatch=hatch,
    )


def render_comparison(
    report: dict,
    output_dir: Path,
    *,
    methods: tuple[str, ...] = PUBLISHED_METHODS,
    dpi: int = 300,
    reference: bool = False,
) -> Path | None:
    """Render measured MSE, pass rate, final K, and complete wall time."""
    metadata, rows = report["metadata"], report["summary"]
    labels = published_method_labels(metadata, SHORT_LABELS)
    datasets = _ordered_datasets(rows, reference=reference)
    if not datasets:
        return None
    index = {(row["dataset"], row["method"]): row for row in rows}
    canonical_k = _synthetic_canonical_k(index, methods)
    tolerance = float(metadata["mse_tolerance"])
    mse_key = "reference_mse_mean" if reference else "mse_mean"
    pass_key = "reference_pass_rate" if reference else "fit_pass_rate"
    metrics = (
        (mse_key, "(a) Mean squared Euclidean error", "MSE", True),
        (pass_key, "(b) Threshold-satisfied curves", "Pass rate", False),
        ("final_k_mean", "(c) Retained internal-knot count", "Mean retained internal K", False),
        ("total_ms_mean", "(d) Complete algorithm time", "Mean time per curve (ms)", True),
    )
    x = np.arange(len(datasets), dtype=float)
    width = 0.82 / len(methods)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(17.5, 11.3))
        fig.subplots_adjust(
            left=0.07,
            right=0.985,
            bottom=0.175,
            top=0.77,
            hspace=0.42,
            wspace=0.24,
        )
        kind = "original-reference points" if reference else "resampled input points"
        fig.suptitle(
            f"Ours v16 versus published-method adaptations | {kind}",
            y=0.975,
            fontsize=18,
        )
        fig.text(
            0.5,
            0.929,
            (
                f"Normalized Euclidean MSE (no square root); threshold = {tolerance:.2e}; "
                f"{_num_points(metadata)} input points per curve"
            ),
            ha="center",
            fontsize=11,
            color="#444444",
        )
        handles = [
            Patch(
                facecolor=COLORS[method],
                edgecolor="white",
                hatch=HATCHES[i],
                label=labels[method],
            )
            for i, method in enumerate(methods)
        ]
        if canonical_k is not None and "Synthetic" in datasets:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="#20252a",
                    marker="D",
                    markerfacecolor="white",
                    markersize=6.5,
                    linestyle="none",
                    label="Synthetic canonical reference K",
                )
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.895),
            ncol=3,
            frameon=False,
            fontsize=9.6,
        )
        if metadata.get("diagnostic_not_final"):
            fig.text(
                0.5,
                0.5,
                _diagnostic_label(metadata),
                ha="center",
                va="center",
                rotation=24,
                fontsize=28,
                color="crimson",
                alpha=0.20,
                weight="bold",
                zorder=100,
            )

        tick_labels = []
        for dataset in datasets:
            counts = sorted(int(index[(dataset, method)]["n"]) for method in methods)
            count = str(counts[0]) if counts[0] == counts[-1] else f"{counts[0]}–{counts[-1]}"
            tick_labels.append(f"{DATASET_LABELS.get(dataset, dataset)}\n$n={count}$")

        for panel, (ax, (key, title, ylabel, log_axis)) in enumerate(
            zip(axes.flat, metrics)
        ):
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#dfe3e8", linewidth=0.8)
            ax.set_xticks(x, tick_labels)
            ax.tick_params(axis="x", length=0, pad=8)
            ax.set_xlim(-0.58, len(datasets) - 0.42)
            values = [
                float(index[(dataset, method)][key])
                for dataset in datasets
                for method in methods
                if index[(dataset, method)].get(key) is not None
                and float(index[(dataset, method)][key]) > 0
            ]
            if key == mse_key:
                values.append(tolerance)
            if key == "final_k_mean" and canonical_k is not None and "Synthetic" in datasets:
                values.append(canonical_k)
            floor = min(values) / 2.0 if values else 1e-12
            if log_axis:
                ceiling = max(values) * 2.0 if values else 1.0
                ax.set_yscale("log")
                ax.set_ylim(floor, max(ceiling, floor * 10.0))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif key == pass_key:
                ax.set_ylim(0, 112)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            else:
                ceiling = max(values) if values else 1.0
                ax.set_ylim(0, ceiling * 1.18)

            for j, method in enumerate(methods):
                offsets = x + (j - (len(methods) - 1) / 2) * width
                for dataset, xpos in zip(datasets, offsets):
                    raw = index[(dataset, method)].get(key)
                    value = None if raw is None else float(raw)
                    if value is not None and key == pass_key:
                        value *= 100.0
                    _plot_value(
                        ax,
                        value=value,
                        xpos=float(xpos),
                        width=width,
                        method=method,
                        hatch=HATCHES[j],
                        log_axis=log_axis,
                        floor=floor,
                    )
            ax.set_title(title, pad=10)
            ax.set_ylabel(ylabel)

        axes[0, 0].axhline(
            tolerance,
            color="#30363d",
            linestyle="--",
            linewidth=1.25,
            label="MSE threshold",
        )
        axes[0, 0].text(
            0.985,
            tolerance * 1.08,
            "MSE threshold",
            transform=axes[0, 0].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#444444",
        )
        if canonical_k is not None and "Synthetic" in datasets:
            synthetic_x = float(datasets.index("Synthetic"))
            axes[1, 0].hlines(
                canonical_k,
                synthetic_x - 0.43,
                synthetic_x + 0.43,
                colors="#20252a",
                linestyles=(0, (3, 2)),
                linewidth=1.05,
                zorder=7,
            )
            axes[1, 0].scatter(
                [synthetic_x],
                [canonical_k],
                marker="D",
                s=55,
                facecolor="white",
                edgecolor="#20252a",
                linewidth=1.25,
                zorder=8,
            )
            axes[1, 0].annotate(
                f"canonical {canonical_k:.2f}",
                (synthetic_x, canonical_k),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.1,
                color="#20252a",
            )

        fig.text(
            0.07,
            0.118,
            (
                "Measured paired benchmark only: timing includes each complete method and the common "
                "endpoint-constrained standard B-spline refit."
            ),
            fontsize=9.6,
        )
        fig.text(
            0.07,
            0.086,
            _network_caption(index, datasets),
            fontsize=9.6,
            color="#137c6b",
        )
        failures = sum(
            int(index[(dataset, method)].get("failed", 0))
            for dataset in datasets
            for method in methods
        )
        detail = (
            published_protocol_caption(metadata)
        )
        if failures:
            detail += f" Failed runs: {failures}; pass rates include them as failures."
        fig.text(0.07, 0.060, detail, fontsize=9.1, color="#765097", wrap=True)
        fig.text(
            0.07, 0.041,
            "MSE / K means include finite threshold misses; time includes failed attempts. Only synthetic has canonical ground-truth K. N/A: no finite fit.",
            fontsize=9.1, color="#555555",
        )
        checkpoint = Path(str(metadata.get("checkpoint", "unknown"))).name
        fig.text(
            0.07,
            0.019,
            (
                f"Source: comparison.json | checkpoint: {checkpoint} | "
                f"experiment fingerprint: {metadata['fingerprint'][:12]}…"
            ),
            fontsize=8.6,
            color="#666666",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "reference" if reference else "input"
        target = output_dir / f"v16_published_methods_{suffix}.png"
        fig.savefig(target, dpi=dpi)
        plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="comparison.json written by scripts/benchmark_v16_datasets.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the input report directory",
    )
    parser.add_argument(
        "--method-set",
        choices=("published", "all"),
        default="published",
        help="'published' draws Ours, Park, Liang, Dung, Kang, and Luo",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also draw metrics evaluated on original real-world reference points",
    )
    parser.add_argument(
        "--allow-unqualified-diagnostic",
        action="store_true",
        help="Permit an unqualified report only with a visible diagnostic watermark",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    report = read_report(args.input)
    methods = PUBLISHED_METHODS if args.method_set == "published" else METHODS
    try:
        validate_v16_benchmark(
            report,
            methods=methods,
            allow_unqualified_diagnostic=args.allow_unqualified_diagnostic,
        )
    except ValueError as error:
        parser.error(str(error))
    output_dir = args.output_dir or args.input.parent
    modes = (False, True) if args.reference else (False,)
    for reference in modes:
        result = render_comparison(
            report,
            output_dir,
            methods=methods,
            dpi=args.dpi,
            reference=reference,
        )
        if result is not None:
            print(result)


if __name__ == "__main__":
    main()
