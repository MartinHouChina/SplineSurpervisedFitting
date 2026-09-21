"""Render paper PNGs exclusively from hash-verified saved benchmark results.

This entry point never loads a checkpoint or reruns any fitting method. Legacy
reports without full measured geometry cannot be reconstructed by this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import struct
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_geometry import case_key, file_sha256, load_geometry_artifact  # noqa: E402
from plot_v16_method_comparison import validate_v16_benchmark  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--benchmark-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-cases-per-dataset", type=int, default=6,
                        help="Independent figure cap; 0 plots all already evaluated cases; never changes benchmark coverage")
    result.add_argument("--selection-seed", type=int, default=20260908)
    result.add_argument("--dpi", type=int, default=180)
    return result


def _number(value, spec=".3e"):
    return "N/A" if value is None else format(value, spec)


def _save_png(figure, path, dpi):
    figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or min(struct.unpack(">II", header[16:24])) < 100:
        raise RuntimeError(f"Invalid or unexpectedly small PNG: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _protocol_text(metadata):
    protocol = metadata.get("published_baseline_protocol", {})
    if protocol.get("baseline_protocol") == "native":
        return "Strict native protocol: unverified original implementations remain UNAVAILABLE / N/A"
    if protocol.get("feasibility_safeguard_enabled"):
        return "Repository adaptations; disclosed common-MSE safeguard enabled (not original-paper reproductions)"
    return "Repository adaptations; no common-MSE safeguard (not verified original-paper reproductions)"


def _plot_case(case, source_arrays, results, metadata, path, dpi):
    methods = metadata["methods"]
    columns = min(3, len(methods))
    rows = (len(methods) + columns - 1) // columns
    dimension = source_arrays["input_points_normalized"].shape[-1]
    figure = plt.figure(figsize=(5.1 * columns, 4.4 * rows + 1.2))
    points = source_arrays["input_points_normalized"]
    reference = source_arrays.get("reference_points_normalized", points)
    labels = metadata.get("method_labels", {})
    for index, method in enumerate(methods):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d" if dimension == 3 else None)

        def line(values, **kwargs):
            axis.plot(*[values[:, i] for i in range(dimension)], **kwargs)

        def scatter(values, **kwargs):
            axis.scatter(*[values[:, i] for i in range(dimension)], **kwargs)

        line(reference, color="0.72", linewidth=1.1, label="Original reference")
        scatter(points, color="0.25", s=6, alpha=.65, label="Model input")
        row, arrays = results.get(method, ({"status": "not_evaluated"}, {}))
        label = labels.get(method, method)
        if row["status"] == "unavailable":
            label = label.split(" (")[0].removesuffix(" [native unavailable]") + " [native unavailable]"
        if row["status"] == "ok":
            line(arrays["dense_curve_normalized"], color="#2166ac", linewidth=1.8, label="Measured B-spline")
            line(arrays["control_points_normalized"], color="#e6ab02", linewidth=.8, linestyle="--", label="Control polygon")
            scatter(arrays["control_points_normalized"], color="#e6ab02", s=13, marker="s")
            if len(arrays["knot_positions_normalized"]):
                scatter(arrays["knot_positions_normalized"], color="#b2182b", s=22, marker="x", label="Internal knot positions")
            verdict = "PASS" if row.get("fit_pass") else "MISS"
            title = (f"{label}\nK={row['final_k']}  {verdict}  MSE={_number(row.get('mse'))}"
                     f"\nmax SE={_number(row.get('max_squared_error'))}  complete={_number(row.get('total_ms'), '.2f')} ms")
            if method == "ours":
                title += f"  network={_number(row.get('network_ms'), '.2f')} ms"
            if row.get("reference_mse") is not None:
                title += f"\nref MSE={_number(row['reference_mse'])}  ref max SE={_number(row.get('reference_max_squared_error'))}"
        else:
            title = f"{label}\n{row['status'].upper()} — metrics/time N/A"
            reason = ("Original-paper implementation not verified/available.\nNo adaptation substituted; no method executed."
                      if row["status"] == "unavailable" else textwrap.fill(str(row.get("error", "No recorded evaluation"))[:160], 50))
            axis.text2D(.02, .02, reason, transform=axis.transAxes, fontsize=6.5) if dimension == 3 else axis.text(
                .02, .02, reason, transform=axis.transAxes, fontsize=6.5)
        axis.set_title(title, fontsize=8)
        if dimension == 2:
            axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("Normalized x", fontsize=8)
        axis.set_ylabel("Normalized y", fontsize=8)
        axis.grid(alpha=.15)
        axis.tick_params(labelsize=7)
        axis.legend(fontsize=6, loc="best")
    figure.suptitle(
        f"{case.get('dataset_label') or case['dataset']} | {case['sample_id']}\n"
        "Corresponding-point squared errors; normalized coordinates",
        fontsize=10,
    )
    figure.text(.02, .012, _protocol_text(metadata), fontsize=7, color="0.4")
    figure.tight_layout(rect=(0, .04, 1, .92))
    return _save_png(figure, path, dpi)


def _plot_dataset_summary(dataset, rows, metadata, path, dpi):
    """No success-only imputation: N/A stays N/A and every denominator is shown."""
    methods = metadata["methods"]
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    fields = (("final_k", "Mean internal knots (successful fits)"),
              ("mse", "Mean input squared error (successful fits)"),
              ("max_squared_error", "Mean per-curve peak squared error (successful fits)"),
              ("total_ms", "Mean complete time, ms (successful fits)"))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    tick_labels = []
    for method in methods:
        values = by_method[method]
        passed = sum(row.get("status") == "ok" and bool(row.get("fit_pass")) for row in values)
        unavailable = sum(row.get("status") == "unavailable" for row in values)
        evaluated = len(values) - unavailable
        title = metadata.get("method_labels", {}).get(method, method).replace(" (", "\n(")
        pass_text = f"{passed}/{evaluated}" if evaluated else "N/A"
        tick_labels.append(f"{title}\npass {pass_text}" +
                           (f"; unavailable {unavailable}" if unavailable else ""))
    for axis, (field, title) in zip(axes.flat, fields):
        for index, method in enumerate(methods):
            valid = [row for row in by_method[method] if row["status"] == "ok"]
            values = [row[field] for row in valid if row.get(field) is not None]
            if values and len(values) == len(valid):
                mean = float(np.mean(values))
                axis.bar(index, mean, color="#2166ac" if method == "ours" else "#8da0cb")
                axis.annotate(_number(mean), (index, mean), ha="center", xytext=(0, 3), textcoords="offset points", fontsize=7)
            else:
                axis.text(index, .04, "N/A", transform=axis.get_xaxis_transform(), ha="center", fontsize=9)
        axis.set_xticks(range(len(methods)), tick_labels, rotation=20, ha="right", fontsize=6)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=.2)
    label = next((row.get("dataset_label") for row in rows if row.get("dataset_label")), dataset)
    figure.suptitle(label, fontsize=12)
    figure.text(.02, .012, _protocol_text(metadata), fontsize=7, color="0.4")
    figure.tight_layout(rect=(0, .04, 1, .95))
    return _save_png(figure, path, dpi)


def run(args):
    if args.max_cases_per_dataset < 0 or args.dpi < 30:
        raise ValueError("Figure cap must be non-negative and DPI at least 30")
    source = args.benchmark_dir / "comparison.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    metadata, measurements = report["metadata"], report["measurements"]
    if not measurements or any(not row.get("geometry_artifact") for row in measurements):
        raise ValueError("Saved measured geometry is missing; cannot reconstruct legacy fits without rerunning, which this plotter forbids")
    # Recompute the metadata fingerprint too, so edited labels or protocols
    # cannot silently relabel historical fits by retaining an old hash string.
    validate_v16_benchmark(report, methods=tuple(metadata["methods"]),
                           allow_unqualified_diagnostic=True)
    grouped = defaultdict(dict)
    dataset_rows = defaultdict(list)
    for row in measurements:
        key = (row["dataset"], row["sample_id"])
        if row["method"] in grouped[key]:
            raise ValueError(f"Duplicate measured method for {key}")
        grouped[key][row["method"]] = row
        dataset_rows[row["dataset"]].append(row)
    selected = []
    for dataset in sorted(dataset_rows):
        keys = sorted(key for key in grouped if key[0] == dataset)
        random.Random(args.selection_seed).shuffle(keys)
        selected.extend(keys[:args.max_cases_per_dataset] if args.max_cases_per_dataset else keys)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for key in selected:
        results = {}
        source_case, source_arrays = None, None
        for method, row in grouped[key].items():
            document, arrays = load_geometry_artifact(args.benchmark_dir, row["geometry_artifact"])
            if document["fingerprint"] != metadata["fingerprint"] or document["measurement"] != {
                name: value for name, value in row.items() if name != "geometry_artifact"
            }:
                raise ValueError(f"Geometry does not match measured row: {key}, {method}")
            case, case_arrays = load_geometry_artifact(args.benchmark_dir, document["case_artifact"])
            if (case["dataset"], case["sample_id"]) != key or case["fingerprint"] != metadata["fingerprint"]:
                raise ValueError("Case geometry identity/fingerprint mismatch")
            if source_case is not None and case != source_case:
                raise ValueError("Paired methods do not reference the same input geometry")
            source_case, source_arrays = case, case_arrays
            results[method] = (row, arrays)
        image = _plot_case(source_case, source_arrays, results, metadata,
                           args.output_dir / f"case_{case_key(source_case)}.png", args.dpi)
        figures.append({"dataset": key[0], "sample_id": key[1], "image": image})
    summaries = []
    for dataset, values in sorted(dataset_rows.items()):
        slug = case_key({"dataset": dataset, "sample_id": "summary"})[:20]
        summaries.append({"dataset": dataset, "image": _plot_dataset_summary(
            dataset, values, metadata, args.output_dir / f"summary_{slug}.png", args.dpi)})
    manifest = {
        "source_comparison": str(source.resolve()), "source_comparison_sha256": file_sha256(source),
        "benchmark_fingerprint": metadata["fingerprint"], "reran_methods": False,
        "diagnostic_not_final": metadata.get("diagnostic_not_final", True),
        "diagnostic_reasons": metadata.get("diagnostic_reasons", []),
        "checkpoint_qualification": metadata.get("checkpoint_qualification"),
        "published_baseline_protocol": metadata.get("published_baseline_protocol", {}),
        "watermark_rendered": False,
        "plot_selection": {"seed": args.selection_seed, "max_cases_per_dataset": args.max_cases_per_dataset,
                           "rule": "seeded random sample without ranking by error, success, knots or runtime"},
        "requested_measurements": len(measurements),
        "evaluated_measurements": sum(row["status"] != "unavailable" for row in measurements),
        "unavailable_measurements": sum(row["status"] == "unavailable" for row in measurements),
        "evaluated_cases": len(grouped),
        "case_figures": figures, "summary_figures": summaries,
    }
    (args.output_dir / "figures.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main(argv=None):
    args = parser().parse_args(argv)
    result = run(args)
    print(f"Saved {len(result['case_figures'])} case PNGs and {len(result['summary_figures'])} summary PNGs; no fitting reruns.")
    return result


if __name__ == "__main__":
    main()
