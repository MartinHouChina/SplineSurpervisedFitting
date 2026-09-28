"""Contact sheets and audited tables from saved, measured benchmark geometry.

This script does not load a checkpoint, solve a spline, or rerun a baseline.
The comparison sheet samples cases without looking at scores.  The Ours sheet
is explicitly a selected-example showcase, never the source of aggregate data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_geometry import file_sha256, load_geometry_artifact  # noqa: E402
from plot_v16_method_comparison import validate_v16_benchmark  # noqa: E402

SOURCES = ("Synthetic", "UJI", "NaturalEarth", "USGS", "IndustrialOffset")
SOURCE_LABELS = {"Synthetic": "Synthetic", "UJI": "UJI handwriting",
                 "NaturalEarth": "Natural Earth coastline", "USGS": "USGS contours",
                 "IndustrialOffset": "Industrial offsets (procedural)"}
COMPARISON_RULE = "one seeded random measured case per source, independent of errors, pass, count and runtime"
SHOWCASE_RULE = (
    "finite Ours fits; prefer both input/reference MSE pass, then ascending "
    "max(input MSE, reference MSE)/tolerance, then reference peak squared error; "
    "within each pass/fail tier choose distinct groups first, then fill that tier before the next; "
    "selected examples are not used to calculate aggregate statistics"
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--benchmark-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20260922)
    result.add_argument("--dpi", type=int, default=180)
    result.add_argument("--allow-incomplete", action="store_true",
                        help="Smoke tests only: keep missing-source/insufficient-case panels explicitly empty")
    return result


def _finite(value):
    return value is not None and isinstance(value, (float, int)) and math.isfinite(value)


def _number(value, spec=".2e"):
    return "--" if not _finite(value) else format(value, spec)


def _label(method, metadata):
    label = metadata.get("method_labels", {}).get(method, method)
    return "Ours" if method == "ours" else label.split(" (")[0].removesuffix(" [native unavailable]")


def _source_rng(seed, source):
    # Stable across Python hash randomization and across M16/M32 reports.
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big"))


def choose_comparison_cases(grouped, seed):
    selected = {}
    for source in SOURCES:
        keys = sorted((key for key in grouped if key[0] == source), key=lambda key: str(key[1]))
        if keys:
            selected[source] = _source_rng(seed, source).choice(keys)
    return selected


def choose_showcase_cases(grouped, tolerance, count=5):
    selected = {}
    for source in SOURCES:
        ranked = []
        for key, by_method in grouped.items():
            row = by_method.get("ours")
            if key[0] != source or not row or row.get("status") != "ok" or not _finite(row.get("mse")):
                continue
            reference = row.get("reference_mse")
            if row.get("has_reference") and not _finite(reference):
                continue
            values = [row["mse"]] + ([reference] if _finite(reference) else [])
            both_pass = bool(row.get("fit_pass")) and (not row.get("has_reference") or bool(row.get("reference_pass")))
            peak = row.get("reference_max_squared_error", row.get("max_squared_error"))
            score = (not both_pass, max(values) / tolerance,
                     peak if _finite(peak) else math.inf, str(key[1]))
            ranked.append((score, key, row.get("group_id") or key[1]))
        ranked.sort(key=lambda item: item[0])
        chosen, groups = [], set()
        for failed_tier in (False, True):
            tier = [item for item in ranked if item[0][0] == failed_tier]
            for _, key, group in tier:
                if len(chosen) == count:
                    break
                if group not in groups:
                    chosen.append(key)
                    groups.add(group)
            for _, key, _ in tier:
                if len(chosen) == count:
                    break
                if key not in chosen:
                    chosen.append(key)
        selected[source] = chosen[:count]
    return selected


def _load_all_geometry(directory, report):
    metadata = report["metadata"]
    validate_v16_benchmark(report, methods=tuple(metadata["methods"]), allow_unqualified_diagnostic=True)
    grouped, cases, geometry = defaultdict(dict), {}, {}
    for row in report["measurements"]:
        key = (row["dataset"], row["sample_id"])
        method = row["method"]
        if method in grouped[key]:
            raise ValueError(f"Duplicate measurement: {key}, {method}")
        if not row.get("geometry_artifact"):
            raise ValueError(f"Measured geometry unavailable: {key}, {method}")
        document, arrays = load_geometry_artifact(directory, row["geometry_artifact"])
        if document["fingerprint"] != metadata["fingerprint"] or document["measurement"] != {
            name: value for name, value in row.items() if name != "geometry_artifact"
        }:
            raise ValueError(f"Geometry/measurement mismatch: {key}, {method}")
        case, source_arrays = load_geometry_artifact(directory, document["case_artifact"])
        if (case["dataset"], case["sample_id"]) != key or case["fingerprint"] != metadata["fingerprint"]:
            raise ValueError(f"Source identity/fingerprint mismatch: {key}")
        if key in cases and cases[key][0] != case:
            raise ValueError(f"Paired methods have different source geometry: {key}")
        grouped[key][method] = row
        cases[key] = (case, source_arrays)
        geometry[(key, method)] = arrays
    return grouped, cases, geometry


def _line(axis, values, dimension, **kwargs):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[-1] != dimension:
        return
    axis.plot(*[values[:, i] for i in range(dimension)], **kwargs)


def _scatter(axis, values, dimension, **kwargs):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[-1] != dimension or not len(values):
        return
    axis.scatter(*[values[:, i] for i in range(dimension)], **kwargs)


def _panel(figure, spec, case, source, row, arrays, title, tolerance):
    dimension = source["input_points_normalized"].shape[-1]
    inner = spec.subgridspec(2, 1, height_ratios=(13, 1), hspace=.04)
    axis = figure.add_subplot(inner[0], projection="3d" if dimension == 3 else None)
    reference = source.get("reference_points_normalized", source["input_points_normalized"])
    _line(axis, reference, dimension, color="0.18", linewidth=1.4, zorder=3)
    _scatter(axis, source["input_points_normalized"], dimension, color="0.45", s=4, alpha=.55, zorder=4)
    # Keep every finite part of the actual output, including failed methods.
    _line(axis, arrays.get("dense_curve_normalized", []), dimension, color="#087f8c", linewidth=1.6, zorder=5)
    controls = arrays.get("control_points_normalized", [])
    _line(axis, controls, dimension, color="#d79412", linewidth=.75, linestyle="--", zorder=1)
    _scatter(axis, controls, dimension, color="#d79412", s=10, marker="s", zorder=2)
    _scatter(axis, arrays.get("knot_positions_normalized", []), dimension, color="#c53b53", s=18, marker="x", zorder=6)
    status = "PASS" if row.get("fit_pass") and row.get("status") == "ok" else "MISS"
    if row.get("status") != "ok":
        status = str(row.get("status", "missing")).upper()
    axis.set_title(title + f"\nK={row.get('final_k', '--')}  {status}  MSE={_number(row.get('mse'))}"
                   f"  MaxSE={_number(row.get('max_squared_error'))}"
                   f"\nTotal={_number(row.get('total_ms'), '.1f')} ms"
                   + (f"  Net={_number(row.get('network_ms'), '.1f')} ms" if row.get("method") == "ours" else "")
                   + (f"\nRef MSE={_number(row.get('reference_mse'))}  Ref MaxSE={_number(row.get('reference_max_squared_error'))}"
                      if row.get("has_reference") else ""), fontsize=7.5, pad=5)
    axis.tick_params(labelsize=6, pad=1)
    axis.grid(alpha=.15, linewidth=.6)
    axis.margins(.08)
    if dimension == 2:
        axis.set_aspect("equal", adjustable="datalim")
    else:
        all_values = [np.asarray(value) for value in (reference, controls, arrays.get("dense_curve_normalized", []))
                      if np.asarray(value).ndim == 2]
        finite = np.concatenate(all_values)
        finite = finite[np.isfinite(finite).all(axis=1)]
        if len(finite):
            extent = np.ptp(finite, axis=0)
            axis.set_box_aspect(np.maximum(extent, max(float(extent.max()), 1e-9) * .02))
    strip = figure.add_subplot(inner[1])
    knots = np.asarray(arrays.get("internal_knots", [])).reshape(-1)
    strip.axhline(0, color="0.6", linewidth=.7)
    strip.scatter(knots, np.zeros_like(knots), marker="|", s=45, color="#c53b53", linewidths=1)
    strip.set_xlim(-.015, 1.015)
    strip.set_ylim(-1, 1)
    strip.set_xticks([0, .5, 1], ["0", ".5", "1"])
    strip.set_yticks([])
    strip.tick_params(axis="x", labelsize=6, length=2, pad=1)
    for spine in strip.spines.values():
        spine.set_visible(False)
    strip.set_ylabel("u", fontsize=7, rotation=0, labelpad=3)


def _save(figure, path, dpi):
    figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _legend(figure):
    handles = [Line2D([], [], color="0.18", label="Reference curve"),
               Line2D([], [], color="0.45", marker=".", linestyle="none", label="Input points"),
               Line2D([], [], color="#087f8c", label="Fitted curve"),
               Line2D([], [], color="#d79412", marker="s", linestyle="--", label="Control vertices / polygon"),
               Line2D([], [], color="#c53b53", marker="x", linestyle="none", label="Internal knots on curve / parameter strip")]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(.5, .023))


def _timing_note(metadata, n):
    config = metadata.get("configuration", {})
    repeats = config.get("end_to_end_repeats", "unspecified")
    hardware = metadata.get("hardware", {})
    device = hardware.get("gpu") or hardware.get("device") or "unspecified device"
    timing = "single complete run per curve" if repeats == 1 else f"{repeats} complete repeats per curve"
    return f"{n} measured curves; {device}; {timing}. Total includes full method execution; Net is separately timed network only."


def _sheet(kind, selected, grouped, cases, geometry, metadata, path, dpi):
    showcase = kind == "showcase"
    columns = 5 if showcase else len(metadata["methods"])
    figure = plt.figure(figsize=(4.05 * columns, 19.8))
    grid = figure.add_gridspec(5, columns, left=.065, right=.992, bottom=.075, top=.94, hspace=.45, wspace=.23)
    displayed = []
    for source_index, source_name in enumerate(SOURCES):
        keys = selected.get(source_name, []) if showcase else [selected.get(source_name)] * columns
        figure.text(.008, .863 - source_index * .181, SOURCE_LABELS[source_name].replace(" ", "\n", 1),
                    rotation=90, va="center", ha="left", fontsize=10, fontweight="bold")
        for column in range(columns):
            key = keys[column] if column < len(keys) else None
            method = "ours" if showcase else metadata["methods"][column]
            if key is None or method not in grouped[key]:
                axis = figure.add_subplot(grid[source_index, column])
                axis.axis("off")
                axis.text(.5, .5, "No additional measured case", ha="center", va="center", fontsize=9)
                continue
            case, arrays = cases[key]
            short = (str(key[1]) if len(str(key[1])) <= 38 else str(key[1])[:18] + "..." + str(key[1])[-17:])
            title = short if showcase else _label(method, metadata)
            _panel(figure, grid[source_index, column], case, arrays, grouped[key][method],
                   geometry[(key, method)], title, metadata["mse_tolerance"])
            displayed.append({"row": source_index, "column": column, "dataset": key[0], "sample_id": key[1],
                              "group_id": case.get("group_id"), "method": method,
                              "geometry_artifact": grouped[key][method]["geometry_artifact"]})
    capacity = metadata.get("knot_capacities", {}).get("network_candidates", "?")
    title = (f"Ours M{capacity}: selected fitting examples (not aggregate results)" if showcase
             else f"M{capacity}: paired six-method comparison across five curve sources")
    subtitle = ("Five selected finite examples per source; input/reference fit quality ranked with group diversity"
                if showcase else "One seeded random test curve per source; all six methods share the same curve")
    figure.suptitle(title + "\n" + subtitle, y=.985, fontsize=16)
    _legend(figure)
    figure.text(.065, .01, f"Normalized coordinates; MSE threshold={metadata['mse_tolerance']:.1e}; MaxSE = largest sampled squared Euclidean residual.\n"
                "Literature baselines are repository adaptations. All finite fitted/control geometry shown; each panel uses its own uncropped scale.",
                fontsize=8, color="0.3")
    return {"image": _save(figure, path, dpi), "panels": displayed}


def summarize(measurements, metadata):
    result = []
    for source in SOURCES:
        for method in metadata["methods"]:
            rows = [row for row in measurements if row["dataset"] == source and row["method"] == method]
            if not rows:
                continue
            executed = [row for row in rows if row.get("status") != "unavailable"]
            valid = [row for row in rows if row.get("status") == "ok" and _finite(row.get("mse"))]
            entry = {"dataset": source, "method": method, "label": _label(method, metadata), "n_requested": len(rows),
                     "n_executed": len(executed), "n_finite_fits": len(valid), "n_failed": len(executed) - len(valid),
                     "n_unavailable": len(rows) - len(executed),
                     "pass_percent": 100 * sum(bool(row.get("fit_pass")) for row in valid) / len(executed) if executed else None}
            for field in ("mse", "reference_mse", "final_k", "total_ms", "network_ms", "max_squared_error", "reference_max_squared_error"):
                values = [float(row[field]) for row in valid if _finite(row.get(field))]
                entry[field + "_valid_count"] = len(values)
                entry[field + "_mean"] = float(np.mean(values)) if values and len(values) == len(valid) else None
                entry[field + "_max"] = max(values) if values and len(values) == len(valid) else None
            result.append(entry)
    return result


def _write_tables(summary, metadata, source_path, directory, dpi):
    headers = ["Source", "Method", "Finite / run", "Pass %", "Mean K", "Mean MSE", "MaxSE (all)", "Total ms", "Net ms"]
    cells = []
    for row in summary:
        cells.append([row["dataset"], row["label"], f"{row['n_finite_fits']}/{row['n_executed']}",
                      _number(row["pass_percent"], ".1f"), _number(row["final_k_mean"], ".2f"),
                      _number(row["mse_mean"]), _number(row["max_squared_error_max"]),
                      _number(row["total_ms_mean"], ".2f"), _number(row["network_ms_mean"], ".2f")])
    measured_rows = json.loads(source_path.read_text(encoding="utf-8"))["measurements"]
    n_curves = len({(row["dataset"], row["sample_id"]) for row in measured_rows})
    capacity = metadata.get("knot_capacities", {}).get("network_candidates")
    stress_ids = sorted({str(row["sample_id"]) for row in measured_rows
                         if row["dataset"] == "Synthetic" and _finite(row.get("source_k"))
                         and _finite(capacity) and row["source_k"] > capacity})
    stress_note = (f"Synthetic includes {len(stress_ids)} source-K > M{capacity} capacity stress curves; "
                   "the all-case mean is not a capacity-matched score.") if stress_ids else ""
    notes = [
        "Four core metrics: mean MSE, input-MSE pass rate, mean retained internal knots, mean complete method time; plus maximum squared error.",
        "MaxSE (all) is max over all sampled input points of all finite completed curves, NOT maximum per-curve MSE or mean peak error.",
        "Means and maxima include every finite completed fit, including threshold failures. Numerical failures stay in executed pass denominators; counts are shown.",
        "Unavailable methods are not executed. A missing number is --, never a fabricated zero. Non-network methods have no Net measurement.",
        f"MSE threshold = {metadata['mse_tolerance']:.8g}; errors are normalized squared Euclidean residuals with no square root.",
        _timing_note(metadata, n_curves),
        "Literature methods are repository adaptations, not verified native reproductions. Synthetic and procedural offsets are not measured real-world data.",
        "Displayed showcase selection has no effect on these aggregate statistics. Full reference-error statistics are retained in JSON.",
    ]
    if metadata.get("diagnostic_not_final"):
        notes.append("The source benchmark is marked diagnostic_not_final; this is a local preview, not a final population benchmark.")
    if stress_note:
        notes.append(stress_note)
    document = {"source_comparison": str(source_path.resolve()), "source_comparison_sha256": file_sha256(source_path),
                "benchmark_fingerprint": metadata["fingerprint"], "all_measured_curves": n_curves,
                "synthetic_overcapacity_sample_ids": stress_ids, "definitions": notes, "rows": summary}
    (directory / "core_metrics_plus_max_error.json").write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    markdown = "# Four core metrics plus maximum squared error\n\n" + "\n\n".join(notes) + "\n\n"
    markdown += "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in cells) + "\n"
    (directory / "core_metrics_plus_max_error.md").write_text(markdown, encoding="utf-8")
    figure, axis = plt.subplots(figsize=(19, max(8, 2.8 + .34 * len(cells))))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=headers, cellLoc="center", loc="upper center",
                       colWidths=[.11, .23, .08, .07, .07, .09, .10, .08, .07])
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2e8")
        cell.set_linewidth(.5)
        if r == 0:
            cell.set_facecolor("#183b56")
            cell.set_text_props(color="white", fontweight="bold")
        elif summary[r - 1]["method"] == "ours":
            cell.set_facecolor("#e3f3f0")
            cell.set_text_props(fontweight="bold")
        elif ((r - 1) // len(metadata["methods"])) % 2:
            cell.set_facecolor("#f4f7f9")
    capacity = metadata.get("knot_capacities", {}).get("network_candidates", "?")
    figure.suptitle(f"M{capacity} | all measured cases: four core metrics + maximum squared error", fontsize=15, y=.985)
    figure.text(.04, .067, f"MSE threshold={metadata['mse_tolerance']:.1e}. Means/maxima: all finite fits, including misses; pass denominator: all executed cases.\n"
                "MaxSE: maximum pointwise squared residual over all finite curves. Total: complete method; Net: separate Ours network latency.\n"
                + _timing_note(metadata, n_curves) + "\nLiterature baselines: repository adaptations. Industrial offsets: procedurally generated."
                + ("\n" + stress_note if stress_note else ""), fontsize=9)
    figure.subplots_adjust(top=.93, bottom=.14, left=.025, right=.975)
    return _save(figure, directory / "core_metrics_plus_max_error.png", dpi)


def run(args):
    if args.dpi < 50:
        raise ValueError("DPI must be at least 50")
    source_path = args.benchmark_dir / "comparison.json"
    report = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    if len(metadata["methods"]) != 6 or "ours" not in metadata["methods"]:
        raise ValueError("Exactly six measured methods including Ours are required")
    grouped, cases, geometry = _load_all_geometry(args.benchmark_dir, report)
    comparison = choose_comparison_cases(grouped, args.seed)
    showcase = choose_showcase_cases(grouped, metadata["mse_tolerance"])
    incomplete = {source: len(showcase.get(source, [])) for source in SOURCES
                  if source not in comparison or len(showcase.get(source, [])) < 5}
    if incomplete and not args.allow_incomplete:
        raise ValueError(f"Five-source contact sheets require at least five finite Ours cases per source: {incomplete}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "six_methods": _sheet("comparison", comparison, grouped, cases, geometry, metadata,
                              args.output_dir / "six_methods_6x5.png", args.dpi),
        "ours_showcase": _sheet("showcase", showcase, grouped, cases, geometry, metadata,
                                args.output_dir / "ours_showcase_5x5.png", args.dpi),
        "metric_table": _write_tables(summarize(report["measurements"], metadata), metadata,
                                      source_path, args.output_dir, args.dpi),
    }
    manifest = {"source_comparison": str(source_path.resolve()), "source_comparison_sha256": file_sha256(source_path),
                "benchmark_fingerprint": metadata["fingerprint"], "checkpoint": metadata["checkpoint"],
                "checkpoint_sha256": metadata.get("checkpoint_sha256"), "seed": args.seed,
                "comparison_rule": COMPARISON_RULE, "showcase_rule": SHOWCASE_RULE,
                "watermark_rendered": False, "reran_fitting": False, "all_geometry_hash_verified": True,
                "diagnostic_not_final": metadata.get("diagnostic_not_final"),
                "diagnostic_reasons": metadata.get("diagnostic_reasons", []),
                "published_baseline_protocol": metadata.get("published_baseline_protocol", {}),
                "incomplete_sources": incomplete, "figures": figures}
    (args.output_dir / "panel_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return manifest


def main(argv=None):
    args = parser().parse_args(argv)
    result = run(args)
    print(f"Saved contact sheets and all-measurement tables to {args.output_dir}; no fitting reruns.")
    return result


if __name__ == "__main__":
    main()
