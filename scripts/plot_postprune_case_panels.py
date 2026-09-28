"""Clean paper panels from saved final fits; no solver or model is run.

The six-method sheet defaults to the same seeded case for all methods; explicit
illustrative-case overrides are recorded separately in the manifest. The Ours
sheet is a deliberately selected, dual-feasible-preferred showcase. Neither
selection is used for aggregate statistics. Numeric measurements and exact
saved geometry are exported separately, not annotated on either image.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_geometry import file_sha256
from plot_dual_error_comparison import (
    SOURCES, PAPER_LABELS, _geometry, _source_random, dual_pass, finite,
    identity, pair_measurements,
)

SOURCE_LABELS = {
    "Synthetic": "Synthetic",
    "UJI": "UJI handwriting",
    "NaturalEarth": "Natural Earth",
    "USGS": "USGS contours",
    "IndustrialOffset": "Industrial offsets",
}
METHODS = tuple(PAPER_LABELS)
COMPARISON_RULE = "one seeded random paired case per source, independent of all measured outcomes"
SHOWCASE_RULE = (
    "five finite Ours cases per source; dual-feasible cases first; within each "
    "feasibility tier prefer distinct group_id, then fill that tier; order by "
    "ascending max(MSE / MSE tolerance, MaxSE / MaxSE tolerance), MSE, sample ID; "
    "no showcase selection is used for aggregate statistics"
)
COLORS = dict(reference="#4d5660", points="#687582", fit="#00877a",
              controls="#c98720", knots="#b64066")


def select_cases(report, seed=20260922, comparison_overrides=None):
    paired = pair_measurements(report)
    if tuple(report["metadata"]["methods"]) != METHODS:
        raise ValueError("This 6 x 5 layout requires the declared six methods in canonical order")
    index = {identity(final): final for _, final in paired}
    if {row["dataset"] for row in index.values()} != set(SOURCES):
        raise ValueError("Both paper panels require all five sources")
    comparison, showcase = {}, {}
    for source in SOURCES:
        ids = {sample for dataset, sample, method in index if dataset == source}
        comparison[source] = _source_random(seed, source, ids)
        candidates = [row for row in index.values() if row["dataset"] == source
                      and row["method"] == "ours" and row.get("status") == "ok"
                      and finite(row.get("mse")) and finite(row.get("max_squared_error"))]
        if len(candidates) < 5:
            raise ValueError(f"Five finite Ours cases are required for {source}")
        metadata = report["metadata"]
        def rank(row):
            return (max(row["mse"] / metadata["mse_tolerance"],
                        row["max_squared_error"] / metadata["max_squared_error_tolerance"]),
                    row["mse"], str(row["sample_id"]))
        chosen, groups = [], set()
        for passed in (True, False):
            tier = sorted([row for row in candidates if dual_pass(row, metadata) == passed], key=rank)
            for row in tier:
                group = str(row.get("group_id") or row["sample_id"])
                if len(chosen) < 5 and group not in groups:
                    chosen.append(str(row["sample_id"]))
                    groups.add(group)
            for row in tier:
                sample = str(row["sample_id"])
                if len(chosen) < 5 and sample not in chosen:
                    chosen.append(sample)
        showcase[source] = chosen
    for source, sample in (comparison_overrides or {}).items():
        if source not in SOURCES:
            raise ValueError(f"Unknown comparison source: {source}")
        sample = str(sample)
        if any((source, sample, method) not in index for method in METHODS):
            raise ValueError(f"Explicit comparison sample must exist for all six methods: {source}={sample}")
        comparison[source] = sample
    return index, comparison, showcase


def _matrix(values, dimension):
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[1] != dimension:
        return np.empty((0, dimension))
    return result


def geometry_bounds(loaded):
    """Include every finite fitted/reference/data/control point, never crop fits."""
    dimension = loaded[0][1]["input_points_normalized"].shape[1]
    values = []
    for _, source, fitted in loaded:
        if source["input_points_normalized"].shape[1] != dimension:
            raise ValueError("Paired curves must have the same coordinate dimension")
        for value in (source["input_points_normalized"], source.get("reference_points_normalized", []),
                      fitted.get("dense_curve_normalized", []), fitted.get("control_points_normalized", []),
                      fitted.get("knot_positions_normalized", [])):
            matrix = _matrix(value, dimension)
            values.append(matrix[np.isfinite(matrix).all(axis=1)])
    joined = np.concatenate(values)
    if not len(joined):
        raise ValueError("No finite geometry to plot")
    low, high = joined.min(axis=0), joined.max(axis=0)
    span = np.maximum(high - low, max(float(np.ptp(joined, axis=0).max()), 1e-9) * .02)
    return low - .07 * span, high + .07 * span


def draw_panel(figure, spec, loaded, bounds):
    _, source, fitted = loaded
    points = source["input_points_normalized"]
    dimension = points.shape[1]
    if dimension not in (2, 3):
        raise ValueError("Panel geometry must be 2D or 3D")
    inner = spec.subgridspec(2, 1, height_ratios=(15, 1), hspace=.035)
    axis = figure.add_subplot(inner[0], projection="3d" if dimension == 3 else None)
    def line(values, **kwargs):
        matrix = _matrix(values, dimension)
        if len(matrix):
            axis.plot(*matrix.T, **kwargs)
    def scatter(values, **kwargs):
        matrix = _matrix(values, dimension)
        if len(matrix):
            axis.scatter(*matrix.T, **kwargs)
    line(source.get("reference_points_normalized", points), color=COLORS["reference"],
         linewidth=.85, linestyle="--", alpha=.8, zorder=2)
    scatter(points, color=COLORS["points"], s=4.3, alpha=.43, zorder=3, linewidths=0)
    line(fitted.get("dense_curve_normalized", []), color=COLORS["fit"], linewidth=1.55, zorder=4)
    controls = fitted.get("control_points_normalized", [])
    line(controls, color=COLORS["controls"], linewidth=.85, linestyle=(0, (4, 3)), alpha=.95, zorder=1)
    scatter(controls, edgecolors=COLORS["controls"], facecolors="white", marker="s", s=13.5,
            linewidths=.9, zorder=5)
    scatter(fitted.get("knot_positions_normalized", []), color=COLORS["knots"], marker="x", s=24,
            linewidths=1.15, zorder=6)
    low, high = bounds
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    if dimension == 2:
        axis.set_aspect("equal", adjustable="box")
    else:
        axis.set_zlim(low[2], high[2])
        axis.set_box_aspect(high - low)
    axis.set_axis_off()
    strip = figure.add_subplot(inner[1])
    knots = np.asarray(fitted.get("internal_knots", []), dtype=float).reshape(-1)
    strip.plot([0, 1], [0, 0], color="#ccd2d6", linewidth=.85, zorder=1)
    strip.scatter(knots, np.zeros_like(knots), marker="|", s=50, linewidths=1.15,
                  color=COLORS["knots"], zorder=2)
    strip.set(xlim=(-.015, 1.015), ylim=(-1, 1), xticks=[0, 1], yticks=[])
    strip.tick_params(axis="x", labelsize=7.5, length=0, pad=1, colors="#68717a")
    for spine in strip.spines.values():
        spine.set_visible(False)
    return axis, strip


def _legend(figure):
    handles = [
        Line2D([], [], color=COLORS["reference"], linestyle="--", linewidth=1, label="Input / reference curve"),
        Line2D([], [], color=COLORS["points"], linestyle="none", marker=".", label="Data points"),
        Line2D([], [], color=COLORS["fit"], linewidth=1.7, label="Fitted curve"),
        Line2D([], [], color=COLORS["controls"], marker="s", markerfacecolor="white", linestyle="--",
               label="Control vertices / polygon"),
        Line2D([], [], color=COLORS["knots"], marker="x", linestyle="none",
               label="Internal knots (curve / parameter strip)"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.525, .012),
                  ncol=5, frameon=False, fontsize=10, handlelength=2.2, columnspacing=1.6)


def _save(figure, path, dpi):
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return {"path": path.name, "sha256": file_sha256(path)}


def draw_sheet(loaded, selection, methods, path, dpi, comparison):
    columns = 6 if comparison else 5
    figure = plt.figure(figsize=(columns * 3.1 + .8, 15.9))
    grid = figure.add_gridspec(5, columns, left=.067, right=.988, bottom=.074,
                              top=.947 if comparison else .986, hspace=.22, wspace=.18)
    for i, source in enumerate(SOURCES):
        samples = [selection[source]] * 6 if comparison else selection[source]
        row_loaded = [loaded[(source, sample, method)] for sample, method in
                      zip(samples, methods if comparison else ["ours"] * 5)]
        shared_bounds = geometry_bounds(row_loaded) if comparison else None
        for j, sample in enumerate(samples):
            method = methods[j] if comparison else "ours"
            case = loaded[(source, sample, method)]
            draw_panel(figure, grid[i, j], case, shared_bounds if comparison else geometry_bounds([case]))
            if i == 0 and comparison:
                position = grid[0, j].get_position(figure)
                figure.text((position.x0 + position.x1) / 2, .97, PAPER_LABELS[method],
                            ha="center", va="center", fontsize=17, fontweight="semibold", color="#183849")
        position = grid[i, 0].get_position(figure)
        figure.text(.023, (position.y0 + position.y1) / 2, SOURCE_LABELS[source],
                    ha="center", va="center", rotation=90, fontsize=12, color="#334956")
    _legend(figure)
    return _save(figure, path, dpi)


def render(report_path, directory, seed=20260922, dpi=300, overwrite=False, comparison_overrides=None):
    report_path, directory = Path(report_path).resolve(), Path(directory).resolve()
    original_hash = file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index, comparison, showcase = select_cases(report, seed, comparison_overrides)
    panel_rows = []
    for source in SOURCES:
        panel_rows.extend(("six_methods_6x5", source, comparison[source], method) for method in METHODS)
        panel_rows.extend(("ours_5x5", source, sample, "ours") for sample in showcase[source])
    keys = sorted({(source, sample, method) for _, source, sample, method in panel_rows})
    names = {key: hashlib.sha256("\0".join(key).encode()).hexdigest()[:24] + ".npz" for key in keys}
    targets = [directory / name for name in ("six_methods_6x5.png", "ours_5x5.png", "case_panels_manifest.json", "six_methods_case_values.md")]
    targets += [directory / "case_panels_data" / names[key] for key in keys]
    if not overwrite:
        existing = next((path for path in targets if path.exists()), None)
        if existing:
            raise FileExistsError(f"Refusing to overwrite panel artifact: {existing}; use --overwrite explicitly")
    loaded, checked_sources = {}, {}
    for key in keys:
        row = index[key]
        if not row.get("geometry_artifact"):
            raise ValueError(f"Saved measured geometry required: {key}")
        loaded[key] = _geometry(report_path.parent, row)
        points = loaded[key][1]["input_points_normalized"]
        if key[:2] in checked_sources and not np.array_equal(points, checked_sources[key[:2]]):
            raise ValueError(f"Paired source geometry differs: {key[:2]}")
        checked_sources[key[:2]] = points
    directory.mkdir(parents=True, exist_ok=True)
    data_directory = directory / "case_panels_data"
    data_directory.mkdir(exist_ok=True)
    exports = {}
    for key in keys:
        _, source, fitted = loaded[key]
        arrays = {"source__" + name: value for name, value in source.items()}
        arrays.update({"fit__" + name: value for name, value in fitted.items()})
        target = data_directory / names[key]
        np.savez_compressed(target, **arrays)
        exports[key] = {"path": target.relative_to(directory).as_posix(), "sha256": file_sha256(target)}
    artifacts = [draw_sheet(loaded, comparison, METHODS, directory / "six_methods_6x5.png", dpi, True),
                 draw_sheet(loaded, showcase, METHODS, directory / "ours_5x5.png", dpi, False)]
    if file_sha256(report_path) != original_hash:
        raise RuntimeError("Source report changed while plotting")
    manifest = {
        "source_report": str(report_path), "source_report_sha256": original_hash,
        "source_metadata": deepcopy(report["metadata"]), "dpi": dpi, "seed": seed,
        "comparison_selection_rule": (
            "explicit user-selected illustrative cases for overridden sources; remaining sources use " + COMPARISON_RULE
            if comparison_overrides else COMPARISON_RULE),
        "comparison_overrides": dict(comparison_overrides or {}),
        "showcase_selection_rule": SHOWCASE_RULE,
        "comparison_selected_sample_ids": comparison, "showcase_selected_sample_ids": showcase,
        "display": {"per_panel_method_or_metrics": False, "one_legend_per_figure": True,
                    "six_methods_shared_geometry_bounds_per_source": True,
                    "geometry": "Exact saved normalized fits, controls, data, and internal knots; no refit or model inference",
                    "industrial_offsets": "Procedural industrial-model offset curves, not measured real-world observations"},
        "artifacts": artifacts, "panels": [],
    }
    for sheet, source, sample, method in panel_rows:
        key = source, sample, method
        manifest["panels"].append({
            "sheet": sheet, "dataset": source, "sample_id": sample, "method": method,
            "measurement": deepcopy(index[key]), "exported_geometry": exports[key],
            "source_case": deepcopy(loaded[key][0]),
        })
    (directory / "case_panels_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["# Six-method figure: exact measured values", "",
             "These are the 30 displayed panels, not aggregate statistics. Errors are normalized squared Euclidean distances; no square root.",
             "Total time includes every measured numerical postprocessing stage. Full records and geometry are in case_panels_manifest.json.",
             "", "Selection: " + manifest["comparison_selection_rule"], ""]
    for source in SOURCES:
        lines += [f"## {source}", "", f"Sample: `{comparison[source]}`", "",
                  "| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |",
                  "|---|---:|---:|---:|---:|---|"]
        for method in METHODS:
            row = index[source, comparison[source], method]
            values = [PAPER_LABELS[method], str(row.get("final_k", "—"))]
            values += [format(row[field], ".10g") if finite(row.get(field)) else "—"
                       for field in ("mse", "max_squared_error", "total_ms")]
            values += ["yes" if dual_pass(row, report["metadata"]) else "no"]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    (directory / "six_methods_case_values.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--comparison-sample", action="append", default=[], metavar="SOURCE=SAMPLE_ID",
                        help="Choose an illustrative case for all six methods in one row; records non-random selection in the manifest")
    args = parser.parse_args(argv)
    if args.dpi < 40:
        parser.error("--dpi must be at least 40")
    overrides = {}
    for item in args.comparison_sample:
        source, separator, sample = item.partition("=")
        if not separator or not source or not sample or source in overrides:
            parser.error("Each --comparison-sample must be a unique SOURCE=SAMPLE_ID")
        overrides[source] = sample
    result = render(args.report, args.output_dir, args.seed, args.dpi, args.overwrite, overrides)
    print(f"Saved 30 paired comparison panels and 25 Ours panels: {args.output_dir}")
    print(f"Exact geometry and full measurements: {len(result['panels'])} panel records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
