"""Render paired raw/final experiments without rerunning any method.

All statistics use every requested case. Contact sheets use a seeded random
case per source, never a fit-quality selection. Errors remain squared.
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
from benchmark_geometry import file_sha256, load_geometry_artifact

SOURCES = ("Synthetic", "UJI", "NaturalEarth", "USGS", "IndustrialOffset")
LABELS = {
    "ours": "Ours V16",
    "park_dominant_point_2007_adaptation": "Park (adaptation)",
    "liang_feature_iki_2017_adaptation": "Liang (adaptation)",
    "dung_direct_knot_2017_adaptation": "Dung (adaptation)",
    "kang_sparse_2015_adaptation": "Kang (adaptation)",
    "luo_linf_de_2022_adaptation": "Luo (adaptation)",
}
PAPER_LABELS = {method: label.split(" (")[0] for method, label in LABELS.items()}
PAPER_LABELS["ours"] = "Ours"


def _post_pruning(metadata):
    return isinstance(metadata.get("ours_post_pruning"), dict)


def _label(method, paper_style=False):
    return (PAPER_LABELS if paper_style else LABELS).get(method, method)


def _final_label(metadata):
    return "Final method" if _post_pruning(metadata) else "Method + common insertion"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value, spec=".2e"):
    return "--" if not finite(value) else format(value, spec)


def identity(row):
    return row["dataset"], str(row["sample_id"]), row["method"]


def dual_pass(row, metadata):
    return (row.get("status") == "ok" and finite(row.get("mse"))
            and finite(row.get("max_squared_error"))
            and 0 <= row["mse"] <= metadata["mse_tolerance"]
            and 0 <= row["max_squared_error"] <= metadata["max_squared_error_tolerance"])


def pair_measurements(report):
    """Reject unpaired identities instead of silently merging unmatched runs."""
    metadata = report["metadata"]
    for field in ("mse_tolerance", "max_squared_error_tolerance"):
        if not finite(metadata.get(field)) or metadata[field] <= 0:
            raise ValueError(f"{field} must be positive and finite")
    methods = metadata.get("methods", [])
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("metadata.methods must contain unique method identities")
    indexes = []
    for key in ("raw_measurements", "measurements"):
        index = {}
        for row in report[key]:
            entry = identity(row)
            if entry in index:
                raise ValueError(f"Duplicate {key} identity: {entry}")
            if row["method"] not in methods:
                raise ValueError(f"Undeclared method: {row['method']}")
            expected = dual_pass(row, metadata)
            if "joint_pass" in row and bool(row["joint_pass"]) != expected:
                raise ValueError(f"joint_pass does not match measured dual thresholds: {entry}")
            index[entry] = row
        indexes.append(index)
    raw, repaired = indexes
    if not raw or raw.keys() != repaired.keys():
        raise ValueError("Raw/repaired identities must match exactly and be nonempty")
    # Every method must have been requested for the same inputs, including
    # explicit algorithm-failure rows. Missing failures cannot improve rates.
    cases = defaultdict(set)
    for source, sample, method in raw:
        cases[(source, sample)].add(method)
    if "cases" in metadata:
        declared = metadata["cases"]
        if not isinstance(declared, list) or any(
                not isinstance(case, dict) or "dataset" not in case or "sample_id" not in case
                for case in declared):
            raise ValueError("metadata.cases must list declared dataset/sample_id identities")
        expected = {(case["dataset"], str(case["sample_id"])) for case in declared}
        if len(expected) != len(declared):
            raise ValueError("metadata.cases contains duplicate case identities")
        if set(cases) != expected:
            raise ValueError("Measured case identities must match metadata.cases exactly; incomplete benchmark cannot be plotted as complete")
    if any(values != set(methods) for values in cases.values()):
        raise ValueError("Each case must contain all declared methods, including failures")
    for key in raw:
        for field in ("source_points_sha256", "input_points_sha256", "source_k", "group_id"):
            left, right = raw[key].get(field), repaired[key].get(field)
            if left is not None and right is not None and left != right:
                raise ValueError(f"Raw/repaired source mismatch ({field}): {key}")
        for source_field, saved_field in (("mse", "raw_mse"), ("max_squared_error", "raw_max_squared_error"),
                                           ("final_k", "raw_final_k"), ("total_ms", "raw_ms")):
            saved = repaired[key].get(saved_field)
            original = raw[key].get(source_field)
            if finite(saved) and finite(original) and not math.isclose(saved, original, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Stored {saved_field} disagrees with raw row: {key}")
    return [(raw[key], repaired[key]) for key in sorted(raw)]


def _statistics(rows, metadata):
    valid = [row for row in rows if row.get("status") == "ok" and finite(row.get("mse"))
             and finite(row.get("max_squared_error"))]
    result = {"n_requested": len(rows), "n_finite": len(valid), "n_failed": len(rows) - len(valid),
              "dual_pass_count": sum(dual_pass(row, metadata) for row in rows)}
    result["dual_pass_percent"] = 100 * result["dual_pass_count"] / len(rows)
    for name in ("mse", "max_squared_error", "reference_mse", "reference_max_squared_error", "final_k"):
        values = [float(row[name]) for row in valid if finite(row.get(name))]
        result[name + "_n"] = len(values)
        result[name + "_mean"] = float(np.mean(values)) if values else None
        result[name + "_max"] = max(values) if values else None
    # A failed algorithm may still have a measured elapsed time. Preserve it.
    for name in ("total_ms", "network_ms", "repair_ms", "added_knots_count", "pruning_ms", "removed_knots_count"):
        values = [float(row[name]) for row in rows if finite(row.get(name))]
        result[name + "_n"] = len(values)
        result[name + "_mean"] = float(np.mean(values)) if values else None
        result[name + "_max"] = max(values) if values else None
    return result


def summarize(report):
    paired = pair_measurements(report)
    observed = {raw["dataset"] for raw, _ in paired}
    sources = ["All"] + [source for source in SOURCES if source in observed]
    sources += sorted(observed - set(SOURCES))
    result = []
    for source in sources:
        for method in report["metadata"]["methods"]:
            pairs = [(a, b) for a, b in paired if a["method"] == method and (source == "All" or a["dataset"] == source)]
            if not pairs:
                continue
            shared = [(a, b) for a, b in pairs if dual_pass(a, report["metadata"]) and dual_pass(b, report["metadata"])]
            deltas = [b["final_k"] - a["final_k"] for a, b in shared
                      if finite(a.get("final_k")) and finite(b.get("final_k"))]
            result.append({"dataset": source, "method": method, "label": LABELS.get(method, method),
                           "raw": _statistics([a for a, _ in pairs], report["metadata"]),
                           "repaired": _statistics([b for _, b in pairs], report["metadata"]),
                           "shared_dual_feasible_n": len(shared),
                           "shared_dual_feasible_k_delta_mean": float(np.mean(deltas)) if deltas else None})
    return result


def _save(figure, path, dpi):
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"path": path.name, "sha256": file_sha256(path)}


def write_tables(summary, metadata, directory, dpi, paper_style=False):
    headers = ["Source", "Method", "N / finite raw→final", "Dual pass % raw→final", "Mean K raw→final",
               "Mean MSE raw→final", "MaxSE raw→final", "Mean total ms", "Mean repair ms", "Mean added K"]
    pruning = _post_pruning(metadata)
    if pruning:
        headers += ["Mean prune ms", "Mean removed K"]
    cells = []
    for row in summary:
        raw, final = row["raw"], row["repaired"]
        def pair(field, spec):
            return number(raw[field], spec) + " → " + number(final[field], spec)
        cells.append([row["dataset"], row["label"], f"{raw['n_requested']} / {raw['n_finite']}→{final['n_finite']}",
                      pair("dual_pass_percent", ".1f"), pair("final_k_mean", ".2f"), pair("mse_mean", ".2e"),
                      pair("max_squared_error_max", ".2e"), number(final["total_ms_mean"], ".2f"),
                      number(final["repair_ms_mean"], ".2f"), number(final["added_knots_count_mean"], ".2f")])
        if pruning:
            cells[-1] += [number(final["pruning_ms_mean"], ".2f"), number(final["removed_knots_count_mean"], ".2f")]
    notes = [
        f"Dual pass requires input MSE ≤ {metadata['mse_tolerance']:.3g} AND maximum point squared Euclidean residual ≤ {metadata['max_squared_error_tolerance']:.3g}; neither quantity is square-rooted.",
        "MaxSE is the maximum over all points of all finite fits in the group, not maximum per-curve MSE and not average peak error.",
        "Every requested case, including numerical failures and unavailable results, remains in the pass denominator. Error/count means include all finite fits, even threshold misses; finite counts are shown.",
        "Arrows compare the saved raw method against that same method followed by common residual-guided knot insertion. All five literature methods are repository adaptations, not verified native reproductions. Repaired Ours is a hybrid, not one-shot deployment.",
        "Total time includes the measured original method and common repair; repair time is listed separately. The report's timing_protocol metadata is authoritative about acquisition and device scopes. Network and numerical methods can use different CUDA/CPU scopes; these are exploratory local measurements, not a controlled hardware-speed claim.",
        "All-case aggregate is curve-weighted, not equal-source-weighted. Synthetic and procedural industrial offsets are not measured real-world data. Contact-sheet selections do not affect statistics.",
        f"Declared unified internal-knot budget: {metadata.get('max_internal_knots', '--')}. A budget-exhausted case is still a failure; no threshold is silently relaxed.",
    ]
    if pruning:
        notes[3] = ("Arrows compare saved raw methods against their saved final measurements. "
                    "The final Ours measurements additionally include the redundant-knot post-pruning specified in ours_post_pruning metadata. "
                    "All five literature methods are repository adaptations, not verified native reproductions; "
                    "the final pipeline is not one-shot network inference.")
        notes[4] = ("Total time includes the measured original method, common repair, and any measured post-pruning; "
                    "repair and prune times are listed separately. Saved timing_protocol and ours_post_pruning metadata "
                    "record acquisition and device scopes; these are exploratory local measurements, not a controlled hardware-speed claim.")
    heading = "Raw methods versus final methods" if pruning else "Raw methods versus common dual-error insertion"
    markdown = "# " + heading + "\n\n" + "\n\n".join(notes) + "\n\n"
    if pruning:
        markdown += "## Post-pruning metadata\n\n```json\n" + json.dumps(metadata["ours_post_pruning"], indent=2, ensure_ascii=False) + "\n```\n\n"
    markdown += "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    markdown += "\n".join("| " + " | ".join(map(str, row)) + " |" for row in cells) + "\n"
    (directory / "dual_error_summary.md").write_text(markdown, encoding="utf-8")
    figure, axis = plt.subplots(figsize=(25, max(7, 3 + .36 * len(cells))))
    axis.axis("off")
    figure_cells = [list(cell) for cell in cells]
    if paper_style:
        for cell, row in zip(figure_cells, summary):
            cell[1] = _label(row["method"], True)
    widths = [.065, .12, .11, .115, .105, .15, .15, .065, .065, .055]
    if pruning:
        widths = [width * .88 for width in widths] + [.06, .06]
    table = axis.table(cellText=figure_cells, colLabels=headers, cellLoc="center", loc="upper center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dfe4")
        cell.set_linewidth(.5)
        if r == 0:
            cell.set_facecolor("#173b50")
            cell.set_text_props(color="white", fontweight="bold")
        elif summary[r - 1]["method"] == "ours":
            cell.set_facecolor("#dff2eb")
        elif summary[r - 1]["dataset"] == "All":
            cell.set_facecolor("#e9eef4")
    figure.suptitle("All measured cases | raw → final" if paper_style or pruning else
                   "All measured cases | raw methods → methods + common insertion", fontsize=17, y=.988)
    if not paper_style:
        figure.text(.03, .023, f"Dual pass: MSE ≤ {metadata['mse_tolerance']:.1e} AND MaxSE ≤ {metadata['max_squared_error_tolerance']:.1e}; squared Euclidean errors, no roots.\n"
                    "All requested cases remain in pass denominators; finite-fit means include misses. MaxSE = worst sampled point across finite cases.\n" +
                    ("Literature adaptations; final Ours includes post-pruning. Total includes base method, repair and pruning; see report metadata."
                     if pruning else "Literature adaptations + disclosed common repair; repaired Ours is hybrid. Total = measured base-method time + repair; device/acquisition scopes follow report metadata."), fontsize=9)
    artifact = _save(figure, directory / "dual_error_summary.png", dpi)
    return notes, artifact


def write_overall_table(summary, metadata, directory, dpi, paper_style=False):
    """Slide-sized aggregate, while the full per-source table remains saved."""
    overall = [row for row in summary if row["dataset"] == "All"]
    headers = ["Method", "N", "Dual pass %\nraw → final", "Mean K\nraw → final", "Mean MSE\nraw → final",
               "Worst MaxSE\nraw → final", "Mean total\n(ms)", "Mean repair\n(ms)"]
    cells = []
    for row in overall:
        raw, final = row["raw"], row["repaired"]
        def pair(field, spec):
            return number(raw[field], spec) + " → " + number(final[field], spec)
        cells.append([_label(row["method"], paper_style), str(raw["n_requested"]), pair("dual_pass_percent", ".1f"),
                      pair("final_k_mean", ".2f"), pair("mse_mean", ".2e"),
                      pair("max_squared_error_max", ".2e"), number(final["total_ms_mean"], ".2f"),
                      number(final["repair_ms_mean"], ".2f")])
    figure, axis = plt.subplots(figsize=(21, max(4.5, 2.6 + .47 * len(cells))))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=headers, cellLoc="center", bbox=(0, .17, 1, .74),
                       colWidths=[.16, .045, .125, .125, .165, .165, .105, .11])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dfe4")
        cell.set_linewidth(.6)
        if r == 0:
            cell.set_facecolor("#173b50")
            cell.set_text_props(color="white", fontweight="bold")
        elif overall[r - 1]["method"] == "ours":
            cell.set_facecolor("#dff2eb")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f5f8")
    pruning = _post_pruning(metadata)
    figure.suptitle("All measured curves | raw → final" if paper_style or pruning else
                   "All measured curves | raw methods → methods + common insertion", fontsize=17, y=.97)
    if not paper_style:
        figure.text(.055, .035, f"Dual pass: MSE ≤ {metadata['mse_tolerance']:.1e} AND MaxSE ≤ {metadata['max_squared_error_tolerance']:.1e}. Errors are squared; MaxSE is worst sampled point across finite cases.\n" +
                    ("All requested cases remain in pass denominators; finite-fit means include misses. Literature methods are adaptations; final Ours includes post-pruning.\n"
                     if pruning else "All requested cases remain in pass denominators; finite-fit means include misses. Literature methods are adaptations; repaired Ours is hybrid.\n") +
                    ("Total includes base method, common repair and post-pruning. Timing scopes and finite-fit counts are recorded in the full report."
                     if pruning else "Total = measured base-method time + common repair. Device/acquisition scopes and finite-fit counts are recorded in the full report."), fontsize=9)
    return _save(figure, directory / "dual_error_overall.png", dpi)


def plot_five_metrics(summary, metadata, directory, dpi, paper_style=False):
    sources = [source for source in SOURCES if any(row["dataset"] == source for row in summary)]
    sources += sorted({row["dataset"] for row in summary} - set(SOURCES) - {"All"})
    methods = metadata["methods"]
    metrics = [("mse_mean", "Mean MSE", True), ("dual_pass_percent", "Dual pass (%)", False),
               ("final_k_mean", "Mean internal K", False), ("total_ms_mean", "Mean total time (ms)", True),
               ("max_squared_error_max", "Maximum squared error", True)]
    figure, axes = plt.subplots(len(sources), 5, figsize=(24, 3.35 * len(sources) + 1.6), squeeze=False)
    colors = plt.get_cmap("tab10")(np.arange(len(methods)))
    for i, source in enumerate(sources):
        by_method = {row["method"]: row for row in summary if row["dataset"] == source}
        for j, (field, title, log) in enumerate(metrics):
            axis = axes[i, j]
            for k, method in enumerate(methods):
                row = by_method.get(method)
                if row is None:
                    continue
                for stage, offset in (("raw", -.19), ("repaired", .19)):
                    value = row[stage][field]
                    if not finite(value):
                        continue
                    axis.bar(k + offset, value, width=.36, edgecolor=colors[k],
                             color="none" if stage == "raw" else colors[k], linewidth=1.3,
                             hatch="//" if stage == "raw" else None)
            if log:
                values = [entry[stage][field] for entry in by_method.values() for stage in ("raw", "repaired")
                          if finite(entry[stage][field]) and entry[stage][field] > 0]
                if values:
                    axis.set_yscale("symlog", linthresh=min(values) * .1)
            if field == "mse_mean":
                axis.axhline(metadata["mse_tolerance"], color="0.45", linestyle="--", linewidth=.8)
            if field == "max_squared_error_max":
                axis.axhline(metadata["max_squared_error_tolerance"], color="0.45", linestyle="--", linewidth=.8)
            if field == "dual_pass_percent":
                axis.set_ylim(0, 105)
            axis.set_xticks(range(len(methods)), [_label(m, paper_style).replace(" (adaptation)", "") for m in methods], rotation=35, ha="right", fontsize=8)
            axis.set_title(title, fontsize=11)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            if j == 0:
                axis.set_ylabel(source, fontsize=12)
    pruning = _post_pruning(metadata)
    figure.suptitle("Raw and final methods | all measured cases" if paper_style or pruning else
                   "Raw methods and common dual-error insertion | all measured cases", fontsize=18, y=.993)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="0.35", hatch="//", label="Raw method"),
               plt.Rectangle((0, 0), 1, 1, facecolor="0.45", label="Final method" if paper_style else _final_label(metadata))]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .973), ncol=2, frameon=False)
    if not paper_style:
        figure.text(.04, .009, ("Literature methods are adaptations; final Ours includes post-pruning. Pass denominators include failures. Errors/counts use all finite fits, not only passes.\n"
                              if pruning else "Literature methods are adaptations; repaired Ours is hybrid. Pass denominators include failures. Errors/counts use all finite fits, not only passes.\n") +
                    ("MaxSE is worst sampled squared residual. Total includes base method, repair and post-pruning; timing/device protocol is in summary JSON."
                     if pruning else "MaxSE is worst sampled squared residual. Total = measured base-method time + repair. See summary JSON for sample counts and timing/device protocol."), fontsize=9)
    figure.tight_layout(rect=(.015, .015 if paper_style else .045, .995, .949))
    return _save(figure, directory / "dual_error_five_metrics.png", dpi)


def _source_random(seed, source, values):
    digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).choice(sorted(values))


def _geometry(root, row):
    document, arrays = load_geometry_artifact(root, row["geometry_artifact"])
    recorded = document["measurement"]
    if identity(recorded) != identity(row):
        raise ValueError(f"Geometry identity mismatch: {identity(row)}")
    for field in ("mse", "max_squared_error", "final_k"):
        if finite(recorded.get(field)) and finite(row.get(field)) and not math.isclose(
                recorded[field], row[field], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(f"Geometry measured {field} disagrees with report: {identity(row)}")
    case_reference = document["case_artifact"]
    # Copied raw/ subtrees can retain case paths relative to their original
    # benchmark root. Only that declared subtree is considered as fallback;
    # hash verification and benchmark-directory containment still apply.
    if not (root / case_reference["path"]).is_file():
        first = Path(row["geometry_artifact"]["path"]).parts[0]
        if first in {"raw", "repaired"}:
            case_reference = dict(case_reference, path=(Path(first) / case_reference["path"]).as_posix())
    case, source = load_geometry_artifact(root, case_reference)
    if (case["dataset"], str(case["sample_id"])) != identity(row)[:2]:
        raise ValueError("Geometry source identity mismatch")
    return case, source, arrays


def plot_contact_sheets(report, root, directory, dpi, seed, paper_style=False):
    paired = pair_measurements(report)
    if not all(row.get("geometry_artifact") for pair in paired for row in pair):
        return {"status": "not_generated", "reason": "At least one raw/repaired geometry artifact is absent; no geometry was fabricated."}
    methods = report["metadata"]["methods"]
    sources = [source for source in SOURCES if any(a["dataset"] == source for a, _ in paired)]
    by_id = {identity(a): (a, b) for a, b in paired}
    selected = {source: _source_random(seed, source, {str(a["sample_id"]) for a, _ in paired if a["dataset"] == source}) for source in sources}
    manifest = {"status": "generated", "selection_rule": "one seeded random paired case per source, independent of measured results", "seed": seed,
                "selected_sample_ids": selected, "images": [], "panels": []}
    checked_inputs = {}
    for stage_index, stage in enumerate(("raw", "repaired")):
        figure = plt.figure(figsize=(4 * len(methods), 3.8 * len(sources) + 1.7))
        grid = figure.add_gridspec(len(sources), len(methods), left=.045, right=.99,
                                  bottom=.025 if paper_style else .065, top=.905, hspace=.55, wspace=.27)
        for i, source in enumerate(sources):
            for j, method in enumerate(methods):
                row = by_id[(source, selected[source], method)][stage_index]
                case, original, arrays = _geometry(root, row)
                points = original["input_points_normalized"]
                source_key = source, selected[source]
                if source_key in checked_inputs and not np.array_equal(checked_inputs[source_key], points):
                    raise ValueError(f"Paired panels have different input points: {source_key}")
                checked_inputs[source_key] = points
                dimension = points.shape[-1]
                inner = grid[i, j].subgridspec(2, 1, height_ratios=(12, 1), hspace=.04)
                axis = figure.add_subplot(inner[0], projection="3d" if dimension == 3 else None)
                def line(values, **kwargs):
                    value = np.asarray(values)
                    if value.ndim == 2 and value.shape[1] == dimension:
                        axis.plot(*value.T, **kwargs)
                def scatter(values, **kwargs):
                    value = np.asarray(values)
                    if value.ndim == 2 and value.shape[1] == dimension and len(value):
                        axis.scatter(*value.T, **kwargs)
                reference = original.get("reference_points_normalized", points)
                line(reference, color="0.15", linewidth=1.2)
                scatter(points, color="0.5", s=4, alpha=.5)
                line(arrays.get("dense_curve_normalized", []), color="#078c82", linewidth=1.4)
                controls = arrays.get("control_points_normalized", [])
                line(controls, color="#ce911c", linestyle="--", linewidth=.65)
                scatter(controls, color="#ce911c", marker="s", s=9)
                scatter(arrays.get("knot_positions_normalized", []), color="#bd3e50", marker="x", s=17)
                status = "DUAL PASS" if dual_pass(row, report["metadata"]) else "MISS"
                if row.get("status") != "ok":
                    status = str(row.get("status", "failed")).upper()
                timing = f"Total={number(row.get('total_ms'), '.1f')} ms"
                if stage_index and not paper_style:
                    timing += f" Repair={number(row.get('repair_ms'), '.1f')} ms"
                    if _post_pruning(report["metadata"]) and method == "ours":
                        timing += f" Prune={number(row.get('pruning_ms'), '.1f')} ms"
                axis.set_title(f"{_label(method, paper_style)} | {status}\nK={row.get('final_k', '--')} MSE={number(row.get('mse'))} MaxSE={number(row.get('max_squared_error'))}\n"
                               + timing, fontsize=7.2)
                axis.tick_params(labelsize=6)
                axis.grid(alpha=.15)
                if dimension == 2:
                    axis.set_aspect("equal", adjustable="datalim")
                if j == 0:
                    axis.set_ylabel(source, fontsize=9)
                strip = figure.add_subplot(inner[1])
                knots = np.asarray(arrays.get("internal_knots", []))
                strip.axhline(0, color="0.7", linewidth=.6)
                strip.scatter(knots, np.zeros_like(knots), color="#bd3e50", marker="|", s=40)
                strip.set(xlim=(-.015, 1.015), ylim=(-1, 1), yticks=[], xticks=[0, .5, 1])
                strip.tick_params(labelsize=6, length=2)
                for spine in strip.spines.values():
                    spine.set_visible(False)
                manifest["panels"].append({"stage": stage, "dataset": source, "sample_id": selected[source], "method": method,
                                           "geometry_artifact": row["geometry_artifact"]})
        title = "Raw methods" if stage_index == 0 else (
            "Final methods (Ours includes redundant-knot post-pruning)" if _post_pruning(report["metadata"])
            else "Methods + common dual-error insertion (repaired Ours is hybrid)")
        figure.suptitle(("Six-method comparison | Raw" if stage_index == 0 else "Six-method comparison") if paper_style else
                       title + "\nSame seeded random case per source; actual measured geometry, including failures", fontsize=16, y=.99)
        legend = [Line2D([0], [0], color="0.15", label="Reference/input curve"), Line2D([0], [0], color="#078c82", label="Fit"),
                  Line2D([0], [0], color="#ce911c", marker="s", linestyle="--", label="Control polygon"),
                  Line2D([0], [0], color="#bd3e50", marker="x", linestyle="None", label="Internal knots")]
        if paper_style:
            legend.insert(1, Line2D([0], [0], color="0.5", marker=".", linestyle="None", label="Data"))
        figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .942), ncol=len(legend), frameon=False)
        if not paper_style:
            figure.text(.045, .018, f"MSE ≤ {report['metadata']['mse_tolerance']:.1e} AND MaxSE ≤ {report['metadata']['max_squared_error_tolerance']:.1e}; normalized squared Euclidean errors, no roots.\n" +
                        ("Literature methods are adaptations; final Ours includes post-pruning. Each panel shows uncropped measured geometry; full protocol is in report metadata."
                         if _post_pruning(report["metadata"]) else "All literature methods are adaptations; common insertion is an added comparison wrapper, not part of the cited papers. Each panel shows uncropped geometry."), fontsize=8)
        manifest["images"].append(_save(figure, directory / f"six_methods_{stage}_6x5.png", dpi))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", default=20260922, type=int)
    parser.add_argument("--dpi", default=180, type=int)
    parser.add_argument("--no-contact-sheets", action="store_true")
    parser.add_argument("--paper-style", action="store_true",
                        help="Use concise figure labels without methodological footnotes; retain all audits in Markdown/JSON.")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = summarize(report)
    directory = args.output_dir or args.report.parent / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    definitions, table = write_tables(summary, report["metadata"], directory, args.dpi, args.paper_style)
    overall = write_overall_table(summary, report["metadata"], directory, args.dpi, args.paper_style)
    metrics = plot_five_metrics(summary, report["metadata"], directory, args.dpi, args.paper_style)
    panels = ({"status": "disabled"} if args.no_contact_sheets else
              plot_contact_sheets(report, args.report.parent, directory, args.dpi, args.seed, args.paper_style))
    document = {"source_report": str(args.report.resolve()), "source_report_sha256": file_sha256(args.report),
                "metadata": report["metadata"], "definitions": definitions, "summary": summary,
                "artifacts": [table, overall, metrics], "contact_sheets": panels,
                "figure_style": "paper" if args.paper_style else "audited"}
    (directory / "dual_error_metrics.json").write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved audited dual-error comparison figures/tables: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
