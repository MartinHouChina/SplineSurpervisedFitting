"""Plot one saved, explicitly selected illustrative case at all ten tolerances."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from benchmark_ours_thresholds import summarize, write_json
from benchmark_geometry import file_sha256, load_geometry_artifact
from plot_dual_error_comparison import _geometry
from plot_postprune_case_panels import draw_panel, geometry_bounds, _legend, _save, plt, SOURCE_LABELS

DEFAULT_REPORT = Path("outputs/comparisons/ours_k64_threshold_sweep_20260923/threshold_sweep.json")


def candidates(report):
    if not report.get("complete"):
        raise ValueError("A completed sweep is required")
    summarize(report)
    if len(report["metadata"]["mse_tolerances"]) != 10:
        raise ValueError("The 2 x 5 sheet requires exactly ten tolerances")
    groups = defaultdict(list)
    for row in report["measurements"]:
        groups[(row["dataset"], str(row["sample_id"]))].append(row)
    eligible = []
    for key, rows in groups.items():
        rows.sort(key=lambda r: r["mse_tolerance"])
        if len(rows) == 10 and all(r["status"] == "ok" and r["joint_pass"] for r in rows):
            eligible.append((key, rows))
    return sorted(eligible, key=lambda item: (-(item[1][0]["final_k"] - item[1][-1]["final_k"]), item[0]))


def verified_geometry(root, rows):
    loaded = []
    for row in rows:
        entry = _geometry(root, row)
        _, source, fitted = entry
        if loaded and not np.array_equal(source["input_points_normalized"], loaded[0][1]["input_points_normalized"]):
            raise ValueError("Input curve differs across tolerances")
        document, _ = load_geometry_artifact(root, row["geometry_artifact"])
        evaluated = BSpline(fitted["full_knot_vector"], fitted["control_points_normalized"], document["spline"]["degree"])(fitted["sample_parameters"])
        squared = ((evaluated - source["input_points_normalized"]) ** 2).sum(-1)
        np.testing.assert_allclose([squared.mean(), squared.max()], [row["mse"], row["max_squared_error"]], rtol=1e-8, atol=1e-12)
        if len(fitted["internal_knots"]) != row["final_k"]:
            raise ValueError("K disagrees with exported geometry")
        loaded.append(entry)
    return loaded


def preview(report_path, directory, eligible, dpi):
    real = [item for item in eligible if item[0][0] in {"USGS", "NaturalEarth", "UJI"}]
    relative = sorted(real, key=lambda item: (item[1][-1]["final_k"] / max(item[1][0]["final_k"], 1), item[0]))
    chosen = dict(real[:4] + relative[:4])
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 4, left=.03, right=.985, bottom=.1, top=.91, hspace=.35)
    manifest = []
    for spec, (key, rows) in zip(grid, chosen.items()):
        entry = _geometry(report_path.parent, rows[0])
        draw_panel(figure, spec, entry, geometry_bounds([entry]))
        position = spec.get_position(figure)
        figure.text((position.x0 + position.x1) / 2, position.y1 + .02,
                    f"{key[0]} | K {rows[0]['final_k']} to {rows[-1]['final_k']}\n{key[1][-30:]}", ha="center", fontsize=10)
        manifest.append(dict(dataset=key[0], sample_id=key[1], counts=[r["final_k"] for r in rows]))
    _legend(figure)
    _save(figure, directory / "candidate_preview.png", dpi)
    write_json(directory / "candidate_preview.json", manifest)


def panel_order(rows):
    """Row-major reading order: loose to strict, 1e-4 down to 1e-5."""
    return sorted(rows, key=lambda row: row["mse_tolerance"], reverse=True)


def render(report_path, directory, key, rows, dpi):
    rows = panel_order(rows)
    loaded = verified_geometry(report_path.parent, rows)
    bounds = geometry_bounds(loaded)
    images = []
    for annotated in (True, False):
        figure = plt.figure(figsize=(20, 9.2))
        grid = figure.add_gridspec(2, 5, left=.025, right=.985, bottom=.105, top=.86, hspace=.42, wspace=.14)
        for i, (row, entry) in enumerate(zip(rows, loaded)):
            draw_panel(figure, grid[i // 5, i % 5], entry, bounds)
            position = grid[i // 5, i % 5].get_position(figure)
            title = f"({chr(97 + i)}) MSE limit {row['mse_tolerance']:.0e}  |  K = {row['final_k']}"
            if annotated:
                title += f"\nMSE {row['mse']:.2e}  |  MaxSE {row['max_squared_error']:.2e}"
            figure.text((position.x0 + position.x1) / 2, position.y1 + .018, title,
                        ha="center", va="bottom", fontsize=10.5, linespacing=1.55)
        figure.suptitle(f"{SOURCE_LABELS[key[0]]} | Deployment under ten error tolerances", fontsize=21, y=.973)
        _legend(figure)
        name = "ours_ten_thresholds_case.png" if annotated else "ours_ten_thresholds_case_clean.png"
        images.append(_save(figure, directory / name, dpi))
    exported = []
    for i, (row, (_, source, fit)) in enumerate(zip(rows, loaded), 1):
        path = directory / f"tier_{i:02d}.npz"
        np.savez_compressed(path, **fit, source_input_points_normalized=source["input_points_normalized"])
        exported.append(dict(measurement=row, geometry_npz=path.name, geometry_sha256=file_sha256(path)))
    write_json(directory / "case_manifest.json", dict(
        source_report=str(report_path.resolve()), source_report_sha256=file_sha256(report_path),
        dataset=key[0], sample_id=key[1], images=images, tiers=exported,
        panel_order="Row-major, top-left to bottom-right, decreasing MSE tolerance (1e-4 to 1e-5)",
        selection="Explicit outcome-selected illustrative example: all ten tiers pass; visually legible curve and clear knot-count reduction. Not used for aggregate statistics.",
        execution="Only saved final deployment geometries plotted; no inference, refit, relocation or pruning rerun.",
        protocol="Fixed 64K checkpoint + dual-bound insertion repair + redundant-knot post-pruning; MaxSE limit = 10 times MSE limit.",
        common_limits=[v.tolist() for v in bounds],
    ))
    lines = [f"# {key[0]}: ten-tolerance deployment example", "", f"Sample: `{key[1]}`", "",
             "An intentionally selected successful example, not an estimate of overall performance. All ten tiers use the same input points and plot limits. Curves are the previously measured outputs, not newly fitted curves.", "",
             "MSE and MaxSE are squared Euclidean errors in normalized coordinates; neither is square-rooted. MaxSE limit = 10 times the MSE limit.", "",
             "| MSE limit | MaxSE limit | Actual MSE | Actual MaxSE | Internal K | Control vertices | Dual pass | Total ms |",
             "|---|---|---|---|---|---|---|---|"]
    for row, (_, _, fit) in zip(rows, loaded):
        lines.append(f"| {row['mse_tolerance']:.0e} | {row['max_squared_error_tolerance']:.0e} | {row['mse']:.7g} | {row['max_squared_error']:.7g} | {row['final_k']} | {len(fit['control_points_normalized'])} | {row['joint_pass']} | {row['total_ms']:.3f} |")
    (directory / "case_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--sample-id")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing case figures and their derived data files")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error("Use a new or empty output directory, or explicitly pass --overwrite")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    eligible = candidates(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview(args.report, args.output_dir, eligible, args.dpi)
    else:
        key = (args.dataset, args.sample_id)
        selected = dict(eligible).get(key)
        if selected is None:
            parser.error("Specify an existing --dataset / --sample-id that passes all ten tiers")
        render(args.report, args.output_dir, key, selected, args.dpi)
    print(args.output_dir)


if __name__ == "__main__":
    main()
