"""Supplement: uniform initialization with exactly Ours' raw K and parameters.

Uses the recorded paired forward cost and reruns only LS/repair/pruning. This
isolates knot placement from initial count without modifying the main report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_paper_evidence import run_chain, timed, refit, failed_snapshots, job_id
from benchmark_ours_thresholds import write_json
from benchmark_geometry import file_sha256, load_geometry_artifact, write_method_geometry
from plot_dual_error_comparison import _geometry
from postprune_saved_comparison import _case_from_arrays
from plot_paper_evidence import summarize, validate, paired_success_summary, plt, save

METHOD = "uniform_matched_count_shared_params"


def matched_initial(case, params, count, degree, paired_components):
    """Only algorithmic initialization/refit is timed, not artifact loading."""
    components = {k: paired_components[k] for k in ("network_ms", "transfer_ms")}
    start = time.perf_counter()
    knots = torch.linspace(0, 1, count + 2, dtype=torch.float64)[1:-1]
    components["initializer_ms"] = (time.perf_counter() - start) * 1000
    fit, components["initial_refit_ms"] = timed(lambda: refit(params, case["points"], knots, degree))
    return fit, components


def run(path):
    base = json.loads(path.read_text(encoding="utf-8"))
    validate(base)
    root, meta = path.parent, dict(base["metadata"])
    project_root = Path(__file__).resolve().parents[1]
    for relative, digest in meta["code_sha256"].items():
        if file_sha256(project_root / relative.replace("\\", "/")) != digest:
            raise ValueError(f"Main experiment implementation changed: {relative}")
    epsilon = meta["ablation_tolerance"]
    meta.update(methods=["ours", METHOD], variants=[], mse_tolerances=[epsilon],
                base_report=str(path.resolve()), base_report_sha256=file_sha256(path),
                supplementary_script_sha256=file_sha256(Path(__file__)),
                intervention="Uniform knots with exactly the raw Ours count and exactly the same learned parameters; paired network/transfer costs reused and charged; subsequent stages freshly executed. Main results unchanged.")
    report = dict(metadata=meta, complete=False, measurements=[r for r in base["measurements"] if r["method"] == "ours" and r["mse_tolerance"] == epsilon])
    torch.set_num_threads(meta["hardware"]["torch_threads"])
    originals = [r for r in report["measurements"] if r["stage"] == "raw"]
    for i, raw in enumerate(originals, 1):
        case_doc, source, fitted = _geometry(root, raw)
        case = _case_from_arrays(case_doc, source)
        record_path = root / "matched_count_runs" / job_id(case, METHOD, epsilon) / "record.json"
        if record_path.exists():
            saved = json.loads(record_path.read_text(encoding="utf-8"))
            if (saved["base_report_sha256"] != meta["base_report_sha256"]
                    or saved.get("supplementary_script_sha256") != meta["supplementary_script_sha256"]):
                raise ValueError("Matched-count resume source or implementation changed")
            report["measurements"].extend(saved["measurements"])
            continue
        start = time.perf_counter()
        params = None
        try:
            if raw["status"] != "ok":
                raise RuntimeError("Cannot construct a paired control from a failed network prediction")
            geometry_doc, _ = load_geometry_artifact(root, raw["geometry_artifact"])
            params = torch.from_numpy(fitted["sample_parameters"]).double()
            count = raw["final_k"]
            fit, components = matched_initial(case, params, count, geometry_doc["spline"]["degree"], raw["timing_components"])
            snapshots, details = run_chain(case, METHOD, fit, params, components, epsilon, meta["peak_ratio"], meta["max_internal_knots"])
        except (ValueError, RuntimeError, KeyError) as error:
            shared_cost = (sum(raw["timing_components"].get(k, 0.) or 0. for k in ("network_ms", "transfer_ms"))
                           if raw["status"] == "ok" else raw["total_ms"])
            snapshots = failed_snapshots(case, METHOD, epsilon, meta["peak_ratio"], error,
                                         shared_cost + (time.perf_counter() - start) * 1000)
            details = dict(error=str(error))
        geometry_doc, _ = load_geometry_artifact(root, raw["geometry_artifact"])
        rows = []
        for row, fit in snapshots:
            folder = record_path.parent / row["stage"]
            artifact = write_method_geometry(folder, case, row, fit, params,
                       case_artifact=geometry_doc["case_artifact"], fingerprint=meta["fingerprint"], dense_points=801,
                       timing_scope=meta["intervention"], pass_criterion="Same dual observation-error bounds")
            artifact["path"] = (folder.relative_to(root) / artifact["path"]).as_posix()
            row["geometry_artifact"] = artifact
            rows.append(row)
        write_json(record_path, dict(base_report_sha256=meta["base_report_sha256"],
                   supplementary_script_sha256=meta["supplementary_script_sha256"], measurements=rows, details=details))
        report["measurements"].extend(rows)
        final = rows[-1]
        print(f"Matched-count [{i}/{len(originals)}] {case['dataset']} K={final['final_k']} pass={final['joint_pass']} MSE={final['mse']}", flush=True)
    report["complete"] = True
    summary = summarize(report)
    write_json(root / "matched_count_control.json", report)
    write_json(root / "matched_count_summary.json", dict(metadata=meta, summary=summary))
    write_json(root / "matched_count_paired_success.json", dict(
        scope="Conditional on both methods passing; report alongside full-denominator pass rates.",
        comparisons=paired_success_summary(report)))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    rows = [next(r for r in summary if r["dataset"] == "All" and r["method"] == m and r["stage"] == "pruned") for m in meta["methods"]]
    for ax, field, title in zip(axes, ("pass_percent", "final_k_mean", "mse_mean", "total_ms_mean"),
                                ("Dual pass (%)", "Mean internal K", "Mean MSE", "Mean full cost (ms)")):
        ax.bar([0, 1], [r[field] for r in rows], color=["#00877a", "#bd7943"])
        ax.set_xticks([0, 1], ["Learned\nlocations", "Uniform\nlocations"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if field == "pass_percent":
            ax.set_ylim(0, 105)
        for j, r in enumerate(rows):
            value = r[field]
            if value is not None:
                ax.annotate(f"{value:.2e}" if field == "mse_mean" else f"{value:.2f}", (j, value), xytext=(0, 4), textcoords="offset points", ha="center")
    fig.suptitle("Matched initial count and identical learned parameters | same dual-error repair and pruning")
    fig.tight_layout(rect=(0, 0, 1, .91))
    save(fig, root / "ablation_matched_initial_count.png")
    write_json(root / "matched_count_plot_provenance.json", dict(
        supplementary_script_sha256=file_sha256(Path(__file__)),
        report_sha256=file_sha256(root / "matched_count_control.json"),
        summary_sha256=file_sha256(root / "matched_count_summary.json"),
        image_sha256=file_sha256(root / "ablation_matched_initial_count.png")))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    run(args.report)
