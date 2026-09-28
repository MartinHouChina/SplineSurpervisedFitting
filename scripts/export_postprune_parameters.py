"""Render a checkpoint- and result-backed experimental-parameter table.

The selected weights are authenticated against the completed comparison report.
Launcher defaults, filenames and the current CUDA device are not parameter evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_local_model_parameters import audit_checkpoint, file_hash

DEFAULT_REPORT = ROOT / "outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json"
DEFAULT_CHECKPOINT = ROOT / "outputs/checkpoints/overnight_stable_k32_3090_r1.pt"
DEFAULT_OUTPUT = ROOT / "outputs/figures/postprune_paper_20260923"


def _required(mapping: dict, name: str):
    if name not in mapping or mapping[name] is None:
        raise ValueError(f"Required audited parameter is not recorded: {name}")
    return mapping[name]


def validate_completed_report(report: dict) -> dict:
    """Require one final row per selected curve and method, including failures."""
    meta = report["metadata"]
    cases = meta["cases"]
    identities = [(c["dataset"], c["sample_id"]) for c in cases]
    methods = meta["methods"]
    if len(identities) != len(set(identities)) or len(methods) != len(set(methods)):
        raise ValueError("Duplicate selected sample or method identity")
    expected = {(d, s, m) for d, s in identities for m in methods}
    actual = [(r["dataset"], r["sample_id"], r["method"]) for r in report["measurements"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Comparison must contain exactly one final result per selected case and method")
    pruning = meta.get("ours_post_pruning", {})
    if pruning.get("enabled") is not True or pruning.get("applies_to") != "ours_only":
        raise ValueError("This export requires the Ours-only post-pruning comparison")
    return meta


def parameter_rows(audit: dict, meta: dict) -> list[dict]:
    """No unrecorded default is substituted when composing the printed table."""
    model, train, data = (audit[k] for k in ("model_config", "training_config", "dataset_config"))
    baseline = meta["baseline_options"]
    degree = _required(model, "degree")
    capacity = _required(model, "max_internal_knots")
    proposal_epochs = _required(train, "proposal_epochs")
    total_epochs = _required(train, "epochs")
    if proposal_epochs > total_epochs:
        raise ValueError("Recorded proposal schedule exceeds total schedule")
    source_k_min = _required(data, "min_control_points") - degree - 1
    source_k_max = _required(data, "max_control_points") - degree - 1
    counts = Counter(c["dataset"] for c in meta["cases"])
    synth_k = sorted(c["source_k"] for c in meta["cases"] if c["dataset"] == "Synthetic")
    mixture = audit["synthetic_training_mixture"]
    rows: list[dict] = []

    def add(section: str, parameter: str, setting: str, evidence: str):
        rows.append(dict(section=section, parameter=parameter, setting=setting, evidence=evidence))

    add("Model", "Selected weights", Path(audit["checkpoint"]).name, "checkpoint SHA256 matched to comparison.metadata.checkpoint_sha256")
    add("Model", "Saved epoch / configured Proposal + Joint", f"{audit['epoch']} / {proposal_epochs} + {total_epochs - proposal_epochs}", "checkpoint.epoch; training_config.proposal_epochs / epochs")
    add("Model", "Spline degree / input points × dimensions", f"{degree} / {meta['num_points']} × {model['point_dim']}", "model_config.degree / point_dim; comparison.metadata.num_points")
    add("Model", "Internal / full knot-vector capacity", f"{capacity} / {capacity + 2 * (degree + 1)}", "model_config.max_internal_knots; full vector = K + 2(p + 1)")
    add("Model", "Hidden width / attention heads", f"{model['hidden_dim']} / {model['attention_heads']}", "model_config.hidden_dim / attention_heads")
    add("Model", "Encoder / selector layers; parameters", f"{model['encoder_layers']} / {model['selector_layers']}; {audit['model_parameter_count']:,}", "model_config; strict-loaded model parameter count")
    add("Model", "Selection / adaptive threshold", f"{model['one_shot_selection_policy']}; {'enabled' if model['one_shot_adaptive_threshold'] else 'disabled'}", "model_config.one_shot_selection_policy / one_shot_adaptive_threshold")
    add("Training", "Synthetic training / validation samples", f"{train['train_size']:,} / {train['val_size']:,}", "training_config.train_size / val_size")
    add("Training", "Synthetic mixture: simple / shape / spline", " / ".join(f"{100 * mixture[k]:g}%" for k in ("simple_fraction", "shape_fraction", "historical_fraction")), "checkpoint.synthetic_training_mixture")
    add("Training", "Historical spline internal K / noise σ", f"{source_k_min}–{source_k_max} / {data['noise_std']:g}", "dataset_config control-point range minus p + 1; noise_std")
    add("Training", "Real-data training / validation", f"{100 * train['real_fraction']:g}% training; {train['real_val_size']} per listed source", "training_config.real_fraction / real_val_size / real_manifest; real_data_role")
    add("Training", "Batch size / resample each epoch", f"{train['batch_size']} / {'yes' if train['resample_train_each_epoch'] else 'no'}", "training_config.batch_size / resample_train_each_epoch")
    add("Training", "Proposal / Joint learning rate", f"{train['lr']:.1e} / {train['joint_lr']:.1e}", "training_config.lr / joint_lr; base rates, not final group rates")
    add("Training", "Teacher greedy steps / trajectory checks", f"{train['teacher_greedy_steps']} / {train['teacher_greedy_trajectory_checks']}", "training_config.teacher_greedy_steps / teacher_greedy_trajectory_checks")
    add("Evaluation", "Paired test curves / compared methods", f"{len(meta['cases'])} / {len(meta['methods'])}", "complete Cartesian comparison.metadata.cases × methods, failures included")
    add("Evaluation", "Synthetic test curves / source K", f"{counts['Synthetic']} / {min(synth_k)}–{max(synth_k)}", "comparison.metadata.cases; actual selected source K")
    add("Evaluation", "UJI / coastline / contours / offset curves", " / ".join(str(counts[k]) for k in ("UJI", "NaturalEarth", "USGS", "IndustrialOffset")), "comparison.metadata.cases counts; offset curves are procedurally generated")
    add("Evaluation", "MSE / maximum squared-point-error bound", f"{meta['mse_tolerance']:.1e} / {meta['max_squared_error_tolerance']:.1e}", "comparison.metadata.mse_tolerance / max_squared_error_tolerance")
    add("Evaluation", "Maximum internal knots, all methods", str(meta["max_internal_knots"]), "comparison.metadata.max_internal_knots (reported failures are not truncated)")
    add("Evaluation", "Common correction / Ours postprocessing", "Residual insertion / verified knot deletion", "comparison.metadata.repair_protocol / ours_post_pruning")
    add("Evaluation", "Ours deletion floor / deletion budget", f"{meta['ours_post_pruning']['min_internal_knots']} / {meta['ours_post_pruning']['max_deletions']}", "comparison.metadata.ours_post_pruning")
    add("Evaluation", "Evaluation GPU / CPU numerical threads", f"{meta['hardware']['gpu']} / {meta['hardware']['torch_threads']}", "comparison.metadata.hardware, not current hardware or checkpoint filename")
    add("Baselines", "Park: shape weight", f"{baseline['park_shape_weight']:g}", "comparison.metadata.baseline_options.park_shape_weight")
    add("Baselines", "Liang: initial / dense knots; curvature weight", f"{baseline['liang_initial_knots']} / {baseline['liang_dense_knots']}; {baseline['liang_curvature_weight']:g}", "comparison.metadata.baseline_options.liang_*")
    add("Baselines", "Dung: scan intervals / optimization steps", f"{baseline['dung_scan_intervals']} / {baseline['dung_optimization_iterations']}", "comparison.metadata.baseline_options.dung_*")
    add("Baselines", "Kang: ADMM / bisections / relocation steps", f"{baseline['paper_admm_iterations']} / {baseline['paper_lambda_bisections']} / {baseline['paper_relocation_iterations']}", "comparison.metadata.baseline_options.paper_*")
    add("Baselines", "Luo: population / generations; η", f"{baseline['luo_de_population']} / {baseline['luo_de_iterations']}; {baseline['luo_eta']:g}", "comparison.metadata.baseline_options.luo_*")
    return rows


def create_payload(report_path: Path, checkpoint_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = validate_completed_report(report)
    checkpoint_sha = file_hash(checkpoint_path)
    if checkpoint_sha != meta.get("checkpoint_sha256"):
        raise ValueError("Checkpoint SHA256 does not match the measured model; refusing to relabel parameters")
    audit = audit_checkpoint(checkpoint_path)
    if audit["sha256"] != checkpoint_sha:
        raise ValueError("Checkpoint changed during audit")
    if audit["epoch"] != meta["epoch"] or audit["model_config"]["max_internal_knots"] != meta["model_capacity"]:
        raise ValueError("Checkpoint identity fields conflict with comparison metadata")
    if audit["model_config"]["degree"] != meta["baseline_options"]["degree"]:
        raise ValueError("Model and numerical-method spline degrees differ")
    return {
        "schema_version": "postprune_audited_parameters_v1",
        "source_report": str(report_path.resolve()),
        "source_report_sha256": file_hash(report_path),
        "checkpoint_audit": audit,
        "comparison_metadata": meta,
        "table_rows": parameter_rows(audit, meta),
        "notes": [
            "The selected epoch is the epoch of these weights. The configured Proposal + Joint schedule is not a claim of completed from-scratch training.",
            "These weights were warm-started with capacity resizing from a 64-internal-knot checkpoint; the full initializer chain is retained in checkpoint_audit.",
            "The 4–24 source-knot range describes only the historical spline component, not every synthetic training sample.",
            "Real data are validation-only in the selected training run. The listed validation manifests are UJI, Natural Earth, and USGS; the procedural industrial-offset dataset is an additional test source.",
            "The comparison has 21 synthetic and 10 per external source, not 10% of each source. Every selected curve remains in the numerical comparison, including failures.",
            "MSE is the mean squared Euclidean input-point residual; MaxSE is the largest single input-point squared Euclidean residual. Both use the normalized coordinates and neither takes a square root. They do not certify continuous-curve or Hausdorff error.",
            "Five literature baselines are repository adaptations, followed by common residual insertion for all six methods. Ours additionally applies verified post-deployment single-knot deletion with fixed t and surviving knot locations; this is not a pure network-only comparison.",
            "Total measured time includes raw fitting, common insertion, and Ours pruning. Network-only time is recorded separately. A single fresh complete-method timing per curve does not establish timing variance.",
            "Maximum internal-knot capacity 32 corresponds to a cubic full clamped vector of 40 entries and at most 36 control points, with four endpoint entries at each side.",
        ],
    }


def render_table(path: Path, rows: list[dict], dpi: int = 300):
    # One table, no methodological watermark or footnote on the PNG. Full audit
    # and extensions remain in the companion Markdown and JSON.
    cell_rows = [[r["section"], r["parameter"], r["setting"]] for r in rows]
    fig, ax = plt.subplots(figsize=(13.9, 11.9), facecolor="white")
    ax.axis("off")
    table = ax.table(cellText=cell_rows, colLabels=["Group", "Parameter", "Setting"],
                     colWidths=[0.10, 0.46, 0.44], cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.7)
    section_colors = {"Model": "#f3f6fa", "Training": "#ffffff", "Evaluation": "#f3f6fa", "Baselines": "#ffffff"}
    for (r, c), cell in table.get_celld().items():
        cell.PAD = .027
        cell.set_linewidth(.55)
        cell.set_edgecolor("#b9c2ca")
        cell.set_text_props(color="#162c42")
        if r == 0:
            cell.set_facecolor("white")
            cell.set_text_props(weight="bold", fontsize=12.0)
            cell.set_linewidth(.95)
        else:
            cell.set_facecolor(section_colors[rows[r - 1]["section"]])
            if c == 0:
                if r > 1 and rows[r - 2]["section"] == rows[r - 1]["section"]:
                    cell.get_text().set_text("")
                else:
                    cell.set_text_props(weight="bold")
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def export_payload(payload: dict, output_dir: Path, dpi: int = 300) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_parameters.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Experiment parameters", "", "Parameters are authenticated against the completed fresh comparison and selected checkpoint, not launcher defaults.", "",
             "| Group | Parameter | Setting |", "| --- | --- | --- |"]
    lines += [f"| {r['section']} | {r['parameter']} | {r['setting']} |" for r in payload["table_rows"]]
    lines += ["", "## Measurement and provenance notes", ""]
    lines += [f"- {n}" for n in payload["notes"]]
    lines += ["", f"- Report: `{payload['source_report']}`", f"- Report SHA256: `{payload['source_report_sha256']}`",
              f"- Checkpoint: `{payload['checkpoint_audit']['checkpoint']}`", f"- Checkpoint SHA256: `{payload['checkpoint_audit']['sha256']}`", ""]
    (output_dir / "experiment_parameters.md").write_text("\n".join(lines), encoding="utf-8")
    render_table(output_dir / "experiment_parameters.png", payload["table_rows"], dpi=dpi)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    payload = create_payload(args.report, args.checkpoint)
    export_payload(payload, args.output_dir, dpi=args.dpi)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "checkpoint_epoch": payload['checkpoint_audit']['epoch'],
                      "parameter_rows": len(payload['table_rows']), "checkpoint_sha256": payload['checkpoint_audit']['sha256']}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
