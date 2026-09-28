"""Export audited checkpoint parameters, not launcher defaults, as PNG/MD/JSON.

This read-only checkpoint audit never trains a model or changes its weights.
The recorded validation snapshot is kept separate from any new test benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spline_fitting.checkpointing import build_model_from_checkpoint

DEFAULT_DIR = ROOT / "outputs/downloads/server_models_20260922_215418"
DEFAULT_FILES = [
    DEFAULT_DIR / "paper_coupled_clean_3090_r2_m16.pt",
    DEFAULT_DIR / "paper_coupled_clean_3090_r3_m32.pt",
]
MISSING = "Not recorded"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initializer_chain(provenance: dict) -> list[dict]:
    """Keep provenance hashes and settings without repeating nested ancestry."""
    chain = []
    while isinstance(provenance, dict) and provenance:
        chain.append({key: provenance[key] for key in (
            "mode", "path", "sha256", "epoch", "stage", "copied_tensor_count",
            "training_config", "real_data_provenance",
        ) if key in provenance})
        provenance = provenance.get("ancestor_initializer_provenance", {})
    return chain


def audit_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model, config, migrated = build_model_from_checkpoint(checkpoint)
    history = checkpoint.get("history", [])
    training = checkpoint.get("training_config", {})
    saved_epoch = checkpoint.get("epoch")
    saved_entries = [entry for entry in history if entry.get("epoch") == saved_epoch]
    saved_entry = saved_entries[-1] if saved_entries else {}
    report = {key: checkpoint.get(key) for key in (
        "objective_version", "architecture_revision", "epoch", "stage",
        "model_config", "training_config", "dataset_config", "training_schedule",
        "deployment_config", "loss_config", "real_data_role", "real_data_provenance",
        "synthetic_data_contract", "synthetic_training_mixture", "qualification",
        "checkpoint_quality", "checkpoint_selection", "simplification_ready",
    )}
    report.update({
        "checkpoint": str(path.resolve()),
        "sha256": file_hash(path),
        "file_bytes": path.stat().st_size,
        "strict_model_load": True,
        "legacy_migration": migrated,
        "effective_model_config": config,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "model_buffer_count": sum(p.numel() for p in model.buffers()),
        "history_length_in_checkpoint": len(history),
        "last_history_epoch_in_checkpoint": history[-1].get("epoch") if history else None,
        "saved_phase": saved_entry.get("phase"),
        "saved_proposal_frozen": saved_entry.get("proposal_frozen"),
        "saved_learning_rates": saved_entry.get("learning_rates"),
        "initializer_chain": initializer_chain(checkpoint.get("initializer_provenance", {})),
        "optimizer_parameter_groups_at_save": [
            {key: value for key, value in group.items() if key != "params"}
            for group in checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
        ],
        "recorded_validation_snapshot": checkpoint.get("validation_metrics"),
        "notes": [],
    })
    if isinstance(saved_epoch, int) and isinstance(training.get("epochs"), int):
        if saved_epoch < training["epochs"]:
            report["notes"].append(
                f"This file stores epoch {saved_epoch}; {training['epochs']} is the configured "
                "schedule, not the epoch of these weights. A best checkpoint alone cannot "
                "establish whether the server later finished training."
            )
    if report.get("qualification", {}).get("formal_reporting_eligible") is False:
        report["notes"].append(
            "The saved checkpoint did not meet its recorded reporting qualification. "
            "Any newly measured test outcomes must be reported as measured, without "
            "substituting this validation snapshot or only selecting successful cases."
        )
    if report["initializer_chain"]:
        report["notes"].append("Warm-started weights; this is not a from-scratch training run.")
    report["notes"].append(
        "GPU model, complete-run wall time, and the terminal epoch reached on the server "
        "are not established by the checkpoint. device=cuda is not proof of an RTX 3090."
    )
    del model, checkpoint
    return report


def fmt(value) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def value(record: dict, section: str, field: str):
    return (record.get(section) or {}).get(field)


def main_rows(records: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []

    def add(label, getter):
        rows.append([label, *[fmt(getter(record)) for record in records]])

    def field(label, section, key):
        add(label, lambda r: value(r, section, key))

    field("Candidate internal-knot capacity", "model_config", "max_internal_knots")
    add("Full clamped knot-vector capacity", lambda r: value(r, "model_config", "max_internal_knots")
        + 2 * (value(r, "model_config", "degree") + 1))
    field("Spline degree", "model_config", "degree")
    add("Ordered input samples x dimensions", lambda r: f"{value(r, 'training_config', 'num_points')} x "
        f"{value(r, 'model_config', 'point_dim')}")
    add("Historical spline source internal K", lambda r:
        f"{value(r, 'dataset_config', 'min_control_points') - value(r, 'model_config', 'degree') - 1}--"
        f"{value(r, 'dataset_config', 'max_control_points') - value(r, 'model_config', 'degree') - 1}")
    field("Hidden width / embedding dimension", "model_config", "hidden_dim")
    field("Encoder layers", "model_config", "encoder_layers")
    field("Attention heads", "model_config", "attention_heads")
    field("Base selector blocks", "model_config", "selector_layers")
    add("Proposal / selection / survivor extra blocks", lambda r: " / ".join(str(value(r, "model_config", k))
        for k in ("proposal_refinement_layers", "selection_refinement_layers", "survivor_refinement_layers")))
    add("Coupled proposal / subset updates", lambda r: " / ".join(str(value(r, "model_config", k))
        for k in ("coupled_proposal_steps", "coupled_subset_steps")))
    field("Selection policy", "model_config", "one_shot_selection_policy")
    field("Adaptive logit threshold", "model_config", "one_shot_adaptive_threshold")
    field("Minimum selected internal knots", "model_config", "min_selected_knots")
    field("Subset geometry", "model_config", "subset_geometry_mode")
    field("Final explicit chord-blend coefficient", "model_config", "parameter_chord_blend")
    add("Synthetic training / validation samples", lambda r: f"{value(r, 'training_config', 'train_size')} / "
        f"{value(r, 'training_config', 'val_size')}")
    field("Real validation samples per listed source", "training_config", "real_val_size")
    field("Real training fraction", "training_config", "real_fraction")
    field("Batch size", "training_config", "batch_size")
    field("Per-epoch training resampling", "training_config", "resample_train_each_epoch")
    add("Configured Proposal + Joint epochs", lambda r: f"{value(r, 'training_config', 'proposal_epochs')} + "
        f"{value(r, 'training_config', 'epochs') - value(r, 'training_config', 'proposal_epochs')}")
    add("Stored weight epoch / phase", lambda r: f"{r['epoch']} / Joint calibration")
    field("Initial Joint proposal-freeze epochs", "training_config", "joint_geometry_calibration_epochs")
    add("Proposal / Joint base learning rate", lambda r: f"{value(r, 'training_config', 'lr'):g} / "
        f"{value(r, 'training_config', 'joint_lr'):g}")
    add("Joint proposal / decoder LR scales", lambda r: f"{value(r, 'training_config', 'joint_proposal_lr_scale'):g} / "
        f"{value(r, 'training_config', 'joint_decoder_lr_scale'):g}")
    field("Weight decay", "training_config", "weight_decay")
    field("Synthetic noise standard deviation", "dataset_config", "noise_std")
    field("Normalized MSE threshold", "deployment_config", "mse_tolerance")
    field("Peak squared-point-error training threshold", "training_config", "max_point_error_tolerance")
    add("Learned model parameters", lambda r: f"{r['model_parameter_count']:,}")
    add("Saved qualification satisfied", lambda r: value(r, "qualification", "formal_reporting_eligible"))
    return rows


def loss_rows(records: list[dict]) -> list[list[str]]:
    keys = list(dict.fromkeys(k for r in records for k in (r.get("loss_config") or {}).get("weights", {})))
    return [[key, *[fmt((r.get("loss_config") or {}).get("weights", {}).get(key)) for r in records]] for key in keys]


def table_png(path: Path, title: str, columns: list[str], rows: list[list[str]], footer: str):
    height = max(5.0, .34 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(13.2, height), facecolor="white")
    ax.axis("off")
    fig.suptitle(title, fontsize=19, color="#163550", x=.045, ha="left", y=.979, weight="bold")
    widths = [.56] + [.44 / (len(columns) - 1)] * (len(columns) - 1)
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left",
                     colWidths=widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7e2e9")
        cell.set_linewidth(.45)
        cell.PAD = .045
        if row == 0:
            cell.set_facecolor("#163550")
            cell.set_text_props(color="white", weight="bold", ha="left")
        else:
            cell.set_facecolor("#eef4f8" if row % 2 else "white")
            if col == 0:
                cell.set_text_props(color="#1e3d55")
    fig.text(.045, .023, footer, fontsize=9.2, ha="left", va="bottom", color="#495c69")
    fig.subplots_adjust(left=.045, right=.97, top=.94, bottom=.078)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def markdown_table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
                      *["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]])


def export_evaluation_parameters(experiment: Path, output_dir: Path):
    meta = json.loads(experiment.read_text(encoding="utf-8"))
    config = meta["configuration"]
    hardware = meta["hardware"]
    datasets = meta["datasets"]
    rows = [
        ["Compared checkpoint", Path(meta["checkpoint"]).name.replace("paper_coupled_clean_3090_", "")],
        ["Stored weight epoch", fmt(meta.get("epoch"))],
        ["Input points per curve", fmt(meta["num_points"])],
        ["Spline degree", fmt(meta["knot_capacities"]["degree"])],
        ["Internal-knot capacity (all methods)", fmt(config["max_internal_knots"])],
        ["Normalized MSE pass threshold", fmt(meta["mse_tolerance"])],
        ["Additional peak metric", "max_i ||C(t_i) - Q_i||^2 (no root)"],
        ["Source families / paired selected curves", f"{len(datasets)} / {sum(d['selected_count'] for d in datasets)}"],
        ["Synthetic source K / curves per K", f"{config['min_knot_count']}--{config['max_knot_count']} / {config['samples_per_knot_count']}"],
        ["External test selection seed", fmt(config["selection_seed"])],
        ["External sampling", "Seeded group round-robin"],
    ]
    for dataset in datasets:
        if "available_test_curves" in dataset:
            rows.append([dataset["dataset"] + " selected / available test curves",
                         f"{dataset['selected_count']} / {dataset['available_test_curves']}"])
    rows.extend([
        ["Compared methods", "Ours + Park + Liang + Dung + Kang + Luo"],
        ["Literature implementation protocol", fmt(config["baseline_protocol"])],
        ["Extra common feasibility repair", fmt(config["published_feasibility_safeguard"])],
        ["ADMM iterations / lambda bisections", f"{config['paper_admm_iterations']} / {config['paper_lambda_bisections']}"],
        ["Kang relocation iterations", fmt(config["paper_relocation_iterations"])],
        ["Liang dense / initial internal knots", f"{config['liang_dense_knots']} / {config['liang_initial_knots']}"],
        ["Dung scan intervals / optimization iterations", f"{config['dung_scan_intervals']} / {config['dung_optimization_iterations']}"],
        ["Luo DE population setting / generations", f"{config['luo_de_population']} / {config['luo_de_iterations']}"],
        ["Network-only warmups / timed repeats", f"{config['network_warmups']} / {config['network_repeats']}"],
        ["Complete-method timed repeats per curve", fmt(config["end_to_end_repeats"])],
        ["GPU used by Ours in this local evaluation", hardware["gpu"]],
        ["CPU numerical backend threads", fmt(hardware["threads"])],
        ["Python / PyTorch", f"{hardware['python']} / {hardware['torch']}"],
        ["Dense plot sampling points", fmt(config["geometry_dense_points"])],
    ])
    payload = {"experiment_json": str(experiment.resolve()), "experiment_json_sha256": file_hash(experiment),
               "experiment_fingerprint": meta.get("fingerprint"), "table": {"columns": ["Parameter", "Value"], "rows": rows},
               "configuration": config, "hardware": hardware,
               "datasets": [{k: v for k, v in d.items() if k != "excluded_training_seed_ranges"} for d in datasets],
               "error_metric_definitions": meta["error_metric_definitions"],
               "timing_protocol": meta["timing_protocol"],
               "note": "Planned/selected cases in experiment.json are not proof all methods have finished. "
               "Final statistics must use completed measured rows and count failures."}
    (output_dir / "evaluation_parameters.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["# 本地六方法对比：实际评估参数", "", markdown_table(["Parameter", "Value"], rows), "",
            "- 参数来源为正在运行/已运行 benchmark 的 experiment.json，不是训练默认参数。所列样本数是已选定数目；全部结果是否完成以最终 rows / summary 为准。",
            "- 本次每外部来源选 10 条，采用固定随机种子、按组轮询，不是每个数据集的十分之一。五列来源为合成、UJI、海岸线、等高线、程序生成等距线。",
            "- IndustrialOffset 为程序生成 CAD 风格等距线，不是实测工业数据。",
            "- 五种文献方法均为仓库适配实现，未经原作者代码一致性认证。额外公共可行性修复关闭。",
            "- MSE 与最大平方点误差均为归一化离散对应点指标，后者不是连续曲线 Hausdorff 距离。通过率仅由 MSE 判断；最大误差单列报告。",
            "- 计时由各方法完整运行得到；Ours 额外报告单独预热测得的 network-only 时间。不能用 GPU network-only 时间代替数值方法完整耗时而宣称端到端公平加速。",
            "- 完整方法每曲线只重复 1 次，适合此次出图诊断，不足以估计运行时间方差；每条曲线内部 network-only 重复 10 次。",
            f"- 来源：`{experiment.resolve()}`", f"- SHA256：`{payload['experiment_json_sha256']}`", ""]
    (output_dir / "evaluation_parameters.md").write_text("\n".join(text), encoding="utf-8")
    table_png(output_dir / "evaluation_parameters.png", "Local six-method evaluation settings", ["Parameter", "Value"], rows,
              "Ten curves per external source (not 10%); paired samples are selected before results. IndustrialOffset is procedural.\n"
              "Literature methods are adaptations; complete-method timing and Ours network-only timing are separate.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures/local_models_paper_20260922")
    parser.add_argument("--experiment", type=Path, help="Optional benchmark experiment.json to audit separately")
    args = parser.parse_args(argv)
    records = [audit_checkpoint(path) for path in args.checkpoints]
    labels = [f"M{r['model_config']['max_internal_knots']}" for r in records]
    columns = ["Parameter", *labels]
    rows = main_rows(records)
    losses = loss_rows(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "purpose": "Audited checkpoint parameters; no new performance measurements",
              "records": records, "parameter_table": {"columns": columns, "rows": rows},
              "loss_table": {"columns": columns, "rows": losses}}
    (args.output_dir / "parameters.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["# 本地检查点：实验参数与来源审计", "",
            "所有参数直接读取指定检查点，不使用一条龙脚本的默认值代替实际记录。模型以严格模式恢复成功；没有训练、修改或覆盖权重。", "",
            "## 核心实验参数", "", markdown_table(columns, rows), "",
            "## 必须与论文结果一起说明的事项", "",
            "- 两个文件均保存第 13 轮（12 轮 Proposal 后第 1 轮 Joint calibration）的权重。12 + 48 是配置计划，不是这些权重已经历的轮数。服务器可能继续训练，但仅凭最佳检查点不能判断整次训练是否结束。",
            "- 两者均为 warm start；M16 / M32 的前置模型和源样条复杂度范围不同，不是严格只改变候选容量的消融。",
            "- 当前训练数据中真实数据比例为 0；真实验证名单只有 UJI、Natural Earth、USGS。工业等距线是额外的程序生成测试来源，不能称为实测工业数据。",
            "- 表中的 source K 范围只描述历史样条生成分量。训练混合含 35% simple、25% shape、40% historical，不能将整个混合描述为只有该 K 范围的随机 B 样条。",
            "- 两个保存检查点的 formal_reporting_eligible 均为 False。此前 validation 数字与本次重新测试数字必须分开；成功精选图不能代替全样本统计。",
            "- 误差均在归一化坐标下。MSE 为平均平方欧氏距离；峰值平方误差为样本点中最大平方欧氏距离，不是连续曲线全域 Hausdorff 最大误差。",
            "- min_selected_knots=4 是部署配置；完整端点夹持结点向量容量为 Kc+8（三次样条），控制顶点数量上限为 Kc+4。",
            "- parameter_chord_blend=0 不等于未使用弦长信息：模型仍包含弦长参考、可信度混合和反事实训练。该字段只是最后显式弦长混合的系数。",
            "- CUDA 已记录，但具体 GPU 型号、服务器最终训练轮数和完整训练耗时未记录在该检查点中，不能凭文件名推定。", "",
            "## 损失权重（检查点原值）", "", markdown_table(columns, losses), "",
            "## 检查点身份与已记录验证（非本次测试）", ""]
    for label, record in zip(labels, records):
        val = record.get("recorded_validation_snapshot") or {}
        text.extend([f"### {label}", "", f"- 文件：`{record['checkpoint']}`",
                     f"- SHA256：`{record['sha256']}`",
                     f"- 保存轮次 / phase：{record['epoch']} / `{record['saved_phase']}`",
                     f"- 已记录验证通过率：{100*val.get('deployment_pass_rate', 0):.3f}%；平均 K：{val.get('keep_count')}；MSE：{val.get('deployment_mse'):.6e}。",
                     f"- 保存时优化器参数组：`{json.dumps(record['optimizer_parameter_groups_at_save'], ensure_ascii=False)}`", ""])
        text.extend([f"- {note}" for note in record["notes"]])
        text.append("")
    (args.output_dir / "parameters.md").write_text("\n".join(text), encoding="utf-8")
    table_png(args.output_dir / "parameters.png", "Checkpoint-audited model and training parameters", columns, rows,
              "Configured schedules are not completed weight epochs. Both files store epoch 13; both are warm-started.\n"
              "MSE and peak squared error use normalized coordinates. Full provenance and qualifications: parameters.json / .md.")
    table_png(args.output_dir / "parameters_losses.png", "Recorded loss weights", columns, losses,
              "Values are read from each checkpoint's loss_config.weights. These are weights, not measured losses.\n"
              "Teacher search settings and remaining configuration are retained in parameters.json.")
    if args.experiment is not None:
        export_evaluation_parameters(args.experiment, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "checkpoints": [
        {"file": Path(r["checkpoint"]).name, "epoch": r["epoch"], "parameters": r["model_parameter_count"],
         "sha256": r["sha256"]} for r in records]}, indent=2))


if __name__ == "__main__":
    main()
