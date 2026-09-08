# 输出产物清单与整理规则

本文档记录 2026-09-02 完成后的实际目录结构。正式权重保持原入口，历史文件已经归档；临时 smoke 先移入可恢复区，没有做不可逆删除。

## 1. 当前必须保护的正式产物

### P0：部署与复现实验入口

| 路径 | 作用 | 备注 |
|---|---|---|
| `outputs/candidate_pruning_one_shot_v12.pt` | 当前 v12 部署 checkpoint | SHA256 `E8C9744A86659CBD3E8C65065427CD7B0B7197F0A0B4279998E447B540E32A50` |
| `outputs/candidate_pruning_one_shot_v12_teacher/` | v12 固定教师缓存 | 保护 `train.pt` 与 `val.pt`，不能拆开重命名 |
| `outputs/comparisons/current/v12_curve_comparison_K4_20/` | 当前 K=4–20 几何个例与 manifest | 展示前保留 |
| `outputs/comparisons/current/v12_three_metrics_K4_20_n20/` | 当前三指标分层实验 | 含 JSON、CSV、PNG 与摘要 |
| `outputs/comparisons/current/v12_verified_chord_compact_K4_20_n20/` | verified chord 分层结果 | 文档当前引用 |
| `outputs/comparisons/current/v12_verified_four_panel_K4_20/` | verified 四栏个例 | 文档当前引用 |
| `outputs/figures/current/v12_pipeline_framework.png` | 当前算法框架图 | PPT/文档引用 |
| `outputs/paper/kang_sparse_reproduction/` | Kang 稀疏法两个完整标量复现案例 | JSON 与 PNG 已实跑 |
| `outputs/comparisons/example_K4/` | 三方法 K=4 单样本连通性实例 | 非正式统计，仅用于检查口径与版式 |

### P1：certified/v13 实验链路

| 路径 | 作用 |
|---|---|
| `outputs/candidate_pruning_certified.pt` | certified 最终 checkpoint；SHA256 `6A4A7CC79DDD1F44B2DC11A229F4A9ACC33FED49D79C987628215441FA4B31EE` |
| `outputs/candidate_pruning_certified_proposal.pt` | proposal 阶段快照 |
| `outputs/checkpoints/stages/certified/candidate_pruning_certified_distill.pt` | distillation 阶段快照 |
| `outputs/checkpoints/stages/certified/candidate_pruning_certified_calibrated.pt` | calibration 阶段快照 |
| `outputs/checkpoints/stages/certified/candidate_pruning_certified_last.pt` | 最后一轮恢复点 |
| `outputs/certified_minimal_teacher/` | certified 教师缓存 |

### P1：v12 可恢复训练链路

以下文件在正式 release 归档完成前保留：

```text
outputs/checkpoints/stages/v12/candidate_pruning_one_shot_v12_distill.pt
outputs/checkpoints/stages/v12/candidate_pruning_one_shot_v12_calibrated.pt
outputs/checkpoints/stages/v12/candidate_pruning_one_shot_v12_last.pt
```

本次整理时不存在 `candidate_pruning_one_shot_v12_proposal.pt`；如后续重训生成，应归入 `outputs/checkpoints/stages/v12/`，不要重新放回 `outputs/` 顶层。

删除任何 checkpoint 前，应同时记录文件大小、SHA256、训练配置、父 checkpoint 和 teacher fingerprint。

## 2. 已执行的目录整理

当前结构如下。为了兼容 README 和已有命令，v12/certified 的最终 checkpoint 与教师缓存仍保留在 `outputs/` 顶层；其余文件按用途归类。

```text
outputs/
  checkpoints/
    stages/v12/
    stages/certified/
    archive/legacy/
    archive/v8/
    archive/v9/
    archive/v10/
    archive/v11/
  figures/
    current/
    archive/v8/
    archive/v9/
    archive/v10/
    archive/v11/
  logs/
    v12/
    archive/v8/
    archive/v9/
    archive/v10/
    archive/v11/
  archive/recoverable_cleanup_20260902/
  predictions/
  comparisons/
    current/
    reruns/
  paper/
    kang_sparse_reproduction/
  tmp/
  ARTIFACTS.md
```

`outputs/checkpoints/archive/` 保存 v8--v11 权重及对应 teacher；`outputs/figures/archive/` 保存旧版 PNG；`outputs/logs/archive/` 保存旧 JSON。大量文档直接引用 `outputs/candidate_pruning_one_shot_v12.pt`，所以该正式入口没有改名或移动。

## 3. 可恢复的 smoke 清理区

以下明确的临时连通性测试已经移入 `outputs/archive/recoverable_cleanup_20260902/`。本次没有永久删除，若发现误归档可直接移回原路径：

```text
v13_smoke*.pt
v13_stable_smoke*.pt
v13_smoke_teacher/
v13_stable_smoke_teacher/
_chord_strata_smoke/
_verified_batch_smoke/
_network_forward_timing_batch_smoke/
_network_forward_timing_strata_smoke/
_legacy_three_metrics_redraw_smoke.png
v12_chord_guard_smoke_8.json
v12_verified_smoke_8.json
v12_latency_smoke.json
v12_latency_training_smoke.json
v12_batch_K4_to_Kmax/
paper_sparse_reproduction_quick/
pdf_review_intermediates/
pdf_python_dependencies/
```

`candidate_pruning_certified*`、`candidate_pruning_one_shot_v12*` 和它们的 teacher 目录名称不含 `smoke`，绝不能被宽泛通配符误删。清理时必须使用上述精确路径并确认解析后的绝对路径位于仓库 `outputs` 内。

下列内容只归档、不删除：

- `outputs/checkpoints/archive/v10/candidate_pruning_v10_smoke*` 与 `outputs/checkpoints/archive/v10/teachers/v10_smoke_teacher/`：属于历史版本，保留在归档区。
- v8–v11 checkpoint、teacher、PNG 和 JSON：均作为历史消融证据归档，不与临时 smoke 一起删除。
- `docs/*.md`：当前文档各自承担架构、训练、部署、数据、延迟或展示说明，不按文件日期清理。

## 4. 文件命名约定

### Checkpoint

```text
{method}_{version}_{role}.pt
```

`role` 只使用 `proposal`、`distill`、`calibrated`、`best`、`last`。对外 release 另保存 manifest，不靠文件名记录全部超参数。

### Teacher cache

```text
teachers/{method}_{version}_{dataset_fingerprint}/train.pt
teachers/{method}_{version}_{dataset_fingerprint}/val.pt
```

不得单独重命名 `train.pt` 或 `val.pt`；数据配置、代码版本和 SHA256 写入同目录 `manifest.json`。

### 对比实验

```text
{version}_{methods}_{Kmin}-{Kmax}_n{samples_per_K}_seed{seed}/
```

目录内统一使用：

```text
config.json
per_sample_results.csv
per_k_summary.csv
summary.md
qualitative_sample_{sample_id}.png
aggregate_metrics.png
```

### 临时测试

所有临时产物必须写入 `outputs/tmp/`，文件或目录以 `_smoke_` 开头，并在测试完成后由精确路径清理。正式结果禁止包含 `smoke`、`test` 或 `latest`。

## 5. 发布前检查

1. `git status --short` 确认 checkpoint、教师和正式图没有遗漏。
2. 对 P0/P1 checkpoint 和 teacher 文件计算 SHA256。
3. 用正式 checkpoint 跑一次最小评估和一次单样本可视化。
4. 检查 JSON 中 checkpoint 路径、seed、MSE 定义和 timing scope。
5. 文档中的所有 `outputs/...` 路径均能解析。
6. 只有在新路径验证通过且引用全部更新后，才清理旧路径。

## 6. 真实数据接入报告（2026-09-04）

`outputs/real_world/` 保存 UJI、Natural Earth 和 USGS 的 learned 部署 JSON/CSV。
当前 `*_100` 文件是固定 seed 的 100 样本域外诊断，不是论文最终全 test 结果；正式结果应
使用对应 manifest 的完整 test split，并保留 checkpoint、manifest 与原始快照 SHA-256。
真实数据本体位于 `data/raw/`、`data/processed/` 和 `data/splits/`，受 `.gitignore`
保护，不随代码仓库再分发。
