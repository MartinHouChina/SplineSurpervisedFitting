# 输出产物规则

本文件只规定目录和保护规则，不记录容易过期的“当前 epoch”快照。训练状态以对应
`.history.json` 和 `scripts/inspect_v16_checkpoint.py` 的实测输出为准。

## 目录

```text
outputs/
  checkpoints/       模型、阶段最佳和恢复点
    archive/          历史权重，只读保留
  comparisons/       定量比较、逐样本记录和汇总图
  figures/           论文/PPT 图；方法指标对比和 Ours 拟合案例
  fits/v16/          用户点云部署
  paper/             论文案例
  logs/              非正式诊断日志
  archive/           可恢复历史清理
  tmp/               可再生成 smoke
```

## 同一训练实验的四类文件

```text
experiment.pt             最佳 Joint 模型，正式入口候选
experiment.proposal.pt    最佳稠密候选生成器
experiment.last.pt        最新模型、优化器和 RNG，供 resume
experiment.history.json   每轮训练与分来源验证记录
```

它们不是重复副本。训练运行时四者都不得移动或删除；正式 benchmark 只能读取通过资格
检查的主 `.pt`。

## 正式 v16 资格

文件名不代表资格。必须运行：

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/EXPERIMENT.pt `
  --required-pass-rate 0.90 `
  --mse-tolerance 2.5e-5
```

返回码 0 才能进入正式对比。检查内容包括：

- Joint 阶段和当前 objective/architecture revision；
- adaptive beta + mass_topk；
- certified synthetic source 数据合同及 MSE→RMS 证书阈值；
- worst-source deployment pass 不低于 90%；
- 不是全部候选保留的退化解；
- proposal gate、消融开关和保存字段一致。

旧固定阈值、未认证合成源、proposal-only、未达标或全保留模型只能作为诊断或 warm-start。

## 必须保留

- 正在训练或可能恢复的 `.pt/.proposal.pt/.last.pt/.history.json` 完整链；
- v15 兼容测试需要的权重；
- `comparisons/` 中已被报告引用的正式 JSON、CSV、PNG；
- `data/raw`、`data/processed`、`data/splits`；
- `archive/`、`checkpoints/archive/` 中的历史证据；
- 工作区唯一论文副本和 `docs/figures/` 当前架构图。

## 可清理

- `__pycache__`、`.pytest_cache`、`.codex_tmp`；
- `outputs/test_tmp*`、`outputs/tmp`；
- 名称明确含 `smoke` 且没有文档引用的日志、图片和临时 checkpoint。

优先移入回收站。不要按日期、版本号或通配符批量删除 checkpoint 和正式结果。

## 命名

- checkpoint：`method_version_role.pt`；
- 正式比较：`version_methods_Kmin-Kmax_nN_seedS/`；
- 临时验证：`outputs/tmp/_smoke_name/`。

每个正式比较结果至少保存 checkpoint/manifest SHA-256、test seed、样本 ID、容量、停止
条件、MSE 定义、network/full-method 时间边界、失败样本和 qualification 状态。

## 正式绘图产物

方法指标图由 `scripts/plot_v16_method_comparison.py` 从完整的
`outputs/comparisons/<experiment>/comparison.json` 只读生成。默认的 published
集合包含 Ours、Park、Liang、Dung、Kang 和 Luo；`--method-set all` 另外加入 Yeh
和统一贪心。正式目录建议为：

```text
outputs/figures/<experiment>/method_comparison/
  v16_published_methods_input.png
  v16_published_methods_reference.png
```

Ours 案例由 `scripts/visualize_v16_ours_cases.py` 生成，建议目录为：

```text
outputs/figures/<experiment>/ours_cases/
  ours__<dataset>__<sample-id>.png
  ours_cases_overview.png
  deployment_visualizations.json
```

单例 PNG 和 JSON 必须成套保留；JSON 是内部节点参数、完整节点向量、控制顶点、MSE
和时间的数值依据，PNG 不能手工挪动节点、控制顶点或曲线。推荐正式权重路径为
`outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt`，但该文件需实际
训练并通过资格检查，文件名本身不构成正式证据。未合格 checkpoint 或 benchmark 报告
只可增加 `--allow-unqualified-diagnostic` 生成带 `DIAGNOSTIC NOT FINAL` 水印的诊断产物，
不得改名移入正式目录。
