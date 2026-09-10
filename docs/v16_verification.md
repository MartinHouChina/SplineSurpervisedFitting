# v16 supervised-only 验证清单

## 1. 当前合同

正式 checkpoint 应记录：

```text
objective_version = candidate_selection_supervised_bspline_v16
architecture_revision = v16_supervised_ordered_assignment_mass_topk
simplification_contract = synthetic_ground_truth_ordered_keep_and_relocation_v4
checkpoint_selection = mean_per_curve_subset_cost_v1
checkpoint_quality = supervised_fit_count_selected
qualification_contract = v16_supervised_synthetic_only_pass_rates_report_only_v4
```

同时满足：`real_fraction=0`、`joint_supervision=synthetic_ground_truth`、`synthetic_count_role=exact`、`online_teacher=false`、`ranked_prefix_teacher=false`、Kc=56、source K=4..56、`knot_min_span=0.01`、最终 safety knots=0。

## 2. 运行前检查

- Python 环境可以导入 PyTorch、NumPy、Matplotlib；
- CUDA 运行时可识别 RTX 3090；
- 三个真实 manifest 存在且指向各自留出 split；
- 新 `RunName` 对应的 checkpoint/log/comparison/figure 路径均不存在；
- 训练与验证 synthetic seed 范围不重叠；
- `Kc>=max source K` 且 K56 boundary validation 至少 32 条。

先打印一条龙命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName v16_supervised_dryrun -DryRun
```

## 3. 单元测试

```powershell
python -m pytest -q
```

重点应覆盖：

- ordered one-to-one assignment 在 `Kc>K*` 与 `Kc=K*` 下的行为；
- supervised Joint 拒绝无标签或真实训练行；
- online Teacher 分支不进入正式 checkpoint 合同；
- mass-TopK 数量和 selected-only 重定位；
- checkpoint v4 合同及 pass-rate-report-only 语义；
- 六方法 benchmark、两张 2×2 图和真实案例图的接口。

只有实际测试输出才能写入验证记录；本文件不预设通过数量。

## 4. 训练后结构审计

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --mse-tolerance 1e-4
```

检查重点：

- `stage=joint` 且 simplification curriculum 已成熟；
- synthetic-only、真值监督和无 online Teacher 字段一致；
- Proposal 高 K 比例 0.5、起点 K=40、有序 assignment 权重大于 0；
- synthetic minimality、K56 boundary audit、容量和 MSE 阈值一致；
- checkpoint selection/quality 与当前合同一致。

aggregate/worst-source pass、count MAE、节点 F1 和 retained K 均应显示，但只作为实验诊断，不决定结构资格。

## 5. 一条龙产物检查

成功运行后至少应有：

| 类别 | 必需产物 |
|---|---|
| 模型 | best `.pt`、`.proposal.pt`、`.last.pt`、`.history.json` |
| 审计 | `inspect_checkpoint.log` 与 pipeline manifest 状态 |
| 表格 | `comparison.json`、`summary.csv`、`measurements.csv`、`report.md` |
| 指标图 | `v16_published_methods_input.png`、`v16_published_methods_reference.png` |
| 案例图 | UJI/Natural Earth/USGS 的六方法 3×2 PNG 与对应记录 |

检查 `comparison.json` 的 fingerprint、checkpoint SHA-256、数据样本 ID 和方法集合一致；图必须从该 JSON 生成，不手工调整数值。

## 6. 结果审查

- MSE 是 mean squared Euclidean，不是 RMS；
- pass 分母包含失败；
- final K 是最终 refit 的内部节点数；
- total time 与 network-only time 不混用；
- 真实 reference 指标只用于评估，不进入部署；
- K=56 boundary 单独报告；
- 五个论文基线标注 adaptation；
- 未跑完的格子写 N/A 或明确失败，不做推测补值。

## 7. 诊断模式

结构审计失败时，一条龙默认停止，不生成正式表图。只有排错时才在单独命令中添加 `--allow-unqualified-diagnostic`；所有产物必须保留 `DIAGNOSTIC NOT FINAL` 标识，不能用于正式汇报。
