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
fine_grained_teacher = certified_source_single_deletion_mse
fine_teacher_error_unit = mean_squared_euclidean
fine_teacher_additional_spline_solves_per_batch = 0
```

同时满足：`real_fraction=0`、`joint_supervision=synthetic_ground_truth`、`synthetic_count_role=exact`、`online_teacher=false`、`ranked_prefix_teacher=false`、source K=4..56、`Kc=72`、候选冗余 16、全保留节点向量长度 80、`knot_min_span=0.01`、最终 safety knots=0。`fine_teacher_weight` 和 `fine_teacher_ranking_weight` 应为正；这表示启用认证删除敏感度监督，不表示启用 online teacher。

## 2. 运行前检查

- Python 环境可以导入 PyTorch、NumPy、Matplotlib；
- CUDA 运行时可识别 RTX 3090；
- UJI、Natural Earth、USGS、IndustrialOffset 四个外部 manifest 存在且指向各自留出 split；
- 新 `RunName` 对应的 checkpoint/log/comparison/figure 路径均不存在；
- 训练与验证 synthetic seed 范围不重叠；
- `Kc=72>max source K=56` 且 K56 boundary validation 至少 32 条；高 K 样本过采样不能替代这 16 个候选容量冗余。

先打印一条龙命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName v16_supervised_dryrun -DryRun
```

Linux 对应检查：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --run-name v16_supervised_dryrun_linux \
  --dry-run
```

## 3. 单元测试

```powershell
python -m pytest -q
```

重点应覆盖：

- ordered one-to-one assignment 在当前 `Kc>K*` 下的行为，以及通用 `Kc=K*` 兼容边界；
- supervised Joint 拒绝无标签或真实训练行；
- certified Synthetic 输出逐节点删除 MSE 和完全一致的有效 mask；
- 删除 MSE 按有序一一匹配准确搬到正候选槽位，且 RMS/MSE 单位不会混用；
- 多尺度 Proposal recall、Keep Dice/CDF、fuzzy negative、parameter log-gap/bias 均为有限值并产生预期梯度；
- structure logits 对 `beta` stop-gradient、count logits 对 candidate importance stop-gradient；两者数值必须与部署 logits 相同，且排序/计数损失不能串改错误分支；
- warp gradient scale 为 `0` 时阻断对应跨任务梯度，为 `0.1` 时只按比例回传且不改变 forward 值；
- 新训练 `relocation_blend` 从 `0.03` 初始化并有非零梯度；旧 checkpoint 加载后仍精确恢复保存值；
- Joint 前 8 代只训练 Selector/decoder，之后按四组学习率解冻；history 的 phase 与 optimizer group LR 必须与实际 `requires_grad` 一致；
- online Teacher 分支不进入正式 checkpoint 合同；
- mass-TopK 数量和 selected-only 重定位；
- checkpoint v4 合同及 pass-rate-report-only 语义；
- 六方法 benchmark、两张 2×2 图和真实案例图的接口。

只有实际测试输出才能写入验证记录；本文件不预设通过数量。

## 4. 训练后结构审计

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised.pt `
  --mse-tolerance 1e-4
```

检查重点：

- `stage=joint` 且 simplification curriculum 已成熟；
- synthetic-only、真值监督和无 online Teacher 字段一致；
- `fine_grained_teacher` 来源、MSE 单位和额外在线 solve 数正确；
- Proposal 高 K 比例 0.5、起点 K=40、有序 assignment 与多尺度 recall 权重大于 0；
- `candidate_knots=72`、source max K=56、冗余槽位为 16，且全保留完整节点向量长度为 80；
- Selector warmup 为 8，并记录 Selector、Proposal、Parameter、decoder 四组 Joint 学习率；
- Keep Dice/CDF、fine-teacher、parameter gap/bias 权重大于 0，warp-gradient scale 与运行命令一致；
- Keep 排序/计数梯度解耦已启用，fine-teacher 风险具有非零标准差/范围而不是全体饱和；重定位初始化为 `0.03`；
- synthetic minimality、K56 boundary audit、容量和 MSE 阈值一致；
- `.proposal.pt` 是验证最佳 Proposal 初始化、`.proposal.final.pt` 是阶段末审计状态、`.pt` 是最佳成熟 Joint、`.last.pt` 是最新可恢复状态；epoch 与 checkpoint selection/quality 均与各自职责一致。

aggregate/worst-source pass、count MAE、节点 F1 和 retained K 均应显示，但只作为实验诊断，不决定结构资格。history 还应包含 Proposal `R@.005/.01/.02`、Keep P/R/F1、`critical_false_delete_rate`、`fine_teacher_mean_risk/std/range`、`fine_teacher_mean_log_delete_margin`、`fuzzy_negative_fraction`、`parameter_bias_mae`、`parameter_gap_loss`、`training_phase` 和当前 `learning_rates` 分组学习率。

## 5. 一条龙产物检查

成功运行后至少应有：

| 类别 | 必需产物 |
|---|---|
| 模型 | 最佳 Joint `.pt`、最佳初始化 `.proposal.pt`、阶段末审计 `.proposal.final.pt`、最新恢复状态 `.last.pt`、逐代 `.history.json` |
| 审计 | `inspect_checkpoint.log` 与 pipeline manifest 状态 |
| 表格 | `comparison.json`、`summary.csv`、`measurements.csv`、`report.md` |
| 指标图 | `v16_published_methods_input.png`、`v16_published_methods_reference.png` |
| 案例图 | UJI/Natural Earth/USGS/IndustrialOffset 的六方法 3×2 PNG 与对应记录 |

检查 `comparison.json` 的 fingerprint、checkpoint SHA-256、数据样本 ID 和方法集合一致；图必须从该 JSON 生成，不手工调整数值。

## 6. 结果审查

- MSE 是 mean squared Euclidean，不是 RMS；
- pass 分母包含失败；
- final K 是最终 refit 的内部节点数；
- total time 与 network-only time 不混用；
- 真实 reference 指标只用于评估，不进入部署；
- K=56 boundary 单独报告；它现在有 16 个候选冗余槽位，不再是 Proposal 容量等于标签数的无余量边界；
- 五个论文基线标注 adaptation；
- 未跑完的格子写 N/A 或明确失败，不做推测补值。
- Dung/Kang/Luo 的保护后结果必须标注 `threshold-safe adaptation`，并保留保护前坍缩诊断；不能把保护层结果写成作者原始算法结果。
- 新损失是否提升 recall、筛选准确率、参数偏差或最终通过率，必须比较重训 checkpoint 与固定旧 checkpoint 才能判断；代码存在和单元测试通过不等于性能已经提高。

## 7. 诊断模式

结构审计失败时，一条龙默认停止，不生成正式表图。只有排错时才在单独命令中添加 `--allow-unqualified-diagnostic`；所有产物必须保留 `DIAGNOSTIC NOT FINAL` 标识，不能用于正式汇报。
