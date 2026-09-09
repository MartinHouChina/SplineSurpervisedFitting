# K=4–20：verified 部署与贪心剪枝对比

> v16 的独立八方法入口见 [v16 部署与评估](../deployment_pipeline.md)，不要把本页 v12 数值迁移到 v16。

> 历史结果说明：本页数值来自旧的 shared-proposal 实验，Greedy 读取了网络候选，因此只属于候选集消融，不再作为独立传统基线。当前脚本的 Ours 主时间已改为纯 `forward_deployment()`；正式的 Kang 稀疏法与“均匀最大节点→删除→梯度重定位”对比请运行 `scripts/compare_knot_methods.py`。

## 实验目标

在源内部节点数 `K=4,5,...,20` 下，对比：

- **Ours verified**：一次性网络给出参数、冗余 proposal、KeepMask 和存活节点重定位；随后用精确标准 B 样条 refit 校验，并按需执行置信度补回和紧凑化。
- **Greedy hard**：从同一份已物化 proposal 出发，逐节点尝试删除并执行精确 refit。

拟合约束为归一化坐标中的

\[
\operatorname{MSE}=\frac{1}{M}\sum_i\lVert C(t_i)-Q_i\rVert_2^2
\le 2.5\times10^{-5},
\]

等价于 `RMS <= 0.005`。

## 实验设置

| 项目 | 设置 |
|---|---:|
| checkpoint | `outputs/candidate_pruning_one_shot_v12.pt` |
| 独立测试 seed | 20000 |
| 源内部节点数 | 4–20 |
| 样本数 | 每个 K 20 条，共 340 条 |
| 每条曲线采样点 | 192 |
| 网络冗余候选数 | 28 |
| 参数化 | Ours 与 Hard 共用弦长参数化及映射后的同一 proposal |
| bootstrap | 5000 次逐曲线重采样，95% CI |
| 本次展示计时 | 每条曲线 1 次，无预热 |

`canonical reference K` 是在真实参数域和同一阈值下得到的贪心简化参考，不是全局最少节点证明。节点匹配时，最终节点先依据采样点对应关系映射回真实参数域，不能跨参数域直接比较。

## 340 条曲线结果

| 方法 | mean K | canonical bias / MAE | mean / P95 MSE | 通过率 | P / R / F1 @0.05 | 中位时间 |
|---|---:|---:|---:|---:|---:|---:|
| Ours verified | 8.132 | +0.703 / 1.309 | 2.082e-5 / 2.483e-5 | 100.0% | 0.625 / 0.684 / 0.653 | 61.80 ms |
| Greedy hard | 7.324 | -0.106 / 0.859 | 1.760e-5 / 2.442e-5 | 100.0% | 0.747 / 0.736 / 0.742 | 206.30 ms |

Ours 的纯 learned 子集有 `42.4%` 已经满足阈值；其余 `57.6%` 触发 proposal 内的置信度补回。由于本实验开启 `--verified-compact`，全部样本在达到阈值后都继续执行精确紧凑化。340 条曲线均未触发动态残差插点或 hard fallback。

结论边界：verified 路径把通过率提高到了 100%，并把平均节点数降到接近 canonical 的 8.132；但它是自适应精确修复，不能称为纯 one-shot。当前节点位置匹配和最终节点数仍略逊于 Greedy hard，后续应以 v13 重新训练 KeepMask 与 survivor relocation，而不是继续修改展示数据。

## 时间口径

- Ours：完整网络 forward，加精确校验、条件补回、紧凑化和最终 refit。
- Hard：参数和 proposal 已物化后的 greedy pruning stage only，包含其内部全部 refit，排除网络 forward。

两者是按既定需求统计的非对称范围，因此 `206.30 / 61.80` 不能写成严格同边界的端到端加速比。本次图用于展示；论文正式计时建议在固定硬件上改为 `--timing-repeats 5 --warmup-repeats 1` 重跑。

## 复现命令

```powershell
python scripts/evaluate_knot_count_strata.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/current/v12_verified_chord_compact_K4_20_n20 `
  --samples-per-knot-count 20 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --scan-size 1024 `
  --seed 20000 `
  --selection-seed 12345 `
  --execution-seed 67890 `
  --timing-repeats 1 `
  --warmup-repeats 0 `
  --bootstrap-replicates 5000 `
  --bootstrap-seed 24680 `
  --mse-tolerance 2.5e-5 `
  --ours-deployment verified `
  --verified-parameterization chord `
  --verified-compact `
  --verified-hard-fallback `
  --verified-residual-fallback `
  --verified-max-residual-insertions 8 `
  --verified-residual-min-gap 0.001 `
  --verified-refit-device cpu `
  --smoothness-weight 1e-6 `
  --torch-num-threads 4 `
  --dpi 600 `
  --overwrite
```

结果见 `outputs/comparisons/current/v12_verified_chord_compact_K4_20_n20/`：PNG、逐样本 CSV、逐 K CSV、完整 JSON 和自动生成的 Markdown 摘要。
