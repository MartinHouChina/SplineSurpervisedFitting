# v11 部署、评估与可视化

## 1. 在线部署的真实计算

对一条有序点云，v11 默认执行：

```text
有序点云
  → 归一化/重采样
  → 网络 forward 一次
      参数 t
      固定 proposal U_prop
      p0 → u1 → p1 → KeepMask → U*
  → 取 U*[KeepMask]
  → 标准开放 B 样条控制顶点 refit 一次
  → 反归一化并输出曲线
```

部署不执行逐节点 Hard-RMS 试删、BIC 分支枚举、多阈值 sweep、第二次网络前向或数据相关迭代。网络内部的截断幂代理 solve 属于一次 `forward` 的固定计算图，不计作标准 B 样条部署 refit。

## 2. 最终节点与控制顶点

网络输出：

```text
proposal_internal_knots   [B,Kc]
deployment_internal_knots [B,Kc]
learned_keep_mask         [B,Kc]
```

最终内部节点为：

\[
U_b=U^*_{b,M_b}.
\]

对 `K_b` 个内部节点，开放三次 B 样条有 `K_b+4` 个控制顶点。部署求解：

\[
P^*=\arg\min_P
\|B(t,U_b)P-Q\|_F^2+
\lambda_s\|D_2P\|_F^2+
\lambda_r\|P\|_F^2,
\]

并强制拟合首尾端点。`fit_tolerance` 只用于报告当前结果是否满足阈值，不会在 one-shot 部署中再次修改 KeepMask。

## 3. 一次性节点数决策

默认 `mass_topk`：

令：

\[
q=\sum_jp_j+s\sqrt{\sum_jp_j(1-p_j)}.
\]

部署节点数为：

\[
\widehat K=
\begin{cases}
0,&q<0.5,\\
\lceil q\rceil,&q\ge0.5,
\end{cases}
\]

并截断到有效候选数。

选择器一次性取最高得分的 `Khat` 个槽位。默认：

```text
selection_policy = mass_topk
safety_sigma     = 0.25
coverage_bins    = 0
```

这些值保存在 checkpoint 中。正常部署推荐使用 `checkpoint` 策略，不在测试时调参：

```text
--one-shot-selection-policy checkpoint
```

运行时覆盖 `safety_sigma` 或 `coverage_bins` 适合做消融，但应在结果中明确标记，不能与 checkpoint 原生部署结果混在一起。

`u1` 和 `u*` 虽由两个位置头生成，但 `--one-shot-max-position-shift` 限制的是最终 `|u*-U_prop|` 合计位移；第二阶段会扣除第一阶段已经使用的预算。

## 4. 独立测试

训练、验证和独立测试的默认 seed 分别是 `42`、`10000` 和 `20000`。正式报告应使用独立测试 seed：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --num-samples 2000 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --knot-tolerance 0.05 `
  --json-output outputs/candidate_pruning_one_shot_v11_evaluation.json
```

评估脚本对每个样本只用 LearnedKeep 做一次标准 refit。JSON `schema_version=12`。

## 5. 指标如何阅读

### 5.1 proposal 覆盖

`candidate_proposal_metrics` 在 `0.005/0.01/0.02/0.05` 下统计固定 proposal 对 canonical 节点的 recall 和 matched MAE。

这是候选生成网络的指标。由于 `Kc` 故意冗余，proposal precision 通常不应作为最终结构质量结论。

### 5.2 三阶段节点匹配

`knot_stage_metrics` 使用同一组容差分别统计：

| 阶段 | 节点集合 | 回答的问题 |
|---|---|---|
| `proposal_full` | 全部 `U_prop` | 候选网络是否覆盖真节点 |
| `selected_pre_update` | `U_prop[mask]` | selector 是否保留正确槽位 |
| `deployment_post_update` | `U*[mask]` | 联合位置更新后的最终结果 |

每个阶段都报告全数据集聚合的 matched/predicted/true count、Precision、Recall、F1 和 matched MAE。

另外：

\[
\Delta R_{\mathrm{position}}
=R_{\mathrm{post}}-R_{\mathrm{pre}},
\]

\[
\Delta P_{\mathrm{position}}
=P_{\mathrm{post}}-P_{\mathrm{pre}}.
\]

正值表示位置更新在该容差下改善指标，负值表示位置更新使匹配变差。它们是诊断结果，不应预设一定为正。

### 5.3 最终部署匹配

顶层：

```text
knot_match_precision
knot_match_recall
knot_match_f1
matched_knot_mae
```

始终对应标准 B 样条部署实际使用的 `U*[mask]`，容差由 `--knot-tolerance` 指定。

节点匹配只有在共享参数化下才有直接意义。报告时应同时给出 `true_parameter_rmse`，避免把参数头误差完全归因于节点头。

### 5.4 拟合与复杂度

重点字段：

```text
one_shot_deployment.retained_count_mean/min/max
one_shot_deployment.refit_rms_mean_curve
one_shot_deployment.refit_rms_pooled
one_shot_deployment.refit_rms_p95
one_shot_deployment.refit_rms_max
one_shot_deployment.threshold_satisfied_fraction
standard_bspline_control_count_mean
```

`mean_curve RMS` 是逐曲线 RMS 的平均，`pooled RMS` 是先汇总 MSE 再开方；二者定义不同。

## 6. 与离线 Hard-RMS 对比

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --num-samples 2000 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --run-hard-diagnostic `
  --json-output outputs/candidate_pruning_one_shot_v11_hard_diagnostic.json
```

Hard-RMS 从固定 `proposal_internal_knots` 开始，反复试删和标准 refit。它用于衡量 LearnedKeep 与离线教师的节点数/RMS 差距、检查 proposal 可行性，以及生成速度/精度对照。

它不使用 v11 联动后的 `U*`，也不会替换默认 one-shot 部署。其计算量远大于一次 refit。

## 7. 单样本可视化

仅看 LearnedKeep：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --seed 20000 `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view learned `
  --dpi 600 `
  --output outputs/v11_learned_000.png
```

四联对比：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --seed 20000 `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view comparison `
  --timing-repeats 5 `
  --dpi 600 `
  --output outputs/v11_comparison_000.png
```

四个面板依次是：

1. 源样条/原始数据；
2. 全部固定 proposal 的冗余拟合；
3. `U*[LearnedKeep]` 的一次性部署；
4. 从固定 proposal 出发的离线 Hard-RMS 删除。

每个拟合面板显示采样点、曲线、控制多边形、内部节点、完整节点向量、节点数、RMS 和时间。

注意：第二幅图是网络生成的全部 proposal 重新拟合，不是严格的 Boehm 等价节点插入。

## 8. 图中时间

- `net`：一次网络前向，包括参数、proposal、两次 Keep 概率和两次固定位置更新。
- `refit`：节点确定后，一次标准 B 样条控制顶点重拟合。
- `prune`：Hard-RMS 的多轮试删和重复标准 refit。

\[
T_{\mathrm{Learned}}=T_{\mathrm{net}}+T_{\mathrm{refit}},
\]

\[
T_{\mathrm{Hard}}=T_{\mathrm{net}}+T_{\mathrm{prune}}.
\]

`--timing-repeats 5` 报告多次 wall-clock 的中位数，不含绘图和 PNG 保存。

## 9. 批量分层生成论文图

`scripts/visualize_batch_comparison.py` 用固定规则抽取多个样本，统一生成四联 PNG 和机器可读清单。推荐命令：

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --output-dir outputs/v11_batch_comparison `
  --num-figures 10 `
  --scan-size 512 `
  --seed 20000 `
  --stratify-by source `
  --knot-counts 4 8 12 16 20 `
  --mse-tolerance 2.5e-5 `
  --selection-seed 12345 `
  --dpi 600 `
  --timing-repeats 5
```

### 9.1 四个面板的节点语义

| 面板 | 节点与参数 | 作用 |
|---|---|---|
| `(a) Original source` | 数据生成时的 source 节点、控制顶点和真参数 | 展示原始样条与带噪观测 |
| `(b) All proposal` | 全部固定 `proposal_internal_knots`，使用预测参数做一次标准 refit | 展示冗余固定候选的整体 refit |
| `(c) Learned deployment` | `deployment_internal_knots[learned_keep_mask]`，使用预测参数做一次标准 refit | v11 实际一次性节点组合与位置 |
| `(d) Traditional hard` | 从与 `(b)` 完全相同的 `proposal_internal_knots` 开始 | 传统贪心单节点删除对照 |

Hard 对照每轮计算所有当前单节点删除方案，选择 refit MSE 最小的一项；仅当该 MSE 不超过阈值时接受，然后继续下一轮。它是在同一 proposal 上运行的传统离线贪心删除，会重复标准 B 样条 refit，不是 LearnedKeep 的在线部署步骤，也不使用联动后的 deployment 位置。

### 9.2 MSE 定义

批量脚本统一报告归一化坐标中的平均平方欧氏误差：

\[
\operatorname{MSE}
=\frac1M\sum_{i=0}^{M-1}
\|C(t_i)-Q_i\|_2^2.
\]

这里对每个点先将各坐标平方误差求和，再对点取平均，最后不开平方。`--mse-tolerance 2.5e-5` 对应历史 RMS 阈值 `0.005` 的平方。若省略该参数，脚本会读取 checkpoint 中的 RMS tolerance 并自动平方。

source 面板使用真参数；all-proposal、Learned 和 Hard 使用网络预测参数。因此 source MSE 与后三者还包含不同参数化的影响，不能只按该数值解释节点选择质量。单样本 `visualize_result.py` 原有标题使用 RMS，不能把其数值与这里的 MSE 直接混写。

### 9.3 source/canonical 分层随机抽样

脚本先扫描 `[0, scan_size)` 的确定性测试样本，并按以下标签分组：

- `--stratify-by source`：按生成样条的 source 内部节点数；
- `--stratify-by canonical`：按阈值删除后的 canonical 内部节点数。

`--knot-counts` 限定允许抽取的层；各层使用同一个 `--selection-seed` 可复现地打乱，再按随机轮转方式均衡取样，且样本索引不重复。若指定层在扫描池中不存在，脚本会报错；此时应增大 `--scan-size` 或调整节点数列表。

例如要按 canonical 数量抽样，可将推荐命令中的两行替换为：

```powershell
  --stratify-by canonical `
  --knot-counts 3 6 9 12 `
```

固定 `--seed`、`--scan-size`、`--stratify-by`、`--knot-counts` 和 `--selection-seed` 后，选图规则可复现，避免先看结果再手工挑选。

### 9.4 输出文件

`--output-dir` 下保存：

```text
comparison_001_sourceK.._canonicalK.._sample....png
comparison_002_sourceK.._canonicalK.._sample....png
...
comparison_manifest.json
comparison_manifest.csv
```

每个 PNG 同时绘制观测点、曲线、控制多边形、内部节点 `C(u)`、节点向量、K、MSE 和 CPU 时间。JSON 保留完整配置、抽样索引、四种节点向量、MSE、Hard 删除轨迹和时间；CSV 将同一信息展平成表格，数组字段以 JSON 字符串保存。

计时固定为 CPU、batch size 1，并报告 `--timing-repeats` 次 wall-clock 的中位数；网络前向时间在 all/Learned/Hard 三个面板间共享。重新写入已有 manifest 或 PNG 时需显式添加 `--overwrite`。当前批量 PNG 仅支持二维 checkpoint。

## 10. 用户点云部署

输入是沿曲线方向排列的坐标，每行一个点：

```csv
x,y
x,y
...
```

运行：

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --point-cloud data/my_curve.csv `
  --fit-tolerance 0.005 `
  --json-output outputs/my_curve_v11.json `
  --figure-output outputs/my_curve_v11.png
```

脚本会检查输入、按训练长度重采样、归一化、网络前向一次、把参数插值回源点分辨率、在全部源点上做一次标准 B 样条 refit，最后反归一化控制顶点和曲线。

`--fit-tolerance` 对 v11 仍只负责报告是否满足阈值。若用户数据与合成训练分布差异较大，满足率和节点匹配不能由训练集结果外推保证。

## 11. v10 兼容

v10 checkpoint 可以继续直接传给评估、可视化和点云脚本。恢复时：

- Gaussian 局部带宽回退到 `0`；
- 联合位置更新关闭；
- proposal 和 deployment 位置保持 v10 的固定语义；
- 仍执行一次 LearnedKeep 和一次标准 refit。

若要训练 v11，应创建新的 v11 teacher cache；旧 v10 cache 不能直接复用。
