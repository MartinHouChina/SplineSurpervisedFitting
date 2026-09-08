# v13 PPT 展示速查

本文给出一套可以直接用于组会或答辩的 8 页叙述顺序。所有结论应以实际 checkpoint 的
独立测试输出为准；不要在结果尚未跑完时预填性能提升。

只需复现两类对比图时，直接查看[对比图运行速查](comparison_plots_quickstart.md)。其中分别说明
K=4–20 四栏几何个例与 MSE/最终 K/时间三项统计图的输出、MSE 和计时边界；旧 2×2 图仍作为附加输出。

## 第 1 页：问题与目标

标题建议：**阈值约束下的最小复杂度 B 样条拟合**。

展示输入和目标：

```text
输入：沿曲线方向排序的二维/三维点云
输出：参数 t、内部节点 U、控制顶点 P 和最终 B 样条
```

核心目标：

\[
\min |U|\quad\text{s.t.}\quad
\operatorname{RMS}(C_U,Q)\le\varepsilon.
\]

一句话说明困难：删除一个节点会同时改变其余节点的合理位置和控制顶点，因此“节点数量”
与“节点位置”不能当成两个完全独立的问题。

## 第 2 页：数据集与独立划分

默认合成数据为开放三次 B 样条：

| 项目 | 默认设置 |
|---|---:|
| train / validation / test seed | 42 / 10000 / 20000 |
| train / validation 样本数 | 10000 / 2000 |
| 每条曲线观测点 | 192 |
| 源内部节点 K | 4–20 |
| 网络冗余候选 Kc | 28 |
| 坐标噪声标准差 | 0.001 |
| 归一化 RMS 目标 | 0.005 |

说明 source 节点、canonical 阈值简化标签和 teacher survivor 标签是三个不同概念：

- source：生成曲线时的原始表示；
- canonical：用于 proposal 几何覆盖监督的阈值简化表示；
- teacher survivor：在网络固定 proposal 上 delete-then-relax 得到的可部署组合与位置。

定性图使用 test seed `20000`，不要用训练集图片替代独立测试。

## 第 3 页：为什么采用冗余候选再消冗

建议画如下数据流：

```text
有序点云
  -> ParameterHead
  -> 高召回冗余 proposal
  -> 一次性 KeepMask
  -> 存活节点重定位
  -> 标准 B 样条 refit
```

强调：当前没有 `CountHead`。候选头先提高召回，选择头从固定长度候选中一次性决定组合；
最终节点数量就是 KeepMask 的元素数。

## 第 4 页：v12/v13 一次性网络

重点展示删除与位置更新的联动：

```text
p0
  -> provisional position feedback
  -> p1
  -> final hard KeepMask
  -> 按最终 survivors 重算邻居、rank、count、覆盖特征
  -> selected-only multi-head attention
  -> 有界、有序的 U_deploy
```

Key/Value 只来自最终存活节点；被删除槽位不能继续影响 survivor relocation。位置输出满足
参数域端点、`min_gap`、存活邻居顺序和相对 proposal 的最大位移。整个 learned 路径仍是
一次固定深度 forward，不是逐节点在线搜索。

## 第 5 页：离线教师与训练阶段

教师流程：

```text
固定 proposal
  -> greedy Hard-MSE 删除
  -> survivor coordinate relaxation
  -> retry delete
  -> 最终 relaxation
  -> 缓存 mask、优化后位置、gap、risk 和 count
```

训练分为：

1. proposal 预训练或载入 v12 高召回 proposal；
2. 离线生成或严格复用 delete-then-relax teacher cache；
3. 蒸馏一次性 KeepMask；
4. 低学习率联合校准 KeepMask 与 survivor relocation。

v13 用实际预测 survivors 与 relocated teacher set 做有序匹配；只有数量相等时才监督完整
gap，并增加槽位无关的 soft set coverage。校准默认关闭截断幂 surrogate，最终在 distill 与
calibrated checkpoint 中按真实标准 B 样条验证指标选优。

## 第 6 页：K=4 到上限的四联定性图

运行：

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/current/v12_verified_four_panel_K4_20 `
  --samples-per-knot-count 1 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --scan-size 512 `
  --seed 20000 `
  --selection-seed 12345 `
  --stratify-by source `
  --mse-tolerance 2.5e-5 `
  --ours-deployment verified `
  --verified-parameterization chord `
  --verified-compact `
  --timing-repeats 3 `
  --timing-warmups 1 `
  --timing-scope end-to-end `
  --dpi 300
```

省略 `--max-knot-count` 会自动取数据集上限；默认生成 `K=4–20` 每层一张。每层两张时
改成 `--samples-per-knot-count 2`。输出目录包含 PNG、
`comparison_manifest.json` 和 `comparison_manifest.csv`。

四栏口径必须按下面顺序讲：

1. **Original**：源 B 样条、原始观测点、源控制多边形和源节点；
2. **Redundant proposal**：网络预测的全部高召回候选及标准 refit；
3. **Ours**：当前图使用 adaptive verified，即 KeepMask/relocation 后执行精确校验与条件修复；
4. **Hard**：从同一份已物化 proposal 出发的传统 greedy hard pruning。

第 2～4 栏共用弦长参数域和映射后的同一 proposal。若要画纯一次性网络消融，将命令改为
`--ours-deployment learned`，且不要把它与 verified 结果混称为同一种部署。

上面的定性图中 Ours 只显示纯 `forward_deployment()` 时间；Hard 的 `--timing-scope`
决定显示 pruning-only 还是带 proposal 物化的诊断时间。两者边界不同，不能写成公平
end-to-end 加速比。
若需固定重画一张或一批已选个例，可改用 `--sample-indices <id...>`，并去掉所有 K 分层抽样参数。

图例按几何对象解释即可：观测点、拟合/源曲线、控制顶点与控制多边形、曲线上的内部节点
`C(u)`；标题和底部文本给出 K、MSE、时间与节点向量。

特别注意：第 2 栏是 network-proposed redundant candidates，不是 Boehm 精确节点插入，
不能写成“保持原曲线完全不变的插节点结果”。

## 第 7 页：定量指标与计时

独立批量评估（质量优先）：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --deployment-mode verified `
  --verified-parameterization chord `
  --verified-compact `
  --verified-refit-device auto `
  --num-samples 128 `
  --batch-size 16 `
  --torch-num-threads 4 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --knot-tolerance 0.05 `
  --json-output outputs/logs/v12/v12_verified_chord_compact_128.json
```

当前 340 条分层测试结果可直接使用：

| 方法 | mean K | mean / P95 MSE | pass rate | Precision | Recall | F1@0.05 | median time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours verified-compact | 8.132 | 2.082e-5 / 2.483e-5 | 100.0% | 0.625 | 0.684 | 0.653 | 61.80 ms |
| Greedy hard | 7.324 | 1.760e-5 / 2.442e-5 | 100.0% | 0.747 | 0.736 | 0.742 | 206.30 ms |

四联图中的 MSE 定义为

\[
\operatorname{MSE}=\frac1M\sum_i\|C(t_i)-Q_i\|_2^2,
\]

在归一化坐标中计算且不取平方根。节点匹配的 `--knot-tolerance 0.01` 与拟合阈值不是
同一个量。

计时按本次实验指定的非对称边界：

- `ours network forward`：只包含 ParameterHead + proposal + KeepMask + relocation，
  排除最终 refit、verified repair、数据搬运和 I/O；
- `hard pruning`：参数和 proposal 已经物化后的 greedy pruning stage only，包含剪枝内部
  全部 refit，排除网络 forward。

因此只能表述为“纯网络预测耗时”和“给定 proposal 后的 hard 剪枝耗时”。
不要把二者直接写成同边界端到端加速比。teacher cache 生成是离线训练成本，不计入 learned
部署时间。

当前 v12 checkpoint 的 340 条、`K=4–20` 独立分层实验结果为：`verified-chord-compact`
通过率 `340/340`、mean/P95 MSE `2.082e-5 / 2.483e-5`、mean K `8.132`，
`P/R/F1@0.05 = 0.625/0.684/0.653`。纯 learned 子集有 `42.4%` 已满足阈值，其余样本在
固定 proposal 内做置信度补回；没有触发动态残差插点或 hard fallback。完整 Ours 路径中位
时间 `61.80 ms/curve`。本次展示计时仅重复 1 次，论文正式计时应固定硬件并增加预热和重复次数。

## 第 8 页：结论、限制与下一步

可以陈述的结构性结论：

- v13 完整训练最终 KeepMask、位置反馈和剩余节点重定位交互链路；
- 离线教师显式产生删除后的优化位置和 survivor gap 标签；
- learned 部署仍保持一次 forward + 一次标准 refit；
- verified 用精确 refit 只修复失败曲线，并把动态残差插点单独报告；
- 四联实验用同一 proposal 对照一次性部署与传统 hard pruning。

不要过度声称：

- pure learned 不保证每个样本都满足阈值；verified 也不保证全局最少节点；
- high proposal recall 不自动等于 high deployment precision；
- straight-through gate 是近似梯度，不是精确离散优化；
- synthetic test 的泛化结论不能直接外推到真实点云；
- 节点 matching 依赖参数化，必须同时报告参数误差；
- hybrid 是额外的慢速 beam/coordinate 搜索，不是 v12 一次性网络的一部分。

建议最后给出两个后续实验：`min_gap` 消融，以及 learned/verified/hard/hybrid 在统一端到端计时边界
下的质量—复杂度—时间 Pareto 对照。
