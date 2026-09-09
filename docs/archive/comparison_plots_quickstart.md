# 对比图运行速查

本文只说明两类论文/PPT 对比图的运行与口径。命令均在仓库根目录的 PowerShell 中执行，不修改模型或 checkpoint。

> 本页原命令主要服务 v12–v15 历史图。v16 的八方法量化对比和真实曲线四宫格请优先使用
> [v16 部署与评估](../deployment_pipeline.md) 中的入口。

## 1. 两类图分别展示什么

| 图 | 脚本 | 输出内容 |
|---|---|---|
| **K=4–20 四栏几何个例** | `scripts/visualize_batch_comparison.py` | 每个 source K 输出若干张 PNG；每张依次展示 Original、Redundant proposal、Ours、Greedy hard 的曲线、观测点、控制多边形、控制顶点和内部节点 |
| **K=4–20 三项分层统计图** | `scripts/evaluate_knot_count_strata.py` | 主输出 `stratified_three_metrics.png`，依次比较 refit MSE、最终内部节点数和每曲线时间；原 2×2 `stratified_comparison.png` 仍作为附加输出 |

第一类回答“某条曲线具体拟合成什么样”，第二类回答“在 K=4–20 的全部分层样本上，MSE、最终 K 和时间如何变化”。

## 2. 四栏几何个例：每个 K 一张

推荐展示 adaptive verified 版本：

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/reruns/v12_verified_four_panel_K4_20 `
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

默认得到 17 张 PNG。若每个 K 需要两张，将 `--samples-per-knot-count` 改为 `2`。如果省略 `--max-knot-count`，脚本使用 checkpoint 数据配置中的最大 source K。

每张图固定包含：

1. **Original**：源 B 样条、带噪观测点、源控制多边形、控制顶点和源内部节点；
2. **Redundant proposal**：网络预测的全部高召回候选及一次标准 B 样条 refit；这不是 Boehm 精确插点；
3. **Ours**：KeepMask、survivor relocation、精确阈值检查及启用的条件修复；
4. **Greedy hard**：从与第 2、3 栏相同的已物化 proposal 出发，逐个尝试删除节点。

使用 `--verified-parameterization chord` 时，第 2～4 栏共用弦长参数和映射后的同一 proposal。若要画纯一次性网络消融，将 `--ours-deployment verified` 改为 `--ours-deployment learned`；此时不能把结果称为 adaptive verified。

输出目录包含：

```text
comparison_*.png
comparison_manifest.json
comparison_manifest.csv
```

manifest 保存每条曲线的节点向量、MSE、时间、参数域及 verified 修复来源。已有同名输出时脚本会停止；确认需要覆盖后再添加 `--overwrite`。

如果已经选好论文个例，可直接按确定的 sample ID 重画。例如：

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/reruns/v12_selected_four_panel `
  --sample-indices 248 350 397 `
  --scan-size 512 `
  --seed 20000 `
  --mse-tolerance 2.5e-5 `
  --ours-deployment verified `
  --verified-parameterization chord `
  --verified-compact `
  --timing-repeats 5 `
  --timing-warmups 1 `
  --timing-scope end-to-end `
  --dpi 600
```

一个 ID 生成一张四栏图，多个 ID 生成显式批次。`--sample-indices` 绕过随机分层选择，不能与 `--knot-counts`、`--samples-per-knot-count` 或 `--min/--max-knot-count` 同时使用。

Ours 的主显示时间固定为纯 `forward_deployment()`；`--timing-scope end-to-end` 只选择第 4 栏 Greedy 的展示边界。manifest 仍保存 Ours 完整质量管线以及 Greedy pruning-only/end-to-end 的诊断时间，便于追溯，但这些诊断值不进入主时间对比。

## 3. 三项分层统计图：MSE、最终 K、时间

正式分层实验建议每个 source K 取 20 条，共 `17×20=340` 条独立测试曲线：

```powershell
python scripts/evaluate_knot_count_strata.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/reruns/v12_verified_stratified_K4_20_n20 `
  --samples-per-knot-count 20 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --scan-size 1024 `
  --seed 20000 `
  --selection-seed 12345 `
  --execution-seed 67890 `
  --mse-tolerance 2.5e-5 `
  --ours-deployment verified `
  --verified-parameterization chord `
  --verified-compact `
  --timing-repeats 5 `
  --warmup-repeats 1 `
  --bootstrap-replicates 5000 `
  --bootstrap-seed 24680 `
  --torch-num-threads 4 `
  --dpi 600
```

输出：

```text
stratified_three_metrics.png # 主图：MSE、最终 K、时间
stratified_comparison.png    # 附加：原 2×2 K/MSE/pass/time 图
stratified_comparison.json   # 完整配置、逐层与总体统计
per_sample_results.csv       # 每条曲线结果
per_k_summary.csv            # 每个 source K 的汇总
summary.md                   # 可直接查阅的实验摘要
```

主图的三项为：

1. 标准 B 样条最终 refit MSE；
2. 最终保留内部节点数，并同时画出 source K 和 canonical reference K；
3. 每条曲线 wall time 的中位数。

原 2×2 图额外保留阈值通过率子图。两张图使用完全相同的样本、proposal、逐 K 汇总和计时数据；阴影均为以独立曲线为单位的 95% percentile-bootstrap 区间。已有结果时需要显式添加 `--overwrite` 才会覆盖。

已有 `stratified_comparison.json` 时，无需重新采样或运行两种方法，可快速重画三项图：

```powershell
python scripts/plot_stratified_three_metrics.py `
  --report outputs/comparisons/current/v12_verified_chord_compact_K4_20_n20/stratified_comparison.json `
  --output outputs/comparisons/current/v12_verified_chord_compact_K4_20_n20/stratified_three_metrics_redrawn.png `
  --dpi 600
```

若省略 `--output`，默认写到报告同目录的 `stratified_three_metrics.png`。已有同名文件时必须明确添加 `--overwrite`。重绘只改变版式和 DPI，不会改变旧 JSON 中的样本、指标或计时口径。

## 4. MSE 口径

两类图都使用归一化坐标上的**平均平方欧氏距离**：

\[
\operatorname{MSE}
=\frac{1}{M}\sum_{i=1}^{M}\left\|C(t_i)-Q_i\right\|_2^2.
\]

具体含义：

- 先对每个点的各坐标误差平方求和，再对点取平均；
- 不取平方根，也不是按全部坐标元素再次平均；
- `MSE = 2.5e-5` 等价于当前 `RMS = 0.005`；
- MSE 在数据集归一化坐标中计算，不代表毫米或米；
- K 只统计内部节点，不包括端点重复节点，也不是控制顶点数；
- knot matching tolerance 是参数域距离，与 MSE/RMS 拟合阈值不是同一指标。

## 5. 计时口径

两类脚本都按 CPU、batch size `1` 报告 wall time，并对重复运行取中位数；数据生成、分层扫描、绘图和文件写入均不计时。

### 四栏图：Ours 仅报告网络前向

在 `visualize_batch_comparison.py` 中使用：

```text
--timing-warmups 1
--timing-repeats 3
--timing-scope end-to-end
```

- **Ours learned / verified 主显示时间**：只包含 `forward_deployment()` 中的参数、proposal、KeepMask 和 relocation；输入在计时前已经驻留，排除最终 refit 与全部 verified repair；
- **Ours 质量结果**：仍可来自 learned refit 或 verified 修复，但完整质量管线时间只写入 manifest 的 diagnostic 字段；
- **Greedy end-to-end**：同样先运行网络生成参数和 proposal，可选地映射到共享弦长域，再执行 greedy pruning 及其全部 refit；
- **Redundant** 栏的时间仍只是全 proposal 的单次 `refit-only`，不代表候选生成总时间。

`--timing-scope historical-asymmetric` 让 Greedy 只显示 proposal 已经物化后的 pruning-only 时间；两种 Greedy 时间都会写入 manifest。无论该参数取何值，Ours 的主时间都只显示网络前向，因此不能由两条曲线直接推导端到端加速比。

### 三项统计图：Ours 同样只显示网络前向

`evaluate_knot_count_strata.py` 当前没有 `--timing-scope` 参数。其 Ours 主时间只包含同步后的纯 `forward_deployment()`；learned/verified 完整质量管线另存为 diagnostic。Greedy 仍只包含已物化参数和 proposal 后的 pruning stage。因此三项图会显式标注不同时间范围，不能直接声称同边界端到端加速比。

该脚本会在每条曲线上分别预热两种方法，并在每个配对计时块中随机交换 Ours/Greedy 的先后顺序。正式论文应固定 CPU、线程数、PyTorch 版本，并保留 JSON 中的完整 timing protocol。用旧 JSON 快速重绘不会改变这一计时边界。

## 6. 快速检查

正式运行前可把统计图命令临时改为：

```text
--samples-per-knot-count 1
--scan-size 512
--timing-repeats 1
--bootstrap-replicates 100
--dpi 150
```

该配置只验证脚本、checkpoint 和输出目录是否连通，不能用于论文数据。
