# 历史模型筛选与六方法双阈值对比

## 本轮范围

不重新训练、不继续被取消的新框架原型。先筛选本地历史权重，再将选定模型与 Park、Liang、Dung、Kang、Luo 的仓库适配实现进行配对实验。

用户确认的共同约束：

\[
E_{\rm mean}=\frac1N\sum_i\|C(t_i)-Q_i\|_2^2\leq5\times10^{-5},\qquad
E_{\rm peak}=\max_i\|C(t_i)-Q_i\|_2^2\leq5\times10^{-4},\qquad K\leq32.
\]

两种误差均不取平方根，使用192个归一化输入点。它们不是连续曲线最大误差或 Hausdorff 距离。原始参考点另行评估，不用于筛选模型、决定插入位置或重拟合。

这里32指**内部节点**上限；三次、两端各4重节点的开放节点向量最多40项，控制顶点最多36个。

## 1. 如何选择历史模型

- 审计105份模型载荷；排除教师缓存、测试/烟雾检查产物、回滚目录副本与候选机制之前的旧架构。具体清单、容量和阶段排除原因见 `outputs/diagnostics/historical_checkpoint_metadata_20260922.json`。
- 52份明确容量不超过32；排除纯 Proposal 等不适合最终部署的载荷后，40份权重进行严格恢复与公共验证。64/96节点模型没有被强行缩容或与32节点模型混为一谈。
- 初筛：21条新种子合成曲线（源K4..24各1条）和四个外部来源各8条 `val` 曲线，共53条。这里合成源K不是最简节点标签；筛选只评价拟合，不使用节点标签。
- 前5名扩展验证：同21条合成曲线和四来源各32条 `val` 曲线，共149条。扩展集包含初筛的验证样本，不是额外的独立测试集。
- 预设排序：原始网络部署的来源等权双阈值通过率、最差来源通过率、失败数、平均K、MSE。公共插入修复不参与选择。最后61条测试曲线不参与选模型。
- 对比前核对选择与测试的样本ID、group和输入内容哈希不重叠。现有外部数据是按group划分的验证/测试集。

扩展验证结果：

| 权重 | 保存epoch | 来源等权双通过率 | 逐曲线双通过率 | 平均内部K |
|---|---:|---:|---:|---:|
| `overnight_stable_k32_3090_r1.pt` | 14 | 66.82% | 65.77% | 20.77 |
| `overnight_anchored_k32_p12_j48_3090_r1.pt` | 13 | 64.91% | 64.43% | 23.46 |
| `universal_m16_m32_3090_r1_m32.pt` | 5 | 62.41% | 61.74% | 22.29 |
| `paper_coupled_clean_3090_r3_m32.pt` | 13 | 61.19% | 59.73% | 21.70 |
| `overnight_anchored_k32_3090_r1.pt` | 5 | 60.51% | 60.40% | 23.47 |

选择 `outputs/checkpoints/overnight_stable_k32_3090_r1.pt`。这是**本次有限公共验证、可用兼容权重范围内的第一名**，不是统计显著的全面最优证明。其验证集最差来源仍只有40.625%的双阈值通过率；不能把“选出了最好的一份”理解为“已经可靠达标”。

完整筛选记录：`outputs/diagnostics/historical_common_validation_20260922.json`、`historical_confirm_validation_20260922.json`。

## 2. 六方法共同采用什么修复

```text
各方法自身产生 t、U、控制点
          ↓
保留并保存原始拟合与计时
          ↓
重新计算真实 MSE 与最大单点平方误差
          ↓
两项达标：原对象直接返回，不插入、不重拟合
未达标且 K<32：在残差较大区域试插一个节点
          ↓
重求标准 B 样条控制点，比较实际双约束违反量
          ↓
继续到两项达标，或容量/有效插入候选耗尽
```

候选优先位置包括残差峰值的参数、相邻样本参数中点和高残差节点区间中点；每轮最多检查12个候选。选择依据为先双可行性，再最坏归一化违反量 `max(MSE/5e-5, MaxSE/5e-4)`，随后MSE与峰值。逐次插入不保证全局最少节点，也不保证达到32节点时一定可行；返回路径中实际观测到的最好状态，并保留完整试探轨迹。

**单纯 Boehm 节点插入不改变曲线，不能降低误差。本封装在增加自由度后重新求控制点，才可能降低残差。**

严格保持：

- 固定各方法原来的参数`t`，已有节点位置和重数不变；不删节点、不移动节点、不使用网络候选来帮助文献对照组。
- 没有旧 safeguard 中的“整体替换成均匀节点”、提高节点预算或放宽阈值。
- 保留原端点约定：Dung/Kang 无强制端点插值，其他四组保持端点插值；重拟合均为无正则、CPU float64标准最小二乘。
- 六组均使用公共插入封装，包括 Ours。必须同时看原始与修复后两套结果。Ours修复后不再是一次性网络部署。
- 原算法本身报错且没有有效拟合时，不凭空生成另一个方法并冒充原方法的结果；失败保留在统计分母。
- 修复后的方法名称应为“仓库适配实现＋公共插入修复”。不能把这层机制归入论文作者原方法。

实现：`src/spline_fitting/evaluation/dual_error_knot_repair.py`。

## 3. 实验范围、计时和保存内容

最终测试复用之前保存并逐文件校验哈希的61条输入：合成K4..24各1条，UJI、Natural Earth、USGS、工业等距线各10条。工业等距线是程序生成的CAD曲线，不是实测工业数据。

此次六方法均**重新执行**，不是将旧基线耗时加到新曲线上。采用相同32节点上限，保留此前的文献适配参数预算，关闭旧的仅MSE safeguard。原始与修复各366条记录，共732条结果。这里的“原始”是修改公共修复之前的仓库实现，不是已验证的作者原生复现。

计时为本机GTX1070网络、CPU4线程数值算法。`total_ms = raw_ms + repair_ms`；两段均来自当前这条曲线的实际调用，指标评估、导出与绘图不计入。`network_ms`是单独重复5次同步前向的中位数，不可替代完整时间。完整方法只计时一次，本机后台工作与短时波动可能影响结果，不宜据此宣称正式GPU加速比。

保存原始/最终参数、完整节点向量、控制顶点、曲线上节点、稠密曲线、逐点残差、原参考点误差、插入节点、修复尝试、端点设置和计时。统计不删失败；修复前后同一组随机案例绘制6×5图，不按效果挑对照样本。

## 4. 实测结果

以下是61条固定测试曲线的逐曲线统计，不是模型选择验证集。通过率同时要求MSE和最大单点平方误差达标。

| 方法（均允许同一公共修复） | 原始双通过率 | 修复后双通过率 | 修复后平均K | 修复后平均MSE | 最坏MaxSE | 完整时间ms/曲线 |
|---|---:|---:|---:|---:|---:|---:|
| 选中历史Ours | 59.0% | 78.7% | 23.89 | 5.842e-5 | 2.984e-3 | 57.95 |
| Park适配 | 44.3% | 59.0% | 21.10 | 1.145e-4 | 1.041e-2 | 109.66 |
| Liang适配 | 62.3% | 67.2% | 19.39 | 8.861e-5 | 3.710e-3 | 71.01 |
| Dung适配 | 8.2% | 80.3% | 16.78 | 5.565e-5 | 3.575e-3 | 1042.13 |
| Kang适配 | 8.2% | 77.0% | 18.31 | 6.434e-5 | 3.578e-3 | 2511.07 |
| Luo适配 | 16.4% | 78.7% | 18.30 | 6.952e-5 | 7.326e-3 | 5762.74 |

Dung在一条海岸线上原始分段产生36个内部节点，超过32预算而明确报错：该例仍计入61条通过率分母，节点/误差均值按其60条有限结果计算。没有擅自截掉节点或另造一条曲线掩盖失败。其余五方法均有61条有限结果。所有均值包含有限但未达标的拟合，故平均MSE可以超过阈值。

共同修复未发生封装异常，所有返回结果均不超过32个内部节点。Ours修复前36条双达标，修复后48条；仍有13条未达标，不宣称固定容量下全通过。`repair_ms`包括真实误差验收、候选检查与必要的插入/重拟合；即使无需插入也有验收开销。

与上一次最新coupled M32的同61条输入比较：原始双通过率55.74%→59.02%，平均K23.82→23.36，平均MSE1.980e-4→8.609e-5。但仅MSE通过率65.57%→59.02%，并非所有指标同时改善。两者不同行为不能只归因于某一个Joint模块。这次使用历史Stable权重，不要将上一轮coupled框架图当作其结构图。

最终独立审计确认365组有限拟合的参数逐值不变、原节点与重数保留、几何重算误差与标记一致；失败1组明确保留。审计记录：`outputs/diagnostics/historical_dual_final_audit_20260922.json`。

结论：此历史模型的**本机实测速度**有价值，但并不具有最少节点优势。Dung/Luo达到相近或更高的双通过率时，平均节点明显少于Ours；Ours的最坏采样点平方误差在这组结果中较小。不能只展示原始坍缩基线而省略公共修复后的较强结果，也不能将本机单次计时解释为正式算法复杂度或3090加速结论。

完整结果位于 `outputs/comparisons/historical_best_dual_error_20260922/`，其中 `raw/` 为原始拟合，主 `geometry/` 为公共修复后拟合。图表位于 `outputs/figures/historical_best_dual_error_20260922/`。

针对修复、历史模型选择/恢复、完整性、失败计数和绘图的联合回归测试：42项通过。

## 5. 运行

本次已经选择好的权重可以直接复现测试和出图，无需重新训练：

```bash
python scripts/benchmark_historical_dual_error.py \
  --selection outputs/diagnostics/historical_confirm_validation_20260922.json \
  --source-benchmark outputs/comparisons/local_models_paper_20260922_m32 \
  --device cuda \
  --mse-tolerance 5e-5 \
  --max-squared-error-tolerance 5e-4 \
  --max-internal-knots 32 \
  --output-dir outputs/comparisons/historical_best_dual_error_rerun

python scripts/plot_dual_error_comparison.py \
  --report outputs/comparisons/historical_best_dual_error_rerun/comparison.json \
  --output-dir outputs/figures/historical_best_dual_error_rerun
```

需要该选择JSON、选中的模型文件，以及源benchmark完整 `geometry/` 文件夹。脚本不会重新随机生成测试输入或自动下载缺失模型。新结果目录不能覆盖旧结果；相同协议中断续跑可加 `--resume`。

公共验证重新执行（只适用于库存清单中的模型文件仍在本地）：

```bash
python scripts/select_historical_checkpoint.py --device cuda \
  --output outputs/diagnostics/historical_screen_rerun.json
python scripts/select_historical_checkpoint.py --device cuda \
  --shortlist-from outputs/diagnostics/historical_screen_rerun.json \
  --shortlist-size 5 --real-count 32 \
  --output outputs/diagnostics/historical_confirm_rerun.json
```

注意：初筛与复核是模型选择数据，不应作为最终测试性能汇报；正式论文还应扩大独立测试规模并重复计时。
