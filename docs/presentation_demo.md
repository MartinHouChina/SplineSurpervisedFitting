# v16 展示用提纲

本文给出 PPT 的推荐讲解顺序。图表和数值必须来自本次实际运行产物，不填示意数字。

## 第 1 页：任务

输入一条有序点云，输出满足误差阈值的三次开放 B 样条：参数化、内部节点、控制顶点和拟合曲线。目标同时考虑：

- `MSE<=1e-4` 的拟合可行性；
- 尽量紧凑的内部节点集合；
- 一次性网络部署。

## 第 2 页：数据协议

| 数据 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| certified Synthetic，source K=4–56 | 是 | 是 | 是 |
| UJI Pen | 否 | 是 | 是 |
| Natural Earth coastline | 否 | 是 | 是 |
| USGS contours | 否 | 是 | 是 |

强调：真实数据没有真节点标签，当前正式训练只使用认证合成数据；真实数据仅检验泛化。

## 第 3 页：为什么合成标签可信

每条合成曲线先固定 source K，再生成干净曲线，并在 512 点网格上检查：完整 source 节点集满足阈值，删除任一 source 节点后都超过带 20% margin 的阈值。未通过则重新生成几何而不改变目标 K。

准确措辞：这是固定参数化和 source 原节点子集内的 threshold-minimal 证书，不是自由重定位下连续全局最优证明。

## 第 4 页：网络流水线

```text
ordered points
 -> GeometryEncoder
 -> ParameterHead
 -> Kc=56 ordered candidates
 -> interactive Selector + adaptive beta
 -> one mass-TopK KeepMask
 -> survivor-conditioned parameter/knot relocation
 -> one standard B-spline refit
```

三次开放样条全保留时完整节点向量为 `4 zeros + 56 internal + 4 ones = 64` 项。

## 第 5 页：Proposal 监督

前 40 epochs 学参数和候选。真节点对候选进行最小代价有序一一匹配：

- Kc 大于真 K：只匹配真 K 个互异候选；
- Kc 等于真 K：严格逐序位匹配；
- coverage 保召回，assignment 防止多对一坍缩；
- 50% Proposal 合成样本来自 K≥40，强化复杂曲线。

Proposal 到期无条件进入 Joint，pass rate 不参与阶段门控。

## 第 6 页：Joint 监督

有序匹配直接产生 target KeepMask，真 K 监督概率质量和节点数，真参数/真节点监督两条 subset 解码路径的参数反馈与 survivor relocation。损失包括 existence BCE、正负 ranking、count、over-count、节点位置和拟合项。

正式 Joint 无在线 Hard-RMS、prefix/counterfactual/oracle Teacher，无 Teacher cache。真实数据不进入 loss。

## 第 7 页：部署

部署输入只有有序点云。一次网络 forward 后执行一次 mass-TopK，再用最终节点做一次 CPU float64 标准 refit。没有逐节点搜索或多次试拟合。

说明时间口径：公平主表使用完整方法时间；Ours 的 network-only 时间另列，不能替代完整时间。

## 第 8 页：六方法公平协议

主表固定六种方法：Ours、Park & Lee、Liang et al.、Dung & Tjahjowidodo、Kang et al.、Luo et al.。五种公开方法均标注 adaptation。

统一条件：相同配对曲线、相同最大 56 内部节点、相同 MSE 定义与 `1e-4` 阈值、相同最终 CPU float64 refit、失败样本保留在通过率分母。

## 第 9 页：必须展示的结果

一条龙运行后选用以下实际产物：

1. `report.md` 或 `summary.csv`：六方法×四数据集的 MSE、pass、final K、total time；
2. `v16_published_methods_input.png`：192 点输入口径的 2×2 指标图；
3. `v16_published_methods_reference.png`：真实原始参考点口径的 2×2 指标图；
4. 真实曲线六方法 3×2 案例图：参考/输入、拟合、控制多边形、控制顶点和内部节点。

如果某阶段未运行完成，就展示流程和已有诊断，不宣称对应性能结果。

## 第 10 页：结论与限制

- 候选生成、KeepMask 和 survivor relocation 在一个固定深度网络内完成；
- 认证合成标签消除了在线 Teacher 的自举偏差和训练搜索开销；
- 真实数据只用于泛化评估；
- source-subset 最简性不等于连续全局最优；
- K=56 是无候选冗余的容量边界，必须单独报告。

## 演示命令

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName candidate_selection_v16_mse1e-4_k56_supervised `
  -Device cuda
```

该入口 fresh-only，已有同名产物时会拒绝覆盖。
