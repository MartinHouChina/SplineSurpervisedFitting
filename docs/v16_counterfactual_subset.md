# v16 有监督候选选择与重定位

> 文件名为历史兼容保留。当前正式 v16 已从在线 counterfactual Teacher 改为 certified Synthetic 的直接监督。

## 1. 目标

给定有序点云，在一次网络前向内预测满足阈值的紧凑内部节点集合，并联合修正参数化与存活节点位置。最终标准三次 B 样条只 refit 一次。

当前合同：

| 字段 | 值 |
|---|---|
| objective | `candidate_selection_supervised_bspline_v16` |
| architecture | `v16_supervised_ordered_assignment_mass_topk` |
| simplification | `synthetic_ground_truth_ordered_keep_and_relocation_v4` |
| checkpoint selection | `mean_per_curve_subset_cost_v1` |
| checkpoint quality | `supervised_fit_count_selected` |
| qualification | `v16_supervised_synthetic_only_pass_rates_report_only_v4` |

## 2. 网络流水线

```text
Q[B,M,D]
  -> geometry tokens H[B,M,d], global g[B,d]
  -> ordered parameters t[B,M]
  -> ordered proposals U_prop[B,Kc] + proposal tokens
  -> interactive Selector -> logits, adaptive beta, keep probabilities
  -> one probability-mass Top-K -> KeepMask[B,Kc]
  -> selected-only parameter and survivor relocation decoder
  -> U_deploy[KeepMask]
  -> standard cubic B-spline refit × 1
```

`Kc=56` 是内部候选容量，不是预测节点数；三次开放节点向量还包含四个 0 和四个 1，因此全保留长度为 64。

## 3. 为什么采用直接监督

旧在线 Teacher 用当前 Selector 排序搜索可行前缀，容易形成“当前错误排序生成自己的标签”的闭环，而且每 batch 需要大量 float64 refit。新协议利用 certified Synthetic 原生拥有的 `t*、U*、K*`：

1. 用动态规划求 `U_prop` 到 `U*` 的最小代价有序一一匹配；
2. 被匹配的候选为正 existence，其余为负；
3. `K*` 直接监督 adaptive mass 和最终离散节点数；
4. 标签 mask 条件解码器直接向 `t*、U*` 学习参数与重定位；
5. 实际部署 mask 同时接受拟合与几何监督，缩小训练/部署差异。

这一路径不运行在线 Hard-RMS、prefix sweep、counterfactual mask search 或 oracle Teacher，不使用 Teacher cache。代码中的 `online_teacher` 只保留为历史消融，不能生成当前正式 checkpoint。

## 4. 两阶段训练

### Proposal（40 epochs）

- 训练参数化与高召回有序候选；
- 50% 合成 draw 来自 source `K=40..56`；
- 同时优化 directed coverage 与 ordered one-to-one assignment；
- 到期无条件进入 Joint。

### Joint（64 epochs）

- 恢复 source `K=4..56` 原抽样分布；
- 联合训练 KeepMask、count/ranking、参数反馈和 survivor relocation；
- 训练数据仍全部为 certified Synthetic；
- UJI、Natural Earth、USGS 只做留出验证。

## 5. 一次性选择

Selector 为每个候选输出 logit，并由曲线级自适应 `beta` 调整整体概率质量。部署计数由 probability mass 产生，然后一次性取最高分的 K 个候选；不是固定 `p>=0.5`，也不是逐节点删减。选定集合进入 selected-only decoder，所有 survivor 彼此交互后得到有序、受界的最终位置。

部署不读取真节点、真 K 或真参数；这些标签只存在于合成训练阶段。

## 6. 验证与输出

正式结论必须来自独立测试：

- Synthetic：按 source K=4..56 分层；
- UJI Pen、Natural Earth、USGS：留出 test split；
- 方法：Ours、Park、Liang、Dung、Kang、Luo；
- 指标：MSE、通过率、最终内部节点数、完整方法时间；Ours 另报 network-only 时间。

一条龙脚本还生成基于输入采样点和 original-reference 点的两张 2×2 汇总图，以及真实曲线六方法案例图。文档不预填尚未测得的性能数字。

## 7. 解释边界

- supervised source K 是当前训练目标，但 source-subset 证书不证明自由重定位空间的连续全局最少 K。
- pass rate 是结果指标；它不控制阶段切换、checkpoint 选择或结构资格。
- K=56 层没有 proposal 冗余容量，应单独报告其 dense 和 deployment 表现。
