# 节点选择方案演进与当前 v16

本文只用来解释设计演进；当前可运行协议以 supervised-only v16 为准。

## 1. 历史问题

| 版本 | 主要思路 | 暴露的问题 |
|---|---|---|
| v3–v6 | Activity/CountHead 直接预测结构 | 概率塌缩、离散计数错误会整体改变节点向量 |
| v7 | 冗余 proposal + 在线 Greedy Hard-RMS | 可解释但需多次串行 refit，部署慢 |
| v8–v10 | 离线 Hard-RMS Teacher 蒸馏一次性 mask | 标签受固定 proposal/搜索策略限制 |
| v11–v12 | Keep 与 survivor relocation 联动 | Teacher 位置目标和删除后邻接关系仍可能偏置 |
| v13–v15 | 集合匹配、参数反馈与部署 MSE 对齐 | 仍依赖离线 Teacher/cache，数据和模型更新易失配 |
| 早期 v16 | 在线反事实子集学习 | 避免 cache，但每 batch 组合搜索昂贵且可能自举错误排序 |

这些历史实现仍可用于消融，但其 checkpoint、Teacher cache 和结果不能改名后并入当前主表。

## 2. 当前 v16 重构

当前正式版本利用 certified Synthetic 的 `t*、U*、K*` 完成直接监督：

```text
Proposal:
  Q -> t -> ordered U_prop
  t*, U* -> parameter + coverage + ordered assignment losses

Joint:
  ordered_match(U_prop,U*) -> target KeepMask
  K* -> adaptive probability-mass/count target
  t*, U* -> deployed/labelled subset parameter + relocation targets

Deployment:
  Q -> one forward -> one mass-TopK -> one standard refit
```

核心改变不是简单“删除 Teacher”，而是把结构标签和位置标签统一到同一个有序匹配：被保留的候选与其目标真节点一一绑定，KeepMask 与 survivor relocation 因而可以共同训练。

## 3. 为什么保留 Proposal + Selector 两个功能模块

Proposal 回答“候选空间是否覆盖真节点”，Selector 回答“这条曲线应保留哪些候选”。分开后可以分别诊断 candidate recall 与 deployment selection；Joint 又通过共同的 ordered target 和 selected-only decoder把两者耦合，避免完全独立优化。

`Kc=56` 是并行候选容量，所有候选在矩阵运算中一次处理；最终 K 由曲线自适应 probability mass 决定。它不是 CountHead 分类，也不是逐次预测。

## 4. 删除与移动如何联动

Selector 先产生离散 KeepMask；subset decoder 只用 survivors 重新构造集合上下文，包括邻距、相对 rank 和存活数量，并只对 survivor 施加位置残差。Joint 同时训练实际部署 mask 和标签 mask 的节点位置及拟合误差，因此删除改变邻接关系后，剩余节点可以重新分布，而不是照搬 proposal 横坐标。

## 5. 当前数据边界

- Proposal 和 Joint 的 optimizer step 全部来自 certified Synthetic；
- source K=4..56 作为 exact 监督目标；
- UJI、Natural Earth、USGS 只用于 validation/test；
- 正式 Joint 无 online Hard-RMS、ranked-prefix、counterfactual 或 oracle Teacher；
- 不读写 Teacher cache。

source-subset 证书不证明自由重定位下的连续全局最小 K，因此“exact label”应理解为当前监督协议的精确生成标签，而非全局最优定理。

## 6. 当前部署与传统剪枝的差别

| 方法 | 组合决策 | 位置调整 | 最终 refit |
|---|---|---|---|
| 当前 Ours | 一次网络 adaptive mass-TopK | 网络内 survivor-conditioned relocation | 1 次 |
| Greedy Hard-RMS | 逐候选试删 | 可选数值梯度/局部优化 | 多次试拟合 + 最终拟合 |
| v8–v15 | 离线搜索 Teacher，在线一次 mask | 版本相关 | 部署通常 1 次 |

Ours 的优势假设是固定深度的一次性组合预测；是否在 MSE、K 或时间上优于传统方法，必须由同一批测试曲线的实测表决定，不能由架构直接宣称。

## 7. checkpoint 兼容边界

当前正式合同为：

```text
candidate_selection_supervised_bspline_v16
v16_supervised_ordered_assignment_mass_topk
synthetic_ground_truth_ordered_keep_and_relocation_v4
```

旧 checkpoint 不能直接 resume 为当前实验。若训练入口明确允许，可只迁移形状兼容的 encoder、ParameterHead 和 CandidateKnotHead 权重；Selector、subset decoder、optimizer 和合同元数据必须重新训练并重新审计。
