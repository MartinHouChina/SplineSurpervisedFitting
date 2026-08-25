# v10 模型与数据流

## 1. 输入与目标

输入是沿曲线方向排列的点：

```text
points [B,M,D], D∈{2,3}
```

目标是在归一化 RMS 阈值 `ε` 下预测尽量少的内部节点。v10 没有 CountHead；最终节点数由 LearnedKeep 概率质量和不确定性共同确定。

## 2. 编码与参数化

`GeometryEncoder` 从坐标、一阶差分和二阶差分得到：

```text
local_features  [B,M,H]
global_features [B,H]
```

`ParameterHead(local_features, global_features)` 预测严格递增参数：

```text
params [B,M]
0=t0<t1<...<t(M-1)=1
```

`params` 同时用于候选 cross-attention 的位置编码、两个截断幂代理求解和最终标准 B 样条重拟合。

## 3. 高召回候选

`CandidateKnotHead` 输入：

```text
global_features [B,H]
local_features  [B,M,H]
params          [B,M]
```

`Kc+1` 个 interval query 对带参数位置编码的局部特征做 cross-attention，并预测正区间：

\[
\Delta_j=\delta+
\left[1-(K_c+1)\delta\right]\operatorname{softmax}(a)_j.
\]

前缀和产生固定预算的严格有序候选：

\[
c_j=\sum_{r=0}^{j-1}\Delta_r,qquad
0<c_1<\cdots<c_{K_c}<1.
\]

输出：

```text
candidate_knots     [B,Kc]
candidate_tokens    [B,Kc,H]
candidate_intervals [B,Kc+1]
```

该阶段优先保证 true→candidate 的单向召回。冗余候选可以由后续选择头删除，漏掉的必要节点无法恢复。

## 4. 第一次截断幂代理求解

先打开全部候选，建立三次截断幂设计矩阵：

\[
\Phi=[1,t,t^2,t^3,(t-c_1)_+^3,\ldots,(t-c_{K_c})_+^3].
\]

正则化最小二乘产生全候选代理重建，并提取：

```text
coefficient_energy            [B,Kc]
analytic_drop_objective_delta [B,Kc]
candidate_local_residual      [B,Kc]
left/right spacing            [B,Kc,2]
```

这些量是便宜的结构证据，不是标准 B 样条 RMS 保证。解析贡献在送入选择头前停止梯度，以避免对线性求解器求二阶梯度。

## 5. InteractivePruningHead：v10 结构化可行选择器

v10 保留 v9 的固定 proposal 几何，并增强 LearnedKeep 的集合选择：

```text
base candidate tokens
  → proposal-only position residual
  → fixed refined positions
  → 固定位置编码 + 解析贡献特征
  → 两层独立 selector self-attention + FFN
  → raw importance r 与曲线阈值 β
  → probability-mass Top-K + coverage anchors
  → 一次性 KeepMask
```

### 5.1 固定 proposal 几何

位置残差只读取 proposal token，不读取 keep 概率或 hard mask。残差最多使用单侧可用间距的
45%，所以节点始终严格有序。离线教师生成后，编码器、参数头、候选头和位置残差头全部冻结；
同一样本的 `refined_candidate_knots` 在整个蒸馏阶段保持不变。

### 5.2 位置感知 selector adapter

独立 selector 将固定精修位置重新编码，并通过自己的候选 self-attention 学习替代、冗余和互补：

\[
r_j=f_{\mathrm{keep}}(z_j^{\mathrm{selector}}),\qquad
\beta=f_{\mathrm{threshold}}(\operatorname{pool}(z^{\mathrm{selector}})),
\]

\[
p_j=\sigma(r_j-\beta),\qquad
\widehat K=\left\lceil\sum_jp_j+s\sqrt{\sum_jp_j(1-p_j)}\right\rceil.
\]

从 raw importance 中选取最高的 `K̂` 个槽位。参数域覆盖锚点保证预测不坍缩到少数局部区间，
剩余名额仍按 LearnedKeep 全局排序分配。节点位置参与 Keep 判断，但 Keep 不再反向移动教师
绑定的位置。

### 5.3 Straight-through mask

训练仍使用 `g_j=m_j+p_j-stopgrad(p_j)`，使第二次截断幂 surrogate 的前向结构与部署 mask
一致。v10 默认把 surrogate fit/violation 权重设为零；它只提供诊断，不参与 checkpoint 选优。

主要诊断输出：

```text
final_raw_importance
adaptive_keep_threshold
final_keep_probability
final_hard_keep_mask
final_hard_st_keep_gate
final_hard_st_keep_context
refined_candidate_knots
```

兼容键 `raw_importance`、`keep_probability` 和 `refined_candidate_knots` 表示最终状态。

## 6. 第二次截断幂代理求解

最终 refined candidates 与 LearnedKeep 的 straight-through gate 再建立一次截断幂矩阵，并求解训练期代理系数。它在 proposal 阶段提供拟合与阈值监督；在 one-shot 选择阶段仍输出诊断值，但默认权重为零，不驱动 KeepMask。

因此一次网络前向内部固定包含两次代理 solve：

1. 全候选 pilot solve：提取贡献证据。
2. final KeepMask solve：计算可微训练重建。

它们都不是部署导出的标准 B 样条曲线。

## 7. 部署

```text
一次网络 forward
  → final KeepMask + refined candidates
  → 选出 mask=True 的有序节点
  → 在全部原始点上执行一次端点约束的标准开放三次 B 样条 refit
  → 输出内部节点、控制点和拟合曲线
```

默认部署不运行逐节点 Hard-RMS 搜索。因此 `RMS≤ε` 是独立测试集上的统计满足率，而不是每条输入的数学硬保证。

## 8. 计算量与兼容性

主要注意力复杂度为：

\[
O(BK_cM)+O(BK_c^2).
\]

两层 selector adapter 增加固定 `Kc×Kc` 注意力；Top-K 与覆盖锚点只处理分数，不调用样条
求解器，也不增加网络 forward 或部署 refit 次数。

- 同时启用 `one_shot_selection_policy=mass_topk`：v10。
- `structure_mode=candidate_pruning_one_shot` 且 `one_shot_fixed_proposal_geometry=True`、threshold mask：v9。
- 同一 structure mode 且该标志为 `False`：v8 固定双向 one-shot 路径。
- `structure_mode=candidate_pruning`：v7 历史路径。
- `one_shot_adaptive=False` 不创建 one-shot 参数，v7 state layout 与数值路径保持不变。
- 早期 v8 checkpoint 缺少 `position_to_keep_feedback` 参数时，可在 strict load 中注入该模块的构造初值。
