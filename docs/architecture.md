# v8 模型与数据流

## 1. 输入与目标

输入是沿曲线方向排列的点：

```text
points [B,M,D], D∈{2,3}
```

目标是在归一化 RMS 阈值 `ε` 下预测尽量少的内部节点。v8 没有 CountHead；最终节点数是 final KeepMask 中 `True` 的数量。

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

## 5. InteractivePruningHead：固定双向链路

候选 token、位置编码和解析证据先融合，再通过 candidate self-attention 建模相邻、替代和冗余关系。随后在一次 `forward` 内按固定次数执行：

```text
base tokens
  → preliminary raw importance / β / p
  → preliminary soft keep context
  → provisional ordered positions
  → position encoding feedback
  → final decision tokens
  → final raw importance / β / p
  → hard-ST KeepMask context
  → final ordered positions
```

这不是循环，也不会根据预测节点数改变计算次数。

### 5.1 Preliminary keep

对 base token 预测第一组重要性和曲线级阈值：

\[
p_j^{(0)}=\sigma(r_j^{(0)}-\beta^{(0)}).
\]

用 `p^(0)` 汇聚 soft keep context，与每个 base token 交互后预测 provisional position residual。残差最多使用单侧可用间距的 45%，所以 provisional positions 严格有序。

### 5.2 Position → keep 反馈

将 provisional position 的正弦位置编码与原位置编码作差，再与 provisional token 融合。独立模块 `position_to_keep_feedback` 把该状态送回最终选择决策：

\[
r_j=f_{\mathrm{keep}}(z_j^{\mathrm{feedback}}),\qquad
\beta=f_{\mathrm{threshold}}(z_{\mathrm{global}}^{\mathrm{feedback}}),
\]

\[
p_j=\sigma(r_j-\beta),\qquad
m_j=\mathbf 1[p_j\ge0.5].
\]

因此 provisional 位置会改变最终保留概率；最终保留状态也会反过来控制最终位置精修。

### 5.3 Final hard-ST context 与位置

训练使用 straight-through gate：

\[
g_j=m_j+p_j-\operatorname{stopgrad}(p_j).
\]

其前向值严格等于 hard mask，反向梯度来自 `p_j`。最终上下文为 hard-ST 加权汇聚；前向只包含被保留候选。若所有候选都被删除，安全分母使上下文为有限的零向量。

最终位置残差以 `g_j` 调制：

- 被删除候选前向不移动；
- 被保留候选读取 hard KeepMask context 后精修；
- 反向梯度仍可更新 keep 概率；
- 每个残差仍受 45% 邻域 slack 限制，最终位置严格有序。

主要诊断输出：

```text
preliminary_raw_importance
preliminary_adaptive_keep_threshold
preliminary_keep_probability
provisional_candidate_positions
position_feedback_tokens
final_raw_importance
adaptive_keep_threshold
final_keep_probability
final_hard_keep_mask
final_hard_st_keep_gate
final_hard_st_keep_context
refined_candidate_knots
```

旧键 `raw_importance`、`keep_probability` 和 `refined_candidate_knots` 表示最终状态。

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

固定双向链路增加的是常数次 head 计算，不增加候选数量或搜索循环。

- `structure_mode=candidate_pruning_one_shot`：v8 固定双向 one-shot 路径。
- `structure_mode=candidate_pruning`：v7 历史路径。
- `one_shot_adaptive=False` 不创建 v8 参数，v7 state layout 与数值路径保持不变。
- 早期 v8 checkpoint 缺少 `position_to_keep_feedback` 参数时，可在 strict load 中注入该模块的构造初值。
