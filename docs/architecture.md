# v7 模型数据流

## 1. 输入与参数化

唯一外部输入是沿曲线方向排列的点：

```text
points: [B,M,D], D∈{2,3}
```

`GeometryEncoder` 读取坐标、弦长归一化一阶差分和二阶差分，输出：

```text
local_features:  [B,M,H]
global_features: [B,H]
```

`ParameterHead(local_features, global_features)` 预测 `M-1` 个正间隔，归一化并累加为：

```text
params: [B,M]
0=t0<t1<...<t(M-1)=1
```

参数头先运行，随后 `params`、局部特征和全局特征同时提供给候选节点路径。没有“先预测节点
个数，再按个数预测位置”的串联瓶颈。

## 2. CandidateKnotHead

输入：

```text
global_features [B,H]
local_features  [B,M,H]
params          [B,M]
```

局部特征先加入参数位置编码。`Kc+1` 个带均匀锚点的 interval query 对它做
cross-attention，并预测 `Kc+1` 个正区间：

\[
\Delta_j=\delta+igl[1-(K_c+1)\delta\bigr]\operatorname{softmax}(a)_j.
\]

前缀和给出固定预算的候选：

\[
c_j=\sum_{r=0}^{j-1}\Delta_r,qquad
0<c_1<\cdots<c_{K_c}<1.
\]

输出：

```text
candidate_knots  [B,Kc]
candidate_tokens [B,Kc,H]
candidate_intervals [B,Kc+1]
```

该头的目标是高召回，不负责直接给出最终数量。

## 3. 截断幂贡献特征

所有候选先保持开启，构造三次截断幂设计矩阵：

\[
\Phi=[1,t,t^2,t^3,(t-c_1)_+^3,\ldots,(t-c_{K_c})_+^3].
\]

正则化最小二乘得到系数和代理重建。每个候选提取：

```text
coefficient_energy                 [B,Kc]
analytic_drop_objective_delta      [B,Kc]
candidate_local_residual           [B,Kc]
left/right spacing                 [B,Kc,2]
candidate position                 [B,Kc]
```

其中 `analytic_drop_objective_delta` 是从正规矩阵逆对角和节点系数计算的精确“删除该列后的
二次目标增量”。它适合作为廉价的结构证据，但不是部署阶段的几何 RMS 判定。

这些解析量在进入消冗头前停止梯度，避免对线性系统求解二阶梯度。候选位置仍从覆盖、位置和
全候选拟合损失获得梯度。

## 4. InteractivePruningHead

消冗头融合候选 token、位置编码和上述解析特征，再用候选 self-attention 判断节点间的替代、
相邻和互补关系。输出：

```text
keep_probability          [B,Kc]      # 辅助诊断
remove_stop_logits        [B,Kc+1]    # Kc 个删除动作 + STOP
predicted_deletion_cost   [B,Kc]      # 辅助排序/回归
position_residual         [B,Kc]
refined_candidate_knots   [B,Kc]
```

位置残差最多使用左右可用间距的 45%，因此精修后节点仍严格有序。STOP 是显式动作；最终零
节点是合法结果，不需要 CountHead 的 `K=0` 类。

训练 forward 使用全部精修候选进行拟合，不把 keep 概率乘进设计矩阵。这样拟合梯度不会再次
把所有 keep 概率推向 1。

## 5. 训练输出与部署输出不同

训练阶段的 `reconstructed_points` 是截断幂代理，目的是给参数和候选位置提供稳定梯度。

部署阶段读取：

```text
params
refined_candidate_knots
```

随后对标准开放三次 B 样条执行逐节点删除和完整控制点重拟合。每次删除只有在实际欧氏
RMS 不超过 `fit_tolerance` 时才接受。最终节点数是硬删除轨迹的长度，而不是：

- CountHead 输出；
- keep 概率之和；
- 固定 0.5 阈值；
- BIC。

## 6. 计算量

网络部分主要复杂度为：

\[
O(BK_cM)+O(BK_c^2).
\]

默认 `Kc=28` 时，固定候选计算用于换取真实节点的高召回。部署硬验证更昂贵：贪心路径每轮
尝试所有剩余单节点删除，逻辑上最多评估 `Kc(Kc+1)/2` 个删除状态。实现把同一轮状态合并成
批量 Cox–de Boor 和批量最小二乘，只单独物化本轮获胜拟合。它不在反向传播中，并以可审计
的阈值保证换取这部分计算。

## 7. 兼容路径

`SplineFittingNetwork` 仍保留 `interactive_dynamic`、`count_conditioned` 和
`hard_concrete`，用于加载 v6/v5/更早 checkpoint。v7 使用独立 objective version 和模块
参数布局，不会把旧权重静默迁移成新结构。
