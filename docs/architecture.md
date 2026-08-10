# 模型内部数据流

以下内容描述 `structure_mode="count_conditioned"` 的 v5 主路径。

## 实际 forward 投喂顺序

代码不是把原始点分别直接投给三个 head。真实调用顺序如下：

```python
# 1. 原始有序点只先进入编码器
local_features, global_features = encoder(points)

# 2. 参数头读取编码后的局部特征和全局特征
parameter_output = parameter_head(local_features, global_features)
params = parameter_output["params"]

# 3. 数量头读取全局特征、局部特征和预测参数
count_output = count_head(global_features, local_features, params)
predicted_count = count_output["predicted_knot_count"]

# 4. 决定本次用于节点解码的数量
selected_count = true_count if training else predicted_count

# 5. 节点头读取相同的几何特征、预测参数和 selected_count
knot_output = knot_head(
    global_features,
    local_features,
    params,
    selected_count,
)

# 6. 用所选节点构造设计矩阵并求拟合系数
design = build_design_matrix(params, internal_knots, knot_mask)
coefficients = solve_coefficients(design, points)
reconstructed_points = design @ coefficients
```

对应的数据依赖关系是：

```text
points
  └─ GeometryEncoder
       ├─ local_features ─┬─ ParameterHead ── params ─┬─ CountHead
       │                  │                            └─ KnotHead
       └─ global_features ┴────────────────────────────┴─ CountHead/KnotHead

CountHead → predicted_count ──┐
true_count（仅训练）───────────┼─ selected_count → KnotHead 选择节点表示
                              └─ 训练取 true，验证/普通推理取 predicted
```

因此 ParameterHead 在 CountHead 之前运行，因为 CountHead 的局部 memory 需要使用 `params` 位置编码。

## 1. 输入

网络输入：

```text
points: [B, M, D]
```

- `B`：batch size；
- `M`：每条曲线的有序采样点数量，默认 64；
- `D`：空间维度，支持 2 或 3。

采样点在进入网络前已经中心化，并按最大点半径缩放。

## 2. GeometryEncoder

编码器为每个采样点构造三类输入：

1. 归一化坐标；
2. 按弦长参数间隔归一化的一阶导数；
3. 按弦长参数间隔归一化的二阶导数。

经过一维卷积、GroupNorm 和 GELU 后得到：

```text
local_features:  [B, M, H]
global_features: [B, H]
```

`local_features` 保留参数方向上的局部几何变化；`global_features` 是沿采样点维度最大池化后的曲线级描述。

## 3. ParameterHead

ParameterHead 的输入不是原始坐标，而是：

```text
local_features:  [B,M,H]
global_features: [B,H]
```

首先把全局特征复制到每个采样位置：

```text
global_expanded: [B,M,H]
```

然后与局部特征拼接：

```text
fused = concat(local_features, global_expanded)
fused: [B,M,2H]
```

MLP 对前 `M-1` 个位置分别输出一个原始参数间隔：

```text
raw_parameter_gaps: [B,M-1]
```

这些间隔经过 softmax、最小间隔约束和累加，得到：

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

输出：

```text
params: [B, M]
```

`params` 有两个作用：

- 与 `true_params` 计算参数监督损失；
- 生成位置编码，供 CountHead 和 KnotHead 读取参数域位置。

## 4. Ordinal CountHead

CountHead 不只读取全局池化特征。它使用多个可学习 count query，对下面的 memory 做 cross-attention：

\[
F_{memory}=F_{local}+\operatorname{PE}(t).
\]

注意力特征与 `global_features` 拼接后产生曲线复杂度证据 \(e\)。有序阈值 \(b_r\) 定义：

\[
s_r=P(K\ge r)=\sigma(e-b_r),\qquad r=1,\ldots,K_{max}.
\]

由相邻 survival probability 得到：

```text
count_probabilities:    [B, Kmax+1]
predicted_knot_count:   [B]
expected_knot_count:    [B]
count_ordinal_logits:   [B, Kmax]
```

`predicted_knot_count` 是类别概率 argmax，不是对节点逐个做阈值判断。

## 5. Shared Count-Conditioned KnotHead

节点解码器维护一组共享 interval query。对于给定数量 \(K\)：

1. 取前 \(K+1\) 个 interval query；
2. 加入全局曲线特征；
3. 加入节点数量 embedding `Embedding(K)`；
4. 对 `local_features + PE(params)` 做 cross-attention；
5. 输出 \(K+1\) 个 interval logits。

区间经过 softmax 和最小间隔约束：

\[
\Delta_j=\delta+[1-(K+1)\delta]\operatorname{softmax}(a)_j.
\]

前缀和生成内部节点：

\[
u_j=\sum_{r=0}^{j-1}\Delta_r,qquad j=1,\ldots,K.
\]

因此无需排序即可保证：

\[
0<u_1<\cdots<u_K<1.
\]

### 如何得到“对应长度”的节点向量

PyTorch batch 中不能让每个样本直接拥有不同的张量宽度，因此实现采用“全部分支定宽保存 + mask 表示真实长度”。网络一次 forward 会计算全部数量分支：

```text
branch_internal_knots: [B, Kmax+1, Kmax]
```

随后根据本次所选数量提取：

```text
internal_knots:       [B, Kmax]
knot_mask:            [B, Kmax]
count_used_for_knots: [B]
```

`knot_mask` 只是把变长节点表示放进固定宽度张量，不是 Activity 门，也不是剪枝结果。

假设 `Kmax=6`，某条曲线选择 `K=3`。内部实际过程是：

```text
K=0 分支 → [0,  0,  0, 0, 0, 0]
K=1 分支 → [u1, 0,  0, 0, 0, 0]
K=2 分支 → [u1, u2, 0, 0, 0, 0]
K=3 分支 → [u1, u2, u3,0, 0, 0]  ← selected_count=3 选择这一行
K=4 分支 → [u1, u2, u3,u4,0, 0]
K=5 分支 → [u1, u2, u3,u4,u5,0]
K=6 分支 → [u1, u2, u3,u4,u5,u6]
```

选择后返回：

```text
internal_knots = [u1,u2,u3,0,0,0]
knot_mask      = [1, 1, 1, 0,0,0]
```

训练代理使用 `knot_mask` 关闭后三列。标准 B 样条部署则执行：

```python
valid_knots = internal_knots[knot_mask]
```

此时才得到物理长度真正为 3 的节点向量 `[u1,u2,u3]`。

### 每个 K 分支怎样生成 K 个节点

`K=3` 时不是直接回归三个无序数，而是生成 4 个正区间：

```text
interval query 数量 = K+1 = 4
预测区间 = [Δ0, Δ1, Δ2, Δ3]
约束       Δj > 0，且 Δ0+Δ1+Δ2+Δ3 = 1
```

然后取前三个前缀和：

```text
u1 = Δ0
u2 = Δ0 + Δ1
u3 = Δ0 + Δ1 + Δ2
```

最后一个区间 `Δ3` 表示 `u3` 到参数域终点 1 的距离。因此输出必然满足 `0 < u1 < u2 < u3 < 1`。

## 6. 训练代理拟合

网络使用所选节点构造固定宽度截断幂基：

\[
\Phi=[1,t,t^2,t^3,m_j(t-u_j)_+^3].
\]

其中 `m_j` 来自确定性的 `knot_mask`。随后在 forward 内可微求解线性系数：

\[
D^*=\arg\min_D\|\Phi D-Q\|_F^2+
\lambda_{poly}\|D_{poly}\|_F^2+
\lambda_{knot}\|D_{knot}\|_F^2.
\]

输出的 `reconstructed_points` 用于训练拟合损失。这个截断幂模型是训练代理，不是最终导出的 CAD B 样条控制多边形。

## 7. forward 中数量从哪里来

### 训练模式

Trainer 调用：

```python
model(points, true_internal_knot_count=true_count)
```

CountHead 仍然正常预测并接受序数监督，但 KnotHead 使用真实 canonical 数量选择表示。这是 teacher-conditioned 节点解码。

### 验证和普通推理

调用：

```python
model(points)
```

此时 KnotHead 使用 `predicted_knot_count`。验证阶段不会读取真实数量来选择节点。

### BIC 部署

模型 forward 后保留所有 `branch_internal_knots`。部署器逐个检查完整数量分支，再选择最终数量。详细过程见 [deployment_pipeline.md](deployment_pipeline.md)。

## 8. 关键输出字段

| 字段 | 含义 |
|---|---|
| `params` | 预测点参数 |
| `count_probabilities` | 0 到 Kmax 的节点数量分布 |
| `predicted_knot_count` | CountHead 原始 argmax |
| `branch_internal_knots` | 所有完整数量分支的节点 |
| `internal_knots` | 当前选中分支的定宽节点张量 |
| `knot_mask` | 当前数量对应的有效槽位 |
| `reconstructed_points` | 截断幂训练代理重建结果 |

## 9. 历史兼容

`checkpointing.py` 根据 objective version 恢复对应模块：

- v5：ordinal local attention + shared count embedding；
- v4：categorical global head + independent branches；
- v3 及更早：Hard-Concrete/ActivityHead。

历史模块不参与新的 v5 训练。
