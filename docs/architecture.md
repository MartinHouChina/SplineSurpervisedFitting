# v6 模型内部投喂顺序

## 1. 完整 forward

```python
# 输入点只先进入几何编码器
local_features, global_features = encoder(points)

# 参数头读取局部特征和全局特征
parameter_output = parameter_head(local_features, global_features)
params = parameter_output["params"]

# 交互式结构头读取参数化局部几何，直接预测最终数量分布
structure_output = structure_head(
    global_features,
    local_features,
    params,
)
predicted_count = structure_output["predicted_knot_count"]

# 训练取真实数量；验证和部署取网络预测数量
selected_count = true_count if training else predicted_count

# K>0 时只解码 selected_count 对应的 K+1 个区间 query；K=0 时跳过
knot_output = knot_head(
    global_features,
    local_features,
    params,
    structure_output["structure_query_features"],
    selected_count,
)

# 使用所选节点进行训练代理拟合
design = build_design_matrix(params, internal_knots, knot_mask)
coefficients = solve_coefficients(design, points)
reconstructed_points = design @ coefficients
```

依赖关系：

```text
points
  ↓
GeometryEncoder
  ├─ local_features ─┬─ ParameterHead ─→ params
  └─ global_features ┘                    │
                                         ↓
                              InteractiveStructureHead
                                ├─ P(K)
                                └─ structure_query_features
                                         │
                 true K（仅训练）或 predicted K
                                         ↓
                                DynamicKnotDecoder
                                         ↓
                                  K 个有序节点
```

## 2. GeometryEncoder

输入：

```text
points: [B,M,D]
```

编码器使用坐标、弦长归一化一阶导数和二阶导数，输出：

```text
local_features:  [B,M,H]
global_features: [B,H]
```

## 3. ParameterHead

ParameterHead 不读取原始点。它把全局特征复制到每个采样位置，与局部特征拼接：

```text
local_features:   [B,M,H]
global_expanded:  [B,M,H]
fused:            [B,M,2H]
```

MLP 对前 `M-1` 个位置预测原始参数间隔：

```text
raw_parameter_gaps: [B,M-1]
```

间隔经 softmax、最小间隔约束和累加得到：

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

输出 `params:[B,M]` 被后续两个模块共同使用。

## 4. InteractiveStructureHead

### 4.1 Cross-attention

构造带参数位置编码的 memory：

\[
F_{memory}=F_{local}+\operatorname{PE}(t).
\]

一次性使用 `Kmax` 个结构 query：

\[
Z=\operatorname{CrossAttention}(Q_{structure},F_{memory}).
\]

这里的 `Kmax` 个 query 是数量判断所需的局部结构探测器，不是最终输出的候选节点。

### 4.2 Query self-attention

结构 query 之间交换信息：

\[
\widetilde Z=\operatorname{SelfAttention}(Z).
\]

它允许模型判断不同局部几何证据是互补还是冗余。

### 4.3 从停止风险得到数量分布

第 \(j\) 个 token 输出条件停止概率 \(h_j\)。例如：

\[
P(K=0)=h_1,
\]

\[
P(K=1)=(1-h_1)h_2,
\]

\[
P(K=2)=(1-h_1)(1-h_2)h_3.
\]

最后一类为一直没有停止：

\[
P(K=K_{max})=\prod_{j=1}^{K_{max}}(1-h_j).
\]

所有类别概率天然非负且和为 1。输出：

```text
structure_query_features:          [B,Kmax,H]
structure_stop_logits:             [B,Kmax]
structure_survival_probabilities:  [B,Kmax]
count_probabilities:               [B,Kmax+1]
predicted_knot_count:              [B]
```

这里没有独立 CountHead，也没有对 activity 做阈值删除。

## 5. DynamicKnotDecoder

输入：

```text
global_features
local_features
params
structure_query_features
selected_count
```

最终节点 query 同时读取：

- 带参数位置编码的局部几何 token；
- 经过交互的 structure query token；
- 全局曲线特征；
- `Embedding(K)`。

### 5.1 只计算所需数量

假设一个 batch 的 `selected_count` 为：

```text
[2,2,4,3,4,2]
```

解码器按数量分组：

```text
K=2：样本 0,1,5 → 每条使用 3 个 interval query
K=3：样本 3     → 使用 4 个 interval query
K=4：样本 2,4   → 每条使用 5 个 interval query
```

不会计算 K=0、1、5、6 的节点表示，也不会产生 `branch_internal_knots`。

输出仍填充成 batch 定宽形式：

```text
internal_knots: [B,Kmax]
knot_mask:      [B,Kmax]
```

填充只用于张量存储。例如 K=3：

```text
internal_knots = [u1,u2,u3,0,0,0]
knot_mask      = [1, 1, 1, 0,0,0]
```

### 5.2 有序节点生成

给定正整数 K，只取前 `K+1` 个共享 interval query，输出正区间；K=0 时直接返回空节点向量：

\[
\Delta_j=\delta+[1-(K+1)\delta]\operatorname{softmax}(a)_j.
\]

节点为前缀和：

\[
u_j=\sum_{r=0}^{j-1}\Delta_r,\qquad j=1,\ldots,K.
\]

因此节点天然严格有序。

## 6. 计算复杂度

结构头固定使用 `Kmax` 个 query：cross-attention 复杂度为 \(O(BK_{max}M)\)，query self-attention 为 \(O(BK_{max}^2)\)。位置解码在 K>0 时只使用每条曲线的 `K+1` 个 query，K=0 时不运行，其主要注意力复杂度为：

\[
O\left(\sum_{b:K_b>0}(K_b+1)(M+K_{max})\right).
\]

因此节点位置解码不再枚举全部数量分支。结构 query 的 self-attention 仍为 \(O(K_{max}^2)\)；当 `Kmax=20` 时是固定的 400 个 query-pair，但不适合无限增大 `Kmax`。

## 7. 训练代理

动态节点通过 `knot_mask` 构造截断幂基：

\[
\Phi=[1,t,t^2,t^3,m_j(t-u_j)_+^3].
\]

forward 内可微求解线性系数并产生 `reconstructed_points`。该曲线用于训练拟合损失，不是最终导出的标准 B 样条控制多边形。

## 8. 训练、验证、部署的数量来源

| 阶段 | selected_count | 说明 |
|---|---|---|
| 训练前期 | canonical 真实数量 | 稳定监督动态位置解码器 |
| 训练后期 | 按 teacher-forcing 比例混合真实数量和预测数量 | 缩小训练与部署的输入差异 |
| 验证 | `argmax count_probabilities` | 不使用 teacher count |
| 部署 | `argmax count_probabilities` | 最终且唯一的数量决策 |

部署流程见 [deployment_pipeline.md](deployment_pipeline.md)。
