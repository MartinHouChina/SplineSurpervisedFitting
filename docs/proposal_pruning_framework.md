# 候选生成与 Boehm 消冗框架（规划）

> 状态：本文件描述下一阶段目标架构，尚未在当前代码中实现。当前可运行模型仍是 v6 `InteractiveStructureHead + DynamicKnotDecoder`，详见 [architecture.md](architecture.md)。

## 1. 目标与外部接口

目标仍然是从有序点云预测简洁的开放三次 B 样条：

```text
输入：有序点云 Q [B,M,D]
输出：点参数 t、最终内部节点 U、标准 B 样条控制点 P
```

部署时用户只提供点云。冗余候选节点和冗余控制点均由系统内部产生，不要求用户提供初始样条。

## 2. 总体数据流

```text
有序点云 Q
  ↓
GeometryEncoder + ParameterHead
  ├─ 局部/全局几何特征
  └─ 严格递增参数 t
  ↓
CandidateKnotHead
  ├─ 参数域节点热力图
  ├─ 候选位置残差
  └─ Kc 个高召回候选节点 C
  ↓
用 Q、t、C 最小二乘求冗余控制点 P~
  ↓
构造候选节点 token
  ↓
InteractivePruningHead
  ├─ keep probability
  └─ final position residual
  ↓
保留必要节点并精修位置
  ↓
标准开放三次 B 样条重拟合
```

对于最终节点数量 4–20，建议 `Kc=24` 或 `28`。候选数必须高于最大真实节点数，否则 20 节点样本没有冗余空间。

## 3. 两个模块的职责

### 3.1 CandidateKnotHead：高召回候选生成

候选头不预测随机 Boehm 插入位置，也不要求立即给出最终节点集合。它只需保证每个真实节点附近至少存在一个候选：

```text
优化目标：candidate recall 高
允许结果：候选数量偏多、存在冗余、位置有小偏差
```

候选头输出：

```text
proposal_score:  [B,L]
proposal_offset: [B,L]
candidate_knots: [B,Kc]
candidate_score: [B,Kc]
```

其中 `L` 是参数域热力图分辨率，`Kc` 是固定候选预算。

### 3.2 InteractivePruningHead：消冗与精修

消冗头读取所有候选节点的局部几何、节点间关系和冗余拟合信息，输出：

```text
keep_probability: [B,Kc]
position_residual: [B,Kc]
```

最终数量不是由独立 CountHead 给出，而是保留节点数：

\[
\widehat K=\sum_j\mathbf 1[p_j\ge\tau].
\]

最终位置为：

\[
u_j^{final}=c_j+\delta_j.
\]

## 4. 候选生成头如何监督

Boehm 插入不会改变曲线几何，随机插入位置不能从点云唯一推断。因此 Boehm 插入节点不能作为 CandidateKnotHead 的回归标签。

候选头只由真实最简节点 \(U^*\) 监督。

### 4.1 参数域热力图

在参数网格 \(g_i\) 上构造：

\[
y_i=\max_j\exp\left[-\frac{(g_i-u_j^*)^2}{2\sigma^2}\right].
\]

使用 focal BCE：

\[
L_{heatmap}=\operatorname{FocalBCE}(s_i,y_i).
\]

### 4.2 局部位置残差

对真实节点邻域内的正样本监督：

\[
\delta_i^*=u_j^*-g_i,
\qquad
L_{offset}=\operatorname{SmoothL1}(\widehat\delta_i,\delta_i^*).
\]

### 4.3 单向覆盖损失

\[
L_{cover}=\frac1{K^*}\sum_{j=1}^{K^*}\min_i|u_j^*-c_i|.
\]

必须使用“真实节点到候选”的单向覆盖，不能使用对称 Chamfer Loss；多余候选是设计允许的，不应被强迫重叠到真实节点上。

### 4.4 候选排斥

\[
L_{rep}=\sum_{i<j}\max(0,d_{min}-|c_i-c_j|)^2.
\]

候选头总损失：

\[
L_{proposal}=L_{heatmap}+\lambda_oL_{offset}
+\lambda_cL_{cover}+\lambda_rL_{rep}.
\]

核心指标是 `candidate recall@0.01/0.02` 和 nearest MAE，而不是 candidate precision。进入联合训练前建议 `candidate recall@0.02 > 0.95`。

## 5. Boehm 数据如何监督消冗头

从不可在目标容差内继续删除的简洁样条 \((U^*,P^*)\) 开始，随机插入额外节点并执行标准 Boehm 更新：

\[
(U^*,P^*)\xrightarrow{\text{Boehm insertion}}(\widetilde U,\widetilde P).
\]

插入前后曲线几何严格相同：

\[
C(t;U^*,P^*)=C(t;\widetilde U,\widetilde P).
\]

原始节点标记 `keep=1`，插入节点标记 `keep=0`。同时用实际删除误差复核标签：

\[
E_{del,j}=L_{fit}(\widetilde U\setminus\widetilde u_j)-L_{fit}(\widetilde U).
\]

候选节点 token 建议包含：

```text
候选位置及左右间隔
候选置信度
对应参数邻域的点云特征
冗余控制点一阶/二阶差分
删除该节点后的拟合误差 E_del
```

节点 token 通过 self-attention 交互后预测 keep/remove，避免把相关节点当作相互独立的 Bernoulli 变量。

## 6. 训练数据必须混合

只用精确 Boehm 插入训练会产生分布差异：部署时由带噪点云和候选头得到的冗余样条通常只有近似冗余性。

建议消冗头训练输入逐步混合：

| 来源 | 作用 |
|---|---|
| 精确 Boehm 候选 | 提供无歧义 keep/remove 监督 |
| Boehm 候选加位置扰动并重拟合 | 模拟近似冗余 |
| CandidateKnotHead 真实输出并重拟合 | 对齐最终部署分布 |

标签生成应基于无噪声曲线；网络输入可独立加入噪声、非均匀采样和点序反转增强。

## 7. 三阶段训练

### 阶段 A：候选头预训练

```text
输入：点云
监督：真实最简节点热力图、offset、coverage
目标：candidate recall
```

### 阶段 B：消冗头预训练

```text
输入：点云 + 精确/扰动 Boehm 冗余表示
监督：keep/remove、删除误差、真实节点位置
目标：pruning precision/recall
```

### 阶段 C：联合微调

逐步提高真实候选头输出的占比，例如：

```text
前期：80% Boehm / 20% proposal
中期：50% Boehm / 50% proposal
后期：20% Boehm / 80% proposal
```

联合目标：

\[
L=L_{proposal}+\lambda_kL_{keep}+\lambda_nL_{count}
+\lambda_fL_{fit}+\lambda_dL_{delete}+\lambda_rL_{refine}.
\]

其中：

\[
L_{count}=\left|\sum_jp_j-K^*\right|.
\]

它只监督保留概率总量，不需要独立 CountHead。

## 8. 点云单输入部署

部署不使用 Boehm 算法，也不需要真实节点：

```text
用户有序点云
  → 归一化/重采样
  → 参数预测
  → 候选头生成 Kc 个候选
  → 冗余控制点最小二乘求解
  → 消冗头保留并精修节点
  → 标准 B 样条重拟合
  → 恢复原坐标控制点
```

Boehm 只用于构造消冗监督，不是部署输入要求。

## 9. 评估指标

必须分层报告：

1. `candidate recall@ε`：真实节点是否被候选集合覆盖；
2. `pruning precision/recall/F1`：在给定候选集合下的消冗能力；
3. `end-to-end knot precision/recall/F1`：点云到最终节点的完整性能；
4. `matched MAE`：匹配节点的位置误差；
5. `count accuracy/MAE`：最终保留数量；
6. 标准 B 样条重拟合 RMS 与控制点数量。

对于 4–20 节点，建议主要报告 `match@0.01` 和 `match@0.02`，不使用过宽的 `0.05` 作为主指标。

## 10. 与当前 v6 的关系

| 功能 | 当前 v6 | 规划框架 |
|---|---|---|
| 外部输入 | 点云 | 点云 |
| 数量决策 | categorical 分布的后验中位数 | keep probability 总数 |
| 位置生成 | interval 前缀和 | 热力图候选 + 局部残差 |
| 冗余监督 | 无 | Boehm 插入与删除误差 |
| 节点筛选 | 无 | InteractivePruningHead |
| 部署 BIC | 无 | 无 |

规划框架不是两个独立部署程序，而是共享编码器的两个可训练头，最终仍封装为一次点云推理。
