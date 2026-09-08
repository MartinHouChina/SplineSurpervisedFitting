# v12–v15 模型与数据流

> v15 沿用 v14_joint 的一次性前向结构，训练校准时新增标准 B 样条精确部署 MSE，并让计数损失监督 `mass_topk` 的实际 requested score。部署结构与网络耗时不变；详见 [v15 部署对齐优化](v15_deployment_aligned_optimization.md)。

> v14 keeps candidate proposal, teacher labels and KeepMask in `t0`. After selection and survivor relocation, `ParameterFeedbackHead` combines an explicit learnable network/chord gap blend with pilot, deletion-risk and survivor feedback to produce `t1`; survivors are then transported through the monotone `t0 -> t1` map. It is part of the same forward and adds no spline solve; see [v14_parameter_feedback.md](v14_parameter_feedback.md).

v13 不改变 v12 的部署计算图；它修正 selector、位置联动和教师集合监督的训练语义。以下网络张量与一次性 forward 对两者相同。

## 1. 输入、输出与约束

输入是沿曲线方向排序的点序列：

```text
points: [B, M, D]，D=2 或 3
```

网络输出：

```text
parameters                  [B,M]   严格递增参数 t
proposal_internal_knots     [B,Kc]  固定冗余候选 U_prop
final_keep_probabilities    [B,Kc]  最终保留概率
final_hard_keep_mask        [B,Kc]  一次性离散子集
deployment_internal_knots   [B,Kc]  v12 调整后的候选位置 U_deploy
```

真正部署的内部节点是：

\[
U=U_{\mathrm{deploy}}[M_{\mathrm{keep}}].
\]

`internal_knots` 是 `deployment_internal_knots` 的兼容别名；不要用 `proposal_internal_knots` 代替最终位置。

## 2. 几何编码与参数预测

`GeometryEncoder` 从归一化点、弦长坐标、一阶差分和二阶差分构造有序几何 token。`ParameterHead` 读取编码后的整条点序列，预测正间隔并累积归一化，从而得到：

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

预测参数的位置编码随后加入点特征，供候选节点 cross-attention 使用。数据顺序因此是：先编码点云并预测 `t`，再由节点头读取带参数位置的特征；不是先预测节点数。

## 3. 高召回冗余候选

`CandidateKnotHead` 使用 `Kc+1` 个 interval query。每个 query 以参数域锚点为中心，对点特征做带 Gaussian 局部偏置的 cross-attention，再输出正 interval gap；累积并归一化后得到严格有序候选：

\[
0<u^{prop}_1<\cdots<u^{prop}_{K_c}<1.
\]

局部带宽由 `--candidate-local-attention-bandwidth` 控制。`0` 退回历史全局 attention；当前训练默认 `0.08`。

候选阶段追求 recall，而不是直接得到最少节点。训练教师生成以后，proposal 几何和槽位身份保持固定，使 `teacher_retained_mask[j]` 始终对应同一个 proposal 槽位。

v13 还把仅用于结构描述的 all-candidate pilot 求解提升为 float64，再把系数能量、解析删除增量和局部残差转回网络 dtype。原因是截断幂正规方程在 `Kc=28` 时条件数可达约 `1e8`；直接使用 CUDA float32 会把微小舍入误差放大到 KeepMask。该设置写入 proposal 指纹，改变后必须重建离线教师缓存。

## 4. 一次性节点组合

v12 保留 v11 的固定深度交互路径：

```text
proposal token + U_prop
  -> preliminary keep p0
  -> provisional position feedback
  -> final keep probability p1
  -> mass-TopK 或 threshold
  -> final KeepMask
```

默认 `mass_topk` 根据概率总质量与 Bernoulli 不确定性储备估计一次性保留数量，再全局选择最高分槽位。这里没有逐节点删除循环，也没有单独 `CountHead`。

训练时使用 straight-through keep gate：前向是硬 mask。v13 另外使用 slot-invariant soft teacher-set coverage，把漏掉目标位置的梯度直接传到 keep 概率；实际 survivors 的有序位置匹配训练部署位置。它让“删谁”和“删完后位置是否合理”发生联系，但不能把离散组合优化变成全局最优求解。

## 5. v12 存活节点重定位

### 5.1 v11 的缺口

v11 的教师位置是 retained proposal 的原值，因此位置监督天然偏向零位移。即使网络中有位置头，也缺少“这个删除组合产生后，剩余节点应该移动到哪里”的直接标签。

### 5.2 final-mask 条件化特征

最终 `KeepMask` 确定后，v12 对每个槽位构造相对存活序列的 8 维几何描述：

- 到前一个、后一个存活节点或端点的距离；
- 两侧存活边界形成的 cell 跨度与局部坐标；
- 在存活序列中的相对 rank；
- 存活数量占有效候选数的比例；
- rank 与当前参数位置的覆盖偏差；
- straight-through keep gate。

这些量使用最终存活邻居，而不是原始稠密 proposal 的相邻槽位。删除若干中间节点后，网络看到的是新的真实邻接关系。

### 5.3 selected-only attention

`survivor_relocation_attention` 的 Key/Value 只开放最终存活节点；被删除槽位不参与存活集合的信息聚合。实现仍保留定长 query 以便批处理，但最终位置残差只施加到存活槽位，删除槽位残差强制为零。

```text
final decision token
  + 当前位置编码
  + 存活序列相对几何
  -> selected-only multi-head attention
  -> feed-forward
  -> relocation raw signal
  -> 有界位置残差
```

attention 使每个存活节点一次性读取其余存活节点的分布，因此节点删除、覆盖均匀性和局部位移可以联动。

### 5.4 有界单调更新

对第 `j` 个存活节点，位移只允许落在前后存活邻居或端点之间，并保留 `min_gap`。同时满足：

\[
|u_j^*-u_j^{prop}|\le \Delta_{max},
\]

其中 `Delta_max` 由 `--one-shot-max-position-shift` 控制，v12 训练默认 `0.15`。这保证输出仍位于 `[0,1]`、严格有序，且不会因连续位置头无限漂移。

新 relocation head 零初始化。因此 v11 权重可严格载入 v12；在其余模型配置相同的兼容性检查中，新增分支初始为恒等映射。正式 v12 训练会把位移上限等配置写入新 checkpoint，只有经过联合校准后才会学到非零 relocation。

## 6. v13 训练监督

v13 使用两套互补监督：

1. canonical 标签监督参数、proposal 覆盖和几何位置；
2. 离线 delete-then-relax Hard-RMS 教师监督最终槽位 mask、保留风险、数量、优化后 survivor 绝对位置及边界/相邻 gap。

实际 `final_hard_keep_mask` 产生的 survivors 与 packed relocation teacher knots 做最小代价有序匹配；预测数和教师数相等时再监督完整的边界/相邻 gap。soft teacher-set coverage 与 proposal 槽位编号无关，可同时更新 Keep logits 和位置。survivor spacing 的相对权重由 `--teacher-survivor-spacing-weight` 控制，默认 `0.25`。

校准阶段默认令截断幂 surrogate fit/threshold 权重为零，并始终用真实标准 B 样条验证 rank 比较 distill 与 calibrated checkpoint，防止位置校准牺牲通过率。

## 7. 网络 forward 与标准 refit

v12 learned 部署只运行一次网络 forward。上述 preliminary keep、position feedback、final mask 与 relocation 都在同一固定计算图内，不发生第二次网络调用，也不根据样本进入 while 循环。

网络内部可计算截断幂代理拟合用于训练或诊断；最终曲线仍由筛选后的节点和预测参数执行一次标准开放三次 B 样条控制顶点 refit。代理误差不等同于部署误差。

## 8. 保证边界

v12 是离线教师蒸馏的一次性近似：

- 不保证每条样本都满足 `epsilon`；
- 不保证得到全局最少节点；
- `--fit-tolerance` 在 learned 部署中只用于报告；
- 用户点云若偏离合成训练分布，拟合与结构准确率需要单独验证。

需要低延迟且逐样本检查阈值时使用 `verified`：它先验证一次性结果，只修复失败曲线，并可选择 compact 或极端残差插点。需要更充分的组合/位置搜索时使用 `hybrid`。两者都会执行多次真实 refit，均不属于纯一次性网络结构。
