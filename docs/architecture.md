# v16 网络架构

## 1. 总览

```text
ordered points
  -> GeometryEncoder
  -> ParameterHead
  -> CandidateKnotHead
  -> interactive Selector + adaptive beta
  -> probability-mass Top-K
  -> survivor-conditioned parameter/relocation decoder
  -> one standard cubic B-spline refit
```

当前 certified Synthetic source 为 `K=4..56`，网络容量为 `Kc=72` 个内部候选。即使 source `K=56`，Proposal 仍有 16 个冗余槽位；三次开放样条全保留时完整节点向量为 `4+72+4=80` 项。网络一次产生全部候选和一次离散组合，不是自回归生成，也不是在线逐节点剪枝。

## 2. 张量与模块

| 模块 | 主要输入 | 主要输出 | 作用 |
|---|---|---|---|
| GeometryEncoder | `Q[B,M,D]` | `H[B,M,d]`, `g[B,d]` | 编码坐标、局部差分及全局几何 |
| ParameterHead | `H,g` 与弦长参考 | `t[B,M]` | 预测严格递增参数 |
| CandidateKnotHead | `g,H,t` | `U_prop[B,Kc]`, candidate tokens | 带参数位置编码的局部 cross-attention 候选生成 |
| Selector | candidate tokens、几何 memory、阈值 embedding | importance、`beta`、部署/结构/计数 probabilities | 候选排序和曲线级数量调节，并在训练时解耦两类梯度 |
| mass-TopK | probabilities | `KeepMask[B,Kc]` | 一次性离散选集 |
| subset decoder | context、KeepMask | 更新后的 `t`、`U[B,Kc]` | 仅基于 survivor 交互，联动参数与节点位置 |
| standard refit | `Q,t,U[KeepMask]` | 控制顶点与曲线 | 最终 CPU float64 三次 B 样条解 |

CandidateKnotHead 的候选按参数域有序；局部 cross-attention 的 memory 为 `H + PosEnc(t)`。Selector 再通过候选 self-attention 和对几何 memory 的 cross-attention 计算 keep 分数。

## 3. 训练标签如何进入网络

正式训练 batch 只有 certified Synthetic，因此每行都有 `t*、U*、K*`，以及最简性认证时得到的逐真节点删除误差 `D*`。标签不作为网络输入；它们只用于计算损失：

```text
U_prop --ordered assignment--> target KeepMask
K* --------------------------> mass/count target
t* --------------------------> ParameterHead 与 subset parameter target
U* --------------------------> proposal/relocation position target
D* --------------------------> assigned positive slot 的 criticality/ranking 权重
```

`D*_r` 是从完整 source 节点集中删除第 `r` 个真节点、重新做标准 B 样条最小二乘后的 mean squared Euclidean error。它随 `U*` 通过同一有序一一匹配搬到对应候选槽位，因此不会发生“Keep 标签指向一个候选、风险值指向另一个候选”的错位。该值在合成样本认证时已经生成，不由当前 Selector 排序产生。

真实曲线不出现在训练 batch。真实 manifest 仅供 epoch 后的 held-out validation 和训练后的 benchmark/作图使用。

## 4. Proposal 与 Joint

Proposal 先把参数域和 72 个候选位置学稳定；高 K 分层保证复杂样本在该阶段被充分看到。有序 assignment 与 directed coverage 同时使用：前者保证一一对应，后者保持召回方向。额外的多尺度 recall 项在训练尺度 `0.0025/0.005/0.010` 上惩罚最近候选距离，并对最差 20% 真节点加权，避免宽松的 `0.02` 指标掩盖精细位置偏差。

Joint 的前 8 个 epoch 冻结 GeometryEncoder、ParameterHead 和 CandidateKnotHead，只训练 Selector 与 subset decoder；随后以分组学习率联合微调。目标 mask 由当前 proposal 与真节点的有序匹配直接构造；Selector 学 existence、ranking 和 K，decoder 在实际部署 mask 及标签 mask 条件下学习参数反馈与 survivor relocation。逐候选 BCE 之外还使用：

- Dice：约束预测概率集合与 target KeepMask 的整体重叠；
- ordered CDF：按候选位置排序后约束累计概率质量，减少节点质量偏向局部区域；
- fuzzy negative：未被匹配但靠近真节点的候选属于身份模糊区，只降低其负类 BCE 权重，不把它改成正标签；
- single-deletion risk：在每条曲线内部把 `D*` 转成 tie-aware 相对分位风险，范围为 `0.25..1`；相同删除误差共享 midrank。它保留关键性排序，同时避免所有认证节点的绝对 margin 都远高于阈值时风险一起饱和到 1。

Selector 前向部署 logit 仍为 `s-beta`，其中 `s` 是候选相对 importance，`beta` 是曲线级数量偏移。训练额外暴露两个数值完全相同的视图：Keep BCE/ranking/Dice/CDF/fine-teacher 通过 `s-stopgrad(beta)` 只更新候选排序；count/over-count 通过 `stopgrad(s)-beta` 只更新数量校准。真实部署继续使用两路梯度都有效的 `s-beta`，所以解耦不会改写前向选择结果。

存活节点重定位的可学习 blend 默认从 `0.03` 初始化，而不是从接近零的 sigmoid 饱和区开始。该 blend 控制参数 warp 后的 Proposal 位置与 survivor rank 参考之间的混合，再叠加有界位置残差；旧 checkpoint 加载时仍由保存值精确覆盖。

参数头同时接受逐点 MSE、相邻参数间隔的 log-gap 损失和每条曲线的有符号均值偏差损失。节点在预测参数域和真参数域之间做分段线性 warp 时，forward 值保持不变，而跨任务梯度由 scale 控制：当前 Proposal 为 `0`，Joint 为 `0.1`；它允许节点重定位给参数头有限反馈，避免节点损失完全主导参数化。

正式 Joint 没有在线 Teacher 搜索、Hard-RMS 循环或 Teacher cache。这里的“细粒度 teacher”仅指认证合成样本携带的静态 single-deletion MSE 标签；其来源与当前网络分数无关，且 loss forward 的额外样条求解数为 0。历史 `online_teacher` 分支只用于显式消融，不能与正式结果合并。

## 5. 部署不变式

- 输入只有归一化有序点云和请求 MSE 阈值；
- 不读取合成标签或真实参考折线；
- 只执行一次网络 forward、一次 Top-K 和一次最终 refit；
- 节点严格位于 `(0,1)` 且有序；
- 三次开放完整节点向量为 `[0,0,0,0] + U + [1,1,1,1]`；
- `Kc` 是容量，最终 K 是 KeepMask 的元素数。

## 6. checkpoint 职责

- `<run>.proposal.pt`：Proposal 阶段依次按 dense subset cost、worst-source/aggregate pass、匹配位置误差、参数误差、dense MSE、F1/recall 选择的最佳初始化；它不一定来自第 64 代，也不是正式部署模型。
- `<run>.proposal.final.pt`：Proposal 第 64 代的阶段末状态，仅用于审计和排障；最佳文件存在时，Joint 不用它覆盖验证最佳初始化。
- `<run>.pt`：成熟 Joint 中选出的最佳模型，独立评测和部署应使用它。
- `<run>.last.pt`：最后完成 epoch 的模型、optimizer、RNG 和课程状态，仅用于断点恢复；它可能劣于最佳 `.pt`。
- `<run>.history.json`：保留每个 epoch、`training_phase` 和 `learning_rates` 分组学习率，用于证明 Joint warmup 与全量微调均已发生。

## 7. 当前主线与历史版本

v8–v15 的核心是 Hard-RMS/离线 Teacher 蒸馏；早期 v16 的 counterfactual 版本仍需在线组合搜索。当前 supervised-only v16 用认证合成标签直接监督选择与移动，并复用认证过程的逐节点删除敏感度；它保留一次性部署结构，同时移除在线 self-teacher 和 cache 依赖。旧 checkpoint 不能靠改名升级为当前合同，只能在明确允许且形状兼容时用作 proposal 初始化。新增损失改变了训练目标，因此其效果必须由重训后的独立测试确认。
