# 结构预测方案演化

本文只说明各版本解决的问题和部署语义。当前主版本是 v11；历史版本仍可通过 checkpoint 兼容层读取。

## v3：独立 activity 与 Hard-Concrete

固定候选分别预测 activity，再用 Hard-Concrete 二值化。它把节点身份、节点数和拟合梯度压在同一组独立 Bernoulli 门上，容易出现全保留、全删除或同一曲线内区分度不足。

## v4：直接 CountHead

```text
CountHead 预测 K
  → 选择 K 专属分支
  → 输出长度为 K 的节点向量
```

它消除了 activity threshold，但训练样本被多个计数分支分散。

## v5：canonical 数量监督与条件分支

v5 引入 canonical 标签、序数 CountHead 和共享 count-conditioned decoder。部署曾用 BIC/先验比较完整数量分支，解释和训练/部署一致性仍不理想。

## v6：交互结构头与动态解码

结构 query 先读取点云，再通过 self-attention 联合预测节点数；动态解码器只生成选定长度的节点向量。它减少无用位置分支，但计数错误仍会立即改变整个节点向量长度。

## v7：冗余 proposal + Hard-RMS 删除

```text
点云
  → 固定预算高召回 proposal
  → 贡献特征与删除建议
  → 标准 B 样条逐节点试删
```

最终节点数由真实 RMS 阈值决定，可解释性强，但每次部署需要多次 refit。

## v8：离线教师 + 一次性 LearnedKeep

把 v7 的 Hard-RMS 搜索移到训练前：

```text
离线：proposal → Hard-RMS → mask/risk/count cache
在线：一次网络 forward → KeepMask → 一次标准 refit
```

速度提升来自不再在线试删；代价是阈值只具统计满足率，不再逐样本硬保证。

## v9：固定 proposal 与独立 selector

v8 的 Keep/位置联动会使教师缓存绑定的候选槽位漂移。v9 将 proposal 几何冻结，selector 只读取固定位置并预测 mask，保证 teacher slot 一致性。

## v10：结构化一次性组合

v10 在 v9 上加入：

- 两层 selector interaction；
- teacher 槽位概率质量的累计分布损失；
- critical retained-slot recall；
- probability-mass Top-K；
- Bernoulli 不确定性安全余量；
- 可选参数域覆盖锚点。

它改善的是组合建模能力，但最终位置仍等于固定 proposal；selector 选对附近槽位后，不能再把它向 canonical 节点校准。

## v11：局部 proposal 与 Keep/位置固定交互

v11 针对两个可分解问题更新。

### 1. proposal 召回

CandidateKnotHead 从“有位置编码但全局的 cross-attention”改为锚点中心 Gaussian 局部 cross-attention，并在多个匹配容差上直接训练 coverage：

```text
局部几何 + 参数位置
  → Gaussian-biased interval queries
  → 严格有序 Kc 候选
  → 0.005/0.01/0.02 多尺度 coverage
```

该设计的目标是让每个 interval query 更稳定地读取对应参数区间，并减少平均 coverage 掩盖少数漏检的情况。是否提升召回必须由独立测试确认。

### 2. selector 与位置更新

v11 保留固定 teacher proposal，同时增加独立 deployment 位置：

```text
U_prop
  → p0
  → 临时位置 u1
  → 位置反馈后的 p1
  → 一次性 KeepMask
  → mask 条件化最终位置 u*
```

这是固定两次概率预测和两次位置预测，不是逐次节点生成或迭代优化。部署仍只执行一次网络前向和一次标准 B 样条 refit。

### 3. 双监督

```text
Hard-RMS teacher：决定固定 proposal 中保留哪些槽位
canonical labels ：监督候选覆盖、辅助选择和最终位置
```

`U_prop` 冻结以保护教师标签；`U*` 单独更新以提高最终节点定位精度。

### 4. 计数与误保留

v11 额外约束：

- teacher false-positive，但豁免 canonical-positive 的可替代槽位；
- 实际 `mass_topk` requested-count score；分数小于 `0.5` 时稳定输出零节点，正数仍使用 `ceil`；
- canonical existence；
- 最终 teacher-retained slot 的位置。

这些项用于缓解“召回较高但 precision 低”和“概率和正确但 `ceil` 后节点数偏大”。它们不构成性能保证。

位置校准还加入小权重截断幂 fit/threshold 信号。fit gate 对选择概率 detach，避免沿门控梯度回到 all-keep；该信号只承担已选组合的位置可行性约束。两阶段位置修正共享同一个相对 `U_prop` 的总位移预算。

## 版本对比

| 版本 | 候选/数量机制 | 位置机制 | 部署 |
|---|---|---|---|
| v3 | 独立 activity + Hard-Concrete | 固定候选 | threshold |
| v4 | categorical CountHead | 数量专属分支 | argmax |
| v5 | ordinal CountHead | 全数量条件分支 | BIC/先验 |
| v6 | 交互结构 query 预测数量 | 仅解码选定长度 | posterior median |
| v7 | 冗余 proposal，RMS 决定数量 | proposal 局部精修 | 在线逐节点 Hard-RMS |
| v8 | 离线教师蒸馏 KeepMask | 固定次数 Keep/位置反馈 | 一次 mask + 一次 refit |
| v9 | 固定 proposal 上的独立 selector | 教师绑定位置不动 | 一次 mask + 一次 refit |
| v10 | mass-TopK + 集合约束 | 固定 proposal | 一次结构化 mask + 一次 refit |
| v11 | 局部 proposal + mass-TopK | `p0→u1→p1→mask→u*` | 一次 forward + 一次 refit |

## 兼容边界

- v10 checkpoint 按原固定位置语义恢复，v11 不会静默改写它。
- v10 proposal 可作为 v11 参数初始化。
- 推荐在加载 v10 proposal 后执行 5–10 轮 v11 proposal 适配，以训练新增的局部 attention 行为。
- `candidate-pretrain-epochs=0` 只复用固定 proposal，不会适配 Gaussian 局部 attention；脚本保留 checkpoint 带宽，并校验 `model_config` 中影响 proposal 的非权重语义。
- 新选出的 proposal checkpoint 会保存 `model_config`；缺少该配置的历史 checkpoint 只能警告并按兼容语义加载。
- v11 必须重新生成 teacher cache；v10 cache 不应直接复用。
- v7 Hard-RMS 仍保留为离线教师和显式 diagnostic。

统一实验应同时报告 proposal recall、三阶段节点 Precision/Recall/F1/MAE、最终标准 B 样条 RMS、阈值满足率、节点数和时间。
