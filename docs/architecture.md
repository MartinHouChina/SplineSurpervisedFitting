# v16 网络架构与训练数据流

当前新实验是 [中小 K 独立配置](v16_small_medium_k24.md)，`Kc=24`、合成 source K=4..20、MSE 阈值 `1e-4`，完整三次节点向量最多 32 项。它保留此前恢复档的交互实现，但从零训练，关闭高 K 采样，不加载 Kc56 完整权重。此前 Kc56 [本地短训验证](v16_local_self_validation.md)尚未通过简化能力检查；缩小范围和接口测试不能替代新档的性能验证。旧检查点的新增开关默认关闭；下文显式状态交互及全局间隔调整由 `--small-medium-k24` 或历史 `--keep-state-recovery` 启用。

## 前向数据顺序

```text
有序点云 Q[B,M,D]
  -> GeometryEncoder -> 点特征 H[B,M,d]、全局特征 g[B,d]
  -> ParameterHead(H,g,弦长参数) -> 严格递增 t[B,M]
  -> CandidateKnotHead(g,H,t) -> Kc 个有序候选及 token
  -> 正间隔全局重分配 -> U_prop[B,Kc]
  -> 初步 Keep 概率 p0 -> (1-p0)E_drop + p0 E_keep
  -> candidate self-attention + point cross-attention
  -> 最终 importance、曲线自适应 beta、概率 p
  -> mass-TopK -> 一个布尔 KeepMask[B,Kc]
  -> survivor decoder(候选、Keep状态、幸存距离/rank/count)
  -> 更新后的 t、存活节点 U[KeepMask]
  -> 标准三次 B 样条最小二乘 refit -> 控制顶点、曲线、MSE
```

ParameterHead 的输入是编码后的每点特征、全局特征及弦长参考参数；节点头再读取这个参数域中的局部 cross-attention 特征。节点数量由最终概率质量的 ceil/Top-K 规则得到，实际完整节点向量为 `[0,0,0,0] + U[KeepMask] + [1,1,1,1]`。三次样条有 K 个内部节点时，对应 K+4 个控制顶点。

## Keep 与节点位置如何交互

每个候选都有自己的概率、特征 token 和最终 KeepMask。两种状态向量 `E_drop/E_keep` 由全体候选共享，候选自身的概率决定混合比例，因此不同节点得到不同状态特征。初步状态进入 Selector 注意力，最终状态进入 survivor decoder；同一前向内完成，没有逐节点反复推演。

Decoder 的特征还包含最近左右幸存节点距离、幸存序号和数量。只有幸存节点能充当 Key/Value；空掩码使用 sentinel。Decoder 联合更新参数间隔和存活节点位置，已删除节点的原始位置不会约束存活节点只能在旧单元中移动。

硬 KeepMask 不可微。连续 Keep 置信度参与真实的 decoder 前向，因而拟合误差能沿此路径更新 KeepHead；离散组合本身由 Teacher 的分类和排序损失训练，没有使用 straight-through 假装离散删除可微。结构概率视图 detach beta，数量校准负责 beta；恢复档启用 `count_structure_coupling`，让数量损失也能改善共享候选表征。

## 候选全局移动与有序性

原 `bounded_anchor_residual` 将每个节点限制在均匀锚点左右半个单元内。恢复档在正间隔上额外预测有界乘法权重，再将剩余长度归一化到整个 `[0,1]`。这允许累计移动超过半单元，仍保留正间隔、最小 gap 和固定 Kc。新权重零初始化时，间隔调整为恒等，完整热启动不破坏旧候选预测。

## Teacher 与两阶段训练

1. Proposal 用合成真参数、真节点、有序一一匹配和多尺度召回损失学习参数及候选位置。当前中小 K 档从零训练 24 代；历史恢复档从完整旧模型开始，先验证并保存初始结果，再进行短 Proposal 微调。
2. 选出 dense 验证较好的 Proposal，冻结编码器、参数头和候选头（含全局间隔调整）。在固定训练曲线上离线搜索满足 MSE 的候选子集，缓存 KeepMask、数量、节点位置、删除风险及局部交换标签。
3. Joint 训练 Selector 和 survivor decoder，中小 K 档默认 24 代（含 4 代 warmup）。每批同时评价网络自由掩码与 Teacher 强制掩码；Teacher 数量来自预测参数/候选域，不能用源真 K 强迫替代。新边界损失关注 Top-K 边界上的错删、错留，已验证可替换的槽位不被强制当成负例。

Teacher 节点处于冻结 Proposal 参数域。计算位置监督前，将 decoder 输出节点从更新后的参数域映射回 Proposal 参数域；拟合损失仍直接用实际部署参数/节点计算。Teacher 从不作为部署输入。

离线缓存绑定训练样本、Proposal 权重、间隔调整配置与阈值。缓存生成后不能继续修改 Proposal 而复用原标签。原 `online_teacher` 和 `synthetic_ground_truth` 路径仍可用于历史对照，但不等于恢复档的离线教师协议。

## 诊断与模型文件

Joint 日志同时展示数值 Teacher 通过率、Teacher 掩码经 decoder 后的通过率、自由 KeepMask 通过率，以及强制正确数量时的 Top-K recall/exact。前两者高而最后低，优先检查筛选排序；dense 本身失败则先改善候选/参数。

`--init-checkpoint` 只迁移 Proposal。`--init-full-checkpoint` 迁移完整模型、记录源训练数据血统，并新增 `.initial.pt`、`.initial.validation.json`；新优化器和历史从本次实验开始。`.proposal.pt` 是 Joint 使用的冻结初始化，`.pt` 为本次选出的 Joint 检查点，`.last.pt` 用于续跑。历史 r2 预训练使用过真实数据，不能因当前微调只用合成数据就改写这个来源。

完整初始化要求来源已经进入 Joint，不能用 `.proposal.pt` 代替。离线损失版本记录为 `offline_teacher_proposal_frame_positions_v2`：旧 Joint 检查点仍可完整初始化新实验，但不允许保留旧 Adam 状态直接 `--resume` 到新损失。

部署始终为一次网络前向、一次选集、一次最终 refit。阈值是训练目标和评价条件，不构成每例必达标的保证；数值修复属于单独计时和报告的方法。
