# v16 回滚诊断后的 Keep 状态交互修正

> 本地实训更新：交互和完整迁移机制已验证，但短训未改善简化能力；最佳模型近乎全留，末轮删点后拟合退化。此档暂不建议直接长训，先看 [本地验证报告](v16_local_self_validation.md)。下文保留该实验的实现与命令，不能当成效果已达标。

本次改动针对 `v16_r2_rollback_fast_k4_56_r1`：验证部署通过率 17.7%，同一批 53 条合成曲线的通过率由旧 r2 的 83.0% 降到 1.9%。新模型普遍保留约 36–40 个节点。这个运行仅迁移 Proposal 的 41 个张量，Selector/Decoder 随机初始化；600 个样本、Batch=128、Joint=12 代实际上只有 60 次优化更新。因此它没有继承旧模型的筛选能力，也不能代表完整 r2 微调的效果。

## 机制核对与修正

| 环节 | 改动前的 v16 | 本次恢复档 |
|---|---|---|
| 每节点 KeepMask | 有，mass-TopK 一次选择 | 保留 |
| 显式保留/删除嵌入 | 没有；旧 InteractivePruningHead 未被 v16 调用 | 恢复两个可学习状态向量，参与 Selector 和 survivor decoder |
| 幸存节点交互 | 有，只有保留节点能作为 decoder 的 Key/Value | 保留，并加入真实前向可微的 Keep 置信度 |
| Teacher | 最新 Linux 主线仍有离线数值 Teacher；旧回滚代码用在线 counterfactual Teacher | 固定 Proposal 后生成离线可行掩码、数量、风险和局部交换标签 |
| Teacher 节点位置监督 | 直接比较冻结 Proposal 域标签和更新参数后的节点 | 将学生节点映射回 Proposal 参数域后比较 |
| 候选移动范围 | 每个均匀锚点最多移动半个单元；Kc=56 时约 ±0.0087 | 增加正间隔的有界全局重分配，仍保持 56 个有序候选 |
| 微调初始化 | `--init-checkpoint` 只复制 Proposal | `--init-full-checkpoint` 保留整个网络，优化器与训练记录从新实验开始 |

新增网络选项对历史检查点默认关闭。`--keep-state-recovery` 显式启用这些改动；旧完整检查点的新增参数全部零初始化。已用实际 r2 检查点核对：113 个原张量全部复制，在同一输入和部署配置下，参数、节点、KeepMask、概率逐项相等。

## 一次性前向中的交互

每个候选 token 先预测初步概率 `p_j`，再加入 `(1-p_j) E_drop + p_j E_keep`，交给已有候选 self-attention 和点特征 cross-attention。Selector 输出最终概率和自适应 beta，mass-TopK 形成一个硬 KeepMask。Decoder 同时读取硬状态嵌入、连续置信度嵌入、相邻幸存节点距离、幸存 rank 和数量，更新参数及节点位置。

硬掩码仍为布尔值；未选节点不能成为 decoder 的 Key/Value。连续概率影响真实前向的特征和几何输出，所以拟合梯度可沿这条路径到 KeepHead。没有给离散删除操作伪造梯度；离散组合仍主要依靠 Teacher 的 BCE、排序与新增边界损失监督。部署是一次网络 forward、一次选集、一次最终标准 B 样条 refit。

新增边界损失重点拉开“最弱必保留节点”和“最强必删除节点”的分数，避免大量容易分类的节点稀释真正决定 Top-K 的误差。它仅用于数值 Teacher 达标的样本，排除已经测得可相互替代的节点。不能因此保证任意多个局部替代同时发生仍可行。

## 训练顺序与 Linux 命令

默认恢复档：源 K=4..56，Kc=56，MSE 阈值 `1e-4`，训练样本 600，合成验证 160、每个外部来源验证 20，Batch=32。完整加载旧 r2 后先评估并保存初始结果；Proposal 8 代学习参数、候选及全局间隔调整；随后冻结 Proposal，构建离线 Teacher；Joint 24 代学习 Keep 状态交互、排序、数量与存活节点更新。Joint 约有 `ceil(600/32)*24=456` 次更新。

Proposal 学习率 `5e-5`，Selector `5e-5`，decoder `1e-5`；Joint warmup 4 代。安全余量固定为旧 r2 的 `0 + 0.03 sigma`，复杂度附加损失为 0，不再用缩短后的安全/复杂度退火日程替代充分的筛选学习。Teacher 有界交换最多探测 8 个替代位置；Joint 高 K 分层比例为 40%。

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --keep-state-recovery \
  --device cuda \
  --run-name candidate_selection_v16_keep_state_recovery_r1
```

首次缺少外部数据时加 `--prepare-real-data`；先查看完整命令可加 `--dry-run`。默认读取：

```text
outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.pt
```

可以用 `--init-full-checkpoint PATH` 指定其他同结构、同容量且已经进入 Joint 的完整 v16 检查点；误传 `.proposal.pt` 会报错，避免复制未经训练的筛选器。容量或核心权重不匹配也会报错，不会悄悄退化为部分加载。`--init-checkpoint` 仍只迁移 Proposal；`--resume-run` 用本次 `.last.pt` 继续同一实验，两者不能混用。

本次离线 Teacher 位置损失的语义版本为 `offline_teacher_proposal_frame_positions_v2`。旧 Joint `.last.pt` 不能携旧优化器状态直接续跑新损失，应使用新运行名和 `--init-full-checkpoint`；旧 Proposal 尚未开始 Joint，可以记录迁移后续跑。新版本自身的 `.last.pt` 正常续训。Teacher 标签定义未变，不会只因损失版本变化重建缓存；Proposal 或样本变化仍需重新校验。

训练使用合成数据；真实数据仅在本次验证/测试使用。但默认旧 r2 预训练曾使用 35% 真实样本，因此完整热启动后的模型血统是 mixed-pretrained，报告强制标为诊断，不能声称整个模型只见过合成数据。要研究纯合成泛化，应换成有清楚纯合成训练来源的完整检查点或从头训练。

## 如何判断改动是否有效

按同一批曲线比较旧 r2 与新模型，查看 MSE、通过率及 K；再分别看 K>=45 和 K=56。高 K 的 dense 仍不通过时，单改 Keep 无法解决。关键训练诊断为：

- `offline_teacher_numerical_pass_rate`：冻结候选上的数值 Teacher 可行率。
- `supervised_target_pass_rate`：相同 Teacher 掩码经过学生 decoder 后的可行率。
- `deployment_pass_rate`：网络自由选集后的可行率。
- `teacher_count_topk_recall/exact_rate`：强制使用 Teacher 数量时，网络排序能否恢复 Teacher 的节点组合；此项不新增样条求解。
- `offline_teacher_parameter_displacement`：decoder 参数相对冻结 Proposal 的变化。

一条龙输出初始验证快照、Proposal/Joint 检查点、教师缓存、history、旧新同样本配对诊断、六方法四指标图和外部案例图。六方法数据沿用实测结果，额外数值修复单列。训练时长和部署通过率尚需 3090 实测，本地机制测试通过不代表已经达标。
