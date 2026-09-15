# Kang 与 Luo 在 MSE=1e-4 下的历史 Kc=56 实现复查

## 1. 复查口径

本记录只用于核查历史实现和解释失效阶段，不代表当前正式容量协议。当前正式协议为 Ours `Kc=72`，Synthetic source 与数值基线最大内部节点数为 56；正式统计仍须
运行完整的六方法 benchmark。

Ours 使用 `synthetic_ground_truth_ordered_keep_and_relocation_v4` 简化合同和
`mean_per_curve_subset_cost_v1` checkpoint 选择。Proposal 64 epochs 后按计划
无条件进入 Joint 64 epochs；Joint 的 complexity `0→max` 与 safety `1→0` 仅随
epoch 变化，不读取数据集 pass。完整性合同
`v16_supervised_synthetic_only_pass_rates_report_only_v4` 明确规定 pass 只报告，不参与
STOP、checkpoint 选择或 benchmark eligibility。

- 三次开放 B 样条，192 个输入点；
- 合成 source internal K 为 4..56（控制顶点 8..60），并启用 source-subset 最简性证书；
- 合成节点显式使用 `knot_min_span=0.01`；底层通用 synthetic 生成器的旧 0.02
  默认无法生成具有 57 个 span 的 K=56 层，只属于历史数据合同；
- Kang 与 Luo 的初始内部节点容量均为 56；对三次开放样条，这对应完整节点
  向量 64 项、控制顶点最多 60 个；
- 最终统一执行 CPU float64、无正则的标准 B 样条 refit；
- 横向判据统一为采样点 mean squared Euclidean error，`MSE <= 1e-4`；
- 失败样本保留在通过率分母中，不按难度单独调参。

## 2. 已修正和已确认的实现

### Kang 适配

旧实现先后存在两个问题：最初会把每个活动节点簇无条件合并成一个简单节点；补上单节点/重节点分支后，又对一般曲线的长活动簇强制使用了 Algorithm 4。后者正是海岸线退化到 1--2 个节点的主要原因。论文 Remark 3.2.1 只在活动节点形成明显分组时推荐 Algorithm 4，一般数据应使用 Algorithm 1；论文的 Chebyshev 例子还明确跳过了第二阶段。现在的重定位会：

1. 由活动簇保留左右边界；
2. 用完整当前节点向量上的最小二乘误差缩小区间；
3. 根据 `2 E_double < E_single` 判定最终保留一个还是两个重合节点；
4. 仅当每个活动簇的长度不超过 `degree+1`、符合紧凑簇假设时执行上述 Algorithm 4 路径；
5. 对海岸线式长活动簇保留稀疏阶段活动节点，不再强行压成 1--2 个节点；
6. 在最终 refit 中保留节点重数，并报告活动簇、采用/跳过重定位的原因，以及 native 稀疏阶段可行而最终公共 refit 失败的转移。

这仍是对公开论文目标的可审计工程适配，不是作者 CVX 代码的逐行复刻。
当前仓库仍未复现 Algorithm 1 所需的重复稀疏凸优化；长簇分支采用“保留活动节点”而不是冒充 Algorithm 1。因此真实数据上的结果仍必须标为 adaptation，不能等同论文实现的理论上限。

### Luo 适配

复查确认 Algorithm 3.1 只遍历完整的三项窗口，因此不把首尾 jump 强行加入局部极大值候选是正确的。新增诊断分别记录：

- 按实验公共容量建立的稠密初值 MSE（当前为 56 个内部节点）；
- 稀疏优化阶段的 MSE；
- 局部极值筛选后、DE 前的候选 refit MSE；
- DE 后的最终 MSE。

Luo 的 DE 固定候选节点个数，只更新位置；其原生目标是最大欧氏距离，而横向表格采用 MSE。两种口径不混写，也不把最大距离数值直接当作 MSE。

### 公共阈值保护层

Dung、Kang、Luo 在原生适配完成后统一检查公共 endpoint-constrained refit。若最终 MSE 仍超阈值，benchmark 默认从论文阶段候选与同容量均匀网格中执行残差引导补点；达到阈值立即停止，且不超过统一节点容量。该保护层不读取网络或真值，耗时全部计入方法时间，并在 JSON 中同时保留原生和保护后 `K/MSE`。它是公平比较 wrapper，不属于任何论文原始算法；使用 `--no-published-feasibility-safeguard` 可关闭并复现原生适配行为。

## 3. 可复现单例诊断

样本来自 seed 20000、192 点、控制顶点 8..28 的 certified synthetic 数据；分层抽样
seed 为 20260908。下表是旧 source K=4..24、旧 K64 内部候选配置留下的
**历史实现核查 smoke**，只用于
定位 Kang/Luo 的阶段失效，不是当前 K56 公平容量结果，也不是最终均值：

| 难度 | 数据索引 / 实际 seed | source K | Kang：K / MSE | Luo：K / MSE |
|---|---|---:|---:|---:|
| 简单 | 2679 / 22679 | 4 | 6 / 3.897e-5（通过） | 6 / 4.484e-5（通过） |
| 中等 | 4016 / 24016 | 11 | 13 / 1.355e-4（失败） | 10 / 8.275e-4（失败） |
| 复杂 | 650 / 20650 | 24 | 4 / 4.413e-3（失败） | 20 / 2.880e-3（失败） |

复杂单例的分阶段证据更关键：

- Kang：native 稠密/稀疏阶段 MSE 均约 4.780e-5，已经满足阈值；61 个活动项形成 4 个长簇，组合压缩、重定位及公共端点约束 refit 后失去可行性。由于前后 refit 约束不同，诊断不把全部误差增量武断归因于重定位。
- Luo：稠密阶段约 4.780e-5、稀疏阶段约 9.980e-5，均满足阈值；局部极值筛选后的候选 refit 升至约 3.156e-3，固定 K 的 DE 无法恢复被删除的组合。

所以“简单曲线很好、复杂曲线突然失败”并非统计时删除了失败样本，也不只是迭代次数不够。简单曲线的小节点集搜索容易；复杂曲线的主要瓶颈位于稀疏表示到最终离散节点组合的压缩阶段。增加迭代可改善数值收敛，但不能保证补回已被筛掉的节点。

## 4. 正式对比预算

- Kang：56 个初始内部节点、ADMM 1000 次、lambda 二分 10 次、重定位 12 次；
- Luo：56 个内部节点上限、`eta=0.5`、DE population 20、iterations 100；
- 每种方法使用相同曲线、参数维数、阈值和最终 refit；
- Dung/Kang/Luo 默认启用上述显式公共阈值保护层；正式名称必须标注 `threshold-safe adaptation`，不得写成作者原始实现；
- 同时报告 MSE、通过率、最终内部节点数和完整方法时间。
- 对五个数值基线，`source K=56` 与其方法容量 56 同时到达边界；Ours 则使用
  `Kc=72`，仍有 16 个 Proposal 冗余槽位。所有方法都必须单独报告 K=56 层结果，
  失败仍留在分母中；该统计不构成 checkpoint 或 benchmark 的资格门槛。

完整运行入口见 [训练与评测流程](training_pipeline.md)；实现位于
[Kang 适配](../src/spline_fitting/evaluation/sparse_knot_paper.py) 和
[Luo 适配](../src/spline_fitting/evaluation/luo_linf_de.py)。

当前 K56 正式 benchmark 必须重新运行；不得把上面的历史 K64 smoke 数字改标签后
直接放入 K56 主表。
