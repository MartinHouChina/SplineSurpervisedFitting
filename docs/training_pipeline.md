# v13–v15 数据与训练流程

> 当前默认 v15 保留 v14_joint 数据流，但把联合校准目标改为实际 `mass_topk` 计数和标准 B 样条部署 MSE；训练与部署差异及推荐命令见 [v15 部署对齐优化](v15_deployment_aligned_optimization.md)。

> 当前默认的 `v14_joint` 在 v13 结构之后先得到结构条件参数 `t1`，再让候选节点读取
> `t1` 位置编码的局部几何，联合更新 Keep logits、节点位置和 token；最终 mask 重新选择
> 后只用最终 survivors 再重定位。完整命令和兼容规则见
> [v14_joint 参数—节点联合反馈](v14_joint_parameter_structure_feedback.md)。

v13 沿用 v12 的 proposal、一次性 KeepMask 和 survivor relocation 部署图，修复训练端的五个问题：完整解冻位置到 Keep 的交互路径；按实际预测 survivors 做有序集合位置匹配；用 relocated teacher set 构造槽位无关的 soft coverage；默认关闭会支配校准的截断幂 surrogate；最终同时比较 distill 与 calibrated checkpoint。旧 v12 checkpoint 仍可加载，但需要重新训练才能获得这些修复。

## 1. 学习目标

训练目标不是机械复原生成曲线时的源节点向量，而是在归一化 RMS 阈值 `epsilon` 下学习较小且可行的节点集合：

\[
\min |U|\quad\text{s.t.}\quad
\operatorname{RMS}(C_U,Q)\le\varepsilon.
\]

同一条曲线可能有多套近似等价的 B 样条表示，因此训练同时使用 canonical 几何标签和固定 proposal 上的离线 Hard-RMS 教师。

## 2. 数据集构成

兼容模式下，`SyntheticCubicBSplineDataset` 对每条样本执行：

1. 随机生成开放三次 B 样条的控制顶点和非均匀内部节点；
2. 生成严格有序但可非均匀的参数，并沿曲线采样；
3. 加入坐标噪声；
4. 中心化并按最大半径归一化；
5. 从源节点出发，用真实参数和标准 B 样条 refit 构造阈值简化后的 canonical 标签；
6. 同时保存源样条、canonical 节点、真实参数和有序点云。

默认训练配置：

| 项目 | 值 |
|---|---:|
| train / validation | 10000 / 2000 |
| train / validation seed | 42 / 10000 |
| 独立测试 seed | 20000 |
| 点数 | 192 |
| 维度 | 2，可选 3 |
| 源控制顶点 | 8–24 |
| 源内部节点 | 4–20 |
| proposal 槽位 | 28 |
| 噪声标准差 | 0.001 |
| RMS 阈值 | 0.005 |

关键样本字段：

```text
points
chord_params
true_params
true_internal_knots
true_internal_knot_mask
source_control_points
source_control_mask
source_knot_vector
source_knot_mask
canonical_fit_rms
sample_id
```

当前数据中的 `true_internal_knots` 表示 canonical 内部节点；`source_*` 表示生成时的原始样条。

正式重训建议增加 `--certified-minimal-source`。该模式先固定目标 K，在无噪声高密度参考曲线上用 float64、无正则标准 B 样条 refit 检查全部单节点删除；只有完整节点集通过阈值且任一删除都超过 `epsilon * (1 + margin)` 才接受。观测噪声在证书和标签固定后才加入，因此不再改变 K 或节点位置标签。

```text
--certified-minimal-source
--minimality-margin 0.2
--minimality-max-attempts 16
--minimality-audit-points 512
--oscillation-amplitude 0.3
```

该证书证明的是 source knot 的所有子集中的阈值最少基数，不代表允许任意连续节点重定位后的全局最优。开启后数据分布与 fingerprint 改变，必须重新训练 proposal 并重建 teacher cache。完整审计、字段和迁移说明见[合成数据最简性修正](synthetic_data_minimality_report.md)。

真实 CAD、等高线、海岸线、道路和手写轨迹不能直接把折线顶点当作节点标签。UJI、Natural Earth 和 USGS 已可用于分组隔离的外部测试；也可以冻结 proposal/selector/relocation，仅对 v14 参数反馈头做无标签几何适配。它们仍不进入节点位置或 KeepMask 真值监督。命令和指标口径见[真实数据训练适配与部署测试](real_world_evaluation.md)。

## 3. v11 教师为什么限制了节点移动

v11 的 greedy teacher 在固定 proposal 上删除节点后，把保留 proposal 位置直接写入 `teacher_internal_knots`：

```text
teacher position = U_prop[retained slots]
```

这能监督“保留哪些槽位”，却没有提供删除后连续调整位置的目标。位置损失因而把零移动视为正确答案，尤其容易保留局部聚集的节点。

v12 将教师和 student 一起改为删除与移动联动。

## 4. 阶段 1：高召回 proposal

proposal 预训练更新 GeometryEncoder、ParameterHead 和 CandidateKnotHead；selector、联合位置模块和 survivor relocation 模块冻结。主要目标包括：

- 真实参数监督；
- canonical 节点最近距离和有序位置监督；
- `0.005 / 0.01 / 0.02` 多尺度 candidate coverage；
- 候选排斥与代理拟合约束。

此阶段强制打开全部候选，避免尚未成熟的 selector 阻断 proposal 学习。proposal 目标是高召回，不负责决定最终数量。

若已有语义一致的 v11 proposal，可使用：

```text
--candidate-pretrain-epochs 0
--proposal-checkpoint <v11 proposal checkpoint>
```

脚本会从 checkpoint 恢复 local-attention 带宽，并检查会改变 proposal 的配置。若要继续适配 proposal，则把预训练 epoch 设为正数；给定 checkpoint 会作为初始化，而不是被忽略。

## 5. 阶段 2：delete-then-relax 离线教师

教师输入是固定 `U_prop`，不是源节点标签。每条样本执行：

```text
固定 proposal
  -> greedy Hard-RMS 单节点删除，直到下一次删除会超阈值
  -> 对当前 survivors 做有序 coordinate 位置优化
  -> 在调整后的位置上再次 greedy 删除
  -> 重复有限个 delete/relax round
  -> 若最后一轮仍删除了节点，再对最终 survivors 做一次位置优化
  -> 计算最终 leave-one-out keep risk
  -> 缓存原 proposal 槽位 mask + 优化后 packed knot positions
```

每个位置优化都执行标准 B 样条 refit，并把结果投影到 student 可达域：

- 保持 `[0,1]` 内严格有序；
- 遵守 `relocation_min_gap`；
- 相对原 proposal survivor anchor 的移动不超过 `relocation_max_shift`。

默认教师参数：

| CLI | 默认值 |
|---|---:|
| `--teacher-survivor-relaxation` | 开启 |
| `--teacher-relaxation-rounds` | 2 |
| `--teacher-relaxation-sweeps` | 2 |
| `--teacher-relaxation-grid-size` | 7 |
| `--teacher-relaxation-restarts` | 1 |
| `--one-shot-max-position-shift` | 0.15 |

`--teacher-relaxation-min-gap` 未设置时严格使用 `--min-knot-gap`，默认 `0.001`，且 CLI 不允许它大于 `--min-knot-gap`，因为固定 proposal 必须先满足该间距。它不使用 `--candidate-match-tolerance`；后者只是匹配/监督容差。若要抑制聚集，直接调大 `--min-knot-gap`，让 proposal、student 和 teacher 同步采用该约束。增大 min-gap 是额外建模先验，不是无损的数值技巧。

旧 proposal 的 min-gap 属于固定几何语义。若从 `0.001` 改为 `0.005`，必须令 `--candidate-pretrain-epochs` 大于零以重新适配 proposal，并使用新的 teacher cache；脚本不会允许在零适配模式下静默改变该值。

v12 cache 还保存：

```text
teacher_retained_mask
teacher_soft_keep_risk
teacher_internal_knots
teacher_internal_knot_mask
teacher_count
teacher_fit_rms / teacher_fit_mse
teacher_deletion_order
teacher_single_deletion_rms
teacher_greedy_count / teacher_greedy_fit_rms
teacher_relocation_mean_abs / teacher_relocation_max_abs
teacher_extra_deleted_after_relocation
```

这些标签绑定 proposal 指纹、数据集指纹、教师参数与张量形状。当前版本还在同目录保存 `train.pt.start_domains.json` 与 `val.pt.start_domains.json`，记录 `calibrated/true/chord` 起始域计数。正式训练默认要求真值/弦长 fallback 为 0；非零 fallback 只用于诊断，因为它得到的 mask 不保证在网络部署参数域仍可行。旧 v11/v12/v13 cache 没有这组认证元数据，不能直接复用；第一次迁移必须使用 `--no-reuse-teacher-cache` 重建，之后仅在 proposal、数据和配置完全一致时才可使用 `--reuse-teacher-cache`。

## 6. 阶段 3：一次性选择蒸馏

proposal 冻结，student 学习：

- 最终 hard KeepMask；
- teacher soft keep risk 与 ranking；
- teacher count / probability-mass count；
- canonical 集合辅助监督；
- 复杂度惩罚。

该阶段的主要职责是学组合。标准 B 样条教师给出真实选择标签，截断幂代理默认只做诊断，避免通过“全部保留”轻易降低代理拟合损失。

`mass_topk` 是默认选择策略。它由 keep probability mass 估计 `K`，然后一次性选 Top-K；不需要 CountHead，也不在部署时逐节点试删。

## 7. 阶段 4：KeepMask 与 survivor relocation 联合校准

校准阶段从最佳蒸馏 checkpoint 开始，用较低学习率更新 selector 和完整位置交互模块。v13 的训练目标为：

```text
final KeepMask
  -> selected-only survivor attention
  -> deployment positions
  -> actual predicted survivors 与 relocated teacher set 有序匹配
  + 等长集合的端点/相邻 survivor gap Smooth-L1
  + slot-invariant soft teacher-set coverage
  + mask、risk、count、复杂度监督
  + 可选的代理 fit/threshold 诊断项（默认权重 0）
```

hard KeepMask 在前向中决定实际 survivors；位置损失训练实际存活节点，soft set coverage 直接向 Keep logits 和节点位置提供可微梯度。Key/Value 只来自最终 survivors，位置残差也只施加到 survivors。

校准学习率为：

\[
lr_{relocation}=lr_{selector}\times\texttt{--relocation-lr-scale}.
\]

默认 scale 为 `0.25`。`--keep-position-calibration-epochs` 默认 20；如果设为 `0`，relocation head 的零初始化不会得到训练，不适合作为 v13 联动效果实验。

最终 checkpoint 始终在 distillation 与 calibration 两个候选中，按真实标准 B 样条验证 rank 选择。若校准破坏通过率或 MSE，脚本会保留 distillation checkpoint，而不是强制采用较差的校准结果。

## 8. checkpoint 选择

验证使用真实标准 B 样条部署指标，而不只看代理 loss：

1. 在通过率尚未达到 `--deployment-pass-rate-target` 前，优先真实 pass rate、mean/P95 RMS；
2. 达到目标后，优先更少的保留节点；
3. 再以拟合和节点定位指标打破平局。

该规则是有限验证集上的模型选择，不构成逐样本误差保证。

## 9. 推荐 PowerShell 命令

从 v12 权重迁移，并为稳定 pilot 路径重建 v13 teacher cache：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 0 `
  --keep-position-calibration-epochs 20 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --teacher-survivor-relaxation `
  --teacher-relaxation-rounds 2 `
  --teacher-relaxation-sweeps 2 `
  --teacher-relaxation-grid-size 7 `
  --teacher-relaxation-restarts 1 `
  --one-shot-max-position-shift 0.15 `
  --relocation-lr-scale 0.25 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v13_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v13.pt
```

v13 的 `stable_pilot_descriptors=True` 使用 detached float64 pilot solve，避免条件数约 `1e8` 的 float32 正规方程把微小 CUDA 舍入误差放大成不同 KeepMask。该语义写入 proposal 指纹。当前版本又增加了教师起始域认证，因此包括旧 v14 在内的历史 cache 都不可直接复用：首次运行须显式使用 `--no-reuse-teacher-cache` 生成 `.pt` 和对应的 `.pt.start_domains.json`；只有同一个 proposal、固定数据样本、教师配置、shape 和认证元数据全部不变时，后续运行才可添加 `--reuse-teacher-cache`。

主要输出：

```text
candidate_pruning_one_shot_v13_distill.pt
candidate_pruning_one_shot_v13_calibrated.pt
candidate_pruning_one_shot_v13.pt
candidate_pruning_one_shot_v13_last.pt
candidate_pruning_one_shot_v13_teacher/train.pt
candidate_pruning_one_shot_v13_teacher/val.pt
candidate_pruning_one_shot_v13_teacher/train.pt.start_domains.json
candidate_pruning_one_shot_v13_teacher/val.pt.start_domains.json
```

当 `candidate-pretrain-epochs > 0` 时，还会生成 `*_proposal.pt`。

## 10. 小规模链路验证

先用少量固定样本检查缓存、蒸馏和校准链路，再开始正式训练：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 4 `
  --candidate-pretrain-epochs 0 `
  --keep-position-calibration-epochs 1 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --train-size 8 `
  --val-size 4 `
  --batch-size 2 `
  --teacher-batch-size 2 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --teacher-relaxation-rounds 1 `
  --teacher-relaxation-sweeps 1 `
  --teacher-relaxation-grid-size 3 `
  --teacher-cache-dir outputs/tmp/_smoke_v13_teacher `
  --no-resample-train-each-epoch `
  --output outputs/tmp/_smoke_v13.pt
```

这条命令只验证代码路径，不代表正式训练配置或性能结论。
