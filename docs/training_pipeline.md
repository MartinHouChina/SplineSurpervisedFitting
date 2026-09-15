# v16 supervised-only 训练流程

本文描述当前正式协议，不描述 v8–v15 的 Teacher-cache 训练。当前协议保留合成最简性证书给出的逐节点删除 MSE 作为细粒度监督，但不恢复在线 self-teacher、subset search 或 Teacher cache。

## 1. 数据边界

训练集只含在线生成的 certified Synthetic。每条训练样本包含：

- 归一化有序点云 `Q`；
- 干净源曲线上的真参数 `t*`；
- 真内部节点 `U*` 与有效 mask；
- 精确源节点数 `K*`；
- source-subset minimality 证书字段；
- 对每个真内部节点的 single-deletion MSE `D*`。

正式参数为 source `K=4..56`、控制顶点 8–60、每条曲线 192 点、`knot_min_span=0.01`、`MSE tolerance=1e-4`。噪声只加到网络观测，标签来自干净源曲线。

`D*_r` 的计算方式是从完整 source 节点集中删除第 `r` 个节点，再用干净审计点和真参数做标准三次 B 样条 refit。它与最简性证书共用同一 CPU float64 结果，单位是 mean squared Euclidean，不是 RMS。`v16_mixed.py` 将该向量与 mask 填充到 `Kc=72`；真实或未认证样本的该字段无效，不能参与这项监督。

UJI Pen、Natural Earth、USGS 和 IndustrialOffset manifest 可以传给训练入口，但只建立留出验证集；正式配置必须是 `--real-fraction 0`。外部样本不进入 optimizer step。

## 2. Proposal 阶段（epoch 1–64）

该阶段训练 GeometryEncoder、ParameterHead 和 CandidateKnotHead，Selector 与 selected-only decoder 尚不承担最终组合学习。

主要监督为：

1. 真参数回归；
2. 真节点到候选集的 recall-direction coverage；
3. 有序、单调、一一匹配的候选位置损失；
4. 多尺度 Proposal recall：训练尺度为 `0.0025/0.005/0.010`，并额外关注最差 20% 真节点；
5. 参数相邻间隔的 log-gap 与整曲线有符号 bias；
6. 全候选可微 B 样条拟合损失。

正式容量为 `Kc=72`、source `K*=4..56`，所以每条样本都满足 `Kc>K*`。一一匹配只选择 `K*` 个互异且保持顺序的候选；即使 `K*=56`，也仍有 16 个冗余槽位用于覆盖位置误差。这与“高 K 样本过采样”是两个不同概念。Proposal 的合成抽样有 50% 来自 `K>=40`，其余来自低 K 区间，以加强容量边界召回。

第 64 个 epoch 后按计划无条件进入 Joint。aggregate pass 不会延长 Proposal 或触发 STOP。

## 3. Joint 阶段（epoch 65–128）

Joint 的前 8 代是 Selector warmup：冻结 GeometryEncoder、ParameterHead 和 CandidateKnotHead，只训练 Selector 与 selected-only decoder，使 Keep 排序、数量预测和存活节点重定位先适配已经收敛的候选集合。warmup 结束后全部模块解冻，并采用分组学习率：Selector `2e-4`、GeometryEncoder/CandidateKnotHead `1e-5`、ParameterHead `5e-5`、selected-only decoder `5e-5`。各组分别执行同一个 `grad_clip` 上限，避免 Selector 的大梯度经共享特征立即破坏 Proposal。

Joint 恢复 source `K=4..56` 的原始抽样分布，并对每条合成样本执行：

```text
U_prop + U* -> 有序最小代价一一匹配 -> target KeepMask
K*                                      -> count target
t*, U*                                  -> parameter/relocation targets
D*                                      -> positive-slot criticality targets
```

直接监督项包括：

- existence/KeepMask 的类别平衡 BCE；靠近真节点但未被有序匹配的候选按距离降低负类权重，避免任意槽位身份切换造成过强惩罚；
- Keep 概率对标签 mask 的 Dice 与按候选位置排序后的 CDF 损失，分别约束集合重叠和空间概率质量；
- 正候选应排在负候选之前的 ranking loss；
- 将 `D*/epsilon` 的 log margin 映射为逐正候选 criticality，按删除风险加权 Keep 与 ranking；
- 自适应 probability mass 对真 K 的 count loss 与 over-count loss；
- 部署 mask 和标签 mask 两条 selected-only 解码路径的参数、节点位置及拟合损失；其中真节点到存活节点的定向覆盖项会显式惩罚漏节点；
- Proposal 阶段的参数、coverage、ordered-assignment、多尺度 recall、parameter log-gap/bias 监督继续保留。

因为 `K*` 是精确标签，正式监督模式固定 `--complexity-weight 0`；否则额外的自由稀疏惩罚会与真节点数监督冲突。

正式 Joint 不调用在线 subset search，不运行 ranked-prefix、counterfactual 或 geometry-oracle Teacher，也不读取/写入 Teacher cache。逐节点删除 MSE 由合成样本生成/认证阶段提供，loss forward 不为它增加样条求解；每批仍只计算 dense、实际部署 mask 和标签 mask 所需的拟合分支。

节点位置监督需要把预测节点从预测参数域 warp 到 `t*` 域。`proposal_parameter_warp_gradient_scale=0` 只阻断 Proposal 节点匹配通过 warp 反推 ParameterHead；`joint_parameter_warp_gradient_scale=0.1` 保留 Joint 的有限跨任务反馈。两者都不改变 forward 中的 warp 数值，只缩放这条梯度路径；参数头仍持续接受直接的真参数、log-gap 和 bias 监督。

## 4. 训练时新增诊断

终端和 history 中应重点查看：

| 指标 | 含义 |
|---|---|
| `proposal_recall_at_005/.010/.020` | 每个真节点是否在相应参数距离内至少有一个候选；这是 Proposal 召回，不是最终 Keep 召回 |
| `proposal_knot_assignment_mae` | 保持顺序的一一匹配位置 MAE；可识别多个真节点共用同一候选的问题 |
| `keep_mask_precision/recall/f1` | 部署离散 mask 相对有序匹配标签的槽位分类指标 |
| `critical_false_delete_rate` | `teacher_risk>=0.75` 的关键正槽位被部署 mask 删除的比例 |
| `fine_teacher_mean_risk` | 认证删除 MSE 映射后的平均关键性 |
| `fine_teacher_mean_log_delete_margin` | `log(D*/epsilon)` 的均值；正值表示单删误差高于工程阈值 |
| `fuzzy_negative_fraction` | 负类中因靠近真节点而被降权的比例 |
| `parameter_bias_mae` | 每条曲线参数误差有符号均值的绝对值再取平均 |
| `parameter_gap_loss` | 相邻参数 gap 的对数相对形状误差 |

这些训练指标用于定位 Proposal、Selector 和 ParameterHead，不替代独立验证的 MSE、通过率、最终 K、count MAE 和节点匹配结果。

## 5. 选择与 checkpoint

Proposal checkpoint 仅作为 Joint 初始化，按以下连续字典序选择：dense subset cost、最差来源通过率、总体通过率、matched-knot MAE、参数 RMSE、dense MSE、节点 F1/recall。这样既避免“72 个候选整体能拟合”完全掩盖候选几何，也不会因 K=56 boundary 暂时为零，就让单个宽容差 recall 的微小波动覆盖后期显著更好的拟合和细尺度位置精度。正式 `synthetic_ground_truth` 模式还强制 `one_shot_coverage_bins=0`，防止固定分箱锚点挤掉有序匹配得到的真节点槽位。

成熟 Joint checkpoint 使用 `mean_per_curve_subset_cost_v1` 排名：

- 单曲线可行时，优先较少节点，MSE 只作有界 tie-break；
- 单曲线不可行时，只按相对阈值的 MSE 惩罚，不奖励少节点；
- 全数据集 pass rate 不作为阶段切换、停止训练或正式资格的硬门槛；Proposal 选模将连续 dense subset cost 置于首位，通过率紧随其后。

正式 checkpoint 还必须记录：supervised objective、synthetic-only、`joint_supervision=synthetic_ground_truth`、`online_teacher=false`、有序标签映射、最终 safety 设置、source K56/Kc72 冗余容量合同，以及已经完成的 Selector warmup 与 Joint 课程。启用细粒度监督时还应记录 `fine_grained_teacher=certified_source_single_deletion_mse`、误差单位 `mean_squared_euclidean`、loss-forward 额外 spline solve 数为 `0`，以及所有新增权重、分组学习率和 warp-gradient scale。

训练会产生：

```text
<run>.pt          最佳成熟 Joint
<run>.proposal.pt 最佳 Proposal
<run>.proposal.final.pt Proposal 阶段末状态（审计/排障）
<run>.last.pt     最新优化器/RNG/训练状态，用于恢复
<run>.history.json
```

## 6. 推荐一条龙命令

Linux/RTX 3090（缺少真实数据 manifest 时由脚本准备）：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised_linux
```

Windows/RTX 3090：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised `
  -Device cuda `
  -Epochs 128 -ProposalEpochs 64 `
  -SelectorWarmupEpochs 8 `
  -SelectorLr 2e-4 -ProposalJointLr 1e-5 `
  -ParameterJointLr 5e-5 -DecoderJointLr 5e-5 `
  -TrainSize 3000 -ValSize 600 -RealValSize 100 `
  -BatchSize 64 -NumWorkers 4
```

四个外部 manifest 必须预先存在，或给一条龙入口加数据准备开关。一条龙入口 fresh-only，检测到同名 checkpoint、log、comparison 或 figure 时会拒绝覆盖；中断训练应使用 `scripts/train_v16.py --resume <run>.last.pt`，且除允许项外必须保持原实验参数和 output 一致。

训练入口的关键显式参数为：

```powershell
python scripts/train_v16.py `
  --epochs 128 --proposal-epochs 64 `
  --selector-warmup-epochs 8 `
  --selector-lr 2e-4 --proposal-joint-lr 1e-5 `
  --parameter-joint-lr 5e-5 --decoder-joint-lr 5e-5 `
  --train-size 3000 --val-size 600 --synthetic-boundary-val-size 32 `
  --real-val-size 100 --real-fraction 0 `
  --min-control-points 8 --max-control-points 60 `
  --candidate-knots 72 --num-points 192 --batch-size 64 `
  --knot-min-span 0.01 --mse-tolerance 1e-4 `
  --certified-minimal-source --minimality-margin 0.2 `
  --minimality-audit-points 512 `
  --proposal-high-k-fraction 0.5 --proposal-high-k-min-knots 40 `
  --proposal-knot-assignment-weight 1 `
  --proposal-multiscale-recall-weight 0.25 `
  --keep-dice-weight 0.5 --keep-cdf-weight 0.25 `
  --fine-teacher-weight 0.5 --fine-teacher-ranking-weight 0.25 `
  --fine-teacher-temperature 0.5 `
  --keep-fuzzy-negative-radius 0.01 --keep-fuzzy-negative-floor 0.1 `
  --parameter-gap-weight 0.05 --parameter-bias-weight 0.1 `
  --proposal-parameter-warp-gradient-scale 0 `
  --joint-parameter-warp-gradient-scale 0.1 `
  --joint-supervision synthetic_ground_truth `
  --synthetic-count-role exact `
  --no-synthetic-geometry-oracle-teacher `
  --one-shot-selection-policy mass_topk `
  --initial-keep-fraction 0.4166666666666667 `
  --one-shot-coverage-bins 0 --min-selected-knots 4 `
  --one-shot-safety-sigma 0 --one-shot-safety-knots 0 `
  --final-safety-sigma 0 --final-safety-knots 0 `
  --complexity-weight 0 `
  --real-manifest data/splits/uji_pen_v2.jsonl `
  --real-manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --real-manifest data/processed/usgs_contours/large_scale/manifest.jsonl `
  --real-manifest data/processed/industrial_offsets/v1/manifest.jsonl `
  --device cuda `
  --output outputs/checkpoints/candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised.pt
```

Linux 直接训练使用相同参数名，只把 PowerShell 续行符替换为反斜杠。例如新增监督部分为：

```bash
python scripts/train_v16.py \
  --epochs 128 --proposal-epochs 64 \
  --selector-warmup-epochs 8 \
  --selector-lr 2e-4 --proposal-joint-lr 1e-5 \
  --parameter-joint-lr 5e-5 --decoder-joint-lr 5e-5 \
  --train-size 3000 --val-size 600 --batch-size 64 \
  --min-control-points 8 --max-control-points 60 \
  --candidate-knots 72 --num-points 192 \
  --knot-min-span 0.01 --mse-tolerance 1e-4 \
  --certified-minimal-source --minimality-margin 0.2 \
  --minimality-audit-points 512 \
  --proposal-high-k-fraction 0.5 --proposal-high-k-min-knots 40 \
  --proposal-knot-assignment-weight 1 \
  --proposal-multiscale-recall-weight 0.25 \
  --keep-dice-weight 0.5 --keep-cdf-weight 0.25 \
  --fine-teacher-weight 0.5 --fine-teacher-ranking-weight 0.25 \
  --fine-teacher-temperature 0.5 \
  --keep-fuzzy-negative-radius 0.01 --keep-fuzzy-negative-floor 0.1 \
  --parameter-gap-weight 0.05 --parameter-bias-weight 0.1 \
  --proposal-parameter-warp-gradient-scale 0 \
  --joint-parameter-warp-gradient-scale 0.1 \
  --joint-supervision synthetic_ground_truth \
  --synthetic-count-role exact --real-fraction 0 \
  --no-synthetic-geometry-oracle-teacher \
  --one-shot-selection-policy mass_topk \
  --initial-keep-fraction 0.4166666666666667 \
  --one-shot-coverage-bins 0 --min-selected-knots 4 \
  --one-shot-safety-sigma 0 --one-shot-safety-knots 0 \
  --final-safety-sigma 0 --final-safety-knots 0 \
  --complexity-weight 0 \
  --real-manifest data/splits/uji_pen_v2.jsonl \
  --real-manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl \
  --real-manifest data/processed/usgs_contours/large_scale/manifest.jsonl \
  --real-manifest data/processed/industrial_offsets/v1/manifest.jsonl \
  --device cuda \
  --output outputs/checkpoints/candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised.pt
```

上面权重是当前 runner 使用的起点，不是已经证实的最优超参数。启用这些项后必须用新 `RunName` 从头训练；旧 `.pt` 仅能按兼容规则初始化 Proposal，不能当作细粒度版本的正式结果。

## 7. 训练后的串行产物

一条龙脚本随后执行：

1. `inspect_v16_checkpoint.py`；
2. `benchmark_v16_datasets.py --method-set published`；
3. `plot_v16_method_comparison.py --reference`；
4. `visualize_v16_real_deployments.py`。

得到 checkpoint、六方法×五数据集表、逐样本记录、input/reference 两张 2×2 图和外部数据六方法案例图。所有数值必须来自生成的 `comparison.json`/CSV；未运行时不得在报告中填写推测结果。
