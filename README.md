# Minimum-Complexity B-Spline Fitting

> The current default training objective is `v15`: it keeps the v14_joint
> one-shot architecture, aligns cardinality supervision with the actual
> `mass_topk` rounding rule, and differentiates through the final standard
> B-spline refit during calibration. See
> [v15 deployment-aligned optimization](docs/v15_deployment_aligned_optimization.md).

> The earlier one-way v14 `ParameterFeedbackHead` remains checkpoint-compatible;
> it is documented separately in [v14 parameter feedback](docs/v14_parameter_feedback.md).

当前目标为部署损失对齐的参数—节点双向联合 v15；v12、v13 和 v14 checkpoint 仍可按各自保存的
配置严格加载，不会被静默解释成新架构：

```text
objective_version = candidate_pruning_deployment_aligned_feedback_v15
structure_mode    = candidate_pruning_one_shot
```

目标是在归一化 RMS 阈值 `epsilon` 下，用尽量少的内部节点拟合有序点云：

\[
\min |U|\quad\text{s.t.}\quad \operatorname{RMS}(C_U,Q)\le\varepsilon .
\]

网络不使用 `CountHead`。它先生成固定长度的高召回冗余候选，再一次性预测 `KeepMask`；最终保留集合确定后，进一步联动调整存活节点位置。

## 为什么从 v11 升级到 v12

v11 虽然有位置更新头，但离线教师的 `teacher_internal_knots` 只是把被保留的 proposal 原位置复制出来。于是位置损失最容易学成“删除后不移动”，存活节点聚集时也缺少明确的重排监督。

v12 同时修正训练标签和网络结构：

- 离线教师改为 `delete -> relax -> retry delete`，缓存优化后的存活节点位置，而不只缓存删除掩码；
- 最终 `KeepMask` 产生后，用只读取存活节点 Key/Value 的 attention 做一次 survivor relocation；
- 位移受相邻存活节点、端点、最小间距和总位移上限约束，保证节点有序；
- 设计上引入 straight-through gate 连接保留决策和位置更新；部署仍是一条固定深度计算图。v13 进一步修正了该梯度链路的实际训练语义。

## v13 修正了什么

v12 的部署结构有效，但旧训练流程存在三个会直接降低泛化通过率的问题：位置反馈分支被零初始化后没有在校准阶段完整解冻；位置损失按教师槽位监督，而实际部署移动的是预测 `KeepMask`；截断幂 surrogate 又压过了真实标准 B 样条可行性目标。v13 因此：

- 解冻“初步位置 -> Keep 反馈 -> 最终位置 -> survivor relocation”的完整链路；
- 将实际预测 survivors 与 relocation teacher 节点做有序集合匹配，不再假设预测 mask 等于教师 mask；
- 增加与槽位编号无关的 soft teacher-set coverage，使漏掉教师节点时 Keep logits 能收到梯度；
- 默认关闭 calibration 中的 surrogate fit/threshold 项；
- 始终用真实标准 B 样条验证指标比较 distill 与 calibrated checkpoint，校准变差时自动回退。

这些修改需要重新训练才能进入权重；它们不会事后改变现有 v12 checkpoint 的预测。

## 当前工作流

```text
有序点云 Q [B,M,D]
  -> GeometryEncoder：坐标、弦长、一阶/二阶差分特征
  -> ParameterHead：严格递增参数 t in [0,1]
  -> CandidateKnotHead：局部 Gaussian cross-attention
       输出 Kc 个有序冗余候选 U_prop 及候选 token
  -> 一次性 selector：位置反馈后的 keep probability
  -> mass-TopK / threshold：最终离散 KeepMask
  -> v12 survivor relocation
       Key/Value 只来自最终存活节点
       读取存活邻居距离、相对 rank、存活数量等特征
       输出有界、有序的 U_deploy
  -> 取 U_deploy[KeepMask]
  -> 标准三次 B 样条 refit 一次
  -> 曲线、控制顶点、节点向量和误差
```

`proposal_internal_knots` 是固定教师槽位；`deployment_internal_knots` 才是最终可移动的部署位置。两者不能混为一谈。

## 数据集

`SyntheticCubicBSplineDataset` 随机生成开放三次 B 样条、严格有序采样参数和有序点云，并加入坐标噪声、非均匀节点与非均匀采样。默认配置为：

| 项目 | 默认值 |
|---|---:|
| 训练 / 验证 / 独立测试 seed | 42 / 10000 / 20000 |
| 训练 / 验证样本 | 10000 / 2000 |
| 每条曲线采样点 | 192 |
| 源控制顶点 | 8–24 |
| 源内部节点 | 4–20 |
| 冗余候选 `Kc` | 28 |
| 坐标噪声标准差 | 0.001 |
| 归一化 RMS 目标 | 0.005 |

数据同时保留源样条、canonical 阈值简化标签、真实参数和点云。canonical 标签用于几何覆盖监督；v13 继续使用 delete-then-relax Hard-RMS 教师，在网络固定 proposal 上构造可部署组合与优化后位置标签。

## 从 v12 proposal 训练 v13

下面的 PowerShell 命令复用已经训练好的 v12 权重作为 proposal 初始化；v13 会启用稳定的 float64 pilot 描述符，因此首次运行必须重建 v13 delete-then-relax teacher cache。selector 先蒸馏，再用 20 个 epoch 联合校准 KeepMask 与 survivor relocation：

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

注意：

- `--candidate-pretrain-epochs 0` 必须提供 proposal checkpoint；脚本会保留并校验该 checkpoint 的 proposal 语义。
- v13 用 float64 求解病态的 pilot 正规方程，再把描述符转回网络 dtype；这消除了 batch 大小/同伴样本对 KeepMask 的异常影响。该开关属于 proposal 指纹的一部分，所以不能复用旧 v12 cache。
- 只有同一个 v13 proposal 指纹、固定数据样本、张量形状和全部 teacher 配置都一致时，后续运行才能添加 `--reuse-teacher-cache`；脚本会严格校验，不一致时应换新目录重建。
- 当前缓存还要求同目录存在 `train.pt.start_domains.json` 与 `val.pt.start_domains.json`，用于证明教师直接来自网络校准参数域。旧缓存没有这两个认证文件，**不能**添加 `--reuse-teacher-cache`；必须先用 `--no-reuse-teacher-cache` 完整重建。正式训练默认不允许回退到真值/弦长参数（`--max-teacher-fallback-fraction 0`），避免在非部署参数域生成看似可行、实际不可部署的监督；非零值只建议用于诊断。
- `--teacher-relaxation-min-gap` 默认等于 `--min-knot-gap`，不是 `--candidate-match-tolerance`；CLI 不允许它大于 `--min-knot-gap`，因为固定 proposal 必须先满足该间距。若要抑制聚集，直接调大 `--min-knot-gap`，让 proposal、student 和 teacher 保持同一约束；这属于“节点不应过近”的建模先验，可能删除更多节点，也可能伤害确实需要近邻节点的曲线。
- 若旧 proposal 的 `min_knot_gap=0.001`，反聚集实验改为 `0.005` 时不能继续使用 `--candidate-pretrain-epochs 0`；应给出正数 proposal 适配 epoch（例如 `10`）并重建 teacher cache。
- `--keep-position-calibration-epochs` 应大于零，否则新 relocation head 保持兼容性的零初始化，基本不会移动节点。

若还没有可复用 proposal，将 `--candidate-pretrain-epochs` 设为正数并去掉 `--proposal-checkpoint`；脚本会先训练 proposal，再生成与之严格绑定的教师缓存。

## 评估

独立测试使用与训练集、验证集不同的 seed：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --num-samples 512 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --knot-tolerance 0.01 `
  --run-hard-diagnostic `
  --json-output outputs/logs/v12/candidate_pruning_one_shot_v12_evaluation.json
```

`learned` 模式中的 `--fit-tolerance` 只用于报告通过率，不会按样本搜索或改写网络 mask。`--run-hard-diagnostic` 是离线传统 greedy Hard-RMS 对照，计算更慢，也不会替换 v12 默认部署。

## 可视化

单样本 LearnedKeep 与 relocation：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --seed 20000 `
  --sample-index 0 `
  --pruning-view learned `
  --fit-tolerance 0.005 `
  --dpi 600 `
  --output outputs/figures/current/v12_learned_000.png
```

按源内部节点数从 `K=4` 到数据集上限逐层随机抽样，并生成四联批量图：

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/current/v12_verified_four_panel_K4_20 `
  --samples-per-knot-count 1 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --scan-size 512 `
  --seed 20000 `
  --selection-seed 12345 `
  --stratify-by source `
  --mse-tolerance 2.5e-5 `
  --ours-deployment verified `
  --verified-parameterization chord `
  --verified-compact `
  --timing-repeats 3 `
  --dpi 300
```

省略 `--max-knot-count` 时，脚本自动使用当前数据集的最大 source K；默认
`8–24` 个控制顶点对应 `K=4–20`。若每个 K 需要两张图，将
`--samples-per-knot-count` 改为 `2`。每张图固定包含：

1. 原始生成 B 样条、原始观测点、源控制多边形与源节点；
2. 网络生成的全部冗余 proposal 及其标准 B 样条 refit；
3. 当前 Ours；`learned` 为一次性 LearnedKeep，`verified` 为精确校验、条件修复和最终 refit；
4. 从同一份已物化 proposal 出发的传统 greedy hard pruning。

当使用 `--ours-deployment verified --verified-parameterization chord` 时，第 2～4 栏共用
弦长参数域和映射后的同一 proposal。每个 source K 输出一张 PNG，同时保存包含全部节点向量、
MSE、时间范围和 verified 修复来源的 JSON/CSV manifest。旧的纯 one-shot 四栏图仍可通过
`--ours-deployment learned` 生成。

第 2 栏是网络预测的高召回冗余候选，不是 Boehm 精确插入。图中误差统一为归一化
坐标上的平均平方欧氏误差
`MSE = mean_i ||C(t_i)-Q_i||_2^2`，不取平方根。

图中 `ours` 时间现在只统计驻留设备上的一次 `forward_deployment()`：包含参数预测、
proposal、KeepMask 和 survivor relocation，排除数据搬运、标准 refit、verified repair、
绘图与文件 I/O；报告同时保存 p50/p95，并在 CUDA 上于每次测量前后同步。`hard` 的旧
shared-proposal 数值仅保留为历史消融诊断，不能作为独立传统基线，也不能与纯网络时间
解释为严格端到端加速比。正式三方法比较请使用下文的独立数值基线脚本。

## K=4–20 三指标对比图

`evaluate_knot_count_strata.py` 会在原来的 2×2 汇总图之外，同时生成
`stratified_three_metrics.png`，依次比较标准 B 样条 MSE、最终保留节点数和时间。
两种方法使用相同测试曲线、相同 proposal；时间图仍明确保留原实验的非对称边界。

这里的 `Greedy hard` 依赖网络 proposal，只用于分析同一候选集上的选择差异。正式基线
从均匀的最大内部节点数开始，完全不读取网络 proposal，并在每次删除后用梯度更新剩余
节点位置；Kang 稀疏法和该数值基线的统一入口见
`scripts/compare_knot_methods.py` 与 `docs/kang_sparse_reproduction.md`。

若已经有 `stratified_comparison.json`，无需重跑模型：

```powershell
python scripts/plot_stratified_three_metrics.py `
  --report outputs/comparisons/current/v12_verified_chord_compact_K4_20_n20/stratified_comparison.json `
  --output outputs/comparisons/current/v12_verified_chord_compact_K4_20_n20/stratified_three_metrics.png `
  --dpi 600 `
  --overwrite
```

## 独立三方法对比

正式传统基线不再读取网络 proposal，而是从 20 个均匀内部节点开始，逐轮删除并用梯度
更新剩余节点位置。下面的脚本同时比较 Ours、Kang 2015 稀疏优化适配与该纯数值基线：

```powershell
python scripts/compare_knot_methods.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/v12_three_methods_K4_20 `
  --ours-mode verified `
  --sample-figures `
  --device auto `
  --network-warmups 10 `
  --network-repeats 50 `
  --overwrite
```

主图依次给出 MSE、最终内部节点数和时间；CSV/JSON 还保存相对 canonical/source 的
节点数 bias/MAE。Ours 只报告纯网络前向 p50/p95，Kang 与 uniform-gradient 报告完整
数值搜索时间，因此时间范围有意不同。完整复现说明见
`docs/kang_sparse_reproduction.md`。

## 质量守卫 verified 模式

`learned` 保持论文中的纯一次性网络路径，但它只给统计意义上的误差，不提供逐曲线阈值保证。实际部署优先使用 `verified`：先精确检查一次性结果；失败时按 Keep 置信度补回固定 proposal，每个候选结果都执行真实标准 B 样条 refit。完整 proposal 仍失败的极少数曲线，可选地在当前最大几何残差位置插入动态节点。该路径是自适应质量守卫，不应表述成纯 one-shot 或全局最少节点证明。

verified 还提供显式参数域策略：`network` 始终使用网络参数；默认 `chord-fallback` 只在网络域完整 proposal 失败后，将 proposal 通过样本对应关系映射到弦长参数域再精确验证；`chord` 则在整个 verified 流程中使用弦长参数域。返回结果会记录 `final_parameters` 与 `final_parameterization`，节点真值指标也先映射回真实参数域再匹配，不能跨参数域直接比较。

最快质量守卫（保留稍多节点）：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --deployment-mode verified `
  --no-verified-compact `
  --verified-parameterization chord-fallback `
  --verified-refit-device auto `
  --num-samples 512 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/logs/v12/v12_verified_fast.json
```

节点更少的质量守卫把 `--no-verified-compact` 改为 `--verified-compact`。默认还会启用最多 8 次残差插点兜底；可用 `--no-verified-residual-fallback` 做消融。`auto` 会让 GPU 负责网络 forward，而把 10～30 列的小规模精确最小二乘放在 CPU；也可用 `--verified-refit-device model` 强制跟随网络设备。

## 质量优先 hybrid 模式

`hybrid` 不是网络的第二个在线头，而是一次网络 forward 后运行的离线/慢速质量搜索：它用 LearnedKeep、proposal 和传统 greedy 作为起点，做 beam 删除、存活节点 coordinate refinement，并对候选组合反复执行标准 B 样条 refit。

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --deployment-mode hybrid `
  --num-samples 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --hybrid-beam-width 4 `
  --hybrid-branch-factor 4 `
  --hybrid-position-sweeps 2 `
  --hybrid-position-grid-size 7 `
  --hybrid-position-restarts 2 `
  --hybrid-position-refine-count-margin 1 `
  --hybrid-position-refine-candidate-multiplier 4 `
  --json-output outputs/logs/v12/v12_hybrid_evaluation.json

python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --seed 20000 `
  --sample-index 0 `
  --pruning-view hybrid `
  --fit-tolerance 0.005 `
  --output outputs/figures/current/v12_hybrid_000.png
```

hybrid 内部使用 MSE 阈值，因此命令行的 RMS `0.005` 会转换为 `MSE <= 2.5e-5`。它可用更多时间探索更少节点与更好位置，但仍是有限 beam 与局部连续搜索，不承诺全局最优。

## 用户有序点云

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --point-cloud data/my_curve.csv `
  --deployment-mode verified `
  --fit-tolerance 0.005 `
  --json-output outputs/predictions/my_curve_v12.json `
  --figure-output outputs/predictions/my_curve_v12.png
```

输入必须按曲线方向排序；CSV/TXT 每行是二维或三维坐标。纯一次性速度实验使用 `learned`；需要更充分的慢速组合/位置搜索时使用 `hybrid`。

## 真实数据集

UJI Pen v2、Natural Earth 和 USGS 等高线已经接入统一 manifest。批量测试示例：

```powershell
python scripts/evaluate_real_world.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --manifest data/splits/uji_pen_v2.jsonl `
  --split test `
  --max-samples 1000 `
  --json-output outputs/real_world/uji_v14_learned_1000.json `
  --overwrite
```

这些数据没有最简 B 样条节点真值，只能报告高密参考线上的几何误差、阈值通过率、节点数和时间；不能报告 knot Precision/Recall/F1。下载、分组划分、无标签反馈头适配和正式测试命令见[真实数据训练适配与部署测试](docs/real_world_evaluation.md)。

## 进一步文档

- [模型与数据流](docs/architecture.md)
- [数学定义](docs/math_formulation.md)
- [数据与训练](docs/training_pipeline.md)
- [合成数据最简性审计与修正](docs/synthetic_data_minimality_report.md)
- [真实曲线数据集、许可与接入协议](docs/real_world_datasets.md)
- [UJI Pen Characters v2 下载、writer-disjoint 划分与读取](docs/uji_pen_integration.md)
- [Natural Earth 与 USGS 等高线下载、投影和分组划分](docs/geospatial_real_world_data.md)
- [真实数据训练适配、批量部署评估与中断恢复](docs/real_world_evaluation.md)
- [10 ms 延迟口径、实测与部署建议](docs/latency_benchmark.md)
- [部署、评估与可视化](docs/deployment_pipeline.md)
- [K=4–20 一次性部署与贪心剪枝对比实验](docs/k4_20_comparison_experiment.md)
- [Kang 稀疏节点复现与独立数值基线](docs/kang_sparse_reproduction.md)
- [两类论文/PPT 对比图运行速查](docs/comparison_plots_quickstart.md)
- [PPT 展示速查](docs/presentation_demo.md)
- [结构方案演化](docs/pruning_redesign.md)
- [代码文件索引](docs/file_guide.md)
