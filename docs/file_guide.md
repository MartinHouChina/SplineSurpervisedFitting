# 文件索引

## 当前版本

```text
objective_version = candidate_pruning_deployment_aligned_feedback_v15
structure_mode    = candidate_pruning_one_shot
```

v13 与 v12 共用部署结构；新版本修正实际 survivor 集合监督、soft set coverage、完整交互解冻和 calibration checkpoint 回退。

## 模型主路径

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/models/geometry_encoder.py` | 有序点、弦长和导数几何编码 |
| `src/spline_fitting/models/parameter_head.py` | 预测严格递增参数 `t` |
| `src/spline_fitting/models/candidate_knot_head.py` | 局部 Gaussian cross-attention 高召回候选 |
| `src/spline_fitting/models/interactive_pruning_head.py` | 一次性 selector、最终 KeepMask、selected-only survivor relocation |
| `src/spline_fitting/models/spline_network.py` | 串联参数、proposal、筛选、位置更新和代理拟合 |
| `src/spline_fitting/losses/deployment_bspline_loss.py` | v15 训练期可微标准 B 样条重拟合 MSE |
| `src/spline_fitting/losses/parameter_feedback_loss.py` | 部署 MSE、实际 mass_topk 计数与联合位置监督 |

v12 关键输出：

```text
proposal_internal_knots
preliminary_keep_probabilities
provisional_candidate_knots
final_keep_probabilities
final_hard_keep_mask
pre_relocation_candidate_knots
relocation_relative_features
relocation_attention_weights
relocation_position_residual
relocation_selected_mask
deployment_internal_knots
internal_knots
```

`internal_knots` 是最终 deployment 位置的兼容别名；固定教师槽位应读取 `proposal_internal_knots`。

## 数据与标签

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/data/synthetic.py` | 合成开放三次 B 样条、有序参数、点云和源表示 |
| `src/spline_fitting/data/dataset.py` | 数据集导出与组织 |
| `src/spline_fitting/evaluation/minimal_knot_pruning.py` | canonical / greedy 节点简化 |
| `src/spline_fitting/data/point_cloud_io.py` | 用户有序 CSV/TXT 点云读取、重采样和归一化 |
| `src/spline_fitting/data/real_world.py` | 真实曲线 JSONL manifest 校验、读取、重采样与原密度参考点访问 |
| `src/spline_fitting/data/uji_pen.py` | UJI v2 官方文本解析、毫米尺度统一与 writer-disjoint 划分 |

数据同时保存 canonical `true_internal_knots` 和 `source_*` 原始生成样条。两者用途不同：前者用于阈值简化后的几何监督，后者用于源曲线诊断与可视化。

## v13 离线教师与损失

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/training/one_shot_teacher.py` | delete-then-relax Hard-RMS 教师、cache v3、指纹与完整性检查 |
| `src/spline_fitting/losses/candidate_pruning_loss.py` | proposal、mask、risk、count、实际 survivor 有序集合位置和 soft teacher-set coverage |
| `src/spline_fitting/evaluation/bspline_inference.py` | 标准开放 B 样条控制顶点 refit |
| `src/spline_fitting/evaluation/verified_knot_repair.py` | exact-refit 检查、置信度补回、可选 compact 与残差插点质量守卫 |
| `src/spline_fitting/spline/bspline_deletion_teacher.py` | 批量单节点删除误差 |

v12 cache 新增或重点使用：

```text
teacher_fit_mse
teacher_greedy_count
teacher_greedy_fit_rms
teacher_relocation_mean_abs
teacher_relocation_max_abs
teacher_extra_deleted_after_relocation
```

位置监督的关键指标是 `teacher_survivor_spacing_loss`。`teacher_internal_knots` 存放 delete-then-relax 后的 packed survivor 位置，不再只是 retained proposal 的复制。

## 训练与 checkpoint

| 文件 | 作用 |
|---|---|
| `scripts/train_candidate_pruning.py` | v13 proposal、teacher、selector distillation、survivor relocation calibration 主入口 |
| `src/spline_fitting/training/trainer.py` | train/validation、真实部署指标和 checkpoint 排序 |
| `src/spline_fitting/checkpointing.py` | v3–v13 配置迁移、objective/version 与 loss schema 恢复 |
| `scripts/train.py` | 历史训练入口，不是 v13 主流程 |

从 v12 proposal 开始：

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

`--teacher-relaxation-min-gap` 未指定时等于 `--min-knot-gap`；不要把它误写成 candidate match tolerance。抑制聚集时优先调大 `--min-knot-gap`，使 proposal、student 与 teacher 约束一致。增大 min-gap 属于结构先验，需要做消融。只有 proposal、样本与 teacher 配置全部一致时才可复用 cache。

## 评估与部署

| 文件 | 作用 |
|---|---|
| `scripts/evaluate_checkpoint.py` | learned/verified/hard/hybrid 独立测试、节点匹配、拟合与复杂度指标 |
| `scripts/fit_point_cloud.py` | 用户有序点云 learned/verified/hard/hybrid 部署 |
| `scripts/visualize_result.py` | 单样本 all/learned/hard/hybrid/comparison PNG |
| `scripts/visualize_batch_comparison.py` | 按节点数分层抽样的源/all-proposal/learned/hard 四联图 |
| `scripts/evaluate_knot_count_strata.py` | K=4–20 分层定量对比、bootstrap CI、CSV/JSON/PNG 报告 |
| `scripts/compare_knot_methods.py` | Ours、Kang 稀疏适配、uniform-Kmax 删除+梯度重定位的独立三方法比较 |
| `scripts/reproduce_sparse_knot_paper.py` | Kang 论文风格标量算例的 JSON/PNG 数值复现 |
| `scripts/prepare_uji_pen.py` | 下载/预处理 UJI Pen Characters v2 并生成无节点标签 manifest |
| `scripts/prepare_natural_earth.py` | 下载/预处理版本固定的 Natural Earth LineString |
| `scripts/prepare_usgs_contours.py` | 下载/缓存/预处理指定区域的 USGS 等高线 |
| `scripts/evaluate_real_world.py` | UJI/Natural Earth/USGS 的纯网络结果，以及可选 reference-MSE certified repair；两类时间分开报告 |
| `src/spline_fitting/evaluation/certified_real_world.py` | reference refit、联合节点移动/增结、容差折线简化与完整折线最终兜底；只认证提供的离散点 |
| `src/spline_fitting/evaluation/hybrid_knot_search.py` | learned 热启动、greedy fallback、beam 删除和 survivor coordinate refinement |
| `src/spline_fitting/evaluation/sparse_knot_paper.py` | Kang (2015) 方程 (12) 的 group-L1 ADMM 二维适配与可行性诊断 |
| `src/spline_fitting/evaluation/gradient_knot_pruning.py` | 不读取网络的均匀最大节点、逐删和梯度位置更新基线 |
| `src/spline_fitting/evaluation/timing.py` | CPU/CUDA 同步的 network-only p50/p95 计时工具 |
| `src/spline_fitting/evaluation/knot_diagnostics.py` | 参数域映射与一维有序节点匹配 |

### LearnedKeep 评估

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --num-samples 512 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --knot-tolerance 0.01 `
  --run-hard-diagnostic `
  --json-output outputs/logs/v12/v12_evaluation.json
```

### Verified 质量守卫

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --deployment-mode verified `
  --no-verified-compact `
  --verified-refit-device auto `
  --num-samples 512 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/logs/v12/v12_verified_fast.json
```

使用 `--verified-compact` 可换取更少节点；残差插点兜底默认开启，只在完整 proposal 仍失败时触发。

### 单样本 LearnedKeep 图

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --seed 20000 `
  --sample-index 0 `
  --pruning-view learned `
  --fit-tolerance 0.005 `
  --output outputs/figures/current/v12_learned_000.png
```

### Hybrid 质量模式

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
```

hybrid 会联合删除和移动节点，但属于一次网络 forward 后的离线多 refit 搜索，不是 v12 快速一次性路径。

### 批量四联图

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/current/v12_sourceK_4_to_max `
  --samples-per-knot-count 1 `
  --min-knot-count 4 `
  --scan-size 512 `
  --seed 20000 `
  --selection-seed 12345 `
  --stratify-by source `
  --timing-repeats 5 `
  --dpi 600
```

该命令省略 `--max-knot-count`，因此自动扫描到数据集的 source K 上限；默认得到
`K=4–20` 每层一张。`--samples-per-knot-count 2` 可改为每层两张。四栏依次为源样条、
网络冗余 proposal、v12 一次性部署和传统 greedy hard pruning；每栏都显示数据点、
拟合/源曲线、控制多边形、内部节点及节点向量。proposal 是网络预测的冗余候选，不是
Boehm 精确节点插入。

误差字段 `source_mse / proposal_mse / learned_mse / hard_mse` 均采用归一化坐标上的
`mean_i ||C(t_i)-Q_i||_2^2`，不取平方根。时间字段的比较边界是：

- learned/ours：同步后的纯 `forward_deployment()` 中位数，包含参数、proposal、
  KeepMask 与 relocation，排除最终 refit 和 verified repair；
- hard：预测参数和 proposal 已物化后的 greedy pruning stage 中位数，包含其内部
  refit，排除网络时间。

JSON manifest 另存完整质量管线的 diagnostic 时间；CSV 便于整理 PPT/论文表格。上述
边界按实验需求故意不对称，不能把两者写成严格同边界的端到端加速比。这里的 hard 还
读取网络 proposal，只是历史消融；正式独立基线见 `gradient_knot_pruning.py`。

### K=4–20 定量对比

固定每个 source K 的样本数、输出 MSE、canonical 节点数误差、参数域对齐后的节点匹配
以及两条指定流程的计时：

```powershell
python scripts/evaluate_knot_count_strata.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/current/v12_stratified_K4_20_n20 `
  --samples-per-knot-count 20 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --scan-size 1024 `
  --seed 20000 `
  --selection-seed 12345 `
  --execution-seed 67890 `
  --timing-repeats 3 `
  --warmup-repeats 1 `
  --bootstrap-replicates 5000 `
  --bootstrap-seed 24680 `
  --torch-num-threads 4 `
  --dpi 600 `
  --overwrite
```

实验口径、实际数值和结论见 `docs/k4_20_comparison_experiment.md`。

### 用户点云

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --point-cloud data/my_curve.csv `
  --deployment-mode learned `
  --fit-tolerance 0.005 `
  --json-output outputs/predictions/my_curve_v12.json `
  --figure-output outputs/predictions/my_curve_v12.png
```

## 主要测试

| 测试 | 覆盖内容 |
|---|---|
| `tests/test_survivor_relocation.py` | selected-only attention、有界单调位移、ST 梯度、v11 中性载入 |
| `tests/test_v12_coupled_teacher.py` | delete-then-relax、最终 relaxation、cache 与位置/gap 监督 |
| `tests/test_one_shot_teacher.py` | Hard-RMS 教师风险、指纹、sample ID 和 cache 完整性 |
| `tests/test_checkpointing.py` | v3–v12 恢复与迁移语义 |
| `tests/test_candidate_pruning_network.py` | proposal/selector/部署节点网络输出 |
| `tests/test_hybrid_knot_search.py` | greedy fallback、beam、位置精修和阈值逻辑 |
| `tests/test_hybrid_checkpoint_evaluation.py` | hybrid 批量评估与 JSON |
| `tests/test_hybrid_point_cloud_deployment.py` | 用户点云 hybrid 部署 |
| `tests/test_batch_pruning_comparison.py` | 分层抽样、四联图、MSE 与节点语义 |
| `tests/test_trainer_selection.py` | 真实 deployment pass/RMS 与 checkpoint 排序 |

## 兼容说明

- v11 权重可严格载入 v12；新增 relocation 模块零初始化，初始为 identity。
- 从 v11 加载只完成初始化，不会自动产生移动效果；必须重新生成 v12 teacher 并执行 calibration。
- 历史 checkpoint 在评估脚本中保持原 objective 语义，不会被静默当成 v12。
