# 文件索引

## v11 版本标识

```text
objective_version = candidate_pruning_joint_refinement_teacher_v11
structure_mode = candidate_pruning_one_shot
one_shot_fixed_proposal_geometry = True
one_shot_joint_position_refinement = True
one_shot_selection_policy = mass_topk
```

## 模型主路径

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/models/geometry_encoder.py` | 编码坐标、一阶差分和二阶差分，输出局部/全局特征 |
| `src/spline_fitting/models/parameter_head.py` | 预测严格递增参数 `t` |
| `src/spline_fitting/models/candidate_knot_head.py` | interval query、参数位置编码、Gaussian 局部 cross-attention 和严格有序候选 |
| `src/spline_fitting/models/interactive_pruning_head.py` | proposal 贡献交互、两层 selector、`p0→u1→p1→mask→u*`、零节点稳定 mass-TopK 和合计位移约束 |
| `src/spline_fitting/models/spline_network.py` | 组装模型；分离 proposal/deployment 节点；执行两次截断幂代理 solve，并 detach fit gate 的选择概率路径 |
| `src/spline_fitting/spline/truncated_power_basis.py` | 构造可微截断幂设计矩阵 |
| `src/spline_fitting/spline/differentiable_solver.py` | 代理系数求解和解析贡献特征 |

关键输出：

```text
proposal_internal_knots
deployment_internal_knots
internal_knots                  # v11 中为 deployment 兼容别名
preliminary_keep_probability    # p0
provisional_candidate_knots     # u1
final_keep_probability          # p1
final_hard_keep_mask
one_shot_requested_count_score  # mass_topk 的连续计数分数
fit_activity_gate               # 代理设计矩阵门；v11 默认 detach 选择概率路径
final_position_residual
```

## 数据与标签

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/data/synthetic.py` | 生成开放三次 B 样条、有序点云、源表示和 canonical 标签 |
| `src/spline_fitting/data/point_cloud_io.py` | 读取、检查、重采样和归一化用户有序点云 |

默认 canonical 阈值与训练 `fit_tolerance` 一致。源节点只描述数据生成表示，不是最终节点数监督。

## 损失

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/losses/candidate_pruning_loss.py` | proposal 多尺度覆盖、teacher mask/risk/ranking/count、canonical 选择和联合 deployment 位置监督 |
| `src/spline_fitting/losses/total_loss.py` | 历史 hard-concrete/count 方案的通用损失 |

v11 新增或启用的主要 loss 字段：

```text
candidate_coverage_tolerances
teacher_false_positive
policy_count
canonical_selection
joint_deployment_position_loss
joint_position_supervision
```

其中 false-positive 只惩罚 teacher-negative 且 canonical-negative 的槽位；联合校准默认再使用 `lambda_joint_fit=0.05` 和 `lambda_joint_threshold_violation=0.5` 约束位置可行性。

## 离线教师与标准 refit

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/training/one_shot_teacher.py` | 生成、保存、加载和严格校验 Hard-RMS teacher cache |
| `src/spline_fitting/evaluation/minimal_knot_pruning.py` | 贪心逐节点标准 B 样条删除及完整轨迹 |
| `src/spline_fitting/spline/bspline_deletion_teacher.py` | 批量单节点删除 MSE，并保留兼容的 RMS 接口 |
| `src/spline_fitting/evaluation/bspline_inference.py` | 端点约束的标准开放 B 样条控制顶点 refit |

teacher cache 绑定：

```text
teacher config fingerprint
+ dataset content fingerprint
+ proposal-only weight/config fingerprint
+ sample ID/order
+ input shapes
```

proposal-only 指纹包含 Gaussian attention 带宽，但不包含独立 selector 和 deployment 位置头。

## 训练

| 文件 | 作用 |
|---|---|
| `scripts/train_candidate_pruning.py` | v11 主入口：proposal 预训练、教师生成、选择蒸馏、联合位置校准 |
| `src/spline_fitting/training/trainer.py` | batch 训练/验证、真实 one-shot 标准 B 样条验证和 checkpoint 排序 |
| `src/spline_fitting/checkpointing.py` | v11 默认配置和 v3–v10 checkpoint 恢复 |
| `scripts/train.py` | 历史 v6 入口，不是 v11 主训练脚本 |

阶段与文件：

| 阶段 | 默认输出 |
|---|---|
| proposal 预训练 | `*_proposal.pt` |
| one-shot 蒸馏 | `*_distill.pt` |
| 联合校准 | `*_calibrated.pt` |
| 全阶段最佳 | 用户指定的 `--output` |
| 最后一轮 | `*_last.pt` |
| 教师缓存 | `teacher_dir/train.pt`、`teacher_dir/val.pt` |

## 评估和部署脚本

| 文件 | 作用 |
|---|---|
| `scripts/evaluate_checkpoint.py` | 独立测试、proposal recall、三阶段节点匹配、一次 refit 和可选 Hard-RMS 对照 |
| `scripts/visualize_result.py` | 输出 all/learned/hard/comparison PNG，显示曲线、点、控制多边形、节点和时间 |
| `scripts/visualize_batch_comparison.py` | 按 source/canonical 节点数分层随机抽样，批量输出四联 MSE PNG、JSON manifest 和 CSV 表 |
| `scripts/fit_point_cloud.py` | 用户有序点云的一次前向和全分辨率一次 refit |
| `src/spline_fitting/evaluation/knot_diagnostics.py` | 一维有序节点匹配和基础诊断 |

评估 JSON v12 新增：

```text
knot_stage_metrics
  proposal_full
  selected_pre_update
  deployment_post_update
  position_recall_delta
  position_precision_delta
```

## 常用命令

### 完整训练

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --selector-calibration-epochs 10 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --candidate-local-attention-bandwidth 0.08 `
  --candidate-coverage-tolerances 0.005 0.01 0.02 `
  --one-shot-selection-policy mass_topk `
  --one-shot-safety-sigma 0.25 `
  --one-shot-selector-layers 2 `
  --one-shot-coverage-bins 0 `
  --one-shot-max-position-shift 0.05 `
  --lambda-one-shot-surrogate-fit 0 `
  --lambda-one-shot-surrogate-threshold 0 `
  --lambda-joint-fit 0.05 `
  --lambda-joint-threshold-violation 0.5 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v11_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v11.pt
```

### v10 proposal 初始化并适配 v11

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 138 `
  --candidate-pretrain-epochs 8 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v10_proposal.pt `
  --selector-calibration-epochs 10 `
  --candidate-local-attention-bandwidth 0.08 `
  --lambda-one-shot-surrogate-fit 0 `
  --lambda-one-shot-surrogate-threshold 0 `
  --lambda-joint-fit 0.05 `
  --lambda-joint-threshold-violation 0.5 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v11_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v11.pt
```

正数 `candidate-pretrain-epochs` 会先加载 v10 权重，再适配 v11 的 Gaussian 局部 proposal。设为 `0` 只复用固定 proposal，不会适配新增局部 attention；此时脚本保留 checkpoint 带宽，并校验已记录 `model_config` 中影响 proposal 的非权重语义。新生成的 `*_proposal.pt` 会保存 `model_config` 和阶段元数据。

### 评估

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --num-samples 2000 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/candidate_pruning_one_shot_v11_evaluation.json
```

### 四联图

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --seed 20000 `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view comparison `
  --timing-repeats 5 `
  --dpi 600 `
  --output outputs/v11_comparison_000.png
```

### 批量分层四联图

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --output-dir outputs/v11_batch_comparison `
  --num-figures 10 `
  --scan-size 512 `
  --seed 20000 `
  --stratify-by source `
  --knot-counts 4 8 12 16 20 `
  --mse-tolerance 2.5e-5 `
  --selection-seed 12345 `
  --dpi 600 `
  --timing-repeats 5
```

批量脚本使用 `MSE=mean_i ||C(t_i)-Q_i||_2^2`，不开平方。all 和 Hard 使用固定 proposal；Learned 使用 deployment 位置与正式 mask。输出目录包含 PNG、`comparison_manifest.json` 和 `comparison_manifest.csv`。

## v10 与缓存兼容

- v10 checkpoint 可按历史语义直接评估和部署。
- v10 权重可初始化 v11。
- 推荐用 5–10 轮 proposal 预训练适配 v11 局部 attention。
- `candidate-pretrain-epochs=0` 不执行该适配，并校验 checkpoint 的 proposal 非权重语义；缺少 `model_config` 时只能警告回退。
- 新 proposal checkpoint 保存 `model_config`，供固定复用和 cache 指纹校验。
- v11 第一次训练必须新建 teacher cache；不能复用 v10 cache。

## 主要测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_candidate_pruning_heads.py` | Gaussian attention、有序候选、`p0→u1→p1→mask→u*`、总位移预算、零节点 mass-TopK、mask 条件位置和历史布局 |
| `tests/test_candidate_pruning_network.py` | v11 输出、离散 fit gate、教师损失、v10 中性初始化和 checkpoint round-trip |
| `tests/test_one_shot_teacher.py` | final-state risk、缓存完整性、指纹和 sample ID |
| `tests/test_candidate_pruning_deployment_scripts.py` | 一次 refit、Hard-RMS 对照和脚本模式 |
| `tests/test_batch_pruning_comparison.py` | MSE 定义、分层随机抽样、proposal/deployment 节点语义、传统贪心删除和四联 PNG |
| `tests/test_trainer_selection.py` | 真实 deployment RMS/pass 指标和 checkpoint 排序 |
| `tests/test_bspline_inference.py` | 标准 B 样条 refit 与端点约束 |

`outputs/` 下的 checkpoint、teacher cache、JSON 和图片是实验产物，不属于源码。
