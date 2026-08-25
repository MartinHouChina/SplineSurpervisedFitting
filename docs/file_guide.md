# 文件索引

## v10 模型主路径

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/models/geometry_encoder.py` | 编码坐标、一阶和二阶几何差分 |
| `src/spline_fitting/models/parameter_head.py` | 预测严格递增点参数 |
| `src/spline_fitting/models/candidate_knot_head.py` | 用带位置编码的 cross-attention 生成固定预算、严格有序的高召回候选 |
| `src/spline_fitting/models/interactive_pruning_head.py` | 固定 proposal 位置精修；两层位置感知 selector、概率质量 Top-K 和参数域覆盖锚点 |
| `src/spline_fitting/models/spline_network.py` | 组装参数、候选、两次截断幂代理 solve 和 final KeepMask |
| `src/spline_fitting/spline/truncated_power_basis.py` | 构造可微截断幂设计矩阵 |
| `src/spline_fitting/spline/differentiable_solver.py` | 求解代理系数并计算解析列删除增量 |
| `src/spline_fitting/losses/candidate_pruning_loss.py` | 教师 BCE+Dice、final risk、CDF 集合分布、关键节点召回、Hard-ST count 和复杂度损失 |

`InteractivePruningHead` 的关键输出：

```text
final_keep_probability
final_hard_keep_mask
final_hard_st_keep_context
refined_candidate_knots
```

旧键 `keep_probability` 和 `refined_candidate_knots` 指向 final 状态。

## 离线 Hard-RMS 教师

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/training/one_shot_teacher.py` | 生成、校验、保存和加载数据集/proposal 指纹绑定的教师缓存；soft risk 来自最终停止状态 |
| `src/spline_fitting/spline/bspline_deletion_teacher.py` | 批量计算单节点删除后的标准 B 样条 RMS |
| `src/spline_fitting/evaluation/minimal_knot_pruning.py` | 完整重拟合的 Hard-RMS 贪心删除与轨迹记录 |
| `src/spline_fitting/evaluation/bspline_inference.py` | 端点约束的标准开放 B 样条控制点 refit |

缓存绑定：

```text
teacher config fingerprint
+ dataset fingerprint
+ proposal weight fingerprint
+ sample IDs/order
+ input shapes
```

因此缓存禁止与 `resample_each_epoch=True` 同时使用。

## 数据、训练与 checkpoint

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/data/synthetic.py` | 合成开放三次 B 样条、归一化和 canonical 诊断标签 |
| `src/spline_fitting/data/point_cloud_io.py` | 用户点云读取、检查、归一化和有序重采样 |
| `src/spline_fitting/training/trainer.py` | 分阶段训练；验证时运行真实一次标准 B 样条 deployment pass；按满足率/RMS 等选择 checkpoint |
| `src/spline_fitting/checkpointing.py` | v10 配置恢复和 v7/v8/v9 checkpoint 迁移 |

当前版本：

```text
objective_version = candidate_pruning_structured_feasible_teacher_v10
structure_mode    = candidate_pruning_one_shot
one_shot_fixed_proposal_geometry = True
one_shot_selection_policy = mass_topk
```

## 训练脚本的四个阶段

| 阶段 | 脚本行为 |
|---|---|
| 最佳 proposal | 候选预训练后恢复验证排序最好的 `*_proposal.pt`；也可用 `--proposal-checkpoint` 跳过 |
| 离线教师 | 对固定数据和固定 proposal 生成 final-state risk、hard mask、count 与 RMS 缓存 |
| 选择蒸馏 | 只训练独立 selector adapter、`keep_head` 和 `adaptive_threshold_head` |
| 可选校准 | 仍只训练 selector；proposal 位置始终冻结 |

校准参数名为：

```text
--selector-calibration-epochs
```

历史别名 `--joint-finetune-epochs` 仍可解析，但当前行为不是端到端 joint fine-tuning，文档和新命令统一使用新名称。

## 命令入口

| 文件 | 作用 |
|---|---|
| `scripts/train_candidate_pruning.py` | 最佳 proposal → 离线教师 → 选择蒸馏 → 可选 selector 校准 |
| `scripts/evaluate_checkpoint.py` | 独立测试、一次标准 refit 指标和可选 Hard-RMS 诊断 |
| `scripts/visualize_result.py` | 绘制 all、learned、hard；comparison 输出 source/冗余/Learned/Hard 四曲线及节点向量、控制顶点和时间 |
| `scripts/fit_point_cloud.py` | 用户有序点云的一次 forward 和一次全分辨率标准 B 样条 refit |
| `scripts/train.py` | 历史 v6 实验入口，不是 v10 主训练脚本 |

训练：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --selector-calibration-epochs 0 `
  --train-size 10000 `
  --val-size 2000 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --one-shot-selection-policy mass_topk `
  --one-shot-safety-sigma 0.5 `
  --one-shot-selector-layers 2 `
  --one-shot-coverage-bins 4 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v10_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v10.pt
```

复用最佳 proposal 和匹配缓存：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 130 `
  --candidate-pretrain-epochs 0 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v10_proposal.pt `
  --selector-calibration-epochs 0 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v10_teacher `
  --reuse-teacher-cache `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v10.pt
```

评估：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --num-samples 2000 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/candidate_pruning_one_shot_v10_evaluation.json
```

可视化：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --sample-index 0 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --pruning-view learned `
  --dpi 600 `
  --output outputs/candidate_pruning_one_shot_v10_sample_000.png
```

## 兼容模块

| 文件 / 模式 | 版本 |
|---|---|
| `InteractivePruningHead(one_shot_adaptive=False)` | v7 原参数布局和 forward |
| `evaluation/minimal_knot_pruning.py` | v7 默认部署；v8/v9/v10 离线教师或显式 diagnostic |
| `interactive_structure_head.py`、`dynamic_knot_decoder.py` | v6 |
| `count_head.py`、`count_conditioned_knot_head.py` | v5/v4 |
| `activity_head.py`、`hard_concrete.py` | v3 及更早 |

## 主要测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_candidate_pruning_heads.py` | v10 mass-Top-K、覆盖锚点、固定几何、hard-ST 和历史兼容 |
| `tests/test_candidate_pruning_network.py` | candidate-pruning forward/backward 与 checkpoint round-trip |
| `tests/test_one_shot_teacher.py` | final-state risk、缓存完整性、fingerprint、sample ID 与禁止 resample |
| `tests/test_candidate_pruning_deployment_scripts.py` | v8/v9/v10 一次 refit、v7 hard 路径和脚本模式识别 |
| `tests/test_trainer_selection.py` | 真实 deployment pass/mean/P95 指标与 v10 验证排序 |
| `tests/test_bspline_inference.py` | 标准 B 样条 refit 和端点覆盖 |

`outputs/` 中的 checkpoint、教师缓存、JSON 和图片均为实验产物，不属于源码。
