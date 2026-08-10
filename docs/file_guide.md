# 文件索引

## 数据

| 文件 | 作用 |
|---|---|
| `data/synthetic.py` | 生成开放三次 B 样条、非均匀参数、采样点及完整真值 |
| `data/dataset.py` | 通用曲线数据、归一化和弦长参数工具 |

## 模型

| 文件 | 作用 |
|---|---|
| `models/geometry_encoder.py` | 编码点、一阶差分和二阶差分 |
| `models/parameter_head.py` | 预测严格递增参数 \(t\) |
| `models/knot_head.py` | 独立节点 query、位置编码 cross-attention 和节点位置预测 |
| `models/activity_head.py` | 结合 query 与节点局部特征预测 existence logits |
| `models/hard_concrete.py` | 计算非零概率、训练门和部署二值门 |
| `models/spline_network.py` | 组装模型、停梯度 gate、设计矩阵和可微线性求解 |

## 损失与训练

| 文件 | 作用 |
|---|---|
| `losses/total_loss.py` | 有序节点匹配及 fit、参数、existence、位置和节点数损失 |
| `training/trainer.py` | 训练/验证、existence 指标和 checkpoint 选择 |
| `checkpointing.py` | 当前模型构建及旧 checkpoint 显式迁移 |

## 样条与部署

| 文件 | 作用 |
|---|---|
| `spline/truncated_power_basis.py` | 构造训练用截断幂设计矩阵 |
| `spline/differentiable_solver.py` | 可微求解训练代理的线性系数 |
| `spline/curve_evaluation.py` | 重建和密集采样训练代理曲线 |
| `evaluation/bspline_inference.py` | 删除节点并重拟合标准开放 B 样条控制点 |
| `evaluation/knot_diagnostics.py` | 节点计数、匹配和拟合诊断 |

## 脚本

| 文件 | 作用 |
|---|---|
| `scripts/train.py` | 生成训练/验证数据并训练当前 v2 objective |
| `scripts/evaluate_checkpoint.py` | 输出网络、existence、节点和部署 B 样条指标，可保存 JSON |
| `scripts/visualize_result.py` | 可视化真值、训练代理、部署曲线和节点概率 |

## 测试

| 文件 | 主要覆盖内容 |
|---|---|
| `tests/test_hard_concrete.py` | Hard-Concrete、二值门和拟合路径 stop-gradient |
| `tests/test_a_scheme_loss_and_gaps.py` | 参数/节点监督、有序匹配和 existence 指标 |
| `tests/test_bspline_inference.py` | 节点删除与标准 B 样条重拟合 |
| `tests/test_knot_diagnostics.py` | 拟合和节点匹配统计 |
| `tests/test_checkpointing.py` | v2 与历史 checkpoint 迁移 |
| `tests/test_trainer_selection.py` | existence F1 优先的 checkpoint 规则 |

## 输出文件

训练 checkpoint 保存以下配置，以保证评估可复现：

- `model_config`；
- `dataset_config`；
- `loss_config`；
- `training_config`；
- `objective_version`。

`outputs/` 中的 `.pt`、`.json` 和图片为实验产物，不属于源代码。
