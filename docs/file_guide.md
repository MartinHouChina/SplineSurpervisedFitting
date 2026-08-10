# 文件索引

## 数据与模型

| 文件 | 作用 |
|---|---|
| `data/synthetic.py` | 生成开放三次 B 样条，并通过容差约束节点删除构造 canonical 标签 |
| `models/geometry_encoder.py` | 编码坐标和弦长归一化一阶/二阶导数 |
| `models/parameter_head.py` | 预测严格递增点参数 |
| `models/count_head.py` | 局部 cross-attention 序数节点计数；兼容 v4 categorical 模式 |
| `models/count_conditioned_knot_head.py` | 共享 query + count embedding 的有序节点解码；兼容 v4 独立分支 |
| `models/spline_network.py` | 组装主网络、可微线性求解和历史结构路径 |
| `models/activity_head.py`、`hard_concrete.py` | v3 及更早 checkpoint 兼容模块 |

## 损失、部署与训练

| 文件 | 作用 |
|---|---|
| `losses/total_loss.py` | 序数数量、过预测、节点位置、参数和拟合损失 |
| `evaluation/bspline_inference.py` | 标准 B 样条重拟合和 BIC 完整分支选择 |
| `evaluation/knot_diagnostics.py` | 节点匹配及拟合统计 |
| `training/trainer.py` | teacher-conditioned 训练、真实推理验证和 checkpoint 选择 |
| `checkpointing.py` | v5、v4、v3 及更早结构的显式迁移 |

## 脚本

| 文件 | 作用 |
|---|---|
| `scripts/train.py` | 训练 v5，并保存模型、数据、损失和训练配置 |
| `scripts/evaluate_checkpoint.py` | 报告网络数量、部署数量、节点匹配和标准 B 样条指标 |
| `scripts/visualize_result.py` | 对比采样点、拟合曲线、控制多边形和数量分布 |

## 测试

| 文件 | 主要覆盖内容 |
|---|---|
| `tests/test_canonical_labels.py` | canonical 删除标签的一致性、容差和确定性 |
| `tests/test_count_conditioned.py` | 序数 CountHead、共享解码、梯度、BIC 分支选择 |
| `tests/test_checkpointing.py` | v5/v4/v3/v2/历史 checkpoint 严格迁移 |
| `tests/test_bspline_inference.py` | 标准 B 样条重拟合 |
| `tests/test_trainer_selection.py` | 几何匹配优先的 checkpoint 规则 |

## 文档与论文

| 文件 | 作用 |
|---|---|
| `README.md` | 当前工作流、数据集和命令入口 |
| `docs/architecture.md` | v5 模型结构 |
| `docs/training_pipeline.md` | 数据、训练与评估协议 |
| `docs/deployment_pipeline.md` | network/BIC 数量选择、标准 B 样条重拟合和指标解释 |
| `docs/math_formulation.md` | 核心数学定义 |
| `docs/pruning_redesign.md` | v3→v4→v5 的设计演化 |
| `paper/` | Computer-Aided Design LaTeX 论文工程 |

`outputs/` 中的 checkpoint、JSON 和图片属于实验产物，不属于源代码。
