# 文件索引

## 数据与模型

| 文件 | 作用 |
|---|---|
| `data/synthetic.py` | 生成开放三次 B 样条，并通过容差约束节点删除构造 canonical 标签 |
| `data/point_cloud_io.py` | 用户有序点云加载、归一化、重采样和预测参数弦长插值 |
| `models/geometry_encoder.py` | 编码坐标和弦长归一化一阶/二阶导数 |
| `models/parameter_head.py` | 预测严格递增点参数 |
| `models/interactive_structure_head.py` | v6 结构 query cross/self-attention、直接数量分类、合法范围和后验中位数决策 |
| `models/dynamic_knot_decoder.py` | v6 仅计算所选 K+1 个 interval query 的动态节点解码 |
| `models/count_head.py` | v5/v4 数量头兼容模块 |
| `models/count_conditioned_knot_head.py` | v5/v4 全数量分支兼容模块 |
| `models/spline_network.py` | 组装主网络、可微线性求解和历史结构路径 |
| `models/activity_head.py`、`hard_concrete.py` | v3 及更早 checkpoint 兼容模块 |

## 损失、部署与训练

| 文件 | 作用 |
|---|---|
| `losses/total_loss.py` | v6 categorical 数量、节点位置、参数和拟合损失；保留历史目标兼容路径 |
| `evaluation/bspline_inference.py` | 严格插值首尾点的标准 B 样条重拟合；保留 v5 BIC 历史兼容函数 |
| `evaluation/knot_diagnostics.py` | 节点匹配及拟合统计 |
| `training/trainer.py` | scheduled teacher forcing、真实推理验证和以数量 MAE 优先的 checkpoint 选择 |
| `checkpointing.py` | v6、v5、v4、v3 及更早结构的显式迁移 |

## 脚本

| 文件 | 作用 |
|---|---|
| `scripts/train.py` | 训练 v6 交互结构头和动态节点解码器 |
| `scripts/evaluate_checkpoint.py` | 报告网络数量、部署数量、节点匹配和标准 B 样条指标 |
| `scripts/visualize_result.py` | 对比采样点、拟合曲线、控制多边形和数量分布 |
| `scripts/fit_point_cloud.py` | 对用户点云推理，并在全部原始点上执行最终重拟合和稀疏输入诊断 |

## 测试

| 文件 | 主要覆盖内容 |
|---|---|
| `tests/test_canonical_labels.py` | canonical 删除标签的一致性、容差和确定性 |
| `tests/test_interactive_dynamic.py` | v6 数量分布、动态 query 数、梯度和单分支部署 |
| `tests/test_count_conditioned.py` | v5 条件数量结构兼容测试 |
| `tests/test_checkpointing.py` | v6/v5/v4/v3/v2/历史 checkpoint 严格迁移 |
| `tests/test_bspline_inference.py` | 标准 B 样条重拟合 |
| `tests/test_trainer_selection.py` | 历史几何选优与当前数量 MAE 优先的 checkpoint 规则 |
| `tests/test_point_cloud_io.py` | 用户点云加载、维度检查、归一化和参数插值 |

## 文档与论文

| 文件 | 作用 |
|---|---|
| `README.md` | 当前工作流、数据集和命令入口 |
| `docs/proposal_pruning_framework.md` | 下一阶段候选生成、Boehm 消冗、三阶段训练与点云单输入部署设计；尚未实现 |
| `docs/architecture.md` | v6 模型投喂顺序和计算复杂度 |
| `docs/training_pipeline.md` | 数据、训练与评估协议 |
| `docs/deployment_pipeline.md` | v6 单次数量决策、动态节点解码和标准 B 样条重拟合 |
| `docs/math_formulation.md` | 核心数学定义 |
| `docs/pruning_redesign.md` | v3→v4→v5→v6 及下一阶段规划的设计演化 |
| `paper/` | Computer-Aided Design LaTeX 论文工程 |

`outputs/` 中的 checkpoint、JSON 和图片属于实验产物，不属于源代码。
