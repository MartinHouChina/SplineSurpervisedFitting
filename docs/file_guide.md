# 文件索引

## v7 主路径

| 文件 | 作用 |
|---|---|
| `src/spline_fitting/models/geometry_encoder.py` | 编码坐标和一、二阶几何差分 |
| `src/spline_fitting/models/parameter_head.py` | 预测严格递增的点参数 |
| `src/spline_fitting/models/candidate_knot_head.py` | 用带位置编码的 cross-attention 生成固定预算的严格有序候选 |
| `src/spline_fitting/models/interactive_pruning_head.py` | 融合解析贡献和候选 self-attention，输出 remove/STOP、keep 诊断和位置精修 |
| `src/spline_fitting/models/spline_network.py` | 组装候选、截断幂代理求解和交互消冗数据流 |
| `src/spline_fitting/losses/candidate_pruning_loss.py` | 候选覆盖、阈值归一化拟合、remove/STOP、位置和辅助保留损失 |
| `src/spline_fitting/spline/bspline_deletion_teacher.py` | 批量计算每个单节点删除后的真实标准 B 样条 RMS 训练标签 |
| `src/spline_fitting/evaluation/minimal_knot_pruning.py` | 每次真实重拟合标准 B 样条的硬 RMS 贪心剔除器 |
| `src/spline_fitting/evaluation/bspline_inference.py` | 端点约束的标准开放 B 样条控制点重拟合 |
| `src/spline_fitting/data/synthetic.py` | 合成曲线、归一化和阈值 canonical 标签 |
| `src/spline_fitting/training/trainer.py` | 训练、全局指标聚合和候选召回优先 checkpoint 选择 |
| `src/spline_fitting/checkpointing.py` | v7 独立版本及 v6/v5/更早 checkpoint 严格兼容 |

## 命令入口

| 文件 | 作用 |
|---|---|
| `scripts/train_candidate_pruning.py` | v7 两阶段正式训练，保存 best 与 last |
| `scripts/evaluate_checkpoint.py` | 独立测试集、候选指标和硬剔除部署指标 |
| `scripts/visualize_result.py` | 绘制 all/learned/hard 节点选择对应的标准 B 样条及对照图 |
| `scripts/fit_point_cloud.py` | 对用户有序点云推演，并在全部原始点上硬剔除/重拟合 |
| `scripts/train.py` | v6 复现实验入口，不是 v7 主训练脚本 |

## 兼容模块

| 文件 | 版本 |
|---|---|
| `models/interactive_structure_head.py`、`dynamic_knot_decoder.py` | v6 |
| `models/count_head.py`、`count_conditioned_knot_head.py` | v5/v4 |
| `models/activity_head.py`、`hard_concrete.py` | v3 及更早 |
| `losses/total_loss.py` | v6 及历史联合损失 |

## 主要测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_candidate_pruning_heads.py` | 候选严格有序、交互特征、精修不交叉和输出梯度 |
| `tests/test_candidate_pruning_network.py` | v7 完整 forward/backward 和 checkpoint round-trip |
| `tests/test_bspline_deletion_teacher.py` | 批量删除 RMS 教师与逐状态部署重拟合的数值一致性 |
| `tests/test_minimal_knot_pruning.py` | 硬阈值、停止条件、轨迹、零节点和端点 |
| `tests/test_bspline_inference.py` | 标准 B 样条重拟合和严格端点覆盖 |
| `tests/test_canonical_labels.py` | canonical 标签确定性和阈值行为 |
| `tests/test_checkpointing.py` | v6/v5/v4/v3/历史 checkpoint 迁移 |

`outputs/` 中的 checkpoint、JSON 和图片是实验产物，不属于源码。
