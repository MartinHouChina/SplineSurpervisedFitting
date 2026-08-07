# 文件功能说明

本说明对应 v0.4。源码包位于 `src/spline_fitting/`。

## 数据

| 文件 | 功能 |
|---|---|
| `data/dataset.py` | 通用有序曲线数据读取、归一化和弦长参数工具 |
| `data/synthetic.py` | 生成开放三次 B 样条合成数据，返回采样点、真实参数、控制点、完整节点向量和真实内部节点 |

`SyntheticCubicBSplineDataset` 的 `true_params` 同时用于当前参数监督；真实节点和控制点仍用于评估与可视化。

## 网络

| 文件 | 功能 |
|---|---|
| `models/geometry_encoder.py` | 用一维卷积从有序点序列提取局部和全局几何特征 |
| `models/parameter_head.py` | 通过严格间距预算构造单调投影参数 \(t\)，并保留 legacy 参数化入口 |
| `models/knot_head.py` | 用带位置编码的局部 cross-attention 预测区间，并通过严格间距预算构造有序候选节点 \(U\) |
| `models/hard_concrete.py` | 实现 Hard-Concrete 采样、闭式 \(P(z>0)\)、评估二值门、温度与全开控制 |
| `models/activity_head.py` | 汇聚候选节点局部上下文，直接预测 \(\log\alpha\) 并调用 Hard-Concrete |
| `models/spline_network.py` | 组装编码器、三个预测头、训练代理基和可微线性求解 |

## 训练代理数学层

| 文件 | 功能 |
|---|---|
| `spline/truncated_power_basis.py` | 构造多项式基、截断幂增量基；A 方案直接乘门，legacy 模式保留平方根门 |
| `spline/differentiable_solver.py` | 求解截断幂系数，并给出约束删除单列后的解析岭目标增量 |
| `spline/curve_evaluation.py` | 根据设计矩阵和截断幂系数重建或密集采样代理曲线 |
| `spline/derivatives.py` | 构造任意阶导数设计矩阵；当前新训练不调用，仅保留历史正交目标兼容能力 |

## 损失与训练

| 文件 | 功能 |
|---|---|
| `losses/total_loss.py` | 计算 fit、期望 \(L_0\)、gap、true-parameter；保留旧 activity/binary/orthogonal 配置兼容入口 |
| `training/trainer.py` | 训练/验证循环、全开预热、稀疏 ramp、温度调度、日志和 checkpoint 保存 |

## 部署与诊断

| 文件 | 功能 |
|---|---|
| `evaluation/bspline_inference.py` | 校验二值门、物理删节点、构造标准开放 B 样条、用增广 LSTSQ 重拟合控制点 |
| `evaluation/knot_diagnostics.py` | 点拟合统计、软概率统计、旧截断幂硬剪枝诊断、节点贡献与真实节点匹配 |
| `checkpointing.py` | 显式迁移 pre-A model/loss 配置、区分新旧目标并恢复门控温度 |

`bspline_inference.py` 是 A 方案最终推演入口；`hard_prune_and_refit` 仍保留用于旧截断幂结果对照，不代表最终标准 B 样条输出。

## 脚本

| 文件 | 功能 |
|---|---|
| `scripts/train.py` | 训练随机开放三次 B 样条数据集上的 A 方案模型，并写入完整配置元数据 |
| `scripts/evaluate_checkpoint.py` | 在验证集报告硬门节点数、网络代理损失和标准 B 样条部署损失，可导出 JSON |
| `scripts/visualize_result.py` | 绘制采样点、真实曲线、硬门代理曲线、标准 B 样条部署曲线和节点概率 |

## 测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_hard_concrete.py` | Hard-Concrete 概率、随机门、二值门、温度保存、局部上下文、pilot drop-cost 公式和直接门控梯度 |
| `tests/test_a_scheme_loss_and_gaps.py` | expected-\(L_0\)、true-parameter 监督、cross-attention、严格参数/节点间距 |
| `tests/test_bspline_inference.py` | \(K=0\)、批内变长节点、二值门校验、标准 B 样条重拟合与增广目标 |
| `tests/test_knot_diagnostics.py` | 拟合统计、阈值计数、旧硬剪枝诊断和节点匹配 |

运行：

~~~bash
python -m unittest discover -s tests -v
~~~

## 文档

| 文件 | 内容 |
|---|---|
| `README.md` | 项目概览、快速开始、A 方案完整流程和指标口径 |
| `docs/architecture.md` | 训练与部署两条架构路径 |
| `docs/math_formulation.md` | 完次数学定义 |
| `docs/training_pipeline.md` | 三阶段训练、验证、部署和异常诊断 |
| `docs/file_guide.md` | 当前文件职责 |

## 配置和输出

- `configs/*.yaml` 当前为空且没有接入运行入口，不是配置真值来源。
- 运行时配置来自命令行。
- 可复现实验配置保存在 checkpoint 的 `model_config`、`dataset_config`、`loss_config` 和 `training_config`。
- `outputs/` 保存 checkpoint、图像和 JSON 报告，不属于源码。

## 兼容性边界

pre-A checkpoint 缺少 `gate_mode`。`checkpointing.py` 会显式设置：

~~~text
gate_mode = legacy_soft
activity_use_local_context = false
gap_parameterization = legacy
l0 = 0
~~~

这样可以读取并诊断旧权重，但旧 sigmoid activity 不能当作新的 Hard-Concrete 概率解释。A 方案论文实验必须重新训练。

旧 checkpoint 缺少 `knot_use_local_cross_attention` 时恢复全局 MLP KnotHead。历史 pilot 配置按原值加载；v0.4 新训练显式保存 `activity_use_pilot_importance=false`。
