# Minimum-Complexity B-Spline Fitting

本项目从沿曲线方向排列的二维或三维点云，预测开放三次 B 样条的参数、内部节点和控制点。当前主版本为 `candidate_pruning_one_shot_teacher_v8`，目标是在归一化 RMS 阈值 `ε` 下，用尽量少的内部节点拟合曲线：

\[
\min |U|\quad\text{s.t.}\quad \operatorname{RMS}(C_U,Q)\le\varepsilon .
\]

模型没有 CountHead，也不把生成曲线时的源节点数当作目标。节点数等于最终 LearnedKeep 掩码中被保留的候选数。

## 当前工作流

```text
有序点云
  → GeometryEncoder：局部/全局几何特征
  → ParameterHead：严格递增参数 t
  → CandidateKnotHead：固定 Kc 个严格有序、高召回候选
  → 第一次截断幂代理求解：提取系数能量、删除增量和局部残差
  → InteractivePruningHead 的固定双向链路
       preliminary keep/β
       → provisional position
       → position feedback
       → final keep/β/p
       → hard-ST KeepMask context
       → final position
  → 第二次截断幂代理求解：记录 surrogate 拟合诊断（选择阶段默认不驱动 KeepMask）
  → 部署时仅保留最终 KeepMask 对应节点
  → 一次标准开放 B 样条控制点重拟合
```

网络内部虽然有两次截断幂代理求解，但整个模型仍只执行一次 `forward`；这两次代理求解不属于部署的标准 B 样条 refit，也不是逐节点搜索。

最终决策为

\[
p_j=\sigma(r_j-\beta),\qquad
m_j=\mathbf 1[p_j\ge0.5].
\]

`β` 是每条曲线自适应预测的 raw-importance logit 阈值，不是概率。最终位置上下文前向只汇聚 `m_j=1` 的候选；训练反向使用 straight-through 概率梯度。位置更新受相邻间距限制，输出不需要排序也保持严格有序。

## 数据集

默认使用在线生成的开放三次 B 样条：

| 配置 | 默认值 |
|---|---:|
| 训练 / 验证样本 | 10000 / 2000 |
| 训练 / 验证 / 独立测试 seed | 42 / 10000 / 20000 |
| 每条曲线采样点 | 192 |
| 源控制点 | 8–24 |
| 源内部节点 | 4–20 |
| 候选节点预算 `Kc` | 28 |
| 坐标噪声标准差 | 0.001 |
| 归一化 RMS 阈值 `ε` | 0.005 |

样本会中心化并按最大半径归一化。离线教师与部署都使用端点约束的标准开放 B 样条重拟合。源节点数只描述数据生成过程，不等于阈值下的最少节点数。

## 训练

训练由三个必需阶段和一个可选阶段组成：

1. 候选预训练并恢复验证集上最好的 proposal。
2. 对固定数据和固定 proposal 生成离线 Hard-RMS 教师；soft risk 来自贪心删除最终停止状态的 leave-one-out RMS。
3. 冻结编码器、参数头和候选头，只蒸馏 `keep_head`、`adaptive_threshold_head` 和 `position_to_keep_feedback`。
4. 可选 keep-position 校准：额外训练 keep-context 与位置残差模块；proposal 主干仍冻结，不是端到端联合微调。

交互式终端默认逐batch显示训练和验证进度条，包括百分比、当前loss、吞吐率和ETA；使用
`--no-progress` 可关闭。输出被重定向到日志文件时会自动退回 `--log-every-batches` 的逐行日志。

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --keep-position-calibration-epochs 10 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --deployment-pass-rate-target 0.97 `
  --selector-lr 5e-4 `
  --positive-keep-weight 1.0 `
  --lambda-keep 1.0 `
  --lambda-teacher-count 2.0 `
  --lambda-complexity 0.25 `
  --teacher-batch-size 4 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v8_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v8.pt
```

已有最佳 proposal 且缓存指纹完全一致时，可跳过候选预训练：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 130 `
  --candidate-pretrain-epochs 0 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v8_proposal.pt `
  --keep-position-calibration-epochs 10 `
  --train-size 10000 `
  --val-size 2000 `
  --candidate-knots 28 `
  --num-points 192 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v8_teacher `
  --reuse-teacher-cache `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v8.pt
```

教师缓存绑定数据集指纹、proposal 权重指纹、配置、样本 ID 与张量形状。使用缓存时禁止 `--resample-train-each-epoch`；不匹配会直接报错，不会静默套用旧标签。

一次性选择阶段默认关闭截断幂 surrogate 的 fit/violation 梯度，KeepMask 由标准 B 样条
Hard-RMS 教师的 hard mask、soft risk、节点数和位置监督。验证会真实执行一次部署路径：当
标准 B 样条阈值满足率达到 `--deployment-pass-rate-target` 后，优先选择平均节点数最少的
checkpoint；未达到约束时才优先提高满足率。这避免“全部保留以换取最低 RMS”的错误选优。

## 评估、可视化与用户点云

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v8.pt `
  --num-samples 2000 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/candidate_pruning_one_shot_v8_evaluation.json
```

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v8.pt `
  --sample-index 0 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --pruning-view learned `
  --dpi 600 `
  --output outputs/candidate_pruning_one_shot_v8_sample_000.png
```

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v8.pt `
  --point-cloud data/my_curve.csv `
  --fit-tolerance 0.005 `
  --json-output outputs/my_curve_fit.json `
  --figure-output outputs/my_curve_fit.png
```

用户点云必须沿曲线方向有序。默认部署是“一次网络前向 + 一次标准 B 样条 refit”，不会逐样本运行 Hard-RMS 剪枝，因此 `RMS≤ε` 是独立测试分布上的统计满足率，不是每条输入的硬保证。

## 文档

- [模型与数据流](docs/architecture.md)
- [数据与训练流程](docs/training_pipeline.md)
- [部署与指标](docs/deployment_pipeline.md)
- [数学定义](docs/math_formulation.md)
- [文件索引](docs/file_guide.md)

运行测试：

```powershell
python -m pytest -q
```
