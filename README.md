# Minimum-Complexity B-Spline Fitting

本项目从沿曲线方向排列的二维或三维点云，预测开放三次 B 样条的参数、内部节点和控制点。当前主版本为 `candidate_pruning_structured_feasible_teacher_v10`，目标是在归一化 RMS 阈值 `ε` 下，用尽量少的内部节点拟合曲线：

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
  → proposal-only 位置精修：生成教师绑定的固定候选位置
  → v10 结构化 selector adapter
       固定位置 + 解析贡献特征
       → 两层候选 self-attention
       → raw importance r 与曲线自适应阈值 β
       → 概率质量 Top-K + 不确定性余量 + 参数域覆盖锚点
       → 一次性 LearnedKeep mask
  → 第二次截断幂代理求解：记录 surrogate 拟合诊断（选择阶段默认不驱动 KeepMask）
  → 部署时仅保留最终 KeepMask 对应节点
  → 一次标准开放 B 样条控制点重拟合
```

网络内部虽然有两次截断幂代理求解，但整个模型仍只执行一次 `forward`；这两次代理求解不属于部署的标准 B 样条 refit，也不是逐节点搜索。

最终决策不再逐槽位独立执行 `p≥0.5`。先计算

\[
p_j=\sigma(r_j-\beta),\quad
\widehat K=\left\lceil\sum_jp_j+s\sqrt{\sum_jp_j(1-p_j)}\right\rceil,
\]

再保留 raw importance 最高的 `K̂` 个候选，并为若干参数域区间保留最高概率锚点。`s` 是不确定性安全余量。它不是 CountHead，也不运行 B 样条搜索。v10 另外用教师节点集合的累计分布损失和关键节点漏检损失训练组合选择；节点位置仍与离线教师严格绑定。

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
3. 冻结完整 proposal 几何，只蒸馏独立 selector adapter、`keep_head` 和 `adaptive_threshold_head`。
4. 可选 selector 校准仍只更新上述选择模块，绝不移动教师绑定的节点位置。

交互式终端默认逐batch显示训练和验证进度条，包括百分比、当前loss、吞吐率和ETA；使用
`--no-progress` 可关闭。输出被重定向到日志文件时会自动退回 `--log-every-batches` 的逐行日志。

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --selector-calibration-epochs 0 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --candidate-match-tolerance 0.01 `
  --deployment-pass-rate-target 0.97 `
  --selector-lr 5e-4 `
  --positive-keep-weight 2.0 `
  --lambda-keep 1.0 `
  --lambda-teacher-ranking 1.0 `
  --lambda-teacher-distribution 2.0 `
  --lambda-teacher-critical-recall 1.0 `
  --teacher-ranking-margin 1.0 `
  --lambda-teacher-count 4.0 `
  --lambda-complexity 0.1 `
  --one-shot-selection-policy mass_topk `
  --one-shot-safety-sigma 0.5 `
  --one-shot-selector-layers 2 `
  --one-shot-coverage-bins 4 `
  --teacher-batch-size 4 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v10_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v10.pt
```

已有最佳 proposal 且缓存指纹完全一致时，可跳过候选预训练：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 130 `
  --candidate-pretrain-epochs 0 `
  --proposal-checkpoint outputs/candidate_pruning_v9_proposal.pt `
  --selector-calibration-epochs 0 `
  --train-size 10000 `
  --val-size 2000 `
  --candidate-knots 28 `
  --num-points 192 `
  --one-shot-selection-policy mass_topk `
  --one-shot-safety-sigma 0.5 `
  --one-shot-selector-layers 2 `
  --one-shot-coverage-bins 4 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v10_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v10.pt
```

教师缓存绑定数据集指纹、proposal 权重指纹、配置、样本 ID 与张量形状。使用缓存时禁止 `--resample-train-each-epoch`；不匹配会直接报错，不会静默套用旧标签。

一次性选择阶段默认关闭截断幂 surrogate 的 fit/violation 梯度，KeepMask 由标准 B 样条
Hard-RMS 教师的 hard mask、最终状态 soft risk、保留/删除成对排序和节点数监督。canonical
位置监督只用于教师生成前的 proposal 预训练；蒸馏阶段不再用第二套位置目标改变教师槽位。
checkpoint 只依据真实标准 B 样条部署指标排序：未达到目标通过率时比较通过率及均值/P95 RMS；
达到目标后优先最少节点，再以真实 RMS 和最终节点匹配作 tie-break；surrogate fit 不参与选优。

## 评估、可视化与用户点云

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --num-samples 2000 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/candidate_pruning_one_shot_v10_evaluation.json
```

旧 v9 权重可先用结构化掩码做部署消融，无需重训：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_v9.pt `
  --num-samples 2000 `
  --one-shot-selection-policy mass_topk `
  --one-shot-safety-sigma 0.5 `
  --one-shot-coverage-bins 4 `
  --json-output outputs/candidate_pruning_v9_structured_mask.json
```

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --sample-index 0 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --pruning-view comparison `
  --timing-repeats 3 `
  --dpi 600 `
  --output outputs/candidate_pruning_one_shot_v10_sample_000.png
```

`comparison` 输出四个同尺度曲线面板：原始/source 样条、全部冗余候选、LearnedKeep 部署、
离线 Hard-RMS 删除。每个面板同时绘制采样点、控制多边形、内部节点在曲线上的位置、完整
节点向量、RMS 和 CPU 时间；Learned/Hard 时间均包含同一次网络前向。

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
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
