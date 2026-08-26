# Minimum-Complexity B-Spline Fitting

本项目从沿曲线方向排列的二维或三维点云，预测开放三次 B 样条的参数、内部节点和控制顶点。当前主版本为：

```text
objective_version = candidate_pruning_joint_refinement_teacher_v11
structure_mode    = candidate_pruning_one_shot
```

优化目标是在归一化 RMS 阈值 `ε` 下，用尽量少的内部节点拟合曲线：

\[
\min |U| \quad \text{s.t.}\quad
\operatorname{RMS}(C_U,Q)\le \varepsilon .
\]

模型没有 `CountHead`。最终节点数由一次性 `LearnedKeep` 掩码确定，而不是部署时逐节点试删。

## 当前数据流

```text
有序点云 Q [B,M,D]
  → GeometryEncoder
      local_features [B,M,H]
      global_features [B,H]
  → ParameterHead
      严格递增 params t [B,M]
  → CandidateKnotHead
      Kc+1 个带锚点的 interval query
      + 参数位置编码
      + 锚点中心 Gaussian 局部 cross-attention
      → Kc 个严格有序候选
  → 截断幂代理求解
      → 系数能量、解析删除增量、局部残差
  → proposal-only 精修
      → 固定 proposal U_prop [B,Kc]
  → InteractivePruningHead 的固定交互
      p0 → 临时位置 u1 → p1 → 一次性 mask → 最终位置 u*
  → 只保留 mask 对应的 u*
  → 一次标准开放 B 样条控制顶点 refit
```

这里有两套位置，不能混用：

- `proposal_internal_knots`：离线 Hard-RMS 教师绑定的固定候选；教师生成后不再改变。
- `deployment_internal_knots`：由最终 KeepMask 条件化更新的位置；实际部署和最终节点匹配使用它。
- `internal_knots`：兼容别名，v11 中等于 `deployment_internal_knots`。

网络内部会执行两次截断幂代理求解，用于贡献特征和训练期诊断；它们都位于同一次网络 `forward` 中，不是标准 B 样条逐节点搜索。在线部署仍是一次网络前向和一次标准 B 样条 refit。

## v11 的两个主要改动

### 高召回候选

`CandidateKnotHead` 为每个 interval query 设置参数域锚点 `a_j`，并向 cross-attention 分数加入 Gaussian 局部偏置：

\[
b_{j,i}=-\frac{(t_i-a_j)^2}{2h^2}.
\]

默认带宽 `h=0.08`；设为 `0` 可恢复历史全局 attention。候选预训练同时在 `0.005/0.01/0.02` 三个尺度施加 coverage hinge，避免只优化平均最近距离而漏掉少量困难节点。

### 删除与位置更新联动

固定次数交互为：

\[
p^{(0)}\rightarrow u^{(1)}\rightarrow p^{(1)}
\rightarrow M\rightarrow u^* .
\]

1. selector 根据固定 proposal 和贡献特征输出初始概率 `p0`。
2. `p0` 的 hard straight-through 上下文生成所有槽位的临时位置 `u1`。
3. `u1` 的位置编码和位移反馈给 selector，得到最终概率 `p1`。
4. `p1` 经一次 `mass_topk` 或 threshold 产生最终 `KeepMask`。
5. 最终位置头读取 KeepMask 的 hard-ST 上下文，只更新被保留节点，得到 `u*`。

这是一条固定计算图，没有 while 循环，也不在部署时反复拟合。`--one-shot-max-position-shift` 限制的是两阶段合计位移 `|u*-U_prop|`，而不是允许每阶段各移动一次完整预算；同时使用邻接间距和最小节点间隔约束，保持被选节点严格有序。

默认 `mass_topk` 令 `score=sum(p1)+s·sqrt(sum(p1(1-p1)))`。当 `score<0.5` 时稳定输出 `K=0`；否则仍取 `ceil(score)` 并截断到有效候选数。这样零节点是显式结果，不依赖概率质量数值下溢。

## 数据集

默认使用在线生成的开放三次 B 样条：

| 配置 | 默认值 |
|---|---:|
| 训练 / 验证样本 | 10000 / 2000 |
| 训练 / 验证 / 独立测试 seed | 42 / 10000 / 20000 |
| 每条曲线采样点 | 192 |
| 点维度 | 2，可选 3 |
| 源控制顶点 | 8–24 |
| 源内部节点 | 4–20 |
| 候选预算 `Kc` | 28 |
| 坐标噪声标准差 | 0.001 |
| 归一化 RMS 阈值 `ε` | 0.005 |

每条样本包含有序点、弦长参数、真参数、源样条表示和 canonical 节点标签。canonical 标签由源节点出发，在相同 RMS 阈值和端点约束下离线删点得到；源节点数本身不是监督目标。

训练时同时使用两类互补监督：

- canonical 监督：参数、proposal 多尺度覆盖、候选位置、选择存在性和最终部署位置。
- Hard-RMS 教师监督：固定 proposal 槽位上的 hard mask、最终状态 soft risk、排序、关键节点召回、误保留惩罚和目标节点数。

canonical 负责“几何位置接近哪里”，教师负责“固定候选组合中保留哪些槽位才能满足阈值”。二者并不保证在所有曲线上给出相同节点集合。

teacher false-positive 项只惩罚“教师未保留且 canonical 也未匹配”的槽位。若一个槽位是 canonical-positive，但只是没有被贪心教师选中，它会被视为可能的等价替代，不会被强行压成负类。

## 训练

默认训练分为三段：

1. 候选预训练：训练参数头和高召回 proposal，并恢复验证集最优的 `*_proposal.pt`。
2. 选择蒸馏：固定 proposal，生成离线 Hard-RMS 缓存，只训练一次性 selector。
3. 联合校准：默认最后 10 轮以较低学习率训练 selector 与独立 deployment 位置头；固定 proposal 和教师槽位仍不变。该阶段默认加入 `0.05` 的截断幂 fit 和 `0.5` 的阈值违反权重，fit gate 对选择概率停止梯度，因此它们用于改善已选组合的位置可行性，不直接鼓励多开节点。

首次训练：

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

交互式终端默认显示训练、验证、样本指纹和教师生成进度。`--no-progress` 可关闭 batch 进度。

### v10 checkpoint 与教师缓存

v10 checkpoint 仍可按原语义加载、评估和部署；v11 新位置模块由版本配置隔离，读取 v10 不会自动启用 v11 联动。

也可以用已有 v10 proposal 初始化 v11：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 138 `
  --candidate-pretrain-epochs 8 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v10_proposal.pt `
  --selector-calibration-epochs 10 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --candidate-local-attention-bandwidth 0.08 `
  --lambda-one-shot-surrogate-fit 0 `
  --lambda-one-shot-surrogate-threshold 0 `
  --lambda-joint-fit 0.05 `
  --lambda-joint-threshold-violation 0.5 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v11_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v11.pt
```

推荐用 5–10 轮 proposal 预训练适配 v11 的 Gaussian 局部 attention；上例的 8 轮使后续蒸馏仍保留 120 轮。`--candidate-pretrain-epochs 0` 只复用固定 proposal，不会训练新增的局部 attention 行为：脚本会沿用 checkpoint 记录的 attention 带宽，并校验 `model_config` 中会改变 proposal 的非权重语义；不匹配时直接报错。缺少 `model_config` 的历史文件只能按全局 attention 回退并给出无法完整校验的警告。新生成的 `*_proposal.pt` 会保存 `model_config`，便于之后安全复用。

不要把 v10 教师缓存直接用于 v11。Gaussian 局部 attention、proposal 指纹和 v11 的监督契约已经变化，第一次 v11 训练必须使用新的缓存目录，且不要加 `--reuse-teacher-cache`。只有 proposal 权重、数据集、教师配置和张量形状全部不变时，后续 v11 训练才能加 `--reuse-teacher-cache`。

## 独立测试

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --num-samples 2000 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --knot-tolerance 0.05 `
  --json-output outputs/candidate_pruning_one_shot_v11_evaluation.json
```

重点查看：

- proposal recall@`0.005/0.01/0.02/0.05`；
- `proposal_full → selected_pre_update → deployment_post_update` 的 Precision/Recall/F1；
- `position_recall_delta` 和 `position_precision_delta`；
- one-shot 节点数、平均/P95/最大 RMS 与阈值满足率；
- 最终 `match@tolerance` 和 matched MAE。

需要离线 Hard-RMS 对照时额外加 `--run-hard-diagnostic`。它只做诊断，会显著增加时间，不会替换 LearnedKeep 部署。

## 可视化

四联对比图：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --seed 20000 `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view comparison `
  --timing-repeats 5 `
  --dpi 600 `
  --output outputs/candidate_pruning_one_shot_v11_comparison_000.png
```

四个面板依次是源曲线、全部 proposal、一次性 LearnedKeep 部署和离线 Hard-RMS 删除。图中 `net` 是网络前向时间，`refit` 是一次标准 B 样条重拟合时间，`prune` 是多次试删与重拟合的离线硬搜索时间。

### 批量分层四联图

论文批量图推荐使用独立脚本：

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

该脚本先按 source 或 canonical 内部节点数分层，再用固定 `selection-seed` 随机轮转抽样，输出若干 PNG 以及 `comparison_manifest.json`、`comparison_manifest.csv`。批量图使用：

\[
\operatorname{MSE}=\frac1M\sum_i\|C(t_i)-Q_i\|_2^2,
\]

不再开平方；因此 `2.5e-5=(0.005)^2`。四个面板中 all-proposal 和传统 Hard 删除都从固定 `proposal_internal_knots` 开始，LearnedKeep 则使用 `deployment_internal_knots[learned_keep_mask]`。Hard 是同一 proposal 上反复执行标准 refit 的传统贪心单节点删除，不是模型的在线部署步骤。

## 用户点云

输入必须是沿曲线方向排列的 `.csv`、`.txt`、`.npy` 或 `.pt` 点序列：

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v11.pt `
  --point-cloud data/my_curve.csv `
  --fit-tolerance 0.005 `
  --json-output outputs/my_curve_v11.json `
  --figure-output outputs/my_curve_v11.png
```

用户点云会按训练长度重采样、归一化、执行一次网络前向，并在原始点分辨率上做一次标准 B 样条 refit。

## 文档

- [模型与数据流](docs/architecture.md)
- [数据与训练流程](docs/training_pipeline.md)
- [部署、评估与可视化](docs/deployment_pipeline.md)
- [数学定义](docs/math_formulation.md)
- [版本演化](docs/pruning_redesign.md)
- [文件索引](docs/file_guide.md)

当前实现提供的是学习式一次性近似，不保证每条新曲线都达到 `ε`，也不保证全局最少节点。实际效果应以独立 seed 的标准 B 样条 refit、节点匹配和耗时结果为准。
