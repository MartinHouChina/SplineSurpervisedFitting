# v8 数据与训练流程

## 1. 学习目标

给定归一化 RMS 阈值 `ε`：

\[
K^*(\varepsilon)=
\min_U |U|\quad\text{s.t.}\quad
\sqrt{\frac1M\sum_i\|C_U(t_i)-q_i\|_2^2}\le\varepsilon.
\]

源样条的节点数不是监督目标。同一条曲线可以通过 Boehm 插入得到不同但等价的节点表示，因此 v8 用标准 B 样条 Hard-RMS 删除轨迹构造固定候选集上的近似最简教师。

## 2. 基础数据

每条合成样本按顺序生成：

1. 随机选择 8–24 个控制点，对应 4–20 个源内部节点。
2. 生成开放三次 B 样条、非均匀参数和有序采样点。
3. 加入坐标噪声。
4. 中心化并按最大半径归一化。
5. 保存真参数、源节点和 canonical 诊断标签。

默认配置：

| 配置 | 值 |
|---|---:|
| 训练 / 验证样本 | 10000 / 2000 |
| 训练 / 验证 seed | 42 / 10000 |
| 每条曲线点数 | 192 |
| 候选预算 `Kc` | 28 |
| 噪声标准差 | 0.001 |
| RMS 阈值 | 0.005 |

canonical 标签用于诊断和候选覆盖监督；最终 KeepMask 教师来自离线 Hard-RMS 搜索。

## 3. 三个必需阶段和一个可选阶段

### 阶段 1：候选预训练并选择最佳 proposal

候选阶段强制所有候选参与拟合，冻结最终 keep 决策模块，主要优化：

- 参数真值监督；
- true→candidate 单向覆盖；
- 候选位置与轻量排斥；
- 全候选截断幂代理拟合；
- 阈值违约惩罚。

每个 epoch 在验证集上比较 candidate recall、nearest MAE、节点匹配和损失。阶段结束后，不使用最后一轮，而是恢复验证排序最好的 `*_proposal.pt`。

若已有可信 proposal，可设置：

```text
--proposal-checkpoint <path>
--candidate-pretrain-epochs 0
```

零候选预训练时 `--proposal-checkpoint` 是必需的。

### 阶段 2：数据 + proposal 指纹绑定的离线教师

恢复最佳 proposal 后，对固定训练集和验证集生成教师：

```text
固定候选
  → 尝试删除每个当前候选
  → 每个方案完整执行标准 B 样条 refit
  → 接受 RMS 最低且 RMS≤ε 的删除
  → 直到下一次删除不再安全
```

缓存包括：

```text
teacher_retained_mask
teacher_soft_keep_risk
teacher_count
teacher_fit_rms
teacher_threshold_satisfied
teacher_deletion_order
teacher_single_deletion_rms
teacher_internal_knots / mask
```

`teacher_single_deletion_rms` 是“全候选状态”的单次删除诊断，不直接作为最终 keep 概率目标。真正的 `teacher_soft_keep_risk` 来自贪心搜索最终停止状态：

- 对最终保留节点，计算该状态下逐节点 leave-one-out RMS；
- 用 `RMS/ε-1` 的 margin 经温度 sigmoid 得到风险；
- 已删除候选的最终风险为 0；
- 最终 hard mask 来自完整删除轨迹。

这避免了“多个冗余节点在初始全候选状态下都可单独删除，却不能同时删除”的错误监督。

### 阶段 3：只训练选择模块的蒸馏

生成缓存后，冻结整个模型，再只解冻：

```text
keep_head
adaptive_threshold_head
position_to_keep_feedback
```

编码器、参数头、候选头、解析特征路径、keep-context 和位置残差头均冻结，因此基础 proposal
槽位与参数不会漂移；最终保留位置另由教师节点位置监督锚定。

学生在一次前向中学习：

```text
preliminary keep
  → provisional position
  → position feedback
  → final keep / β / p
```

主要监督来自标准 B 样条 Hard-RMS 教师：hard mask 使用无偏置 BCE + soft Dice，另有
final-state soft risk、Hard-ST 节点数、教师节点位置和复杂度项。Hard-ST 数量的前向值等于
实际二值节点数，反向使用概率梯度，避免“所有概率均低于 0.5，但概率和恰好正确”的漏洞。

截断幂 surrogate 仍随 forward 计算并记录，但一次性选择阶段默认设置
`--lambda-one-shot-surrogate-fit 0` 和 `--lambda-one-shot-surrogate-threshold 0`，不再驱动
KeepMask。标准 B 样条 refit 是离散运算，不直接反向传播；它通过离线教师标签提供监督，并在
验证中作为真实部署指标。

### 阶段 4：可选 keep-position 校准

`--keep-position-calibration-epochs > 0` 时，以 `0.1×selector-lr` 继续训练：

```text
keep_head
adaptive_threshold_head
position_to_keep_feedback
keep_context_projection
keep_context_norm
position_residual_head
```

GeometryEncoder、ParameterHead 和 CandidateKnotHead 始终冻结。该阶段只校准 keep 与位置头的耦合，不是端到端 joint fine-tuning，也不会改变缓存绑定的 proposal 主干。

若总 epoch 为 `E`、候选预训练为 `Ep`、校准为 `Ec`，蒸馏 epoch 为：

\[
E_d=E-E_p-E_c\ge1.
\]

设置 `--keep-position-calibration-epochs 0` 即运行三阶段流程。

## 4. 缓存完整性约束

教师缓存严格校验：

- 教师数值配置及其 fingerprint；
- proposal 模型权重 fingerprint；
- 数据集配置和内容 fingerprint；
- split、seed、样本 ID 与顺序；
- 参数、点云、候选张量形状；
- 标签键、dtype、有限性、mask/count 一致性。

首次运行会先逐条生成固定数据并计算内容指纹。对于 `10000+2000`、192点、4–20源节点的
配置，这一步主要在CPU执行canonical删点，可能持续数十分钟；终端会分别显示
`fingerprint train`、`fingerprint validation`进度、速度和ETA。样本同时写入内存缓存，随后
Hard-RMS教师阶段不会再次生成这些曲线，并会显示独立的`Hard-RMS teacher`进度条。

因此：

> 使用离线教师缓存时禁止按 epoch 重采样训练集。

`--resample-train-each-epoch` 会使同一 sample ID 对应不同曲线，脚本会直接拒绝。只有所有指纹完全相同时，`--reuse-teacher-cache` 才会成功；不会静默复用旧标签。

## 5. 一次训练 forward 内的两个代理 solve

每次网络前向固定执行两次截断幂代理求解：

1. 全候选 pilot solve：生成 coefficient energy、analytic deletion delta 和局部残差。
2. final hard-ST KeepMask solve：生成可微代理重建；选择阶段默认仅作诊断。

这两次 solve 都不是标准 B 样条 refit。验证选优会额外按真实部署规则，对 final KeepMask 执行一次标准 B 样条 refit。

## 6. 验证与 checkpoint 选优

蒸馏和校准阶段的验证会真实执行：

```text
one-shot final KeepMask
  → 一次标准开放 B 样条 refit
  → deployment RMS 与 RMS≤ε
```

验证采用约束式选优。默认目标是：

\[
\min \mathbb E[K]\quad
\text{s.t.}\quad
P(R_{\mathrm{B\text{-}spline}}\le\varepsilon)\ge0.97.
\]

达到 `--deployment-pass-rate-target` 的 checkpoint 先按平均保留节点数排序，再比较教师 mask
F1、教师数量 MAE 和真实部署 RMS；尚未达到约束时，优先提高真实标准 B 样条满足率。最初
`--selector-checkpoint-warmup-epochs 10` 轮不参与选优，防止轻微保守初始化成为“全保留”的
伪最佳结果。

蒸馏和校准分别保存最佳 checkpoint，最后按同一验证排序选择正式输出。

## 7. 正式训练命令

训练与验证默认显示实时batch进度条：

```text
train      [############----------------]  250/625   40.00% | loss=0.842100 | 3.20 batch/s | ETA 01:57
validation [############################]   63/63   100.00% | loss=0.731200 | 5.80 batch/s | ETA 00:00
```

`--no-progress` 可关闭进度条；非交互式日志环境会自动使用 `--log-every-batches` 输出，避免在日志中产生大量回车刷新记录。

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --keep-position-calibration-epochs 10 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --log-every-batches 20 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --deployment-pass-rate-target 0.97 `
  --selector-lr 5e-4 `
  --selector-checkpoint-warmup-epochs 10 `
  --initial-keep-probability 0.55 `
  --positive-keep-weight 1.0 `
  --lambda-keep 1.0 `
  --lambda-teacher-count 2.0 `
  --lambda-complexity 0.25 `
  --lambda-one-shot-surrogate-fit 0 `
  --lambda-one-shot-surrogate-threshold 0 `
  --candidate-match-tolerance 0.02 `
  --teacher-risk-temperature 0.1 `
  --teacher-batch-size 4 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v8_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v8.pt
```

首次运行会保存最佳 proposal：

```text
outputs/candidate_pruning_one_shot_v8_proposal.pt
```

使用同一数据、配置、proposal 和缓存继续实验：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 130 `
  --candidate-pretrain-epochs 0 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v8_proposal.pt `
  --keep-position-calibration-epochs 10 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v8_teacher `
  --reuse-teacher-cache `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v8.pt
```

小规模链路测试：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 4 `
  --candidate-pretrain-epochs 1 `
  --keep-position-calibration-epochs 1 `
  --train-size 16 `
  --val-size 8 `
  --batch-size 4 `
  --num-points 64 `
  --min-control-points 4 `
  --max-control-points 8 `
  --candidate-knots 8 `
  --hidden-dim 32 `
  --encoder-layers 1 `
  --teacher-batch-size 2 `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v8_smoke.pt
```

该命令只验证 proposal、缓存、蒸馏、校准和 checkpoint 链路，不代表精度。
