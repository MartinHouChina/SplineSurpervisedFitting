# v11 数据与训练流程

## 1. 训练目标

训练目标不是复原生成曲线时的源节点数，而是在归一化 RMS 阈值 `ε` 下学习一个较小且可行的节点集合：

\[
\min |U| \quad \text{s.t.}\quad
\operatorname{RMS}(C_U,Q)\le\varepsilon.
\]

同一条曲线可以有多套等价或近似等价的 B 样条表示。因此 v11 不把“源节点向量”直接当作唯一结构答案，而是组合使用 canonical 几何标签和固定 proposal 上的 Hard-RMS 教师。

## 2. 数据集构成

`SyntheticCubicBSplineDataset` 对每条样本执行：

1. 随机生成开放三次 B 样条控制顶点和非均匀内部节点。
2. 生成严格有序的曲线参数和空间采样点。
3. 加入坐标噪声。
4. 中心化，并按最大半径归一化。
5. 从源节点出发，在 `canonical_knot_tolerance` 下贪心删除可移除节点。
6. 保存源表示、canonical 表示、真参数和有序点云。

默认配置：

| 项目 | 默认值 |
|---|---:|
| 训练集 | 10000，seed 42 |
| 验证集 | 2000，seed 10000 |
| 独立测试集 | 由评估脚本生成，seed 20000 |
| 每条曲线点数 | 192 |
| 点维度 | 2，可选 3 |
| 源控制顶点 | 8–24 |
| 源内部节点 | 4–20 |
| 网络候选预算 `Kc` | 28 |
| 噪声标准差 | 0.001 |
| RMS 阈值 | 0.005 |

关键样本字段：

```text
points                         [M,D]
chord_params                   [M]
true_params                    [M]
true_internal_knots            [Kmax]
true_internal_knot_mask        [Kmax]
source_control_points          [Pmax,D]
source_control_point_mask      [Pmax]
source_knot_vector             [Umax]
source_knot_mask               [Umax]
canonical_fit_rms              scalar
sample_id                      scalar
```

`true_internal_knots` 在当前数据集里表示 canonical 内部节点，而 `source_*` 保留生成时的原始样条，主要用于诊断和可视化。

## 3. 两类监督

### 3.1 canonical 监督

canonical 标签来自源节点上的阈值删点，使用真参数和端点约束的标准 B 样条 refit。它监督：

- `ParameterHead` 的真参数；
- proposal 的最近距离和有序匹配位置；
- `0.005/0.01/0.02` 多尺度候选覆盖；
- selector 的 canonical existence 辅助目标；
- 联合校准时，教师保留槽位对应的最终 deployment 位置。

### 3.2 Hard-RMS 教师

教师输入不是源节点，而是模型产生的固定 `proposal_internal_knots`。对每条曲线运行离线贪心删除：

```text
固定 proposal
  → 标准 B 样条 refit
  → 尝试每个单节点删除
  → 接受 RMS≤ε 的最佳删除
  → 重复，直到没有可接受删除
```

缓存标签包括：

```text
teacher_retained_mask
teacher_soft_keep_risk
teacher_internal_knots
teacher_internal_knot_mask
teacher_count
teacher_fit_rms
teacher_threshold_satisfied
teacher_deletion_order
teacher_single_deletion_rms
```

hard mask 给出教师最终组合；soft risk 来自最终停止状态的 leave-one-out 风险，而不是初始全候选状态。

### 3.3 为什么需要双监督

教师 mask 与固定 proposal 槽位严格绑定，擅长回答“哪些候选组合能满足阈值”，但不会自动把候选移动到 canonical 节点附近。canonical 标签提供位置和几何覆盖目标。

v11 的处理方式是：

```text
proposal U_prop：冻结，维持教师槽位合法
deployment U*  ：独立更新，向 canonical 节点校准
```

这样删除和位置更新可以联动，而不会让离线 teacher cache 失效。

## 4. 阶段 1：高召回 proposal 预训练

候选预训练时：

- `Kc+1` 个 interval query 使用锚点中心 Gaussian 局部 cross-attention；
- 参数头、候选头、proposal self-attention 和 proposal-only 位置头参与训练；
- selector 和 v11 的两个 deployment 位置头冻结；
- 实际截断幂设计矩阵强制保留全部候选，避免早期错误 mask 阻断 proposal 学习。

proposal 损失包括：

\[
L_{\mathrm{proposal}}=
\lambda_tL_t+
\lambda_cL_{\mathrm{coverage}}+
\lambda_mL_{\mathrm{multiscale}}+
\lambda_uL_{\mathrm{position}}+
\lambda_rL_{\mathrm{repulsion}}+
\lambda_fL_{\mathrm{fit}}+
\lambda_\varepsilon L_{\mathrm{violation}}.
\]

多尺度项对每个 canonical 节点的最近候选距离 `d` 使用：

\[
L_{\mathrm{multiscale}}
=\frac{1}{|\mathcal T|}
\sum_{\tau\in\mathcal T}
\left[\max\left(\frac d\tau-1,0\right)\right]^2,
\quad
\mathcal T=\{0.005,0.01,0.02\}.
\]

它在进入容差后饱和，目标是提升各尺度的 proposal recall，而不是无限压缩所有最近距离。

每个 epoch 验证 proposal recall、最近 MAE、节点匹配和损失。阶段结束后恢复验证排序最好的：

```text
outputs/..._proposal.pt
```

## 5. 阶段 2：生成离线教师

恢复最佳 proposal 后：

1. 关闭 force-open 模式。
2. 把 selector 重置到接近 `p=0.55` 的中性状态。
3. 计算 proposal 权重指纹和训练/验证数据内容指纹。
4. 从 `proposal_internal_knots` 生成 train/val Hard-RMS 缓存。

proposal 指纹只包含会影响参数和固定 proposal 的模块与配置，包括：

- encoder 和 ParameterHead；
- CandidateKnotHead；
- proposal 的解析特征投影、self-attention 和位置头；
- Gaussian attention 带宽、最小间隔、代理正则等元数据。

它不包含独立 selector 和 deployment 位置头，因此后两者的训练不会使 teacher cache 自己失效。

缓存还严格校验：

- 教师数值配置；
- 数据集内容、split、seed 和样本顺序；
- sample ID；
- 输入张量形状；
- 标签键、dtype、有限性和 mask/count 一致性。

固定缓存与 `resample_each_epoch=True` 不兼容，所以 v11 要求：

```text
--no-resample-train-each-epoch
```

## 6. 阶段 3：一次性选择蒸馏

此阶段冻结参数头、候选头和 `U_prop`，只训练：

- 两层 selector attention；
- `keep_head`；
- `adaptive_threshold_head`。

v11 固定交互模块存在于 forward 中，但初始和最终位置头保持零初始化且冻结，因此此阶段主要先学稳定的 KeepMask。

主要损失为：

- teacher hard mask：加权 BCE + soft Dice；
- teacher soft risk；
- retained/remove 成对排序；
- teacher critical recall；
- teacher false-positive，但只作用于 teacher-negative 且 canonical-negative 的槽位；
- teacher count；
- `mass_topk` 实际 requested-count score；
- teacher 槽位分布；
- canonical existence；
- complexity。

canonical-positive 槽位可能是贪心 Hard-RMS 未选择的邻近等价替代，因此不会仅因 teacher mask 为零就被 false-positive 项强行压低。

其中 policy count 直接约束请求分数：

\[
\sum_jp_j+s\sqrt{\sum_jp_j(1-p_j)}
\]

部署规则在分数小于 `0.5` 时输出 `K=0`，否则使用 `ceil`。训练把 teacher `K=0` 的连续目标放在 `0.25`，把正数 `K` 的目标放在 `K-0.25`，即各自稳定决策区间的内部，避免概率和正确但部署节点数仍偏大，也不要求零节点样本把所有概率压到数值下溢。

默认：

```text
--lambda-one-shot-surrogate-fit 0
--lambda-one-shot-surrogate-threshold 0
```

纯选择蒸馏时截断幂 surrogate 仍计算和记录，但不驱动 KeepMask，避免通过全保留轻易降低代理误差。

## 7. 阶段 4：Keep/位置联合校准

默认最后 10 轮：

```text
--selector-calibration-epochs 10
```

在阶段 3 参数的基础上，以 `selector_lr × 0.1` 训练 selector 和以下独立模块：

```text
joint_preliminary_position_*
joint_position_to_keep_*
joint_final_position_*
```

该阶段默认额外启用小权重位置可行性信号：

```text
--lambda-joint-fit 0.05
--lambda-joint-threshold-violation 0.5
```

它们作用于截断幂代理拟合和代理 RMS 阈值违反。进入设计矩阵的 hard-ST fit gate 会对选择概率 detach，因此代理 fit 不能沿门控捷径直接打开更多节点；其设计用途是校准已选组合的位置可行性。KeepMask 仍主要由 teacher/canonical/count 等结构监督决定，最终可行性仍以验证时的标准 B 样条 refit 为准。

前向顺序固定为：

```text
p0 → u1 → p1 → final mask → u*
```

`--one-shot-max-position-shift` 是 `u*` 相对固定 `U_prop` 的两阶段合计预算。第二个位置头只能使用第一阶段剩余的预算，而不是再获得一次完整上限。

联合位置损失只对 teacher retained slots 生效，再与 canonical 节点做一维有序最小代价匹配：

\[
L_{\mathrm{joint-pos}}
=\operatorname{SmoothL1}(U^*_{\mathrm{teacher\ slots}},
U_{\mathrm{canonical}}).
\]

未被教师保留的槽位不承担这一位置目标。`U_prop` 始终冻结，teacher mask 的槽位身份保持不变。

若设置 `--selector-calibration-epochs 0`，会跳过位置校准；此时 v11 架构存在，但新增位置头保持中性，结果主要等价于固定 proposal 上的选择蒸馏。

## 8. 验证与 checkpoint 选择

每轮验证都执行真实的一次性部署：

1. 网络前向一次；
2. 读取最终 KeepMask 和 `deployment_internal_knots`；
3. 标准开放 B 样条 refit 一次；
4. 计算每条曲线 RMS。

选优不使用 surrogate fit：

- 未达到 `--deployment-pass-rate-target` 时，优先提高真实阈值满足率，再比较均值/P95 RMS 和复杂度。
- 达到目标满足率后，优先减少平均内部节点数，再以满足率、真实 RMS、最终节点匹配和 mask/count 指标作 tie-break。

这是验证集上的约束式排序，不是逐样本保证，也不是全局最优证明。

## 9. 首次完整训练

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

输出通常包括：

```text
candidate_pruning_one_shot_v11_proposal.pt
candidate_pruning_one_shot_v11_distill.pt
candidate_pruning_one_shot_v11_calibrated.pt
candidate_pruning_one_shot_v11.pt
candidate_pruning_one_shot_v11_last.pt
candidate_pruning_one_shot_v11_teacher/train.pt
candidate_pruning_one_shot_v11_teacher/val.pt
```

最终 `.pt` 是在蒸馏和校准候选 checkpoint 中按相同验证排序选出的最佳结果，不一定是最后一轮。

## 10. 用已有 proposal 开始 v11

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 138 `
  --candidate-pretrain-epochs 8 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v10_proposal.pt `
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
  --lambda-one-shot-surrogate-fit 0 `
  --lambda-one-shot-surrogate-threshold 0 `
  --lambda-joint-fit 0.05 `
  --lambda-joint-threshold-violation 0.5 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v11_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v11.pt
```

v10 checkpoint 的参数布局可以兼容加载。推荐用 5–10 轮 proposal 预训练适配 v11 的 Gaussian attention；上例用 8 轮，并将总轮数设为 138，从而保留 120 轮蒸馏和 10 轮联合校准。

`--candidate-pretrain-epochs 0` 只复用固定 proposal，不会适配新增局部 attention。此时脚本会：

1. 沿用 checkpoint `model_config` 中记录的 proposal attention 带宽，而不是静默套用命令行的新带宽；
2. 校验点维度、隐藏维度、候选数、最小间隔、参数化、代理正则和 proposal 特征布局等非权重语义；
3. 发现语义不一致时停止并要求使用匹配参数或启用 proposal 适配。

缺少 `model_config` 的历史 checkpoint 只能回退为全局 attention，并提示无法完成其余语义校验。当前训练脚本在选出新的 `*_proposal.pt` 后会补写 `model_config`、数据配置和阶段元数据，供后续固定 proposal 复用与指纹校验。

v11 的 proposal 行为、监督和指纹契约已经变化。因此：

- 不要复用 `outputs/...v10_teacher`；
- v11 首次运行使用新缓存目录；
- 首次不要加 `--reuse-teacher-cache`；
- 只有完全相同的 v11 proposal、数据和教师配置才能复用 v11 缓存。

## 11. 小规模流程验证

以下命令只验证流程和文件完整性，不用于报告性能：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 4 `
  --candidate-pretrain-epochs 1 `
  --selector-calibration-epochs 1 `
  --train-size 32 `
  --val-size 16 `
  --batch-size 8 `
  --min-control-points 8 `
  --max-control-points 12 `
  --candidate-knots 8 `
  --num-points 64 `
  --teacher-batch-size 2 `
  --teacher-cache-dir outputs/v11_smoke_teacher `
  --no-resample-train-each-epoch `
  --output outputs/v11_smoke.pt
```

小样本结果不应与正式实验比较。
