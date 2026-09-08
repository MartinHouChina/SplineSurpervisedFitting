# 合成数据的最简性修正

## 1. 旧数据的实际语义

旧生成器先随机选择控制顶点数，再生成平滑随机控制多边形、随机内部节点和带噪采样点。这里的
`source_internal_knot_count` 只是生成时使用的节点数，不等于阈值下的最少节点数。

训练脚本默认令 `canonical_knot_tolerance = fit_tolerance = 0.005`，然后只在 source 节点的
子集上执行一条贪心删除路径。`true_internal_knots` 因而是 greedy canonical 标签，而不是
source 标签。它解决了部分冗余，但仍有四个边界：

1. 平滑随机游走在归一化后可能用很少的节点近似，因此 source K 与真实几何复杂度脱节；
2. canonical 标签直接由带噪、非均匀采样点生成，临界样本会随噪声改变节点数或删除路径；
3. 单路径贪心不保证在 source 的所有子集中得到全局最小基数；
4. 只删除原节点，不搜索任意连续位置上的更优节点。

以默认 K=4--20、192 点、噪声 0.001、阈值 0.005 对 seed=42 的 100 条样本做审计：source
平均 K=11.66，greedy canonical 平均 K=7.29，96% 的 source 被继续删除，平均删除 4.37
个节点。这证明随机 source K 不能直接作为最简标签。对相同潜在曲线分别用干净点和带噪点
生成 canonical 标签时，14% 的节点集合发生变化，其中 8% 的节点数发生变化。该数字只是
一次小规模诊断，不应当当作总体置信区间。

## 2. 新的 opt-in 数据模式

`SyntheticCubicBSplineDataset` 新增 `certified_minimal_source=True`。开启后流程为：

```text
先固定目标 K
  -> 生成 complexity-aligned 干净曲线
  -> 用干净点和 float64 无正则 refit 检查全部 K 个单节点删除
  -> 不满足带 margin 的最简性条件则拒绝并重采样，但不重抽 K
  -> 接受后固定 source knots 为监督标签
  -> 最后才向网络输入点加入噪声
```

复杂度对齐控制多边形由单调主方向和交替横向细节组成，并施加随机正交变换。它避免高 K
曲线在归一化后退化成几乎相同的低频随机游走。构造方式只提高通过证书的概率；是否接受仍
完全由数值证书决定。

设完整 source 节点集为 U，阈值为 epsilon，margin 为 m。接受条件是：

```text
RMS(U) <= epsilon
min_j RMS(U without u_j) > epsilon * (1 + m)
```

证书使用 endpoint-constrained、smoothness=0、ridge=0 的 float64 标准 B 样条最小二乘。
在 source 节点的子集域中，任一更小子集都包含于某个单节点删除后的样条空间；无正则最小
二乘误差随空间缩小不会下降。因此全部单节点删除均失败时，任何 source proper subset 都
不可能通过阈值。这给出该离散子集域内的最少基数证书。

## 3. 证书边界

该模式保证的是：

- 在指定的干净审计采样和归一化尺度上成立；
- 在 source knot vector 的所有子集中成立；
- 标签不受随后加入的观测噪声影响；
- K 在拒绝采样期间保持不变，因此不会把数据分布偏向较小 K。

它不保证：

- 任意连续节点位置重定位后的全局最优；
- 对所有可能参数化均最优；
- 对真实世界连续曲线的解析全局最优。

论文中应称为 `source-subset threshold-minimal certificate`，不能称为 continuous global
minimum。连续位置最优化仍应由 delete-then-relax teacher、verified repair 或独立的全局
优化基线评估。

## 4. 参数和输出字段

数据集构造参数：

| 参数 | 建议值 | 作用 |
|---|---:|---|
| `certified_minimal_source` | `True` | 开启最简 source 模式 |
| `minimality_margin` | `0.2` | 要求单删除误差至少超过阈值 20% |
| `minimality_max_attempts` | `16` | 固定 K 后的最大重采样次数 |
| `minimality_audit_points` | `512` | 用均匀干净点生成证书；`0` 表示沿用训练采样参数 |
| `oscillation_amplitude` | `0.3` | complexity-aligned 横向细节强度 |

新增样本字段：

| 字段 | 含义 |
|---|---|
| `clean_points` | 与输入同一归一化尺度的无噪点 |
| `source_minimality_certified` | 是否通过证书 |
| `source_full_fit_rms` | 完整 source 节点的干净 refit RMS |
| `source_min_single_deletion_rms` | 所有单节点删除 RMS 的最小值 |
| `source_minimality_required_rms` | `epsilon * (1 + margin)` |
| `source_generation_attempts` | 固定 K 后实际生成次数 |
| `source_observation_fit_rms` | source 节点对带噪网络输入的 refit RMS |

训练 CLI 已完成接线，建议使用：

```powershell
python scripts/train_candidate_pruning.py `
  --certified-minimal-source `
  --minimality-margin 0.2 `
  --minimality-max-attempts 16 `
  --minimality-audit-points 512 `
  --oscillation-amplitude 0.3 `
  --fit-tolerance 0.005
```

## 5. 迁移规则

该功能默认关闭，所以旧 checkpoint 和旧 seed 的样本流保持原语义。开启后属于一个新的数据
分布，必须：

1. 将全部新参数写入 checkpoint 的 `dataset_config`；
2. 使用新的 train/validation 数据集 fingerprint；
3. 重建 proposal 和离线 teacher cache，不能复用 v12/v13 旧 cache；
4. 验证集固定 seed 且不逐 epoch 重采样；
5. 同时报告 clean certificate 指标和 noisy observation 部署指标。

建议混合训练而非只训练高频合成曲线：certified-minimal 合成数据负责明确结构监督，原平滑
随机曲线负责形状覆盖，真实等距线/轮廓线负责域外验证。三类数据应分别报告结果，不能只汇总
一个平均数。

## 6. 验证要求

每个 certified 样本至少断言：

```text
source_minimality_certified == True
true K == source K
source_full_fit_rms <= epsilon
source_min_single_deletion_rms > source_minimality_required_rms
```

还应对同一个 seed 分别设置 `noise_std=0` 和非零噪声，验证 `true_internal_knots` 与
`clean_points` 完全一致，而网络输入 `points` 不同。现有测试已覆盖冗余多项式反例、证书
条件、K 不变以及噪声与标签解耦。
