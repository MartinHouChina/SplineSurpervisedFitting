# 数据生成与训练流程

## 1. 数据生成

每条源曲线按以下步骤生成：

1. 从 5–10 中随机选择源控制点数量；
2. 生成平滑随机控制多边形；
3. 生成非均匀开放三次 B 样条节点向量；
4. 生成非均匀采样参数 `true_params`；
5. 计算有序采样点并加入坐标噪声；
6. 对采样点做中心化和尺度归一化。

默认划分：

| 划分 | 数量 | seed | 用途 |
|---|---:|---:|---|
| train | 10000 | 42 | 参数更新 |
| validation | 1000 | 10000 | checkpoint 选择 |
| evaluation | 命令指定，默认 128 | 10000 | 独立报告 |

## 2. Canonical 标签生成

随机生成器使用的源节点数不是可靠监督目标，因为同一条 B 样条曲线可以通过节点插入获得更冗余的等价表示。

因此数据集对源内部节点执行贪心删除：

```text
从完整源节点集合开始
  → 分别尝试删除每一个剩余节点
  → 对每种删除结果重新最小二乘求控制点
  → 找到重拟合 RMS 最小的删除方案
  → 若 RMS ≤ canonical_knot_tolerance，则接受删除
  → 否则停止
```

默认容差：

```text
canonical_knot_tolerance = 0.005
```

最终剩余节点构成：

```text
true_internal_knots
true_internal_knot_mask
true_knot_vector
true_control_points
```

同时保留以下诊断字段：

```text
source_internal_knot_count
source_num_control_points
canonical_fit_rms
```

注意：这里删除的是数据标签中的源节点，不是网络预测节点。

canonical 样本第一次生成后缓存在 Dataset 实例中，后续 epoch 直接复用。

## 3. 单个训练 batch

Trainer 从 batch 中读取：

```text
points
chord_params
true_params
true_internal_knots
true_internal_knot_mask
```

真实节点数量由 mask 计算：

\[
K^*=\sum_j\mathbf 1[\text{true knot slot }j\text{ valid}].
\]

训练 forward 使用：

```python
output = model(points, true_internal_knot_count=true_count)
```

完整步骤：

1. GeometryEncoder 产生局部和全局特征；
2. ParameterHead 预测严格递增 `params`；
3. CountHead 预测序数数量分布；
4. KnotHead 使用真实 \(K^*\) 选择节点表示；
5. 共享解码器直接生成 \(K^*\) 个有序节点；
6. 构造截断幂设计矩阵；
7. forward 内可微求解线性拟合系数；
8. 计算联合损失；
9. 反向传播、梯度裁剪并执行 AdamW 更新。

teacher count 只决定本次使用哪个节点表示，不会替代 CountHead 的监督。CountHead 仍然产生概率并计算数量损失。

## 4. 损失函数

默认总目标：

\[
L=L_{fit}+0.05L_t+0.005L_{ordinal}
+0.002L_{over}+0.05L_{knot}.
\]

| 损失 | 具体作用 |
|---|---|
| `fit_loss` | 截断幂代理重建点与输入点的均方欧氏距离 |
| `true_parameter_loss` | 预测参数与真实采样参数的 MSE |
| `count_loss` | 对 \(P(K\ge r)\) 的序数 BCE |
| `over_count_loss` | 惩罚期望节点数高于真实数量 |
| `knot_position_loss` | 所选真实数量表示与 canonical 有序节点的 Smooth-L1 |

位置损失不需要 Hungarian matching：预测节点和 canonical 节点都已经严格有序、数量相同，可直接逐位置比较。

## 5. 为什么训练使用真实数量

如果训练一开始就使用错误的预测数量：

- 节点位置张量与真值数量不同；
- 位置损失难以定义；
- CountHead 的早期错误会阻止正确数量表示学习。

因此训练使用 teacher-conditioned 解码，使数量学习和条件位置学习分别获得稳定监督。

这不代表验证结果使用了真值。验证阶段调用 `model(points)`，完全使用网络自己的数量预测。

## 6. 验证流程

每个验证 batch 执行：

```python
output = model(points)
```

此时：

1. CountHead 预测 `predicted_knot_count`；
2. KnotHead 使用预测数量提取节点；
3. 计算预测数量下的拟合和结构指标；
4. 在共享参数域下，用容差 0.05 匹配预测节点与 canonical 节点。

checkpoint 排序顺序：

1. 节点匹配 F1 更高；
2. F1 相同时 precision 更高；
3. 再相同时匹配节点 MAE 更低；
4. 最后比较总验证损失。

因此保存的 checkpoint 面向结构准确性，而不是只追求低拟合误差。

## 7. 启动训练

默认训练：

```powershell
python scripts/train.py `
  --epochs 100 `
  --output outputs/count_conditioned_v5.pt
```

常用参数：

```text
--train-size                   训练样本数，默认 10000
--val-size                     验证样本数，默认 1000
--batch-size                   默认 32
--hidden-dim                   默认 128
--max-knots                    默认 6
--canonical-knot-tolerance     默认 0.005
--lambda-count                 默认 0.005
--lambda-over-count            默认 0.002
--lambda-knot-position         默认 0.05
--lambda-true-params           默认 0.05
```

小规模功能检查：

```powershell
python scripts/train.py `
  --epochs 2 `
  --train-size 128 `
  --val-size 64 `
  --hidden-dim 32 `
  --output outputs/v5_debug.pt
```

小规模命令仅用于检查代码链路，不代表正式性能。

## 8. checkpoint 内容

checkpoint 保存：

- `model_state_dict`；
- `model_config`；
- `dataset_config`；
- `loss_config`；
- `training_config`；
- 最佳 epoch 和结构指标；
- 完整训练历史；
- objective version。

评估脚本优先使用 checkpoint 内的数据配置，保证标签容差和训练设置一致。
