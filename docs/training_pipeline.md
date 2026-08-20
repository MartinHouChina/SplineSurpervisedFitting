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
| train | 10000/epoch | 42 加 epoch 偏移 | 参数更新；默认每个 epoch 重新生成 |
| validation | 1000 | 10000 | 固定不变，用于 checkpoint 选择 |
| test/evaluation | 命令指定，默认 128 | 20000 | 与训练、验证独立的最终报告 |

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

样本在当前 epoch 内缓存。默认训练集切换 epoch 时清空缓存并使用新的确定性 seed 重新生成；验证集不切换 epoch，因此始终固定。

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

训练 forward 根据当前 teacher-forcing 比例选择数量来源：

```python
output = model(
    points,
    true_internal_knot_count=true_count,
    teacher_forcing_ratio=current_ratio,
)
```

完整步骤：

1. GeometryEncoder 产生局部和全局特征；
2. ParameterHead 预测严格递增 `params`；
3. InteractiveStructureHead 用 cross-attention 和 self-attention 预测数量分布；
4. DynamicKnotDecoder 前期接收真实 \(K^*\)，后期逐渐混入网络预测数量；
5. 当所选 \(K>0\) 时，动态解码器只运行 \(K+1\) 个 interval query 并生成 \(K\) 个有序节点；\(K=0\) 时跳过位置解码；
6. 构造截断幂设计矩阵；
7. forward 内可微求解线性拟合系数；
8. 计算联合损失；
9. 反向传播、梯度裁剪并执行 AdamW 更新。

teacher count 只决定动态位置解码器运行多少个 interval query，不会替代结构数量监督。InteractiveStructureHead 仍然产生数量概率并计算 continuation loss。默认前 10 个 epoch 的 teacher-forcing 比例为 1，之后线性降低到 0.5；未使用 teacher 的样本按网络预测数量解码，从而减小训练与验证之间的 exposure gap。

## 4. 损失函数

默认总目标：

\[
L=L_{fit}+0.05L_t+0.005L_{structure}
+0.002L_{over}+0.05L_{knot}.
\]

| 损失 | 具体作用 |
|---|---|
| `fit_loss` | 截断幂代理重建点与输入点的均方欧氏距离 |
| `true_parameter_loss` | 预测参数与真实采样参数的 MSE |
| `count_loss` | 对结构 survival probability \(P(K\ge r)\) 的 BCE |
| `over_count_loss` | 惩罚期望节点数高于真实数量 |
| `knot_position_loss` | 所选真实数量表示与 canonical 有序节点的 Smooth-L1 |

位置损失不需要 Hungarian matching：预测节点和 canonical 节点都已经严格有序、数量相同，可直接逐位置比较。

## 5. 为什么使用渐进 teacher forcing

如果训练一开始就使用错误的预测数量：

- 节点位置张量与真值数量不同；
- 位置损失难以定义；
- 结构数量头的早期错误会让动态解码器收到错误的 query 数量。

因此训练前期使用 teacher-conditioned 解码，使数量学习和条件位置学习分别获得稳定监督；后期逐渐使用预测数量，让动态解码器适应部署时可能出现的数量误差。

这不代表验证结果使用了真值。验证阶段调用 `model(points)`，完全使用网络自己的数量预测。

## 6. 验证流程

每个验证 batch 执行：

```python
output = model(points)
```

此时：

1. InteractiveStructureHead 预测 `predicted_knot_count`；
2. DynamicKnotDecoder 在预测 `K>0` 时只运行对应的 `K+1` 个 interval query；预测 `K=0` 时跳过；
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
  --output outputs/interactive_dynamic_v6.pt
```

常用参数：

```text
--train-size                   训练样本数，默认 10000
--val-size                     验证样本数，默认 1000
--batch-size                   默认 32
--hidden-dim                   默认 128
--max-knots                    默认 6
--canonical-knot-tolerance     默认 0.005
--structure-attention-heads    默认 4
--train-seed                   默认 42
--val-seed                     默认 10000
--resample-train-each-epoch    默认启用；每个 epoch 生成新训练曲线
--teacher-forcing-final        默认 0.5
--teacher-forcing-warmup-epochs 默认 10
--weight-decay                 默认 1e-4
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
  --output outputs/v6_debug.pt
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
