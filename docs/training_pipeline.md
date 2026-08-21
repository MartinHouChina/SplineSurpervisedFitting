# 数据生成与训练流程

> 第 1–8 节记录当前已实现的 v6。第 9 节概述下一阶段候选生成与 Boehm 消冗训练协议；该部分尚未进入代码。

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

teacher count 只决定动态位置解码器运行多少个 interval query，不会替代结构数量监督。InteractiveStructureHead 仍然产生数量概率并计算 categorical cross-entropy。默认前 5 个 epoch 的 teacher-forcing 比例为 1，之后线性降低到 0.5；未使用 teacher 的样本按网络预测数量解码，从而减小训练与验证之间的 exposure gap。

## 4. 损失函数

默认总目标：

\[
L=L_{fit}+0.05L_t+0.005L_{structure}
+0.05L_{knot}.
\]

| 损失 | 具体作用 |
|---|---|
| `fit_loss` | 截断幂代理重建点与输入点的均方欧氏距离 |
| `true_parameter_loss` | 预测参数与真实采样参数的 MSE |
| `count_loss` | 合法数量类别上的 categorical cross-entropy |
| `over_count_loss` | 历史兼容项；新训练默认权重为 0 |
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

checkpoint 只在 teacher-forcing 已开始退火后参与排序，默认顺序：

1. 数量 MAE 更低；
2. 数量准确率更高；
3. 节点匹配 F1 和 precision 更高；
4. 匹配节点 MAE 更低；
5. 最后比较总验证损失。

这避免了严格节点容差偶然选择仍处于 100% teacher forcing 的早期权重。验证和部署的数量决策均为合法范围内的后验中位数；argmax 众数只作为诊断输出。

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
--structure-count-mode         默认 categorical；hazard 仅用于历史实验
--train-seed                   默认 42
--val-seed                     默认 10000
--resample-train-each-epoch    默认启用；每个 epoch 生成新训练曲线
--teacher-forcing-final        默认 0.5
--teacher-forcing-warmup-epochs 默认 5
--checkpoint-selection-start-epoch 默认在 teacher-forcing 开始退火后
--weight-decay                 默认 1e-4
--lambda-count                 默认 0.005
--lambda-over-count            默认 0
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

### 4–20 个源内部节点

三次 B 样条的内部节点数等于控制点数减 4，因此 4–20 个源节点对应 8–24 个控制点。在线 canonical 贪心删除在该范围非常慢，并且删除后的数量不保证仍为 4–20。若实验目标是先训练精确的“源节点 4–20”范围，应关闭 canonical 删除并使用快速路径：

```powershell
python scripts/train.py `
  --epochs 150 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --log-every-batches 20 `
  --min-control-points 8 `
  --max-control-points 24 `
  --max-knots 20 `
  --num-points 192 `
  --canonical-knot-tolerance 0 `
  --structure-count-mode categorical `
  --resample-train-each-epoch `
  --teacher-forcing-warmup-epochs 10 `
  --teacher-forcing-final 0.25 `
  --lambda-over-count 0 `
  --knot-match-tolerance 0.02 `
  --output outputs/knot_4_20_categorical.pt
```

`canonical-knot-tolerance=0` 会直接使用生成器的源节点和归一化源控制点，不执行逐节点最小二乘删除，因此应保留每 epoch 重采样来减少对固定曲线的记忆。该模式速度快、数量范围准确，并自动把合法预测范围设为 4–20。

但“源节点数量”并不是由点云唯一决定的：Boehm 插入可增加节点而不改变曲线。categorical 修复解决的是概率分解与非法边界塌缩，不保证任意源表示都可被精确反演。若目标是几何上可辨识的最简结构，应使用 canonical 标签或第 9 节的冗余输入消冗任务。

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

## 9. 规划中的候选生成与 Boehm 消冗训练

下一阶段不再要求一个节点头同时决定数量和全部连续位置，而是分为共享编码器下的两个头：

```text
CandidateKnotHead：由真实最简节点监督，优化候选覆盖率
InteractivePruningHead：由 Boehm 冗余和删除误差监督，优化保留精度
```

### 9.1 标签与输入分离

```text
无噪声最简样条 → 真实节点标签 U*
无噪声最简样条 + Boehm插入 → 消冗监督
独立加入噪声/非均匀采样 → 网络点云输入 Q
```

随机 Boehm 插入位置不能作为 CandidateKnotHead 的回归标签，因为插入不改变点云且位置不由几何决定。候选头只学习覆盖真实最简节点。

### 9.2 三阶段训练

1. **候选预训练**：热力图、offset、单向 coverage 和轻量 repulsion；以 `candidate recall@0.02` 为 checkpoint 主指标。
2. **消冗预训练**：输入精确及扰动 Boehm 冗余表示，监督 keep/remove、删除误差和位置精修。
3. **联合微调**：逐步从 Boehm 理想候选切换到 CandidateKnotHead 真实输出，同时优化最终节点 F1 和标准 B 样条拟合。

### 9.3 4–20 节点建议

```text
真实节点范围：4–20
候选预算 Kc：24或28
点云采样数：128–192，推荐192
主匹配容差：0.01和0.02
```

最终数量来自 keep probability 的保留数量，不使用独立 CountHead。详细损失、token 特征和混合比例见 [proposal_pruning_framework.md](proposal_pruning_framework.md)。
