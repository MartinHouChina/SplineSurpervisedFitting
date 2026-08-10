# 数据、训练与评估

## 数据集构成

`SyntheticCubicBSplineDataset` 按索引确定性生成开放三次 B 样条。样本随机种子为：

```text
sample_seed = dataset_seed + sample_id
```

默认划分：

| 划分 | 数量 | seed |
|---|---:|---:|
| train | 512 | 42 |
| validation | 128 | 10000 |
| evaluation | 由 `--num-samples` 指定，默认 128 | 由 `--seed` 指定，默认 10000 |

### 生成步骤

每条曲线按以下顺序生成：

1. 从 5–10 中均匀抽取控制点数 \(N\)。
2. 用方向相关的随机步进生成平滑控制多边形，并施加轻微各向异性缩放。
3. 构造定义域为 \([0,1]\) 的开放夹持三次节点向量。
4. 内部节点由均匀跨度和随机正跨度混合得到；每个跨度至少为 0.02。
5. 生成 64 个严格有序、可非均匀的真实参数。
6. 用 Cox–de Boor 基函数计算曲线采样点。
7. 加入标准差为 0.001 的高斯噪声。
8. 对采样点中心化，并按最大点半径缩放；真实控制点使用相同变换。

三次 B 样条的真实内部节点数为：

\[
K^*=N-4.
\]

默认控制点数为 5–10，因此每条曲线有 1–6 个真实内部节点。

### 样本字段

设采样点数为 \(M\)，维度为 \(d\)，最大控制点数为 \(N_{max}\)。

| 字段 | 形状 | 用途 |
|---|---|---|
| `points` | `[M, d]` | 模型输入 |
| `chord_params` | `[M]` | 可选弦长先验，当前默认权重为 0 |
| `true_params` | `[M]` | ParameterHead 监督 |
| `true_internal_knots` | `[N_max-4]` | 节点位置和 existence 监督 |
| `true_internal_knot_mask` | `[N_max-4]` | 标记有效内部节点 |
| `true_control_points` | `[N_max, d]` | 评估和可视化 |
| `true_control_mask` | `[N_max]` | 标记有效控制点 |
| `true_knot_vector` | `[N_max+4]` | 完整真值节点向量 |
| `true_knot_mask` | `[N_max+4]` | 标记有效节点向量元素 |
| `center` | `[d]` | 坐标反归一化 |
| `scale` | 标量 | 坐标反归一化 |
| `num_control_points` | 标量 | 真值结构统计 |

补齐位置必须与对应 mask 一起使用。

## 训练流程

每个 batch 执行：

1. 编码有序采样点。
2. 预测参数 \(t\)。
3. 用独立 query 预测并排序 \(K\) 个候选内部节点。
4. 将真实内部节点与候选节点做有序最小代价匹配。
5. 生成每个 query 的 existence 标签和位置目标。
6. ActivityHead 输出 Hard-Concrete 非零概率和训练门。
7. 用停梯度的门构造截断幂设计矩阵。
8. 可微求解线性系数并重建采样点。
9. 计算总损失并更新网络。

默认模型有 8 个候选 query；数据最多有 6 个真实内部节点。`--max-knots` 不得小于数据集最大真实节点数。

### 默认损失

| 配置 | 默认值 |
|---|---:|
| `lambda_true_params` | 0.01 |
| `lambda_existence` | 0.005 |
| `lambda_knot_position` | 0.01 |
| `lambda_count` | 0.002 |
| `lambda_l0` | 0 |
| `lambda_parameter_prior` | 0 |
| `gap` | 0 |

Hard-Concrete 温度默认从 \(2/3\) 线性退火到 0.5。当前默认不做全开 gate warm-up。

## checkpoint 选择

验证阶段基于匹配标签计算 existence Precision、Recall 和 F1。保存规则为：

1. existence F1 更高；
2. F1 相同时，总损失更低。

checkpoint 保存：

- `model_state_dict`；
- `model_config`、`dataset_config`、`loss_config`、`training_config`；
- epoch、验证指标和训练历史；
- Hard-Concrete 温度与部署阈值；
- objective 版本和 checkpoint 选择指标。

## 评估与部署

`evaluate_checkpoint.py` 在固定 seed 的合成数据上报告：

- 网络前向拟合误差；
- existence Precision/Recall/F1；
- 期望节点数和硬保留节点数；
- 节点匹配 Precision/Recall/F1 与匹配 MAE；
- 参数 RMSE；
- 标准 B 样条重拟合误差；
- 不同固定阈值下的节点数分布。

最终部署结果不是训练截断幂代理，而是：

```text
固定阈值二值化
  → 删除节点
  → 构造开放三次节点向量
  → 增广最小二乘求控制点
  → 标准 B 样条
```

## 常用命令

```bash
python scripts/train.py --epochs 100 --output outputs/independent_queries_v2.pt
```

```bash
python scripts/evaluate_checkpoint.py --checkpoint outputs/independent_queries_v2.pt --json-output outputs/independent_queries_v2_evaluation.json
```

```bash
python scripts/visualize_result.py --checkpoint outputs/independent_queries_v2.pt --output outputs/independent_queries_v2.png
```
