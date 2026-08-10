# 部署与结果解释

部署的目标是把网络输出转换成可以正常评估、绘制和导出的标准开放三次 B 样条。

## 1. 部署输入

当前脚本直接从 synthetic evaluation dataset 读取归一化有序点：

```text
points: [B, M, D]
```

对于实际数据，应保持以下条件：

- 点沿曲线方向有序；
- 使用与训练一致的中心化和尺度归一化；
- 记录 `center` 和 `scale`，以便将控制点变换回原坐标系；
- 点数和噪声范围最好接近训练数据，或者先进行对应的数据增强训练。

## 2. 网络 forward

部署不提供真实节点数量：

```python
with torch.no_grad():
    output = model(points)
```

网络输出：

- 预测参数 `params`；
- 数量分布 `count_probabilities`；
- CountHead 原始数量 `predicted_knot_count`；
- 所有完整数量表示 `branch_internal_knots`；
- argmax 数量对应的 `internal_knots` 和 `knot_mask`。

节点头不会在预测出 \(K\) 后临时创建一个新网络层。共享解码器在同一次 forward 中计算 `K=0...Kmax` 的全部完整表示，并填充为统一宽度；`predicted_knot_count` 或后续 BIC 只负责从这些表示中选择一项。选择后通过 `knot_mask` 提取前 \(K\) 个有效值，得到真正长度为 \(K\) 的内部节点向量。

部署有两种数量选择方式。

## 3. 方式一：network

```text
K = argmax count_probabilities
```

然后直接使用该数量对应的节点表示。

优点：

- 只需网络 forward 和一次标准 B 样条重拟合；
- 速度快；
- 可直接测量 CountHead 本身的性能。

缺点：

- 如果数量预测错一类，会整体切换到另一个节点表示；
- 不利用实际重拟合误差校正数量。

命令：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/count_conditioned_v5.pt `
  --count-selection network
```

## 4. 方式二：BIC，v5 默认

网络已经生成所有数量分支：

```text
K = 0, 1, ..., Kmax
```

部署器对每个完整分支执行：

1. 取该分支的前 \(K\) 个内部节点；
2. 构造开放三次 B 样条节点向量；
3. 使用预测 `params` 和输入点重新最小二乘求控制点；
4. 计算坐标 SSE；
5. 计算 BIC 和 CountHead 先验联合分数。

分数为：

\[
S_K=N\log(\operatorname{SSE}_K/N)
+d_K\log N-2\eta\log p_K.
\]

- 第一项奖励低拟合误差；
- 第二项惩罚更多节点和控制点；
- 第三项加入 CountHead 对该数量的先验概率；
- `count_prior_weight` 对应 \(\eta\)，默认 1.0。

最终选择：

\[
\widehat K=\arg\min_K S_K.
\]

这里比较的是多个完整 B 样条模型，不是先预测固定节点再逐个删除。

命令：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/count_conditioned_v5.pt `
  --count-selection bic `
  --count-prior-weight 1.0 `
  --json-output outputs/count_conditioned_v5_evaluation.json
```

`--count-selection auto` 是默认值：v5 使用 BIC，v4 使用 network，历史门控模型使用原有阈值路径。

## 5. 标准 B 样条重拟合

确定最终数量和内部节点后，构造：

\[
U=[0,0,0,0,u_1,\ldots,u_K,1,1,1,1].
\]

根据预测参数计算标准 B 样条基矩阵 \(B\)，重新求解控制点：

\[
P^*=\arg\min_P
\|BP-Q\|_F^2+
\lambda_s\|D_2P\|_F^2+
\lambda_r\|P\|_F^2.
\]

这一阶段输出真正的：

- 标准开放节点向量；
- B 样条控制点；
- 控制多边形；
- 重建采样点；
- 拟合 RMS 和坐标 RMSE。

它与网络 forward 内的截断幂训练代理不同。

## 6. 评估脚本具体报告

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/count_conditioned_v5.pt `
  --num-samples 128 `
  --batch-size 32 `
  --json-output outputs/count_conditioned_v5_evaluation.json
```

### Network forward model

这里报告网络原始 forward 的联合目标和截断幂代理拟合误差，不是最终标准 B 样条误差。

### Supervised knot count

- `network count accuracy`：CountHead argmax 与 canonical 数量完全相等的比例；
- `network count MAE`：原始数量绝对误差；
- `expected count mean`：数量概率分布的期望；
- `network count histogram`：CountHead 原始数量分布；
- `deployment count accuracy/MAE`：执行 network/BIC 选择后的实际数量指标；
- `deployment count histogram`：最终部署数量分布。

不要把 network count accuracy 和节点 precision 当成同一个指标。

### Standard B-spline deployment

- 最终内部节点数量均值、最小值、最大值；
- 零节点比例和最大节点比例；
- 标准 B 样条重拟合 loss/RMS；
- 平均控制点数量。

### Ground-truth diagnostics

节点匹配使用共享参数域和指定容差，默认 0.05：

- precision：预测节点中匹配到 canonical 节点的比例；
- recall：canonical 节点中被匹配到的比例；
- F1：precision 和 recall 的调和平均；
- matched MAE：成功匹配节点的位置误差。

低拟合 RMS 不代表节点结构正确，因此必须同时报告结构指标。

## 7. 可视化

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/count_conditioned_v5.pt `
  --sample-index 0 `
  --count-selection bic `
  --output outputs/count_conditioned_v5_sample_000.png
```

左图包含：

- 输入采样点；
- 网络截断幂代理曲线；
- 最终标准 B 样条曲线；
- 重拟合控制多边形。

右图显示节点数量概率：

- CountHead 原始 argmax；
- 实际部署选择的数量；
- 每个数量的概率。

终端同时打印预测节点、canonical 真值节点、匹配指标和两种拟合误差。

## 8. 实际数据导出时的坐标恢复

训练数据使用：

\[
q_{norm}=\frac{q-center}{scale}.
\]

部署得到归一化控制点 \(P_{norm}\) 后，应恢复：

\[
P=P_{norm}\cdot scale+center.
\]

节点参数位于 \([0,1]\)，不需要做坐标尺度恢复。

## 9. 推荐的检查顺序

1. 先用 `--count-selection network` 检查 CountHead 泛化；
2. 再用 `--count-selection bic` 检查最终部署性能；
3. 比较两份 JSON 中的数量 MAE、节点 F1 和重拟合 RMS；
4. 可视化数量预测错误和节点位置误差最大的样本；
5. 最后再调整 `count_prior_weight`，不要根据单条曲线调参。
