# v6 部署与结果解释

v6 部署只有一次节点数量决策：InteractiveStructureHead 的数量分布 argmax。没有 BIC、阈值剪枝或全部数量分支枚举。

## 1. 部署输入

```text
points: [B,M,D]
```

实际数据需要：

- 点沿曲线方向有序；
- 使用与训练一致的中心化和尺度归一化；
- 保存 `center` 和 `scale`，用于恢复最终控制点坐标。

## 2. 网络推理

部署不提供真实数量：

```python
with torch.no_grad():
    output = model(points)
```

内部顺序：

```text
points
  → GeometryEncoder
  → ParameterHead 得到 params
  → InteractiveStructureHead 得到 P(K)
  → predicted_knot_count = argmax P(K)
  → predicted K>0 时 DynamicKnotDecoder 只运行 K+1 个 query；K=0 时跳过
  → 得到 predicted K 个有序节点
```

关键输出：

```text
params
count_probabilities
predicted_knot_count
structure_survival_probabilities
internal_knots
knot_mask
decoded_interval_query_count
```

其中：

```text
decoded_interval_query_count = (
    predicted_knot_count + 1 if predicted_knot_count > 0 else 0
)
```

它可以直接验证动态解码器没有计算全部数量分支。

## 3. Batch 中的变长节点

假设预测数量为：

```text
[2,4,2]
```

解码器实际执行：

```text
K=2 组：两条样本，各运行 3 个 interval query
K=4 组：一条样本，运行 5 个 interval query
```

输出为定宽张量：

```text
internal_knots = [
  [u1,u2,0, 0],
  [u1,u2,u3,u4],
  [u1,u2,0, 0],
]

knot_mask = [
  [1,1,0,0],
  [1,1,1,1],
  [1,1,0,0],
]
```

标准部署提取：

```python
valid_knots = internal_knots[knot_mask]
```

定宽填充只是 batch 存储格式，不产生额外分支计算。

## 4. 标准 B 样条重拟合

对每条曲线只执行一次重拟合。内部节点为：

\[
U_{int}=[u_1,\ldots,u_{\widehat K}].
\]

构造开放三次节点向量：

\[
U=[0,0,0,0,U_{int},1,1,1,1].
\]

根据预测参数建立标准 B 样条基矩阵，并求控制点：

\[
P^*=\arg\min_P
\|BP-Q\|_F^2+
\lambda_s\|D_2P\|_F^2+
\lambda_r\|P\|_F^2.
\]

最终得到：

- 标准开放节点向量；
- \(\widehat K+4\) 个三次 B 样条控制点；
- 控制多边形；
- 标准 B 样条重建曲线；
- 拟合 RMS。

网络 forward 内的截断幂曲线只是训练代理，标准 B 样条重拟合结果才是部署输出。

## 5. 评估命令

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/interactive_dynamic_v6.pt `
  --num-samples 128 `
  --seed 20000 `
  --batch-size 32 `
  --json-output outputs/interactive_dynamic_v6_evaluation.json
```

对于 v6，`--count-selection auto` 和 `--count-selection network` 等价。显式使用 `--count-selection bic` 会报错，因为 v6 不生成全部数量分支。

这里默认 `seed=20000` 是独立测试集。`seed=10000` 对应训练过程中使用的验证集，只适合复核验证结果，不能作为最终测试成绩。

## 6. 指标解释

### Network forward model

这里报告截断幂训练代理的联合目标和拟合误差，不是最终标准 B 样条误差。

### Supervised knot count

- `network count accuracy`：预测数量与 canonical 数量完全相等的比例；
- `network count MAE`：节点数量绝对误差；
- `expected count mean`：数量概率分布期望；
- `network count histogram`：预测数量分布。

v6 只有一次数量决策，因此 deployment count 与 network count 相同。

### Standard B-spline deployment

- 最终内部节点数量；
- 标准 B 样条重拟合 loss/RMS；
- 平均控制点数量；
- 零节点和最大节点比例。

### Ground-truth diagnostics

在共享参数域和指定容差下匹配预测节点与 canonical 节点：

- precision：预测节点中正确匹配的比例；
- recall：canonical 节点中被找到的比例；
- F1：precision 和 recall 的调和平均；
- matched MAE：成功匹配节点的位置误差。

低拟合 RMS 不代表节点结构正确，必须同时检查数量和节点匹配指标。

## 7. 可视化

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/interactive_dynamic_v6.pt `
  --seed 20000 `
  --sample-index 0 `
  --output outputs/interactive_dynamic_v6_sample_000.png
```

左图显示采样点、训练代理、标准 B 样条和控制多边形；右图显示唯一的数量分布及最终 argmax 数量。

## 8. 用户自选点云

点云必须按照曲线方向排列，支持 `.csv`、`.txt`、`.xyz`、`.json`、`.npy`、`.pt` 和 `.pth`：

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/interactive_dynamic_v6.pt `
  --point-cloud data/my_curve.csv `
  --json-output outputs/my_curve_fit.json `
  --figure-output outputs/my_curve_fit.png
```

CSV 每行一个点：二维为 `x,y`，三维为 `x,y,z`。输入维度必须与 checkpoint 一致，至少需要 4 个点。脚本默认按弦长重采样到 checkpoint 记录的训练点数，也可用 `--num-points` 显式指定。脚本没有节点真值，因此只报告预测数量、节点、控制点和标准 B 样条拟合 RMS，不报告 precision/recall。若点序相反，可增加 `--reverse-points`。

## 9. 恢复实际坐标

输入归一化为：

\[
q_{norm}=\frac{q-center}{scale}.
\]

得到控制点后恢复：

\[
P=P_{norm}\cdot scale+center.
\]

节点参数位于 \([0,1]\)，不需要尺度恢复。
