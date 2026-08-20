# Interactive Structure B-Spline Fitting

本项目从有序采样点预测开放三次 B 样条的点参数、内部节点数量、内部节点位置和控制点。

当前 v6 主路径为：

```text
几何编码
  → 点参数预测
  → 交互式结构 query 预测节点数量 K
  → K>0 时动态解码器只运行 K+1 个 interval query；K=0 时跳过
  → 得到恰好 K 个严格有序节点
  → 标准 B 样条控制点重拟合
```

主路径中没有：

- 独立 CountHead；
- Activity threshold 或 Hard-Concrete；
- 固定候选节点删除；
- 全部数量分支枚举；
- BIC 或第二次数量选择。

## 实际数据流

```text
points [B,M,D]
  ↓
GeometryEncoder
  ├─ local_features  [B,M,H]
  └─ global_features [B,H]
  ↓
ParameterHead(local_features, global_features)
  ↓
params [B,M]
  ↓
InteractiveStructureHead(global_features, local_features, params)
  ├─ structure_query_features [B,Kmax,H]
  ├─ count_probabilities [B,Kmax+1]
  └─ predicted_knot_count [B]
  ↓
DynamicKnotDecoder(..., selected_count)
  ├─ selected_count>0 时只计算 selected_count+1 个区间 query
  ├─ selected_count=0 时跳过位置解码
  ├─ internal_knots [B,Kmax]
  └─ knot_mask [B,Kmax]
  ↓
标准开放三次 B 样条重拟合
```

训练时 `selected_count` 使用 canonical 真实数量；验证和部署时使用网络的 `predicted_knot_count`。

## 节点数量如何产生

`Kmax` 个结构 query 先对带参数位置编码的局部特征做 cross-attention，再通过 self-attention 交换节点必要性信息。每个 query 输出一个条件停止概率，由此构造唯一、归一化的数量分布：

\[
P(K=0),P(K=1),\ldots,P(K=K_{max}).
\]

最终数量只有一个来源：

\[
\widehat K=\arg\max_KP(K).
\]

## 节点位置如何产生

预测 \(K\) 后，若 \(K>0\)，动态解码器只取前 \(K+1\) 个共享 interval query，生成 \(K+1\) 个正区间。取前 \(K\) 个前缀和得到节点，因此天然满足；\(K=0\) 时无需预测位置：

\[
0<u_1<\cdots<u_K<1.
\]

一个 batch 中不同样本具有不同数量时，代码按 `selected_count` 分组计算，再填充回 `[B,Kmax]` 张量。填充不代表计算了其他数量分支。

## 默认数据

| 项目 | 默认值 |
|---|---:|
| 训练 / 验证样本 | 10000 / 1000 |
| 训练 / 验证 / 测试 seed | 42 / 10000 / 20000 |
| 每条曲线采样点 | 64 |
| 源控制点数量 | 5–10 |
| 最大内部节点数 | 6 |
| 噪声标准差 | 0.001 |
| canonical 节点删除容差 | 0.005 RMS |

canonical 节点删除只用于构造监督标签，不参与网络部署。

默认训练集每个 epoch 按确定性新 seed 重新生成，验证集保持固定。前 10 个 epoch 完全使用真实数量训练位置头，随后将 teacher-forcing 比例线性降到 0.5，使训练逐渐接近验证和部署时的预测数量输入。

## 运行

```powershell
python scripts/train.py `
  --epochs 100 `
  --output outputs/interactive_dynamic_v6.pt
```

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/interactive_dynamic_v6.pt `
  --seed 20000 `
  --json-output outputs/interactive_dynamic_v6_evaluation.json
```

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/interactive_dynamic_v6.pt `
  --sample-index 0 `
  --output outputs/interactive_dynamic_v6_sample_000.png
```

```powershell
python -m pytest -q
```

用户自己的点云必须沿曲线方向有序，支持 CSV、TXT、XYZ、JSON、NPY 和 PT：

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/interactive_dynamic_v6.pt `
  --point-cloud data/my_curve.csv `
  --json-output outputs/my_curve_fit.json `
  --figure-output outputs/my_curve_fit.png
```

CSV 每行是一个点，例如二维数据为 `x,y`。脚本默认按弦长重采样到 checkpoint 的训练点数，使用训练时相同的中心化和尺度归一化，并在输出中将控制点恢复到原坐标系。

## 文档

- [模型内部投喂顺序](docs/architecture.md)
- [数据生成与训练流程](docs/training_pipeline.md)
- [部署与指标解释](docs/deployment_pipeline.md)
- [数学定义](docs/math_formulation.md)
- [结构演化记录](docs/pruning_redesign.md)
- [文件索引](docs/file_guide.md)

## 兼容性

当前 objective 为 `interactive_structure_dynamic_knots_v6`。v5、v4、v3 及更早 checkpoint 仍按各自历史结构严格加载，但不能自动转换为 v6 权重。
