# B-Spline 节点数量与位置预测

本项目输入一条曲线的有序采样点，输出开放三次 B 样条的：

- 严格递增点参数 \(t\)；
- 内部节点数量 \(K\)；
- 严格有序内部节点 \(u_1<\cdots<u_K\)；
- 最小二乘重拟合后的标准 B 样条控制点。

当前 v5 不使用 ActivityHead、Hard-Concrete 或候选节点阈值。网络先估计节点数量，再直接生成对应数量的节点。

## 三条流程不要混淆

| 阶段 | 节点数量来源 | 是否删除预测节点 | 输出用途 |
|---|---|---:|---|
| 数据标注 | 在源 B 样条上做容差约束节点删除 | 仅删除源标签中的冗余节点 | 构造 canonical 监督真值 |
| 网络训练 | 使用 canonical 真实数量选择解码结果 | 否 | 稳定训练数量头和节点位置头 |
| 验证 | 使用 CountHead 自己预测的数量 | 否 | 选择 checkpoint，测量真实泛化性能 |
| 部署 | network argmax 或 BIC 选择完整数量分支 | 否 | 生成最终标准 B 样条 |

因此，“节点删除”只发生在离线标签构造阶段，不是模型的推理机制。

## 总体数据流

```text
有序采样点 points [B,M,D]
  │
  ├─ GeometryEncoder
  │    ├─ local_features [B,M,H]
  │    └─ global_features [B,H]
  │
  ├─ ParameterHead → params [B,M]
  │
  ├─ Ordinal CountHead
  │    ├─ count_probabilities [B,Kmax+1]
  │    └─ predicted_knot_count [B]
  │
  ├─ Shared Count-Conditioned KnotHead
  │    ├─ 所有数量分支 branch_internal_knots
  │    └─ 所选分支 internal_knots + knot_mask
  │
  ├─ 训练代理：截断幂基 + 可微线性求解
  │
  └─ 部署：选择完整 K 分支 + 标准 B 样条控制点重拟合
```

这里的严格执行顺序是：先编码点，再用 `local_features + global_features` 预测参数；CountHead 随后读取 `global_features`、`local_features` 和参数位置编码；最后 KnotHead 读取这些特征及所选数量。节点张量内部保持 `Kmax` 宽度，通过 `knot_mask` 表示有效的前 K 个位置，部署时再提取成真正长度为 K 的向量。

详细说明：

- [模型内部数据流](docs/architecture.md)
- [数据生成与训练流程](docs/training_pipeline.md)
- [部署与结果解释](docs/deployment_pipeline.md)
- [数学定义](docs/math_formulation.md)
- [文件索引](docs/file_guide.md)

## 默认数据配置

| 项目 | 默认值 |
|---|---:|
| 训练 / 验证样本 | 10000 / 1000 |
| 每条曲线采样点 | 64 |
| 空间维度 | 2D，可选 3D |
| 源控制点数量 | 5–10 |
| 最大内部节点数 | 6 |
| 采样噪声标准差 | 0.001 |
| canonical 删除容差 | 0.005 RMS |

## 快速运行

训练新 v5 模型：

```powershell
python scripts/train.py --epochs 100 --output outputs/count_conditioned_v5.pt
```

使用默认 BIC 部署策略评估：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/count_conditioned_v5.pt `
  --json-output outputs/count_conditioned_v5_evaluation.json
```

可视化一条验证曲线：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/count_conditioned_v5.pt `
  --sample-index 0 `
  --output outputs/count_conditioned_v5_sample_000.png
```

只检查 CountHead 原始 argmax，不使用 BIC：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/count_conditioned_v5.pt `
  --count-selection network
```

运行测试：

```powershell
python -m pytest -q
```

## checkpoint 兼容

- v5：canonical 标签、局部序数 CountHead、共享条件解码器；
- v4：恢复 categorical CountHead 和独立数量分支；
- v3 及更早：恢复 ActivityHead/Hard-Concrete 历史结构。

旧 checkpoint 可以继续评估，但不能直接转换成 v5 权重。
