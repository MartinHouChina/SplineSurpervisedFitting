# Sparse Spline Fitting

本项目从有序曲线采样点预测：

- 点参数 \(t\)；
- 候选内部节点 \(U\)；
- 每个候选节点的存在概率；
- 删除无效节点后得到的标准开放三次 B 样条。

当前方案采用 **独立节点 query + 有监督 existence + Hard-Concrete**。训练不使用 pilot 或事后节点重要性评分。

## 数据集

训练使用程序生成的开放三次 B 样条数据。默认配置如下：

| 项目 | 默认值 |
|---|---:|
| 训练集 / 验证集 | 512 / 128 条 |
| 每条曲线采样点 | 64 |
| 空间维度 | 2D，可选 3D |
| 控制点数 | 5–10 |
| 真实内部节点数 | 1–6 |
| 采样噪声标准差 | 0.001 |
| 节点非均匀度 | 0.65 |
| 采样非均匀度 | 0.45 |

每条样本由随机平滑控制多边形、非均匀开放节点向量和非均匀参数采样生成。采样点随后中心化，并按最大半径缩放到单位尺度。

主要字段：

| 字段 | 含义 |
|---|---|
| `points` | 归一化后的有序采样点 |
| `chord_params` | 弦长参数，仅作为可选先验 |
| `true_params` | 生成曲线时的真实采样参数 |
| `true_internal_knots` + mask | 补齐后的真实内部节点及有效掩码 |
| `true_control_points` + mask | 补齐后的真实控制点及有效掩码 |
| `true_knot_vector` + mask | 完整开放节点向量及有效掩码 |
| `center`, `scale` | 恢复原坐标所需的归一化量 |

同一 `seed + sample_id` 总是生成同一条曲线。详细说明见 [数据与训练流程](docs/training_pipeline.md)。

## 当前工作流

```text
有序采样点 Q
  → GeometryEncoder：点、一阶差分、二阶差分
  → ParameterHead：预测严格递增参数 t
  → K 个独立 knot query 对带 t 位置编码的局部特征做 cross-attention
  → 每个 query 独立预测节点位置 u_j
  → 排序节点，并同步重排 query 特征
  → 与真实内部节点做一维有序最小代价匹配
  → ActivityHead 预测 existence logit
  → Hard-Concrete 得到非零概率 π_j 和训练门 z_j
  → 使用 stop-gradient(z) 构造截断幂设计矩阵并求解系数
  → 联合优化拟合、参数、节点位置、existence 和节点数
```

关键设计：

- 每个节点独立回归位置；删除一个节点不会重新分配其他节点位置。
- existence 正样本来自匹配节点，未匹配 query 为负样本。
- Hard-Concrete 门仍参与拟合，但拟合损失不能通过门更新 ActivityHead。
- checkpoint 优先选择验证集 existence F1；F1 相同时选择总损失更低者。

默认损失权重：

| 损失 | 权重 |
|---|---:|
| 曲线拟合 | 1.0 |
| `true_params` | 0.01 |
| existence BCE | 0.005 |
| 匹配节点位置 Smooth-L1 | 0.01 |
| 期望节点数 | 0.002 |

## 部署

评估时使用固定阈值将 \(\pi_j\) 二值化，物理删除未保留节点，再用剩余节点构造标准开放三次 B 样条并重新求解控制点：

```text
π_j ≥ threshold
  → 删除无效内部节点
  → 构造 [0,0,0,0, U_keep, 1,1,1,1]
  → 增广最小二乘重拟合标准 B 样条控制点
  → 输出可部署 B 样条
```

## 运行

安装依赖：

```bash
pip install torch numpy matplotlib
```

训练：

```bash
python scripts/train.py --epochs 100 --output outputs/independent_queries_v2.pt
```

评估：

```bash
python scripts/evaluate_checkpoint.py --checkpoint outputs/independent_queries_v2.pt --json-output outputs/independent_queries_v2_evaluation.json
```

可视化：

```bash
python scripts/visualize_result.py --checkpoint outputs/independent_queries_v2.pt --output outputs/independent_queries_v2.png
```

测试：

```bash
python -m unittest discover -s tests -v
```

## 文档

- [模型与工作流](docs/architecture.md)
- [数据、训练和评估](docs/training_pipeline.md)
- [核心数学形式](docs/math_formulation.md)
- [文件索引](docs/file_guide.md)

## 兼容性

当前 objective 为 `independent_query_supervised_hard_concrete_v2`。旧 checkpoint 会按保存时的结构和损失语义加载；旧权重可以继续评估，但不会自动获得当前 stop-gradient 训练行为。新实验应重新训练并保存到新的输出文件。
