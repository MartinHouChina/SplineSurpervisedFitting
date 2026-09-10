# v16 部署与评测流程

## 1. 单条曲线部署

输入是沿曲线顺序排列的 2D/3D 点。处理顺序为：

1. 读取并校验点文件；
2. 平移/尺度归一化，并重采样为模型需要的 192 点；
3. 一次网络 forward 预测参数、候选、概率质量和重定位结果；
4. 一次 probability-mass Top-K 得到 KeepMask；
5. 仅使用存活节点执行一次 CPU float64 标准三次 B 样条 refit；
6. 映射回源坐标并输出曲线、控制顶点、节点向量、MSE 和时间。

部署没有真参数、真节点或真 K，也不执行 Teacher、阈值扫描、逐节点删除或多次 refit 搜索。

```powershell
python scripts/fit_v16_point_cloud.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --point-cloud data/my_curve.csv `
  --output-dir outputs/fits/my_curve `
  --mse-tolerance 1e-4 --device cuda
```

`data/my_curve.csv` 只是示例，文件必须实际存在。

## 2. checkpoint 先审计

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --mse-tolerance 1e-4
```

正式 checkpoint 应满足 supervised objective、K56 数据/容量、synthetic-only training、direct ground-truth Joint、无 online Teacher、成熟 Joint、mass-TopK 和最终 safety 合同。通过率、最终 K 与节点匹配精度是报告量，不是结构资格门槛。

结构不完整 checkpoint 只能配合 `--allow-unqualified-diagnostic` 排错，输出必须保留诊断水印。

## 3. 六方法、四数据集比较

正式集合为 Ours、Park–Lee、Liang、Dung–Tjahjowidodo、Kang、Luo；数据为 Synthetic、UJI、Natural Earth、USGS。所有方法处理同一批配对曲线，使用相同 56 内部节点容量和相同最终 CPU float64 refit。

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --output-dir outputs/comparisons/v16_supervised_six_methods `
  --method-set published `
  --samples-per-knot-count 5 --min-knot-count 4 --max-knot-count 56 `
  --real-samples-per-dataset 20 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 --max-internal-knots 56 `
  --paper-initial-knots 56 --liang-dense-knots 56 `
  --device cuda
```

输出包含 `comparison.json`、`summary.csv`、`measurements.csv` 和 `report.md`。四项主指标为：

| 指标 | 定义 |
|---|---|
| MSE | `mean_i ||C(t_i)-Q_i||²` |
| pass rate | 包含失败样本在分母内的 `MSE<=1e-4` 比例 |
| final K | 最终 refit 使用的内部节点数 |
| total time | 从归一化点开始到最终 refit 结束，不含 I/O/绘图 |

Ours 另报 `network_ms`，但不能用它替代公平主表中的完整方法时间。

## 4. 两张 2×2 指标图

```powershell
python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_supervised_six_methods/comparison.json `
  --output-dir outputs/figures/v16_supervised_six_methods/four_metrics `
  --method-set published --reference --dpi 300
```

同一条命令生成：

- `v16_published_methods_input.png`：在 192 个网络输入采样点上计算指标；
- `v16_published_methods_reference.png`：在真实数据的原始密度参考点上计算指标。

两张图均为 MSE、通过率、最终 K、完整时间的 2×2 布局。Synthetic 没有独立 original-reference 图项。

## 5. 真实六方法案例图

```powershell
python scripts/visualize_v16_real_deployments.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --output-dir outputs/figures/v16_supervised_six_methods/real_cases `
  --real-samples-per-dataset 2 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 --device cuda --dpi 300
```

每个 3×2 案例图共享同一条留出真实曲线，展示原始参考、网络输入点、最终拟合、控制多边形、控制顶点、内部节点、MSE、K 和时间。

## 6. 一条龙入口

`scripts/run_v16_mse1e-4_3090.ps1` 串行执行训练、审计、benchmark、两张 2×2 图和真实案例图，并按 checkpoint SHA-256 将正式/诊断输出隔离。它拒绝覆盖同名实验，因此每次新实验应使用新的 `-RunName`。
