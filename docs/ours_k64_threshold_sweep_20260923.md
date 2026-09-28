# 64K 模型多阈值部署测试

## 实验口径

- 固定检查点：`outputs/checkpoints/candidate_selection_v16_mse5e-5_k64.pt`，第62轮，训练配置的内部节点容量64、MSE阈值5e-5。不是把32K模型扩容，也不重新训练或根据本次测试选择权重。
- 复用上一轮对比的全部61条曲线：21条合成曲线，以及 UJI、Natural Earth、USGS、IndustrialOffset 各10条。IndustrialOffset 为程序生成的工业模型等距线，并非实测数据。
- MSE 阈值：1e-5、2e-5、3e-5、4e-5、5e-5、6e-5、7e-5、8e-5、9e-5、1e-4。最大单点平方误差阈值同步设为 MSE 阈值的10倍。
- 每个“样本 × 阈值”重新运行：网络预测 → 固定参数的端点约束 B 样条最小二乘拟合 → 未达双阈值时插入修复 → 逐次删除冗余节点、重拟合控制顶点并检查双阈值。
- 内部节点总上限64，剪枝最多接受64次删除，不沿用旧的32次删除限制。后剪枝不移动存活节点，也不更新参数，不保证全局最少节点。
- 不同阈值之间不复用拟合结果。所有失败和超限样本保留在统计中。MSE 是归一化空间中的平均平方欧氏距离，MaxSE 是采样点平方欧氏距离的最大值，两者均不开方。

这是同一64K模型的部署敏感性测试，不是十个分别训练的模型。双阈值共同改变，不是固定最大误差约束的单因素消融。与之前32K结果比较时，权重与容量均不同，不能仅将差异归因于容量。误差约束仅针对给定采样点，不是连续曲线误差保证。

## 重跑指令（Linux / PowerShell 单行均可）

```bash
python scripts/benchmark_ours_thresholds.py --checkpoint outputs/checkpoints/candidate_selection_v16_mse5e-5_k64.pt --allow-checkpoint-change --max-internal-knots 64 --output-dir outputs/comparisons/ours_k64_threshold_sweep_new_run --device cuda --mse-tolerances 1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5 1e-4 --peak-ratio 10
```

`--allow-checkpoint-change` 明确允许更换原始32K对比记录中的模型，仅复用其测试输入。实际检查点哈希、容量、输入哈希及代码哈希均写入新报告。输出目录必须新建或为空，避免覆盖实验。

## 输出

本次目录：`outputs/comparisons/ours_k64_threshold_sweep_20260923/`。

- `ours_threshold_sweep.png`：平均MSE、平均与最坏MaxSE、双阈值通过率、平均内部节点数。
- `threshold_summary.md` / `threshold_summary.json`：总体及各数据来源的完整十档统计，包含耗时。
- `threshold_sweep.json`：610次部署的最终结果、网络原始结果及剪枝前结果；`complete=true` 才表示全部完成。
- `geometry/`：共同输入；`tier_1/` 至 `tier_10/`：各档节点向量、控制顶点、采样参数、拟合曲线、逐点误差及来源哈希。

时间统计包含网络与首次重拟合、插入修复和后剪枝，不含结果导出及绘图；单次测量只用于探索，不应与不同硬件或不同计时范围的性能数字直接比较。

## 独立核验

```bash
python scripts/audit_ours_thresholds.py outputs/comparisons/ours_k64_threshold_sweep_20260923/threshold_sweep.json
```

使用 SciPy 根据保存的节点向量和控制顶点独立计算曲线，逐样本检查实际平方残差、误差统计、通过率、节点上限、剪枝前后可行性及计时分项；同时校验输入一致性和几何文件哈希。只对 `complete=true` 的完整实验生成 `geometry_audit.json`。

## 本次完成结果

610/610次部署和610份几何独立核验全部完成，未出现修复/剪枝执行异常。35项相关回归测试通过。统计包含全部61条曲线的有限输出，不剔除超限案例。

| MSE阈值 | 全样本平均MSE | 双阈值通过率 | 平均内部节点数 |
|---|---:|---:|---:|
| 1e-5 | 1.24112e-5 | 56/61（91.8%） | 27.90 |
| 2e-5 | 2.04472e-5 | 57/61（93.4%） | 23.82 |
| 3e-5 | 2.86301e-5 | 58/61（95.1%） | 21.95 |
| 4e-5 | 3.56010e-5 | 59/61（96.7%） | 20.51 |
| 5e-5 | 4.46185e-5 | 59/61（96.7%） | 19.39 |
| 6e-5 | 5.27903e-5 | 60/61（98.4%） | 18.44 |
| 7e-5 | 6.03264e-5 | 60/61（98.4%） | 17.67 |
| 8e-5 | 6.71892e-5 | 60/61（98.4%） | 17.33 |
| 9e-5 | 7.41516e-5 | 60/61（98.4%） | 16.67 |
| 1e-4 | 8.31147e-5 | 61/61（100.0%） | 16.43 |

这些是“64K网络 + 插入修复 + 后剪枝”的完整结果，不是纯网络一次性预测的通过率。例如1e-5档：原始网络12/61达标；修复后56/61达标、平均32.69个内部节点；后剪枝保持56/61达标并降至27.90个。1e-4档的100%仅描述本批61条曲线，不构成对其他输入的保证。

## 单曲线十档部署案例图

案例：Natural Earth 海岸线 `natural_earth_10m_coastline_b7529bb1babd_w000_b7529bb1`。该案例经过结果筛选用于展示：十档全部满足双阈值，具有多个可辨认转折，节点数随阈值放宽从48降到23。它不是随机样本，也不替代上面的全部61曲线统计。

目录：`outputs/figures/ours_k64_ten_thresholds_coastline_20260923/`。

- `ours_ten_thresholds_case.png`：2×5、300dpi，含阈值、实际MSE、MaxSE和内部节点数。
- `ours_ten_thresholds_case_clean.png`：简洁版，仅在各面板标注阈值和内部节点数；两版均使用一个全局图例，没有水印。
- 两版均按从左上到右下逐行阅读排列为 `1e-4、9e-5、8e-5、7e-5、6e-5；5e-5、4e-5、3e-5、2e-5、1e-5`。数据表、清单和导出NPZ同步采用该顺序；原始实验记录不修改。
- 所有面板使用相同输入点与坐标范围，展示输入折线、采样点、已保存的拟合曲线、控制顶点及控制多边形、曲线上的节点和[0,1]参数轴上的节点分布。
- `case_metrics.md` 保存误差、节点数、控制顶点数量和原实验计时；`case_manifest.json` 保存来源、筛选说明及哈希；`tier_01.npz`至`tier_10.npz` 保存对应的精确几何数据。
- 不重新预测或拟合；选中案例的十档误差由SciPy独立求值核验后绘图。

重绘到新目录：

```bash
python scripts/plot_ours_threshold_case.py --dataset NaturalEarth --sample-id natural_earth_10m_coastline_b7529bb1babd_w000_b7529bb1 --output-dir outputs/figures/ours_k64_ten_thresholds_coastline_new --dpi 300
```
