# Self-Supervised Spline Fitting（当前 v16 主线）

本仓库当前主线是 **v16 supervised-only**：从有序点云一次性预测三次开放 B 样条的参数化、候选内部节点、KeepMask 和存活节点位置，再做一次标准 B 样条最小二乘 refit。

> 这里的 supervised-only 指训练协议。正式训练只使用带真参数、真内部节点和真节点数的认证合成曲线；UJI Pen、Natural Earth 和 USGS 只用于留出验证与测试，不参与梯度更新。

## 当前实验合同

| 项目 | 当前设置 |
|---|---|
| objective | `candidate_selection_supervised_bspline_v16` |
| architecture | `v16_supervised_ordered_assignment_mass_topk` |
| simplification | `synthetic_ground_truth_ordered_keep_and_relocation_v4` |
| 训练数据 | certified Synthetic only |
| 真实数据 | validation/test only |
| 合成 source K | 4–56 个内部节点（8–60 个控制顶点） |
| 网络容量 | `Kc=56` 个内部候选；全保留时完整三次节点向量为 64 项 |
| 主阈值 | `MSE <= 1e-4`，其中 MSE 不开方、不除以坐标维数 |
| 训练长度 | Proposal 40 + Joint 64 = 104 epochs |
| 部署 | 1 次网络 forward + 1 次 mass-TopK + 1 次标准 refit |

## 数据流

```text
有序点云 Q
  -> GeometryEncoder
  -> ParameterHead：严格递增参数 t
  -> CandidateKnotHead：Kc 个有序候选 U_prop
  -> Selector：keep logits + 曲线自适应 beta
  -> probability-mass Top-K：离散 KeepMask
  -> selected-only decoder：联合更新 t 与存活节点位置
  -> 标准三次 B 样条 refit × 1
  -> 曲线、控制顶点、内部节点向量、MSE
```

Proposal 阶段用真参数、真节点和有序一一匹配监督候选位置；Joint 阶段由同一匹配直接产生 existence/KeepMask 标签，并用真 K、真参数和真节点位置监督计数、排序及重定位。正式路径没有在线 Hard-RMS、ranked-prefix、oracle 或反事实 Teacher，也不生成 Teacher cache。

## 一条龙运行

先准备三个真实数据 manifest；训练不会使用其样本更新权重，但验证和最终比较需要它们：

```text
data/splits/uji_pen_v2.jsonl
data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl
data/processed/usgs_contours/large_scale/manifest.jsonl
```

在 Linux/RTX 3090 上运行（缺少真实数据时加 `--prepare-real-data`）：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_mse1e-4_k56_supervised_linux
```

在 Windows/RTX 3090 上运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName candidate_selection_v16_mse1e-4_k56_supervised `
  -Device cuda
```

只检查命令和路径：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName v16_supervised_dryrun -DryRun
```

脚本按顺序生成：

1. best/proposal/last checkpoint、history 与逐阶段日志；Windows 入口另写 `pipeline_manifest.json`；
2. checkpoint 结构完整性审计；
3. Ours、Park、Liang、Dung、Kang、Luo 在 Synthetic、UJI、Natural Earth、USGS 上的四指标表；
4. `v16_published_methods_input.png` 与 `v16_published_methods_reference.png` 两张 2×2 指标图；
5. 三个真实数据集上的六方法案例图。

输出数据不会被作图脚本修改、缩放或替换。未实际运行完成前，文档不预设任何 MSE、通过率、节点数或速度结论。

## 单条点云部署

```powershell
python scripts/fit_v16_point_cloud.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --point-cloud path/to/ordered_points.csv `
  --output-dir outputs/fits/my_curve `
  --mse-tolerance 1e-4 `
  --device cuda
```

输入必须是真实存在的 `.csv`、`.npy` 等有序点文件；示例路径不会自动创建数据。正式结果必须使用通过结构合同检查的 Joint checkpoint。`--allow-unqualified-diagnostic` 仅用于带水印的排错图。

## 结果解释边界

- 数据集通过率只是报告量，不控制 Proposal→Joint、checkpoint 保存或 benchmark 资格。
- 单曲线 `MSE<=1e-4` 仍是该曲线是否满足工程阈值的判据。
- 合成 source K 是认证生成表示的精确监督标签；该认证只证明固定参数化、原 source 节点子集内的阈值最简性，不等于连续自由重定位下的全局最少节点证明。
- 五个论文对照是按公开描述实现的 adaptation，不是作者代码的逐行复刻。
- 真实数据没有节点真值，因此只报告拟合、复杂度和时间，不报告真实节点 precision/recall。

## 文档入口

- [训练流程](docs/training_pipeline.md)
- [v16 算法](docs/v16_counterfactual_subset.md)
- [数学定义](docs/math_formulation.md)
- [网络架构](docs/architecture.md)
- [部署与评测](docs/deployment_pipeline.md)
- [合成数据最简性](docs/synthetic_data_minimality_report.md)
- [六方法适配复现](docs/published_knot_methods_reproduction.md)
- [验证清单](docs/v16_verification.md)
- [文件索引](docs/file_guide.md)
