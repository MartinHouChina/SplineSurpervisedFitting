# Self-Supervised Spline Fitting（当前 v16 主线）

本仓库当前主线是 **v16 supervised-only**：从有序点云一次性预测三次开放 B 样条的参数化、候选内部节点、KeepMask 和存活节点位置，再做一次标准 B 样条最小二乘 refit。

> 这里的 supervised-only 指训练协议。正式训练只使用带真参数、真内部节点、真节点数和逐节点删除 MSE 的认证合成曲线；UJI Pen、Natural Earth、USGS 和工业型线等距线只用于留出验证与测试，不参与梯度更新。逐节点删除 MSE 来自合成最简性认证，是固定的细粒度监督标签，不是在线自监督伪标签。

## 当前实验合同

| 项目 | 当前设置 |
|---|---|
| objective | `candidate_selection_supervised_bspline_v16` |
| architecture | `v16_supervised_ordered_assignment_mass_topk` |
| simplification | `synthetic_ground_truth_ordered_keep_and_relocation_v4` |
| 训练数据 | certified Synthetic only |
| 真实数据 | validation/test only |
| Teacher | 禁用 online prefix/counterfactual self-teacher；启用 certificate-derived single-deletion MSE 监督 |
| 合成 source K | 4–56 个内部节点（8–60 个控制顶点） |
| 网络容量 | `Kc=72` 个内部候选，对 source 最大 `K=56` 保留 16 个冗余槽位；全保留时完整三次节点向量为 80 项 |
| 主阈值 | `MSE <= 1e-4`，其中 MSE 不开方、不除以坐标维数 |
| 训练长度 | Proposal 64 + Joint 64 = 128 epochs |
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

Proposal 阶段用真参数、真节点、有序一一匹配和多尺度 recall 损失监督候选位置。当前 `Kc=72` 且 `K*<=56`，因此最大复杂度样本仍有 16 个冗余候选可供筛选和重定位，不再要求 56 个候选逐一精确复刻 56 个真节点。Joint 阶段由同一匹配直接产生 existence/KeepMask 标签，并用真 K、真参数和真节点位置监督计数、排序及重定位；Keep 概率还接受 Dice、沿候选顺序的 CDF 和邻近真节点的模糊负例约束。最简性认证保存的逐节点 single-deletion MSE 用于区分关键正候选，并细化 Keep 与 ranking 强度。参数头增加相邻参数间隔的 log-gap 和整曲线有符号 bias 监督，Joint 的节点位置损失以受限梯度回传给参数头。

Joint 开始后的前 8 代只更新 Selector 和 selected-only decoder，先把 Keep 排序、数量与存活节点重定位接上已经训练好的 Proposal；随后再以分组学习率联合微调：Selector `2e-4`、GeometryEncoder/CandidateHead `1e-5`、ParameterHead `5e-5`、selected-only decoder `5e-5`。这避免 Joint 刚开始时随机 Selector 以同一学习率拖坏 Proposal。

正式路径仍不运行在线 Hard-RMS、ranked-prefix、oracle 或反事实 subset-search Teacher，也不生成 Teacher cache。细粒度删除标签在合成样本认证时计算，loss forward 不增加在线样条求解；部署仍是一次 forward、一次 mass-TopK 和一次 refit，计时口径不变。`Kc` 从 56 增至 72 会增加一定网络计算量，必须由新的实测 latency 报告，不能沿用旧值。

## 一条龙运行

先准备四个外部数据 manifest；训练不会使用其样本更新权重，但验证和最终比较需要它们：

```text
data/splits/uji_pen_v2.jsonl
data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl
data/processed/usgs_contours/large_scale/manifest.jsonl
data/processed/industrial_offsets/v1/manifest.jsonl
```

在 Linux/RTX 3090 上运行（缺少真实数据时加 `--prepare-real-data`）：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised_linux
```

在 Windows/RTX 3090 上运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised `
  -Device cuda `
  -PrepareRealData
```

只检查命令和路径：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName v16_supervised_dryrun -DryRun
```

脚本按顺序生成：

1. best/proposal/last checkpoint、history 与逐阶段日志；Windows 入口另写 `pipeline_manifest.json`；
2. checkpoint 结构完整性审计；
3. Ours、Park、Liang、Dung、Kang、Luo 在 Synthetic、UJI、Natural Earth、USGS、IndustrialOffset 上的四指标表；
4. `v16_published_methods_input.png` 与 `v16_published_methods_reference.png` 两张 2×2 指标图；
5. 四个外部数据集上的六方法案例图。

输出数据不会被作图脚本修改、缩放或替换。未实际运行完成前，文档不预设任何 MSE、通过率、节点数或速度结论。

这批监督项改变了训练目标；旧 checkpoint 不会自动获得改进。必须重新训练，再用独立 Synthetic 与四个外部数据集验证 Proposal `R@.005/.01/.02`、Keep P/R/F1、关键节点误删率、参数偏差以及最终 MSE/通过率/节点数。

## 单条点云部署

```powershell
python scripts/fit_v16_point_cloud.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised.pt `
  --point-cloud path/to/ordered_points.csv `
  --output-dir outputs/fits/my_curve `
  --mse-tolerance 1e-4 `
  --device cuda
```

输入必须是真实存在的 `.csv`、`.npy` 等有序点文件；示例路径不会自动创建数据。正式结果必须使用通过结构合同检查的 Joint checkpoint。`--allow-unqualified-diagnostic` 仅用于带水印的排错图。

## 结果解释边界

- 数据集通过率不再作为 Proposal→Joint、停止训练或 benchmark 资格的硬门槛；Proposal checkpoint 先比较连续的 dense subset cost，再比较最差来源/总体通过率、位置与参数误差、F1/recall，避免 K=56 boundary 全为零时由宽容差 recall 的微小抖动选回早期模型。
- 单曲线 `MSE<=1e-4` 仍是该曲线是否满足工程阈值的判据。
- 合成 source K 是认证生成表示的精确监督标签；该认证只证明固定参数化、原 source 节点子集内的阈值最简性，不等于连续自由重定位下的全局最少节点证明。
- 五个论文对照是按公开描述实现的 adaptation，不是作者代码的逐行复刻。
- 真实数据没有节点真值，因此只报告拟合、复杂度和时间，不报告真实节点 precision/recall。

## 文档入口

- [训练流程](docs/training_pipeline.md)
- [细粒度合成教师与新增监督](docs/v16_fine_grained_supervision.md)
- [工业型线等距线数据集](docs/industrial_offset_dataset.md)
- [v16 算法](docs/v16_counterfactual_subset.md)
- [数学定义](docs/math_formulation.md)
- [网络架构](docs/architecture.md)
- [部署与评测](docs/deployment_pipeline.md)
- [合成数据最简性](docs/synthetic_data_minimality_report.md)
- [六方法适配复现](docs/published_knot_methods_reproduction.md)
- [验证清单](docs/v16_verification.md)
- [文件索引](docs/file_guide.md)
