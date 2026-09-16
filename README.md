# Self-Supervised Spline Fitting（当前 v16 主线）

本仓库当前 3090 试验主线是 **v16 fixed-Proposal + offline feasible-subset Teacher**：从有序点云一次性预测三次开放 B 样条的参数化、候选内部节点、KeepMask 和存活节点位置，再做一次标准 B 样条最小二乘 refit。已拉取的 `Kc=72` 结果显示，数值修复虽能压低 MSE，却常把节点数推高；现在推荐先跑可选的 `Kc=56` 快速诊断档，验证 Count/Keep 与教师机制，而不是直接投入另一轮大规模训练。新档效果须重训后实测。

> 训练只使用带真参数、真内部节点、真节点数和逐节点删除 MSE 的认证合成曲线；UJI Pen、Natural Earth、USGS 和工业型线等距线只用于留出验证与测试，不参与梯度更新。冻结 Proposal 后，在**预测候选/参数域**上离线搜索可行子集并缓存标签；这是训练时教师，不在网络部署时间内。

推荐的 Linux/3090 快速诊断入口如下；它训练源 `K=4..44`、候选 `Kc=56`、Proposal 48 + Joint 24 epoch、训练/验证 1500/300、Batch 64，测试仍覆盖源 `K=4..56`，其中 K=45..56 明确为训练范围外压力测试。它启用真节点锚定的离线教师和 Count–Keep 梯度耦合；旧默认行为与检查点仍保留。详细协议见 [Kc56 快速诊断说明](docs/v16_kc56_fast_pilot.md)。

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --pilot-kc56 \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_mse1e-4_pilot_sourcek44_kc56_linux_r1
```

## 原 Kc72 大规模实验合同（保留用于对照）

| 项目 | 原设置 |
|---|---|
| objective | `candidate_selection_feasible_teacher_bspline_v16` |
| architecture | `v16_decoupled_keep_count_kc72_mass_topk` |
| simplification | `fixed_proposal_offline_feasible_subset_v1` |
| 训练数据 | certified Synthetic only |
| 真实数据 | validation/test only |
| Teacher | 固定 Proposal 后离线搜索预测候选域的可行 KeepMask/K；认证 source 删除 MSE 仍是辅助监督 |
| 合成 source K | 4–56 个内部节点（8–60 个控制顶点） |
| 网络容量 | `Kc=72` 个内部候选，对 source 最大 `K=56` 保留 16 个冗余槽位；全保留时完整三次节点向量为 80 项 |
| 主阈值 | `MSE <= 1e-4`，其中 MSE 不开方、不除以坐标维数 |
| 训练长度 | Proposal 64 + Joint 64 = 128 epochs |
| 原始部署 | `ours_one_shot`：1 次网络 forward + 1 次 mass-TopK + 1 次标准 refit；核验/修复如启用须单独计时命名 |

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

Proposal 阶段用真参数、真节点、有序一一匹配和多尺度 recall 监督候选位置。`Kc=72` 对最大源 `K*=56` 保留 16 个冗余槽位。冻结 Proposal 后，针对固定合成 Joint 样本在**该 Proposal 预测的参数和候选域**中离线搜索 `MSE<=1e-4` 的删除方案；Joint 学习缓存的可行 KeepMask/K，真源标签继续监督几何与参数。真 `K*` 不再强迫与预测域中的可行节点数相等。

Joint 固定 GeometryEncoder、ParameterHead、CandidateKnotHead（Joint 学习率均为 `0`），只更新 Selector/Count 校准与 selected-only decoder；前 8 代仍保留 Selector/decoder warmup 课程。若让 Proposal 在缓存生成后继续漂移，离线教师的槽位标签就会失配，不能把该训练称为固定 Proposal 协议。

离线教师缓存只属于固定训练样本、Proposal 权重和阈值；旧缓存/旧 Joint checkpoint 不能直接复用。原始部署仍是一次 forward、一次 mass-TopK、一次 refit；这一支的 MSE 不能保证每例过阈值。批量评测用 `--include-verified-ours` 添加单独的 `ours_verified` 数值核验/修复行，报告完整时间、额外 refit 和节点数，不与原始 `ours` 行混写。四指标图仍画原六方法，修复数据在汇总表和逐例记录中。`Kc=72` 的 latency 必须重新测量。

## 原 Kc72 一条龙运行（历史复测）

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
  --benchmark-profile quick \
  --run-name candidate_selection_v16_mse1e-4_sourcek56_kc72_feasible_teacher_linux_r1
```

Proposal 完成后若教师阶段中断，同步新代码并确认同名 `.last.pt`、`.proposal.pt` 仍在，使用原名字继续运行：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --resume-run \
  --device cuda \
  --benchmark-profile quick \
  --run-name candidate_selection_v16_mse1e-4_sourcek56_kc72_feasible_teacher_linux_r1
```

离线教师默认 Batch=8，与 Joint Batch=64 分开；缓存构建可能很久，且当前仅在完整构建后保存。[续跑与产物说明](docs/v16_feasible_teacher_workflow.md)。

下面 Windows PowerShell 入口仍是旧 supervised-only 对照，**不等同于新的离线可行教师协议**：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName candidate_selection_v16_mse1e-4_sourcek56_kc72_supervised `
  -Device cuda `
  -PrepareRealData
```

在 Linux 上只检查新一条龙命令和路径，不执行训练/评测：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --benchmark-profile quick \
  --run-name v16_feasible_teacher_dryrun \
  --dry-run
```

脚本按顺序生成：

1. best/proposal/last checkpoint、history 与逐阶段日志；Windows 入口另写 `pipeline_manifest.json`；
2. checkpoint 结构完整性审计；
3. Ours、Park、Liang、Dung、Kang、Luo 在 Synthetic、UJI、Natural Earth、USGS、IndustrialOffset 上的四指标表；
4. `v16_published_methods_input.png` 与 `v16_published_methods_reference.png` 两张 2×2 指标图；
5. 四个外部数据集上的六方法案例图。

输出数据不会被作图脚本修改、缩放或替换。未实际运行完成前，文档不预设任何 MSE、通过率、节点数或速度结论。

新教师改变训练目标；旧 checkpoint 不会自动获得改进。必须重新训练，再用独立 Synthetic 与四个外部数据集验证教师掩码可行率、网络一次性 MSE/通过率/K、Proposal `R@.005/.01/.02`、Keep P/R/F1 和参数偏差。`quick` 只作跑通诊断，不能作为论文估计。

## 单条点云部署

```powershell
python scripts/fit_v16_point_cloud.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_sourcek56_kc72_feasible_teacher_linux_r1.pt `
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

- [当前 v16 可行教师工作流与 Linux 指令](docs/v16_feasible_teacher_workflow.md)
- [旧 supervised-only 训练流程](docs/training_pipeline.md)
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
