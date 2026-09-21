# Minimum-Complexity B-Spline Fitting

最新：[M16/M32 两个通用模型及一条龙指令](docs/overnight_m16_m32.md)、[Granularity 架构与实验说明](docs/overnight_granularity.md)、[这次训练结果审计](docs/overnight_results_audit_20260921.md)、[当前算法论文框架](paper/overnight/README.md)。容量路线只训练最大16、32个内部候选的两个通用模型，不再按数据集分别训练；均使用混合合成训练、真实验证/测试，默认三段各加2层并启用峰值误差损失。旧检查点行为不变，新模型尚需重新训练验证，未宣称效果已提升。

最新短程改进见 [Overnight Anchored：稳定紧凑解码](docs/overnight_anchored.md)：K32、MSE `5e-5`，锚定式参数/节点残差、紧凑教师专项监督及 Joint 内冻结候选几何校准。新增 `--anchored-selection` 一条龙；旧检查点保持原行为，效果须重新验证。

新增 [Overnight K32：统一容量与最大误差](docs/overnight_k32.md)：显式将网络与全部五个数值对照组上限设为 32 个内部节点，支持 K64 权重缩容后重新训练；增加最大单点平方误差、五指标图与六方法案例标注。含 Linux 训练—评测—绘图一条龙，旧实验不覆盖。

当前改进见 [Overnight Stable](docs/overnight_stable.md)：依据“19 13”训练结果，增加可达删点轨迹教师；部署仍一次性。评测及六方法案例默认覆盖 UJI、Natural Earth、USGS 和工业等距线，含现有 checkpoint 补测与 Linux 训练一条龙。

新增 [Overnight Compact](docs/overnight_compact.md)：`--compact-selection` 从 Reliable best 权重显式 warm-start，采用逐曲线可行性简化、有预算的 greedy Teacher、数量储备对齐和合成混合训练；默认 24 epochs，不增加部署搜索。含更新包上传、Linux 一条龙与续跑说明，不承诺最少节点或达标。

当前短程改进实验见 [Overnight Reliable](docs/overnight_reliable.md)：保留一夜版流程，新增参数 trust gate、排序之外的教师候选与独立真实验证；`--reliable-selection` 提供 32-epoch warm-start 及六方法四指标一条龙。对照组采用明确标注的 threshold-safe adaptation，保留原生结果和修复代价。下面的主线历史参数不覆盖该档配置。

1070 历史架构的 Linux 一键训练、六方法四指标比较及案例图入口见 [Linux 历史配置流程](docs/overnight_1070_linux.md)。这是独立 warm-start 新实验，不是旧 epoch 17 精确续训，也不是当前主线的离线 teacher 流程。

可选的 [Overnight Plus 增强训练](docs/overnight_plus.md) 保留同一网络结构，用 `--enhanced-selection` 加强在线 teacher 与节点选择，并支持从已训练 overnight 模型完整迁移权重到新实验；不加开关仍为原历史配置。

本仓库研究：给定沿曲线方向排序的二维或三维点，在满足归一化拟合误差约束的前提下，用尽量少的三次 B 样条内部节点表示曲线。

默认误差定义为

\[
\operatorname{MSE}=\frac{1}{M}\sum_i\|C(t_i)-Q_i\|_2^2,
\]

不取平方根，也不除以坐标维数。默认单曲线阈值为 `2.5e-5`，工程验收指标为每个验证来源至少 90% 的曲线通过该阈值。

## 当前 v16 流程

> 下文为历史 v16 通用设计与旧参数示例；本轮 Overnight 的实际开关、K32/专用容量、零安全储备、Teacher与合成训练比例，以顶部最新实验说明及checkpoint配置为准。不要把旧K96建议与本轮容量消融混用。

```text
有序点 Q + MSE 阈值 ε
  → GeometryEncoder
  → ParameterHead：弦长参考 + 有界残差 → t0
  → CandidateKnotHead：局部 cross-attention → Kc 个有序候选 U0
  → Contextual Selector：候选自注意力 + 点特征交叉注意力
  → 候选相对重要度 s；曲线级动态阈值 β
  → p = sigmoid((s-mean(s))-β)
  → K̂ = ceil(Σp + 0.25·sqrt(Σp(1-p)) + 2)
  → 一次 Top-K，并保留参数域覆盖锚点 → KeepMask
  → Subset Decoder：只读取存活候选，联合更新参数和节点位置
  → 一次端点约束的标准 B 样条 refit
  → 控制顶点、曲线、内部节点和实际 MSE
```

部署仍然只有一次网络前向、一次离散 Top-K 和一次最终 refit。没有 CountHead、Hard-Concrete、BIC、阈值扫描或逐节点试删。多次反事实子集 refit 只存在于训练阶段。

v16 新筛选器恢复了 v10–v15 中有效的动态 `β + mass_topk` 思路，同时增加：

- 概率质量数量监督；
- 教师保留/删除的相对排序监督；
- 误删重要节点的非对称惩罚；
- 上尾困难曲线损失；
- 只有验证通过率达到安全裕量后才逐步开启复杂度惩罚；
- 删除后的参数更新与存活节点重定位，且重定位从恒等映射开始。

旧 v16 checkpoint 的配置仍按历史 `p>=0.5` 加载，不会被静默解释成新策略。请用新的输出文件重新训练，不要续训旧 `fast90.last.pt`。

## 节点数量的正确含义

`--candidate-knots Kc` 表示内部候选节点数。对三次开区间 B 样条：

\[
N_{\text{full}}=K_{\text{internal}}+8,\qquad
N_{\text{control}}=K_{\text{internal}}+4.
\]

因此：

| 配置 | 内部候选 | 全保留时完整节点向量 | 控制顶点数 |
|---|---:|---:|---:|
| `--full-knot-vector-size 64` | 56 | 64 | 60 |
| `--candidate-knots 64` | 64 | 72 | 68 |
| `--candidate-knots 96` | 96 | 104 | 100 |

真实地理曲线的复杂度跨度远大于合成 K=4–20 数据。建议正式工程模型使用 `Kc=96`，并将 `Kc=64/96` 作为容量消融；完整向量64即内部56，反而低于当前64内部候选。

主对比中不能为不同方法选择各自有利的最大节点数。所有可设置容量的方法必须统一内部节点上限；另一张副表才可以报告论文原始或推荐超参数。

## 训练

推荐 Kc=96、90%工程验收配置：

```powershell
$v16Args = @(
  '--epochs', '60',
  '--proposal-epochs', '20',
  '--train-size', '2400',
  '--val-size', '500',
  '--real-val-size', '100',
  '--batch-size', '16',
  '--num-points', '192',
  '--min-control-points', '8',
  '--max-control-points', '24',
  '--candidate-knots', '96',
  '--mse-tolerance', '2.5e-5',
  '--certified-minimal-source',
  '--minimality-margin', '0.2',
  '--minimality-max-attempts', '16',
  '--minimality-audit-points', '512',
  '--oscillation-amplitude', '0.3',
  '--proposal-pass-target', '0.90',
  '--deployment-pass-target', '0.90',
  '--one-shot-selection-policy', 'mass_topk',
  '--one-shot-safety-sigma', '0.25',
  '--one-shot-safety-knots', '2',
  '--one-shot-coverage-bins', '4',
  '--min-selected-knots', '4',
  '--real-fraction', '0.5',
  '--real-manifest', 'data/splits/uji_pen_v2.jsonl',
  '--real-manifest', 'data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl',
  '--real-manifest', 'data/processed/usgs_contours/large_scale/manifest.jsonl',
  '--init-checkpoint', 'outputs/checkpoints/candidate_selection_v16.proposal.pt',
  '--device', 'cuda',
  '--output', 'outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt'
)
python scripts/train_v16.py @v16Args
```

这里的 `--init-checkpoint` 只迁移形状兼容的 Encoder、ParameterHead 和候选生成参数；Kc=96 新 query、Selector 与 subset decoder 仍从头训练。它适合尽快得到工程模型，但 Kc=64/96 的严格容量消融应使用相同初始化条件、分别从头训练。

`proposal-pass-target=0.90` 配合默认 `complexity-pass-margin=0.02`，意味着进入 Joint 阶段前的实际最差数据源 dense gate 是 92%。若未达到，脚本只保留 `.proposal.pt`、`.last.pt` 和 `.history.json`，不会伪造主 `.pt`。

### 12 小时无人值守实验：Kc=64、MSE=5e-5

下面的入口会串行完成训练或断点续训、checkpoint 资格检查、八方法配对测试、方法对比图和 Ours 案例图：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_overnight_12h.ps1
```

默认合同为：64 个**内部候选**；合成 source 内部节点 `K=4..24`，即控制顶点 `8..28`、最大 source 完整节点向量长度 32；固定合成训练/验证集为 1500/500；MSE 阈值 `5e-5`。注意 Kc=64 全保留时网络完整节点向量长度是 72，不是 32。

脚本会先等待当前正在写同一 checkpoint 的训练进程退出，再严格按 `.last.pt` 中保存的配置续训；不会同时启动第二个写入者。当前已有运行若采用 batch=32、proposal=4，就会原样续用，不能被 fresh 默认的 batch=64、proposal=12 静默改写。逐 epoch 历史、恢复权重、每阶段日志及流水线状态均会保存。

默认先训练到 56 epochs；若仍未通过正式资格且预算还剩至少 6.5 小时，才续训到 64 epochs。随后对 source K=4..24 每档 2 条及三个真实来源各 8 条进行 66 曲线比较，并生成 Ours/Kang/Park/Liang/Dung/Luo/Yeh/Greedy 指标图和 6 个 Ours 真实案例。12 小时是依据当前机器截至 epoch 8 的实测速度制定的预算，不是硬超时保证；GPU、CPU 和数据缓存状态会改变实际时间。

若希望关闭当前终端后继续运行，可启动独立隐藏进程；进度仍写入日志：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_overnight_12h.ps1 -Detach
```

`-Detach` 只派生一个隐藏 runner 并立即返回；隐藏 runner 内的现有进程/GPU 守卫仍会照常执行。主状态文件为 `outputs/logs/candidate_selection_v16_mse5e-5_k64/overnight_manifest.json`。资格检查通过时结果进入 `formal_<checkpoint-hash>/`；否则流程仍完成比较与画图，但进入 `diagnostic_<checkpoint-hash>/`，并强制标记 `DIAGNOSTIC NOT FINAL`，不能用于正式结论。参数、恢复规则和完整产物树见[训练流程](docs/training_pipeline.md#9-12-小时无人值守训练比较与案例图)。

如需验证“完整节点向量最多64项”的容量实验，把 `--candidate-knots 96` 换成：

```powershell
'--full-knot-vector-size', '64'
```

这会严格转换为56个内部候选。不同 Kc 的参数形状不同，必须使用新的 output；不能把 Kc=64 的 `.last.pt` 恢复成 Kc=96。

## 检查和单条点云部署

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --required-pass-rate 0.90 `
  --mse-tolerance 2.5e-5

python scripts/fit_v16_point_cloud.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --point-cloud data/my_curve.csv `
  --mse-tolerance 2.5e-5 `
  --output-dir outputs/fits/v16/my_curve
```

只有 joint 阶段且通过资格检查的 checkpoint 才能作为正式结果。`.proposal.pt`、未达标权重或显式 `--allow-unqualified-diagnostic` 的结果只能用于排错。

## 与论文方法统一对比

统一入口包含：

- Ours v16；
- Park & Lee 2007 dominant-point adaptation；
- Kang et al. 2015 sparse/ADMM adaptation；
- Liang et al. 2017 feature-integral + IKI adaptation；
- Dung & Tjahjowidodo 2017 serial direct-knot adaptation；
- Luo et al. 2022 \(l_{\infty,1}\) + differential-evolution adaptation；
- Yeh 2020 feature-CDF；
- 统一最大节点数的 greedy deletion + gradient relocation。

Park、Kang、Liang、Dung、Luo 和 Yeh 是公开论文目标的可审计适配实现，不冒充作者原始软件；uniform greedy 是本仓库的传统数值控制组。所有方法处理相同曲线，最终统一用 CPU `float64`、无正则、端点插值的标准 B 样条 refit，并报告 MSE、通过率、最终内部节点数和完整耗时；Ours 另外报告纯网络时间。

Kc=96 的正式公平容量命令：

> `candidate_selection_v16_simplified_certified_k96.pt` 是当前简化合同的训练输出目标，不是随仓库附带的现成权重。旧 `adaptive_k96` 运行缺少 certified-source 和计数精度合同，只能作为诊断或 warm-start，不能改名代替新权重。

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --output-dir outputs/comparisons/v16_certified_k96_published `
  --samples-per-knot-count 20 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --real-samples-per-dataset 100 `
  --mse-tolerance 2.5e-5 `
  --max-internal-knots 96 `
  --paper-initial-knots 96 `
  --liang-dense-knots 96 `
  --device cuda

python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_certified_k96_published/comparison.json `
  --output-dir outputs/figures/v16_certified_k96/method_comparison `
  --dpi 300
```

该绘图入口只读取 benchmark 保存的实测 JSON，不会重新运行算法或修改数值；默认比较 Ours、Park、Liang、Dung、Kang 和 Luo，并输出 MSE、通过率、最终内部节点数和完整方法时间四个子图。若还要加入 Yeh 与统一贪心基线，增加 `--method-set all`。

Ours 的论文案例图单独生成：

```powershell
python scripts/visualize_v16_ours_cases.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --real-samples-per-dataset 2 `
  --selection-seed 20260909 `
  --mse-tolerance 2.5e-5 `
  --device cuda `
  --dpi 300 `
  --output-dir outputs/figures/v16_certified_k96/ours_cases
```

脚本为每条留出测试曲线生成一张结构详图，并额外生成 `ours_cases_overview.png`。详图明确标注输入采样点、部署 B 样条、控制多边形、控制顶点、曲线上的内部节点和参数域节点条；对应数值保存在 `deployment_visualizations.json`。

上述正式命令要求主 `.pt` 已完成训练并通过 `inspect_v16_checkpoint.py`。仓库当前不能仅凭目标文件名声称已有正式图；未合格权重只能显式增加 `--allow-unqualified-diagnostic` 生成带 `DIAGNOSTIC NOT FINAL` 水印的排错图。

如果使用完整节点向量64的公平实验，必须先训练 `--full-knot-vector-size 64`（即 Kc=56）的配套 checkpoint；随后 benchmark 使用该 checkpoint、`--full-knot-vector-size 64`，并将 `--paper-initial-knots` 与 `--liang-dense-knots` 都设为56。Kc=96 checkpoint 与数值上限56不属于公平主表，脚本默认会拒绝这种组合。

## 主要文档

- [文档总索引](docs/README.md)
- [v16算法、训练与容量策略](docs/v16_counterfactual_subset.md)
- [训练流程](docs/training_pipeline.md)
- [部署流程](docs/deployment_pipeline.md)
- [论文方法适配与公平协议](docs/published_knot_methods_reproduction.md)
- [数学定义](docs/math_formulation.md)
- [合成曲线最简性证书](docs/synthetic_data_minimality_report.md)
- [真实数据集](docs/real_world_datasets.md)
- [文件索引](docs/file_guide.md)

输出约定：正式权重放在 `outputs/checkpoints/`，统一比较放在 `outputs/comparisons/`，可视化放在 `outputs/figures/`，用户点云部署放在 `outputs/fits/v16/`。
