# v12/v13/v14_joint 部署、评估与可视化

> `v14_joint` 先由初始结构反馈修正 `t0 -> t1`，再让候选节点读取带 `t1` 编码的
> 局部特征并联合更新 KeepMask 与位置；重新选择后只用最终存活节点完成重定位。
> 网络仍是一条固定深度前向。详见
> [v14_joint_parameter_structure_feedback.md](v14_joint_parameter_structure_feedback.md)。

## 1. 快速 learned 部署

对一条有序点云，v12 默认执行：

```text
读取并归一化有序点云
  -> 网络 forward 一次
       预测参数 t
       生成固定 proposal U_prop
       预测 final KeepMask
       基于 final survivors 一次性重定位 U_deploy
  -> 取 U_deploy[KeepMask]
  -> 标准开放三次 B 样条 refit 一次
  -> 反归一化并输出曲线、控制顶点与节点
```

网络内部的初始 keep、位置反馈、最终 keep 和 relocation 是同一固定深度计算图，不是多次调用网络。快速部署不逐节点试删，不执行 beam search，也不根据误差循环。

`--fit-tolerance` 在 learned 模式仅用于报告阈值通过率，不能在部署时修改 mask。因此 v12 是统计近似，不提供逐样本阈值保证。

## 2. 节点张量语义

| 张量 | 含义 |
|---|---|
| `proposal_internal_knots` | 高召回、与教师槽位绑定的固定候选 |
| `provisional_candidate_knots` | 最终选择前的位置反馈结果 |
| `pre_relocation_candidate_knots` | final KeepMask 已知、survivor relocation 之前的位置 |
| `final_hard_keep_mask` | 一次性最终离散子集 |
| `relocation_position_residual` | v12 最后一阶段的存活节点位移 |
| `deployment_internal_knots` | 最终位置，真正交给标准 B 样条 refit |
| `internal_knots` | `deployment_internal_knots` 的兼容别名 |

最终节点向量是 `deployment_internal_knots[final_hard_keep_mask]`，而不是从 proposal 原位置直接取子集。

## 3. 一次性节点数量

默认 `mass_topk` 的顺序是：

1. 预测每个候选的 keep probability；
2. 汇总 probability mass，并加入 `--one-shot-safety-sigma` 控制的不确定性储备；
3. 得到一次性保留数量；
4. 全局选择分数最高的 K 个有效槽位。

可用 `--one-shot-selection-policy threshold` 做阈值策略消融。`--activity-threshold` 是历史参数，对 v8–v12 默认部署不起作用。

## 4. 独立评估

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --num-samples 512 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --knot-tolerance 0.01 `
  --json-output outputs/logs/v12/candidate_pruning_one_shot_v12_evaluation.json
```

训练、验证和独立测试默认 seed 分别为 `42 / 10000 / 20000`。不要用训练集或验证集结果代替独立测试结果。

建议同时报告：

- proposal recall：候选是否覆盖 canonical 节点；
- deployment precision / recall / F1：最终节点与 canonical 节点的一维有序匹配；
- `selected pre->post |shift|`：最终 KeepMask 下 survivor relocation 的平均/最大位移与实际移动比例；
- mean / P95 / max RMS 与 threshold-satisfied fraction；
- 平均、最小、最大保留节点数及直方图；
- 标准 refit 的 MSE；
- ParameterHead 的参数误差。

节点匹配只有在共享或足够接近的参数化下才有直接几何含义。`--knot-tolerance` 是匹配容差，不是拟合阈值，也不是最小节点间距。

## 5. 传统 Hard-RMS 对照

在 learned 评估中附加传统 greedy 对照：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --num-samples 128 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --run-hard-diagnostic `
  --json-output outputs/logs/v12/v12_learned_vs_hard.json
```

Hard-RMS 从固定 proposal 开始，反复测试单节点删除并执行标准 B 样条 refit。它是慢速诊断，不使用 v12 relocation 后的位置，也不会替换 checkpoint 的默认 learned 部署。

需要把 hard 作为实际部署结果时，显式使用：

```text
--deployment-mode hard
```

## 6. verified：快速质量守卫

`verified` 位于纯一次性 `learned` 和慢速 `hybrid` 之间：

```text
网络 forward 一次
  -> 对 LearnedKeep + relocation 结果做标准 B 样条精确 refit
  -> 若满足 RMS 阈值：直接返回
  -> 否则保留 learned proposal 身份，并按 keep probability 补回未选候选
  -> 对置信度前缀做少量精确 refit，返回第一个实测可行状态
  -> 可选：从该较小可行集出发做 greedy compact
  -> 若完整 proposal 仍失败：可选残差引导动态插点
```

前缀二分只用于减少试验次数；由于平滑项存在时 MSE 不保证随节点数严格单调，任何最终接受状态都必须通过真实标准 B 样条 refit 检查。该流程是自适应部署，不是纯 one-shot，也不证明全局最少节点。

参数域必须显式记录。`--verified-parameterization network` 始终使用网络预测参数；默认 `chord-fallback` 仅在网络域完整 proposal 失败后，把节点按采样点对应关系映射到弦长域再验证；`chord` 在整个 verified 流程使用弦长域。结果中的节点只能配合 `final_parameters` 解释和求值；评估时先把部署节点映射回真实参数域，再计算节点匹配指标。

### 两个速度/复杂度档位

最快档保留较多节点，不执行二次压缩：

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --deployment-mode verified `
  --no-verified-compact `
  --verified-parameterization chord-fallback `
  --verified-refit-device auto `
  --num-samples 512 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/logs/v12/v12_verified_fast.json
```

复杂度优先档使用 `--verified-compact`，会从已验证可行的 learned 子集或置信度前缀继续删除冗余节点。两档默认均启用 `--verified-residual-fallback`，仅当完整 proposal 精确失败时，才在当前最大点残差的合法参数位置逐次插入节点；默认最多 8 次，且每次插入后重新精确 refit。消融时可关闭：

```text
--no-verified-residual-fallback
```

残差插点可能使最终节点数超过网络 `Kc`，因此报告会把 immutable proposal slots 与 dynamic inserted knots 分开记录。它优先保证拟合阈值，不用于声称一次性固定长度或最小复杂度。

### 求解设备与时间

`--verified-refit-device auto` 默认让 CUDA 执行网络 forward、CPU 执行小规模 rank-revealing 最小二乘。可选值为 `auto / cpu / model`。JSON 同时记录 model device、refit device、网络外 repair 时间、精确 refit 次数、残差兜底比例和动态插点数量。

完整部署诊断仍应单独记录检查、补回、可选 compact 和残差兜底的时间；当前论文/PPT 主时间按用户指定只报告同步后的纯 `forward_deployment()`。因此 verified 的最终 MSE 可以来自网络外质量守卫，但其主显示时间不能称为 verified 端到端延迟。

## 7. hybrid：离线质量模式

hybrid 用更多推理时间联合搜索“删谁”和“剩余节点移动到哪里”：

```text
网络 forward 一次并固定参数 t
  -> proposal、LearnedKeep、传统 greedy 构造搜索起点
  -> learned 起点先做位置精修
  -> beam 展开单节点删除组合
  -> 在 greedy 停止边界附近先精修多个 child，再截断 beam
  -> 对 survivors 做有序 coordinate refinement
  -> 每个候选状态执行标准 B 样条 refit并测量真实 MSE
  -> 字典序选择：满足阈值优先，其次 K 少，再其次 MSE 低
```

`--hybrid-position-refine-candidate-multiplier` 控制 beam 截断前接受位置精修的 child 数量，默认是 `4 × beam width`。它避免某个删除组合只因“尚未移动时误差较高”而过早被丢弃。

hybrid 的 coordinate refinement 会实际返回移动后的 `final_fit.internal_knots`。当前可视化把 refined knots 画在最终部署曲线上，并在结构轴显示 `proposal u -> refined u*`，不再只用 retained proposal 索引着色。

### 阈值定义

hybrid 核心使用平均平方欧氏误差：

\[
\operatorname{MSE}=\frac1M\sum_i\|C(t_i)-Q_i\|_2^2=\operatorname{RMS}^2.
\]

CLI 的 `--fit-tolerance` 延续 RMS 语义，进入 hybrid 后自动平方：

```text
--fit-tolerance 0.005  <=>  hybrid MSE tolerance = 2.5e-5
```

传统 greedy 的可行结果保留为 fallback，但有限 beam 和离散网格 refinement 仍不构成全局最优证明。

### 默认参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--hybrid-beam-width` | 4 | 每层保留状态数 |
| `--hybrid-branch-factor` | 4 | 每个父状态展开的删除数；`0` 为全部 |
| `--hybrid-position-sweeps` | 2 | coordinate sweep 次数 |
| `--hybrid-position-grid-size` | 7 | 单坐标奇数网格大小 |
| `--hybrid-position-restarts` | 2 | 确定性位置重启次数 |
| `--hybrid-position-refine-count-margin` | 1 | 精修覆盖到 greedy 边界上方的 K 层数 |
| `--hybrid-position-refine-candidate-multiplier` | 4 | beam 截断前精修候选倍数 |

### 批量评估

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --deployment-mode hybrid `
  --num-samples 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --hybrid-beam-width 4 `
  --hybrid-branch-factor 4 `
  --hybrid-position-sweeps 2 `
  --hybrid-position-grid-size 7 `
  --hybrid-position-restarts 2 `
  --hybrid-position-refine-count-margin 1 `
  --hybrid-position-refine-candidate-multiplier 4 `
  --json-output outputs/logs/v12/v12_hybrid_evaluation.json
```

时间允许时可以增大 beam、全部展开分支、增加 sweep/grid/restart，但应先在少量独立样本上测量耗时；文档不预设这些参数一定提高所有样本。

## 8. 单样本图

快速 learned：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --seed 20000 `
  --sample-index 0 `
  --pruning-view learned `
  --fit-tolerance 0.005 `
  --timing-repeats 5 `
  --dpi 600 `
  --output outputs/figures/current/v12_learned_000.png
```

慢速 hybrid：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --seed 20000 `
  --sample-index 0 `
  --pruning-view hybrid `
  --fit-tolerance 0.005 `
  --hybrid-position-refine-candidate-multiplier 4 `
  --dpi 600 `
  --output outputs/figures/current/v12_hybrid_000.png
```

图中的时间含义：

- `net`：一次网络 forward；
- `prune`：hard/hybrid 的离线组合与位置搜索；
- `refit`：最终标准 B 样条控制顶点求解。

GPU 操作是异步的，正式论文计时建议预热并设置 `--timing-repeats 3` 或 `5`，同时报告硬件和 batch size。

## 9. 批量四联对比图

下面的命令按 source 内部节点数，从 `K=4` 到数据集上限，每层随机抽取一条独立
测试曲线。省略 `--max-knot-count` 时上限自动从数据集配置读取；默认控制顶点范围
`8–24` 对应 source `K=4–20`。

```powershell
python scripts/visualize_batch_comparison.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output-dir outputs/comparisons/current/v12_sourceK_4_to_max `
  --samples-per-knot-count 1 `
  --min-knot-count 4 `
  --scan-size 512 `
  --seed 20000 `
  --selection-seed 12345 `
  --stratify-by source `
  --timing-repeats 5 `
  --dpi 600
```

需要每个 K 两个随机样本时，把 `--samples-per-knot-count` 改为 `2`。如果明确只画
到某个上限，可加 `--max-knot-count 20`。`--scan-size` 是先扫描的确定性测试池大小；
若某个指定 K 在池中没有足够样本，脚本会明确报错，而不会拿其他层补齐。

四个面板分别为：

1. 源 B 样条、原始观测点、源控制顶点/控制多边形及源内部节点；
2. 网络产生的全部冗余 proposal、使用预测参数得到的标准 B 样条 refit、控制顶点及节点；
3. v12 LearnedKeep + survivor relocation 的一次性部署结果、最终 refit、控制顶点及节点；
4. 从同一份已物化 proposal 出发的传统 proposal-only greedy hard pruning、控制顶点及节点。

第 2 栏表示 `CandidateKnotHead` 预测的高召回冗余候选，不是 Boehm 算法对源样条做的
精确等形插入。第 2–4 栏共享同一次网络输出的预测参数和 proposal；第 1 栏使用数据生成
时的真实参数，所以它是源曲线/噪声基线，不应与后三栏解释成完全相同的参数估计任务。

### 9.1 MSE 口径

四栏标题、CSV 和 JSON manifest 中的误差统一为：

\[
\operatorname{MSE}=\frac1M\sum_{i=0}^{M-1}
\|C(t_i)-Q_i\|_2^2.
\]

它是在归一化坐标中对“每个点的平方欧氏距离”取平均，不开平方，也不是先对所有坐标
元素取平均的 coordinate-wise MSE。若未显式传入 `--mse-tolerance`，传统 hard pruning
会把 checkpoint 中历史 RMS 阈值平方；例如 `RMS=0.005` 对应 `MSE=2.5e-5`。

### 9.2 计时边界

图和 manifest 使用预热后的 batch-size 1 wall time，并对 `--timing-repeats` 次运行取
中位数。建议展示使用 `3` 或 `5` 次，并同时记录 CPU/GPU 型号。

- `ours network forward`：只统计 `forward_deployment()`，包含 ParameterHead、冗余
  proposal、KeepMask 与 survivor relocation；输入提前驻留，排除标准 refit、verified
  repair、数据搬运、绘图和 I/O。完整质量管线时间仅作为 diagnostic 保存。
- `hard pruning`：只从已经物化的预测参数和同一份 proposal 开始计时。它包含 greedy
  单节点删除、停止判断及其内部全部标准 B 样条 refit，但明确排除生成 proposal 的网络
  forward。

这两个数字服从本实验指定的非对称边界：前者是纯网络预测，后者是给定 proposal 后的
传统剪枝阶段。可以用于回答各自指定范围花了多久，但不能声称它们是边界完全一致的
端到端加速比。独立传统基线应改用均匀最大节点初始化的删除+梯度重定位流程，而不是
读取网络 proposal。

脚本按 source 或 canonical 节点数分层随机抽样，输出 PNG、JSON manifest 和 CSV。面板误差统一使用 MSE，不开平方。该批量脚本的第四栏是传统 hard，对 hybrid 应另用 `visualize_result.py --pruning-view hybrid`。

## 10. 用户有序点云

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --point-cloud data/my_curve.csv `
  --deployment-mode verified `
  --verified-refit-device auto `
  --fit-tolerance 0.005 `
  --json-output outputs/predictions/my_curve_v12.json `
  --figure-output outputs/predictions/my_curve_v12.png
```

脚本会检查文件、按 checkpoint 训练长度重采样、归一化、执行一次 forward，并在全部源点上做最终标准 refit。输入必须沿曲线方向排序；`--reverse-points` 可整体反向。

`verified` 会在全部源点上执行阈值检查与必要修复。改为 `learned` 可测纯一次性延迟；改为 `hybrid` 会启用更慢的组合与位置搜索。对于明显偏离合成训练分布的实测点云，必须单独检查曲线覆盖、端点、MSE 和节点分布，不能由合成测试指标外推。

## 11. 真实数据集批量测试

`prepare_uji_pen.py`、`prepare_natural_earth.py` 和 `prepare_usgs_contours.py` 生成统一 JSONL manifest。批量 learned 部署使用：

```powershell
python scripts/evaluate_real_world.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --manifest data/splits/uji_pen_v2.jsonl `
  --split test `
  --max-samples 1000 `
  --json-output outputs/real_world/uji_v14_learned_1000.json `
  --overwrite
```

该脚本在 192 点输入上求控制点，在 manifest 保存的原始密度参考线上评价 MSE、P95、Chamfer 和 Hausdorff。主时间仅包括 `forward_deployment()`；标准 B 样条 refit、读取与指标计算不计时。真实折线没有节点真值，因此不输出 knot Precision/Recall/F1。完整协议见[真实数据训练适配与部署测试](real_world_evaluation.md)。

真实域纯网络不达标时，可显式选择效果优先路径：

```powershell
python scripts/evaluate_real_world.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --manifest data/splits/uji_pen_v2.jsonl `
  --split test `
  --deployment-mode certified `
  --certified-mse-target 1e-5 `
  --json-output outputs/real_world/uji_v14_certified.json `
  --overwrite
```

它始终保留 `learned_*` 纯网络结果，再独立记录 reference refit、节点联合移动/增结和
兜底耗时。常规搜索失败后先构造满足实测 MSE 的简化折线 B 样条，仅在必要时才精确
表示完整输入折线。认证范围仅是 manifest 提供的 normalized 离散参考点，不是未知连续
真曲线；简化折线和完整折线兜底都可能超过 adaptive 节点上限，必须连同最终 K、控制
点数和 fallback 类型一起报告。
