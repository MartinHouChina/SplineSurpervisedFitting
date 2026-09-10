# v16 训练与验证流程

本文描述当前 `MSE=1e-4、Kc=56` 的 v16 主协议。更完整的教师、计时和公开方法复现边界见 [v16 在线反事实子集学习](v16_counterfactual_subset.md)。目标 checkpoint 尚需训练并通过资格检查，本文不预设实验结果。

## 1. 实验合同

| 项目 | 当前设置 |
|---|---:|
| 输入 | 192 个归一化有序点 |
| 样条 | 三次开放 B 样条 |
| 合成源控制顶点 | 8～60 |
| 合成源内部节点 | 4～56 |
| 合成节点最小 span | 0.01（命令必须显式给出） |
| 网络内部候选容量 | 56 |
| 拟合阈值 | `MSE <= 1e-4`，不开方 |
| 最差数据源通过率目标 | 90% |
| 训练集/合成验证集 | 3000/600 |
| 每个真实来源验证上限 | 100 |
| Batch | 64 |

`Kc=56` 是高召回容量，不是部署节点数。三次开放样条全部保留时完整节点向量有 64 项、控制顶点有 60 个；最终 K 由每条曲线的自适应选择概率质量决定。

## 2. 数据怎样进入网络

每个 batch 约 65% 为在线生成的 certified Synthetic，35% 从 UJI、Natural Earth 和 USGS 三个真实训练 split 中抽取。

Synthetic 返回点序列、真采样参数、真内部节点及有效 mask。认证只保证源节点在**固定源位置的所有子集**中不可继续删除；允许节点重定位后，源 K 只作为计数上界，不作为全局最少节点的精确标签。

当前生成命令显式使用 `--knot-min-span 0.01`。`K=56` 有 57 个 span，底层
通用 synthetic 生成器的旧默认 0.02 会要求总长度至少 1.14，数学上不可行；
该底层默认只为兼容历史调用。checkpoint/manifest 必须记录本轮实际值 0.01。

真实曲线没有真节点向量，不构造伪标签，只使用：

- 点重建与稠密候选可行性；
- 在线教师的组合比较；
- 阈值约束与复杂度学习。

数据前向顺序为：

```text
有序点 Q
  -> GeometryEncoder：局部几何 + 全局几何
  -> ParameterHead：严格递增参数 t0
  -> CandidateKnotHead：56 个有序高召回候选 U0
  -> InteractiveSelector：候选重要度 + 逐曲线 beta
  -> mass-TopK：一次生成 KeepMask
  -> selected-only parameter feedback + survivor relocation
  -> 最终有序内部节点 Udeploy
  -> 一次 float64、端点约束标准 B 样条 refit
```

部署没有 CountHead、Hard-Concrete、BIC、逐次删除或教师搜索。

## 3. 两个训练阶段

### 3.1 Proposal：前 16 个 epoch

全保留 56 个候选，优先学习：

- 稳定的几何编码与参数化；
- 覆盖全参数域的候选位置；
- 四个验证数据源上的 dense proposal 拟合可行性。

Selector 在 Proposal 阶段不更新，因此新实验把 Joint 初始保留比例设为
`30/56=0.5357142857142857`；Kc=56 时目标初始概率质量约为 30。它只是训练
起点，不是部署最终 K；部署仍由逐曲线 beta 与 mass-TopK 决定。

### 3.2 Joint：余下 64 个 epoch

Joint 同时训练候选排序、一次性数量选择、参数反馈和存活节点重定位。训练教师池包含：

1. 当前一次性 deployment mask；
2. `K=4..16` 每个整数计数的精确 ranked-prefix 检查；
3. 更高 K 的粗到细搜索及边界附近的增、删、交换组合；
4. 全保留组合；
5. certified Synthetic 的几何 oracle mask 与 `Ktrue+2` 安全扩展 mask；扩展量
   截断到 Kc，故 Ktrue=56 时就是全候选 mask。

几何 oracle 先把 proposal 节点映射到真参数域，再做分数无关的一维单调一一匹配。它打破早期错误 Selector 排名的自监督闭环，仅在合成训练数据中启用。

教师在可行组合中先选 K 最小者，再用 MSE 打破平局。`--synthetic-count-role upper_bound` 允许重定位后找到比 source K 更小的可行组合。

当前主配置关闭强制覆盖分区：`--one-shot-coverage-bins 0`。这避免简单曲线关键节点集中在局部时，低 K 预算被四个区间锚点占用；复杂曲线仍由候选召回、教师可行性和拟合损失保护。

## 4. 为什么能兼顾简单与复杂曲线

- 复杂曲线：56 个候选覆盖 source K=4..56；当拟合不安全时，验证反馈会降低复杂度压力并恢复安全储备。
- 简单曲线：初始质量约 30、低 K 逐计数扫描、oracle 教师和 source-K 上界语义共同提供更细的简化信号。
- 容量边界：source K=56 与 Kc=56 数值相同但语义不同；该层没有冗余候选余量，必须单独报告 dense/deployment pass。
- 所有曲线：曲线级 `beta` 决定概率质量，`mass_topk` 一次决定活动 K；不是按数据集名称手工选择节点数。

复杂度权重最多放大到 6 倍。只有 worst-source deployment pass 达到目标及安全余量后才加大简化压力；跌破 90% 时会回滚。最终安全配置为 `sigma=0.03、额外节点=0`。

## 5. RTX 3090 训练

```powershell
python scripts/train_v16.py `
  --epochs 80 --proposal-epochs 16 `
  --train-size 3000 --val-size 600 --real-val-size 100 `
  --batch-size 64 --num-points 192 `
  --min-control-points 8 --max-control-points 60 `
  --knot-min-span 0.01 `
  --candidate-knots 56 --mse-tolerance 1e-4 `
  --knot-match-tolerance 0.01 `
  --certified-minimal-source `
  --minimality-margin 0.2 --minimality-max-attempts 16 `
  --minimality-audit-points 512 --oscillation-amplitude 0.3 `
  --proposal-pass-target 0.90 --deployment-pass-target 0.90 `
  --one-shot-selection-policy mass_topk `
  --initial-keep-fraction 0.5357142857142857 `
  --one-shot-safety-sigma 0.20 --one-shot-safety-knots 2 `
  --final-safety-sigma 0.03 --final-safety-knots 0 `
  --safety-anneal-epochs 12 `
  --one-shot-coverage-bins 0 --min-selected-knots 4 `
  --relocation-blend 0 `
  --teacher-prefix-search-steps 7 --teacher-low-count-sweep 16 `
  --synthetic-count-role upper_bound `
  --synthetic-geometry-oracle-teacher --oracle-teacher-extra-knots 2 `
  --policy-samples 2 --counterfactual-edits 4 `
  --count-weight 2.0 --supervised-count-weight 1.0 `
  --supervised-over-count-weight 1.0 `
  --true-parameter-weight 0.1 `
  --proposal-knot-coverage-weight 1.0 `
  --selected-knot-position-weight 1.0 --knot-position-beta 0.01 `
  --complexity-weight 0.05 `
  --complexity-ramp-epochs 12 --complexity-max-scale 6.0 `
  --complexity-pass-margin 0.02 `
  --real-fraction 0.35 `
  --real-manifest data/splits/uji_pen_v2.jsonl `
  --real-manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --real-manifest data/processed/usgs_contours/large_scale/manifest.jsonl `
  --resample-train-each-epoch `
  --num-workers 4 --torch-num-threads 4 --device cuda `
  --output outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt
```

Windows 若在 worker 启动处异常，改用 `--num-workers 0`。旧 K64/K96 权重与本实验合同不同，不可直接 `--resume`；中断恢复只能使用本次生成的 `.last.pt`，并保持结构与数据参数一致。

若旧 `outputs/checkpoints/candidate_selection_v16_mse5e-5_k64.proposal.pt` 存在，可在新训练中追加：

```powershell
  --init-checkpoint outputs/checkpoints/candidate_selection_v16_mse5e-5_k64.proposal.pt
```

这是 proposal warm start：先校验 objective 与关键 proposal 结构合同，再迁移 Encoder、ParameterHead 及 CandidateHead 张量；65 个旧 interval query 沿参数域插值为 57 个，K56 固定锚点重新生成，Selector、联合解码器和优化器新训。它不继承旧实验的阈值、教师状态或资格；合同不兼容或关键张量缺失时会直接终止。

恢复时原样重跑上面的完整命令（`--epochs` 表示新的总轮数），并在末尾增加：

```powershell
  --resume outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.last.pt
```

除 `epochs/device/num-workers/torch-num-threads/log-every-batches` 等恢复白名单外，不要改数据、阈值、容量、教师或输出参数。

3090 主要加速网络部分。Joint 在线教师会为低 K 前缀、oracle 与反事实集合执行多轮逐样本 float64 refit；Kang/Luo 评测也主要使用 CPU，因此增大 batch 或更换显卡不会把整条流水线等比例加速。建议先用唯一的测试 `RunName` 做短诊断，正式 `RunName` 不要执行 `-DryRun`。一键脚本是 fresh-only；若训练已经成功而后续 benchmark/绘图失败，直接执行第 7 节的分步命令。

## 6. 训练后资格检查

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --required-pass-rate 0.90 `
  --mse-tolerance 1e-4
```

正式权重至少应满足：

- worst-source dense 和 deployment pass 均不低于 90%；
- Joint 和简化课程已经成熟；
- 最终 safety knots 为 0、safety sigma 为 0.03；
- 平均 K 明显低于 56；
- Synthetic 的 count bias、knot-match F1、matched MAE 没有退化；
- oracle 教师的可行率/入选率已记录；
- source K=56 边界层的 dense/deployment pass 已单列，且失败没有被总体均值掩盖。

返回码 0 才能正式汇报。返回码 2 的 checkpoint 只能使用显式 diagnostic 模式，不能把诊断图当正式结果。

## 7. 六方法四指标评测

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --method-set published `
  --samples-per-knot-count 5 `
  --min-knot-count 4 --max-knot-count 56 `
  --real-samples-per-dataset 20 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 `
  --max-internal-knots 56 `
  --paper-initial-knots 56 --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 --paper-relocation-iterations 12 `
  --liang-dense-knots 56 --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 `
  --torch-num-threads 4 --device cuda `
  --output-dir outputs/comparisons/v16_mse1e-4_k56_six_methods
```

六方法为 Ours、Park、Liang、Dung、Kang 和 Luo；Kang/Luo 是公开方法的适配复现。每个数据源分别报告 MSE、通过率、最终 K 和完整方法时间，Ours 另报 network-only 时间。

绘制同一报告的 2×2 四指标图：

```powershell
python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_mse1e-4_k56_six_methods/comparison.json `
  --method-set published --reference --dpi 300 `
  --output-dir outputs/figures/v16_mse1e-4_k56_six_methods/metrics
```

绘制三个真实数据集上的六方法曲线、采样点、控制多边形、控制顶点与内部节点：

```powershell
python scripts/visualize_v16_real_deployments.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --real-samples-per-dataset 2 --selection-seed 20260910 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 `
  --max-internal-knots 56 `
  --paper-initial-knots 56 --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 --paper-relocation-iterations 12 `
  --liang-dense-knots 56 --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 `
  --torch-num-threads 4 --device cuda `
  --dpi 300 `
  --output-dir outputs/figures/v16_mse1e-4_k56_six_methods/real_cases
```

一键串行执行同一套训练、检查、比较和绘图：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_mse1e-4_3090.ps1
```

脚本拒绝覆盖同名实验；checkpoint 不合格时默认停在资格检查。仅排错时使用 `-Diagnostic`，仅在 Windows worker 异常时使用 `-NumWorkers 0`。

## 8. 历史配置说明

旧 source K=4..24、旧 K64、`Kc=96、MSE=2.5e-5` 与旧的 `MSE=5e-5` 无人值守脚本仍可作为范围、容量或严格容差消融记录，但不是当前主协议。它们的 checkpoint、通过率和节点数不能与本轮 K56/1e-4 结果混写，也不能被描述为新结构已经达到的结果。当前“完整节点向量 64 项”由 `56+8` 得到，不表示内部候选仍为 64。底层通用 synthetic 生成器的 `knot_min_span=0.02` 默认只为历史兼容；当前 K=4..56 必须显式使用 0.01，因为 K=56 会产生 57 个 span。
