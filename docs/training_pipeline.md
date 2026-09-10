# v16 supervised-only 训练流程

本文描述当前正式协议，不描述 v8–v15 的 Teacher-cache 训练。

## 1. 数据边界

训练集只含在线生成的 certified Synthetic。每条训练样本包含：

- 归一化有序点云 `Q`；
- 干净源曲线上的真参数 `t*`；
- 真内部节点 `U*` 与有效 mask；
- 精确源节点数 `K*`；
- source-subset minimality 证书字段。

正式参数为 source `K=4..56`、控制顶点 8–60、每条曲线 192 点、`knot_min_span=0.01`、`MSE tolerance=1e-4`。噪声只加到网络观测，标签来自干净源曲线。

UJI Pen、Natural Earth 和 USGS manifest 可以传给训练入口，但只建立留出验证集；正式配置必须是 `--real-fraction 0`。真实样本不进入 optimizer step。

## 2. Proposal 阶段（epoch 1–40）

该阶段训练 GeometryEncoder、ParameterHead 和 CandidateKnotHead，Selector 与 selected-only decoder 尚不承担最终组合学习。

主要监督为：

1. 真参数回归；
2. 真节点到候选集的 recall-direction coverage；
3. 有序、单调、一一匹配的候选位置损失；
4. 全候选可微 B 样条拟合损失。

当 `Kc>K*` 时，一一匹配只选择 `K*` 个互异且保持顺序的候选；当 `Kc=K*` 时每个候选必须与相同序位的真节点对应。Proposal 的合成抽样有 50% 来自 `K>=40`，其余来自低 K 区间，以加强容量边界召回。

第 40 个 epoch 后按计划无条件进入 Joint。aggregate pass 不会延长 Proposal 或触发 STOP。

## 3. Joint 阶段（epoch 41–104）

Joint 恢复 source `K=4..56` 的原始抽样分布，并对每条合成样本执行：

```text
U_prop + U* -> 有序最小代价一一匹配 -> target KeepMask
K*                                      -> count target
t*, U*                                  -> parameter/relocation targets
```

直接监督项包括：

- existence/KeepMask 的类别平衡 BCE；
- 正候选应排在负候选之前的 ranking loss；
- 自适应 probability mass 对真 K 的 count loss 与 over-count loss；
- 部署 mask 和标签 mask 两条 selected-only 解码路径的参数、节点位置及拟合损失；其中真节点到存活节点的定向覆盖项会显式惩罚漏节点；
- Proposal 阶段的参数、coverage 和 ordered-assignment 监督继续保留。

因为 `K*` 是精确标签，正式监督模式固定 `--complexity-weight 0`；否则额外的自由稀疏惩罚会与真节点数监督冲突。

正式 Joint 不调用在线 subset search，不运行 ranked-prefix、counterfactual 或 geometry-oracle Teacher，也不读取/写入 Teacher cache。每批仅计算 dense、实际部署 mask 和标签 mask 所需的拟合分支。

## 4. 选择与 checkpoint

Proposal checkpoint 按 dense proposal 指标选择，仅作为初始化。成熟 Joint checkpoint 使用 `mean_per_curve_subset_cost_v1` 排名：

- 单曲线可行时，优先较少节点，MSE 只作有界 tie-break；
- 单曲线不可行时，只按相对阈值的 MSE 惩罚，不奖励少节点；
- 全数据集 pass rate 只报告，不参与排名或训练门控。

正式 checkpoint 还必须记录：supervised objective、synthetic-only、`joint_supervision=synthetic_ground_truth`、`online_teacher=false`、有序标签映射、最终 safety 设置、K56 数据/容量合同，以及已经完成的 Joint 课程。

训练会产生：

```text
<run>.pt          最佳成熟 Joint
<run>.proposal.pt 最佳 Proposal
<run>.last.pt     最新优化器/RNG/训练状态，用于恢复
<run>.history.json
```

## 5. 推荐一条龙命令

Linux/RTX 3090（缺少真实数据 manifest 时由脚本准备）：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_mse1e-4_k56_supervised_linux
```

Windows/RTX 3090：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName candidate_selection_v16_mse1e-4_k56_supervised `
  -Device cuda `
  -Epochs 104 -ProposalEpochs 40 `
  -TrainSize 3000 -ValSize 600 -RealValSize 100 `
  -BatchSize 64 -NumWorkers 4
```

三个真实 manifest 必须预先存在。一条龙入口 fresh-only，检测到同名 checkpoint、log、comparison 或 figure 时会拒绝覆盖；中断训练应使用 `scripts/train_v16.py --resume <run>.last.pt`，且除允许项外必须保持原实验参数和 output 一致。

训练入口的关键显式参数为：

```powershell
python scripts/train_v16.py `
  --epochs 104 --proposal-epochs 40 `
  --train-size 3000 --val-size 600 --synthetic-boundary-val-size 32 `
  --real-val-size 100 --real-fraction 0 `
  --min-control-points 8 --max-control-points 60 `
  --candidate-knots 56 --num-points 192 --batch-size 64 `
  --knot-min-span 0.01 --mse-tolerance 1e-4 `
  --certified-minimal-source --minimality-margin 0.2 `
  --minimality-audit-points 512 `
  --proposal-high-k-fraction 0.5 --proposal-high-k-min-knots 40 `
  --proposal-knot-assignment-weight 1 `
  --joint-supervision synthetic_ground_truth `
  --synthetic-count-role exact `
  --no-synthetic-geometry-oracle-teacher `
  --one-shot-selection-policy mass_topk `
  --one-shot-coverage-bins 0 --min-selected-knots 4 `
  --one-shot-safety-sigma 0 --one-shot-safety-knots 0 `
  --final-safety-sigma 0 --final-safety-knots 0 `
  --complexity-weight 0 `
  --real-manifest data/splits/uji_pen_v2.jsonl `
  --real-manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --real-manifest data/processed/usgs_contours/large_scale/manifest.jsonl `
  --device cuda `
  --output outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt
```

## 6. 训练后的串行产物

一条龙脚本随后执行：

1. `inspect_v16_checkpoint.py`；
2. `benchmark_v16_datasets.py --method-set published`；
3. `plot_v16_method_comparison.py --reference`；
4. `visualize_v16_real_deployments.py`。

得到 checkpoint、六方法×四数据集表、逐样本记录、input/reference 两张 2×2 图和真实六方法案例图。所有数值必须来自生成的 `comparison.json`/CSV；未运行时不得在报告中填写推测结果。
