# v16 细粒度合成监督

## 1. 定位

当前正式 v16 仍是 `joint_supervision=synthetic_ground_truth`。训练只使用 certified Synthetic，外部数据只做 validation/test；`online_teacher=false`，不执行 ranked-prefix、counterfactual、oracle subset search，也不使用 Teacher cache。

这里的细粒度 teacher 是**合成最简性证书附带的逐节点标签**：认证 source 节点集时，保存删除每个真节点并重新拟合后的 MSE。它不依赖当前 Selector 分数，因而不是 self-teacher。

## 2. 标签如何流动

```text
clean source (t*, U*, K*)
  -> full refit + every single-knot-deletion refit
  -> source-subset minimality certificate
  -> D*[r] = MSE after deleting true knot r
  -> v16_mixed padding: target_single_deletion_mse/mask/valid
  -> ordered one-to-one assignment U_prop <-> U*
  -> D* and binary Keep label move to the same candidate slot
```

风险值为

\[
r_j=\sigma\left(\log((D_j^*+\delta)/\epsilon)/T\right)y_j,
\]

其中 `epsilon` 是 MSE 阈值，当前 `T=0.5`。风险越高，删除该正候选后越可能严重越过阈值；它会加强正槽位 Keep 损失和正负 ranking，但不会替代二值标签或改变真 K。

## 3. 四组细化

| 目标 | 当前实现 | 主要诊断 |
|---|---|---|
| 候选召回 | directed coverage + ordered assignment + `0.0025/0.005/0.010` 多尺度距离和最差 20% 尾部项 | `proposal_recall_at_005/.010/.020`、assignment MAE |
| 节点筛选 | 平衡 BCE + Dice + ordered CDF；靠近真节点的未匹配候选作为 fuzzy negative 降权 | Keep P/R/F1、fuzzy-negative fraction |
| 关键节点保护 | certificate single-deletion MSE 转为连续风险并加权 Keep/ranking | mean risk、log delete margin、critical false-delete rate |
| 参数偏差 | 逐点真参数损失 + interval log-gap + 整曲线 signed bias；Joint warp 梯度缩放为 `0.1` | parameter gap loss、parameter bias MAE |

fuzzy negative 只降低负类 BCE 权重，不把多余候选标成正类。Dice 约束整体集合重叠；CDF 按候选位置排序后约束累计概率质量，减少 Keep 概率集中在参数域局部的问题。

普通 ranking 与细粒度 teacher ranking 使用同一 `negative_confidence`，因此近真节点的可替代负候选不会在 BCE 中被放宽、又在排序项中被全强度压低。正式 direct-label 模式固定 `one_shot_coverage_bins=0`；分箱锚点属于 legacy 部署约束，可能在固定真 K 下用负槽位挤掉正槽位，与精确 KeepMask 标签不相容。

warp-gradient scale 采用 `stopgrad(t)+alpha*(t-stopgrad(t))`：forward 值不变，仅缩放节点位置损失回到 ParameterHead 的梯度。当前 Proposal `alpha=0`，Joint `alpha=0.1`。

## 4. 运行

Linux：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_fine_teacher_linux
```

PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -PrepareRealData `
  -Device cuda `
  -RunName candidate_selection_v16_fine_teacher_windows
```

两个 runner 当前都显式传入：

```text
proposal_multiscale_recall_weight=0.25
keep_dice_weight=0.5
keep_cdf_weight=0.25
fine_teacher_weight=0.5
fine_teacher_ranking_weight=0.25
fine_teacher_temperature=0.5
keep_fuzzy_negative_radius=0.01
keep_fuzzy_negative_floor=0.1
parameter_gap_weight=0.05
parameter_bias_weight=0.1
proposal_parameter_warp_gradient_scale=0
joint_parameter_warp_gradient_scale=0.1
```

这些是当前实验起点，不是已验证的最优权重。必须使用新的 run name 重训；旧 checkpoint 不会因代码升级而自动具备新效果。

## 5. 计算与结论边界

- single-deletion refit 在合成样本认证/生成阶段完成；loss forward 复用标签，额外 teacher spline solve 数为 0。
- 生成认证样本仍有 CPU 单删审计成本；“loss 内无额外 solve”不等于数据生成免费。
- 部署不读取 `D*`、真节点、真 K 或真参数，仍是一次 network forward、一次 mass-TopK 和一次最终标准 refit。
- 证书只证明固定真参数化、原 source 节点所有严格子集在阈值下不可行，不证明允许自由重定位后的连续全局最少 K。
- 是否真正提升候选 recall、Keep 准确率、参数偏差和最终 MSE/通过率，必须由重训后的独立 Synthetic 与外部数据 benchmark 确认；本文不预设改善幅度。
