# 真实数据训练适配与部署测试

## 当前接入范围

三类真实数据现在共用 `manifest.jsonl + curves/*.npy`，并可由
`RealWorldCurveDataset` 读取：

| 数据源 | 曲线 | 划分单位 | 节点标签 | 当前用途 |
|---|---|---|---|---|
| UJI Pen v2 | 18,219 个连续 pen-down stroke | writer | 无 | 无标签反馈头适配、外部测试 |
| Natural Earth 10m | 3,908 条海岸线窗口 | 666 个地理 tile | 无 | 无标签反馈头适配、外部测试 |
| USGS TNM | 820 条 1:24,000 等高线窗口 | 12 个地理 tile | 无 | 无标签反馈头适配、外部测试 |

这三类数据都不能把折线顶点当作 B 样条节点真值。因此它们不参与 proposal
coverage、KeepMask 真值或 knot Precision/Recall/F1 监督。当前允许的真实数据训练仅是：
冻结合成数据学到的 proposal、selector 和 survivor relocation，只用几何拟合、阈值违约、
弦长先验和 identity 约束校准 v14 `ParameterFeedbackHead`。

## 1. 数据准备

```powershell
# UJI：保留官方 writer-disjoint test，并从训练 writer 中划 val
python scripts/prepare_uji_pen.py --download

# Natural Earth：固定 v5.1.2 URL 和文件哈希
python scripts/prepare_natural_earth.py `
  --resolution 10m `
  --layer coastline `
  --reference-points 768

# USGS：示例区域；原始服务响应会缓存
python scripts/prepare_usgs_contours.py `
  --bbox-file configs/usgs_contour_regions.example.json `
  --max-features-per-region 2000 `
  --reference-points 768 `
  --output-dir data/processed/usgs_contours/large_scale
```

不要用只包含一个 tile 的 USGS smoke 数据做 train/val/test；整个 tile 只能进入一个 split。
正式实验至少准备多个互不重叠区域。

## 2. 不做真实训练，直接测试合成模型的域外泛化

UJI test：

```powershell
python scripts/evaluate_real_world.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --manifest data/splits/uji_pen_v2.jsonl `
  --split test `
  --max-samples 1000 `
  --batch-size 16 `
  --json-output outputs/real_world/uji_v14_learned_1000.json `
  --overwrite
```

Natural Earth 与 USGS 可在一次运行中评估；`--manifest` 可以重复：

```powershell
python scripts/evaluate_real_world.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest data/processed/usgs_contours/large_scale/manifest.jsonl `
  --split test `
  --max-samples 2000 `
  --batch-size 16 `
  --json-output outputs/real_world/geospatial_v14_learned.json `
  --overwrite
```

JSON/CSV 同时记录每条曲线和按数据源汇总的：

- 原始密度参考点上的 normalized mean squared Euclidean error；
- RMS、P95/最大点距离、对称 Chamfer MSE 和 Hausdorff；
- 阈值通过率和最终内部节点数；
- 只含 `forward_deployment()` 的网络时间。

控制点只在 192 点网络输入网格上求解，高密度参考点不参与最小二乘，因此
`reference_mse` 不是把同一批评价点重新拟合后的训练误差。标准 B 样条 refit、文件读取、
预处理和指标计算均不计入 network-only 时间。

## 3. 效果优先的 reference-certified 部署

纯网络在真实域通过率低时，可启用独立的慢速质量路径：

```powershell
python scripts/evaluate_real_world.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --manifest data/splits/uji_pen_v2.jsonl `
  --split test `
  --max-samples 1000 `
  --batch-size 16 `
  --deployment-mode certified `
  --certified-mse-target 1e-5 `
  --certified-max-knots 64 `
  --json-output outputs/real_world/uji_v14_certified_1000.json `
  --overwrite
```

这里的阈值是 **mean squared Euclidean MSE**，不是 RMS：

\[
\operatorname{MSE}_{ref}=\frac{1}{M}\sum_{i=1}^{M}
\lVert C(t_i)-Q_i\rVert_2^2\le 10^{-5}.
\]

Certified 路径固定使用 normalized reference 点和弦长参数，在 CPU float64 中依次执行：

1. 保存纯 `forward_deployment()` 的 KeepMask、节点、reference MSE 和 network-only 时间；
2. 用网络存活节点在全部 reference 点上重新求控制点；
3. 对全部存活节点做联合坐标重定位；
4. 若 reference 点数允许，在 `--certified-max-knots` 内尝试标准 averaging
   interpolation knot vector；
5. 若仍失败，同时评估高残差位置、未选网络 proposal 和空区间中点，逐个增加节点；
6. 每增加若干节点后再次联合移动全部节点，而不是只移动新节点；
7. 在参数域递归加入当前线性插值误差最大的 reference 点，直到整条折线的实测 MSE
   达标；再逐点尝试删除冗余折线顶点。将简化折线写成 C0 复合三次 Bézier；
8. 只有第 7 步未通过时，才把完整输入折线逐段精确写成 C0 复合三次 Bézier：
   每个内部弦长断点重复 3 次，
   每条线段控制点为 `Pi, Pi+Δ/3, Pi+2Δ/3, Pi+1`。

第 7 步名为 `tolerance_polyline_fallback`。若保留 `V` 个折线顶点，它产生
`3(V-2)` 个内部节点和 `3(V-1)+1` 个控制点；停止和压缩条件均为全部 reference 点上的
mean squared Euclidean MSE。它通常远小于完整折线，但仍允许超过 adaptive knot 上限，
且不保证节点数全局最少。可分别用以下参数关闭该层或关闭其贪心压缩：

```powershell
--no-certified-tolerance-polyline-fallback
--no-certified-tolerance-polyline-compact
```

第 8 步名为 `polyline_exact_fallback`。它能精确表示**提供的输入折线**，但可能产生
`3(M-2)` 个内部节点和 `3(M-1)+1` 个控制点，并允许超过第 4–6 步的 adaptive knot
上限。它不是节点简化、不是网络预测，也不保证采样点之间潜在的物理真曲线。若实验要求
严格限制节点数，应关闭它：

```powershell
--no-certified-polyline-exact-fallback
```

此时搜索未达到目标会写入 `certified_status=not_reached_within_limits`、
`reference_certificate_valid=false` 和具体 `certified_failure_reason`，不会伪装成通过。
`certificate_valid=true` 只表示最终曲线在本条 manifest 保存的 normalized reference 点上
经独立复算满足 MSE，并且普通 least-squares 路径满列秩；构造式 polyline fallback 不涉及
least-squares rank。它不证明全局最少节点数，也不证明未知连续曲线误差上界。

JSON/CSV 同时保留两套字段：

- `learned_*`：纯网络节点和原部署 reference MSE；
- `normalized_reference_mse`、`retained_internal_knots`：repair 后最终结果；
- `network_amortized_time_ms`：只含网络；
- `certified_postprocess_time_ms`：reference refit、增结、联合重定位和兜底，单独计时；
- `certified_status`、`reference_certificate_valid`、rank、refit 次数、节点重数和控制点数；
- `certified_tolerance_polyline_*`：该兜底是否触发、保留折线顶点数、实测 MSE、插入与
  压缩次数，以及是否超过 adaptive knot 上限。

20 条固定随机 UJI test smoke（仅检查链路，不是论文统计）中，纯网络平均 reference MSE
为 `1.2788e-4`、平均 K 为 `15.35`；repair 后平均 MSE 为 `3.9621e-6`、20/20 实测
通过、平均 K 为 `21.95`，GTX 1070 主机的 CPU repair 平均约 `479 ms/curve`。其中没有
触发 polyline fallback。5 条 Natural Earth 连通性 smoke 中，第 7 步使全部样本达到
`1e-5`，平均 K 为 `109.2`、CPU repair 约 `216 ms/curve`；此前完整折线兜底的极端样本
会达到 K=`2298`。这些都不是论文统计，正式表格必须单列两级 fallback 比例、控制点数、
节点数和 repair 时间，不能只展示 100% 通过率。

## 4. 可选的无标签真实数据适配

可重复提供 manifest，将多个真实训练 split 混合；`--train-size/--val-size` 是确定性上限：

```powershell
python scripts/finetune_parameter_feedback.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --real-world-manifest data/splits/uji_pen_v2.jsonl `
  --epochs 10 `
  --train-size 4000 `
  --val-size 1000 `
  --batch-size 16 `
  --output outputs/checkpoints/current/v14_uji_feedback_adapted.pt
```

若输入 checkpoint 已经启用 v14 feedback，应从对应的 v12/v13 结构 checkpoint 开始；
现有脚本的默认职责是“挂载并校准一个新 feedback head”。真实适配 checkpoint 仍需在未参与
训练的 test writer/tile 上用上一节命令独立评价。

## 5. 本次中断训练的恢复

原运行已经完成结构训练，`*_calibrated.pt` 只有权重、没有完整配置。以下命令用
`*_proposal.pt` 补齐模型/数据元数据，保留 calibrated 的 selector/relocation 权重，丢弃尚未
训练的 feedback 状态，并只运行反馈校准：

```powershell
python scripts/finetune_parameter_feedback.py `
  --checkpoint outputs/candidate_pruning_one_shot_v14_calibrated.pt `
  --metadata-checkpoint outputs/candidate_pruning_one_shot_v14_proposal.pt `
  --epochs 10 `
  --train-size 4000 `
  --val-size 1000 `
  --batch-size 16 `
  --learning-rate 1e-4 `
  --feedback-max-logit-shift 0.5 `
  --initial-chord-blend 0.6 `
  --output outputs/candidate_pruning_one_shot_v14.pt `
  --overwrite
```

这条恢复路径不读取离线 Hard-RMS teacher，也不会重训前 100 个结构 epoch。
修复后的主训练脚本会给新生成的 `*_distill.pt` 和 `*_calibrated.pt` 同步写入完整
model/dataset/deployment 元数据；`--metadata-checkpoint` 只用于恢复这次修复前生成的旧中间文件。

## 6. 已完成的连通性结果

这些数值只用于验证数据链路，不作为论文最终表格：

| 数据 | 样本 | normalized reference MSE | RMS≤0.005 | 平均 K | network-only |
|---|---:|---:|---:|---:|---:|
| UJI 官方 test 随机子集 | 100 | `1.4983e-4` | 16.0% | 13.68 | 1.654 ms/curve |
| Natural Earth 10m test 随机子集 | 100 | `6.4670e-4` | 1.0% | 17.56 | 1.514 ms/curve |
| USGS 三地区 test 随机子集 | 100 | `7.4929e-4` | 33.0% | 18.42 | 1.730 ms/curve |

计时来自当前 GTX 1070、batch size 16（地理 smoke 实际 batch 10），不能直接外推为 RTX
3090 的 batch-size-1 延迟。低通过率是真实的 synthetic-to-real domain gap，不能为了展示而
修改误差；下一步应先做无标签 feedback 适配并在严格隔离的 test group 上复测。

### `MSE <= 1e-5` 认证回归

以下是固定随机种子、每个 test split 100 条的实现回归；checkpoint 仍是旧 v14，因此这些
数字验证的是 certified 安全层，不代表尚未完整训练的 `v14_joint` 网络精度。为缩短回归，
adaptive 阶段使用 `--certified-max-knots 28 --certified-position-sweeps 0
--certified-insertion-candidates 4`，之后允许 tolerance-polyline fallback：

| 数据 | 最大逐曲线 MSE | certificate 通过 | 最终平均 K | tolerance-polyline | exact-polyline | CPU repair/curve |
|---|---:|---:|---:|---:|---:|---:|
| UJI test | `9.99825e-6` | 100/100 | 25.49 | 18% | 0% | 55.9 ms |
| Natural Earth test | `9.98675e-6` | 100/100 | 81.49 | 86% | 0% | 276.0 ms |
| USGS test | `9.99673e-6` | 100/100 | 123.74 | 56% | 0% | 244.0 ms |

对应报告为 `outputs/real_world/v14_certified_{uji,natural_earth,usgs}_100.json`。
这里 100% 的含义仅是“保存的输入 reference 折线逐点复算通过”；较高的 fallback 比例同时
说明纯网络的跨域误差尚未达到该阈值，正式实验必须连同 fallback 比例和最终节点数一起报告。
