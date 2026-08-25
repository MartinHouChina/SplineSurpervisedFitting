# v10 部署与指标

## 1. 输入

用户提供沿曲线方向有序的二维或三维点云：

```text
ordered points [M,D], D∈{2,3}
```

不需要初始节点、控制点、Boehm 插入结果或真实节点数。点云先按训练规则归一化；只有网络输入会重采样到 checkpoint 记录的长度，最后的标准 B 样条 refit 使用全部原始点。

## 2. 默认部署流程

```text
有序点云
  → 一次网络 forward
      ParameterHead
      CandidateKnotHead
      全候选截断幂 pilot solve
      proposal-only 固定位置精修
      position-aware selector adapter
      final keep / β / p
      probability-mass Top-K + uncertainty reserve + coverage anchors
      final KeepMask 截断幂代理 solve
  → 将参数插值回全部原始点
  → 选出 final KeepMask=True 的 refined knots
  → 一次端点约束的标准开放三次 B 样条 refit
  → 恢复原坐标系
```

部署契约：

```text
network_forward_passes = 1
forward_internal_surrogate_solves = 2
standard_bspline_refits = 1
hard_rms_pruning_at_deployment = False
```

两次内部 solve 使用截断幂代理：第一次提取结构贡献，第二次生成网络代理重建。它们不是两次网络前向，也不计作最终标准 B 样条 refit。

## 3. 最终节点决策

最终反馈 token 产生：

\[
p_j=\sigma(r_j-\beta),\qquad
\widehat K=\left\lceil\sum_jp_j+s\sqrt{\sum_jp_j(1-p_j)}\right\rceil.
\]

`β` 是每条曲线的 raw-importance logit 阈值，`s` 是 Bernoulli 不确定性安全余量。最终从
raw importance 中一次性选择最高的 `K̂` 个候选；若启用 coverage bins，每个非空参数区间先
保留一个最高概率锚点，再按全局分数填满剩余名额。最终内部节点数为：

\[
\sum_j m_j=\widehat K.
\]

没有 CountHead、BIC、Hard-Concrete，也不会调用拟合器遍历节点子集。Top-K 和覆盖锚点只构造
一次离散掩码，随后仍只有一次标准 B 样条 refit。

节点位置在教师生成前已经确定，部署 KeepMask 不会再次移动节点。所有节点均被删除时，最终可得到无内部节点的三次多项式 B 样条。

## 4. 标准 B 样条 refit

LearnedKeep 只决定内部节点，最终控制点重新求解一次。开放三次 B 样条固定：

\[
P_0=Q_0,\qquad P_{n-1}=Q_{M-1}.
\]

因此最终曲线覆盖输入首尾点。ridge 与平滑项只作用于未知内部控制点，固定端点贡献会移到最小二乘右端。截断幂代理系数不会作为最终控制点导出。

## 5. 阈值保证的边界

`fit_tolerance=ε` 用于：

- 构造离线 Hard-RMS 教师；
- 训练截断幂代理的阈值违约损失；
- 验证和独立测试的标准 B 样条部署满足率；
- 可选 Hard-RMS 诊断。

默认部署不重新运行 Hard-RMS 搜索，所以它提供的是统计保证：

> 在独立测试分布上，报告一次性预测经过一次标准 B 样条 refit 后满足 `RMS≤ε` 的比例。

它不是每条新输入的硬保证。若业务必须逐条满足阈值，需要额外执行 Hard-RMS 检查或回退；这会产生多次标准 B 样条 refit，不再属于默认 one-shot 部署。

## 6. Checkpoint 评估

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --num-samples 2000 `
  --batch-size 32 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/candidate_pruning_one_shot_v10_evaluation.json
```

核心指标分三组：

### 候选能力

- true→candidate recall@0.005/0.01/0.02；
- nearest candidate MAE；
- 全候选标准 B 样条 RMS。

若全候选拟合已超阈值，应先改进参数化和候选覆盖，而不是调 KeepMask。

### 最终结构

- 相对独立测试集 canonical 诊断标签的节点数 accuracy / MAE；
- final keep probability 与 adaptive `β`；
- refined knot match precision / recall / F1 和 matched MAE。

teacher-mask F1 属于带离线缓存的训练/验证日志；普通独立评估不把训练教师标签重新生成一遍。

### 真实部署

- 一次标准 B 样条 refit RMS mean / P95 / max；
- `RMS≤ε` 满足率；
- 最终节点数 mean / min / max；
- 端点误差和控制点数。

训练 checkpoint 选优同样真实执行该一次性标准 B 样条 deployment pass。未达到目标通过率
时按通过率及 mean/P95 RMS 改善；达到目标后优先最少节点，再用真实 fit 与节点 recall/F1
作 tie-break。截断幂 surrogate loss 不决定正式 checkpoint。

## 7. 可选 Hard-RMS 诊断

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --num-samples 128 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --run-hard-diagnostic `
  --json-output outputs/candidate_pruning_one_shot_v10_hard_diagnostic.json
```

该选项从候选集运行昂贵的逐节点 Hard-RMS 搜索，只用于衡量 one-shot 学生与离线教师的差距，不替换默认部署结果。

## 8. 可视化

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --seed 20000 `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view learned `
  --dpi 600 `
  --output outputs/candidate_pruning_one_shot_v10_sample_000.png
```

`--pruning-view`：

- `all`：全部候选的标准 B 样条 refit；
- `learned`：final LearnedKeep 子集；
- `hard`：离线 Hard-RMS 对照；
- `comparison`：2×2 同尺度显示原始/source 样条、全部冗余候选、LearnedKeep 和 Hard-RMS。

四个 comparison 面板都包含曲线、采样点、控制顶点/控制多边形、内部节点位置、完整开放
节点向量和 RMS。时间为 CPU wall time；冗余、Learned 和 Hard 面板报告
`network forward + refit/prune`，使用 `--timing-repeats 3` 可取三次中位数。
合成数据集会额外保留生成采样点之前的 source 控制顶点和完整节点向量；因此第一幅图不是
经过 RMS canonicalization 后的监督标签。旧 checkpoint 配置缺少这些字段时才回退到 canonical 标签。

对照图：

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --seed 20000 `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view comparison `
  --timing-repeats 3 `
  --dpi 600 `
  --output outputs/candidate_pruning_one_shot_v10_comparison_000.png
```

## 9. 用户点云推演

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_one_shot_v10.pt `
  --point-cloud data/my_curve.csv `
  --fit-tolerance 0.005 `
  --json-output outputs/my_curve_fit.json `
  --figure-output outputs/my_curve_fit.png
```

支持 `.csv`、`.txt`、`.xyz`、`.json`、`.npy`、`.pt` 和 `.pth`。CSV 每行一个 `x,y` 或 `x,y,z`；点序相反时添加 `--reverse-points`。

## 10. v7 兼容路径

`structure_mode=candidate_pruning` 的 v7 checkpoint 仍使用历史逐节点 Hard-RMS 部署。v8/v9/v10
的 `structure_mode=candidate_pruning_one_shot` 都使用“一次 forward + 一次标准 refit”；v9 固定
proposal 几何，v10 在此基础上增加结构化集合监督和概率质量 Top-K。
