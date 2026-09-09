# v16 部署、评估与可视化

v16 的部署路径固定为一次网络前向和一次标准 B 样条 refit。训练时的随机集合、反事实编辑、策略梯度和多次 refit 都不会进入部署。

## 1. 用户输入

输入是沿曲线方向排序的 CSV 或 TXT 点云，每行包含二维或三维坐标。脚本会：

1. 检查文件存在、维度一致且数值有限；
2. 按累计弦长重采样到模型要求的 M 个点；
3. 中心化并按尺度归一化；
4. 把归一化点云和 MSE 阈值送入网络。

无序点集不能直接使用；本项目没有在部署中求解点的拓扑顺序。

## 2. 固定深度部署

~~~text
normalized points + ε
  → forward_deployment() × 1
       encode geometry
       predict t0
       generate Kc candidates
       adaptive β + probability-mass Top-K → KeepMask
       update t1 and relocate selected knots
  → materialize U = internal_knots[KeepMask]
  → endpoint-constrained standard cubic B-spline refit × 1
  → denormalize curve and control points
~~~

KeepMask 使用一次性结构化规则：

\[
p_j=\sigma((s_j-\bar s)-\beta),\qquad
\hat K=\left\lceil\sum p_j+0.25\sqrt{\sum p_j(1-p_j)}+2\right\rceil,
\]

再按分数执行一次 Top-K。β由每条曲线的候选池、全局几何和误差阈值共同预测；这不是部署后的阈值扫描。部署不执行：

- CountHead；
- BIC；
- Hard-Concrete 采样；
- 逐节点删除；
- beam search；
- 根据最终 MSE 回补节点。

所以它给出统计意义的可行率，而不是逐曲线误差保证。未通过阈值的样本必须如实计入失败。

容量统一按内部节点计数。三次样条的完整节点向量还包含4个零和4个一，所以完整长度64等价于内部上限56；benchmark 可用 `--full-knot-vector-size 64` 显式采用这种记法。

## 3. checkpoint 正式资格

部署和论文对比入口默认执行统一资格审计；当前 `V16_FORMAL_PASS_RATE=0.90`。正式 checkpoint 需要：

| 条件 | 要求 |
|---|---|
| objective | candidate_selection_counterfactual_bspline_v16 |
| stage | joint |
| 配置的 proposal / deployment target | 均不低于 0.90 |
| 实测 worst-source deployment pass | 不低于 0.90 |
| proposal_ready | true |
| allow-infeasible-proposals | false |
| 元数据一致性 | 旧质量字段与实测指标不冲突 |

旧 `fast90` 联合训练 checkpoint 使用固定0.5离散化，不能被静默解释成新的动态策略。新的动态β模型必须输出到新文件。若只需要排查旧模型，可添加：

~~~text
--allow-unqualified-diagnostic
~~~

诊断模式必须在 JSON、Markdown 和 PNG 上显示 DIAGNOSTIC NOT FINAL，不能用于论文结论。新建 90% 实验可按工程协议取得正式资格；旧 97% checkpoint 不能通过 `--resume` 修改 target 后重新解释。

## 4. 点云部署命令

正式 checkpoint：

~~~powershell
python scripts/fit_v16_point_cloud.py --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt --point-cloud data/my_curve.csv --mse-tolerance 2.5e-5 --output-dir outputs/fits/v16/my_curve
~~~

当前 proposal 仅作诊断：

~~~powershell
python scripts/fit_v16_point_cloud.py --checkpoint outputs/checkpoints/candidate_selection_v16.proposal.pt --point-cloud data/my_curve.csv --mse-tolerance 2.5e-5 --allow-unqualified-diagnostic --output-dir outputs/fits/v16/diagnostic_my_curve
~~~

输出：

| 文件 | 内容 |
|---|---|
| fit.png | 输入点、拟合曲线、控制多边形和节点在曲线上的位置 |
| report.json | 参数、内部节点、控制顶点、MSE、计时、checkpoint 资格和输入信息 |

## 5. MSE 与通过率

默认指标为归一化点上的平均平方欧氏误差：

\[
\operatorname{MSE}
=\frac{1}{M}\sum_{i=0}^{M-1}
\lVert C(t_i)-q_i\rVert_2^2.
\]

它不取平方根。MSE≤2.5e-5 等价于 RMS≤0.005。

90% 验收条件是 `worst-source pass rate≥0.90`，其中每条曲线仍只有在 MSE≤2.5e-5 时才计为通过。降低数据集通过率要求不会改变 MSE 阈值，也不会修改失败曲线的误差。

真实数据还应报告 original-reference MSE：在原始密度参考折线上衡量重采样之外的形状保真度。训练和 checkpoint 选择当前只使用 M 个输入点的 MSE，original-reference MSE 是独立部署诊断。

通过率必须同时给出：

- 每个数据源单独的 pass rate；
- worst-source pass rate；
- 总体 pass rate。

正式验收以前两项中的 worst-source 为准，防止大量简单合成样本掩盖地理曲线失败。

## 6. 时间口径

报告保存两种 Ours 时间：

| 指标 | 范围 |
|---|---|
| network time | 一次 forward_deployment；排除数据搬运、refit、绘图和 I/O |
| full deployment time | 预处理后的网络前向、节点物化和一次最终 refit |

CUDA 测量在计时前后同步并先预热。论文速度比较应使用同一设备、dtype、batch size 和计时边界。若数值基线报告完整搜索时间，就应与 Ours 的 full deployment time 比较；纯 network time 只能标成网络延迟，不能解释成端到端加速比。

3090 通常会显著降低网络前向时间，但对 CPU 小矩阵 refit、数据读取和绘图帮助有限；最终数字仍应在目标硬件重新测量。

## 7. 八方法配对测试

~~~powershell
python scripts/benchmark_v16_datasets.py --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt --samples-per-knot-count 2 --real-samples-per-dataset 20 --mse-tolerance 2.5e-5 --max-internal-knots 96 --paper-initial-knots 96 --liang-dense-knots 96 --torch-num-threads 1 --output-dir outputs/comparisons/v16_K4_20_n2_real20
~~~

同一批曲线比较：

1. Ours v16；
2. Park–Lee 适配；
3. Liang 适配；
4. Dung–Tjahjowidodo 适配；
5. Kang 稀疏适配；
6. Luo–Kang–Yang 适配；
7. Yeh 特征 CDF 适配；
8. 从统一均匀最大节点初始化的贪心删除＋位置更新。

这些论文方法是根据公开目标重新实现的可审计 adaptation，不应称为作者官方代码。各方法容量、停止条件、MSE 和时间边界写入 comparison.json。完整协议见 [公开方法复现](published_knot_methods_reproduction.md)。

中断后可在完全相同实验指纹下增加 `--resume`。正式方法对比图只读取实测 JSON，不重新运行任何方法，也不得人工缩放、裁剪或替换 Ours 的误差：

~~~powershell
python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_K4_20_n2_real20/comparison.json `
  --output-dir outputs/figures/v16_simplified_certified_k96/method_comparison `
  --dpi 300
~~~

默认 `--method-set published` 比较 Ours、Park、Liang、Dung、Kang 和 Luo，输出 `v16_published_methods_input.png`。`--reference` 默认启用；若 JSON 含原始真实曲线参考指标，还会生成 `v16_published_methods_reference.png`，可用 `--no-reference` 关闭。增加 `--method-set all` 可把 Yeh 和统一贪心也放入同一张 2×2 图。四个子图依次是 MSE、阈值通过率、最终内部节点数和完整算法时间；Ours 的 network time 只作为独立文字注释，不能替代端到端时间柱。

## 8. Ours 拟合案例

只展示当前方法的论文案例时使用专用入口：

~~~powershell
python scripts/visualize_v16_ours_cases.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --real-samples-per-dataset 2 `
  --selection-seed 20260909 `
  --mse-tolerance 2.5e-5 `
  --device cuda `
  --dpi 300 `
  --output-dir outputs/figures/v16_simplified_certified_k96/ours_cases
~~~

每条留出测试曲线生成一个 `ours__<dataset>__<sample-id>.png`，同时生成 `ours_cases_overview.png` 和 `deployment_visualizations.json`。单例图包含：

- 原始参考折线（仅用于评价）和网络实际读取的采样点；
- Ours 最终部署 B 样条；
- 控制多边形、带索引的控制顶点 `P_i`；
- 曲线上的内部节点 `C(u_i)`、带索引的节点位置和完整参数域节点条；
- 最终内部节点数、输入点 MSE、参考点 MSE、是否通过阈值、network time 与 network+refit 时间。

总览图用于快速展示多条曲线的拟合形态，精确节点值、完整节点向量和控制顶点坐标以 JSON 为准。脚本默认使用 UJI、Natural Earth 和 USGS 的独立 test split；可重复传入 `--manifest NAME=PATH` 指定其他已准备的真实数据 manifest。

## 9. 真实曲线四方法诊断图

几何可视化故意只展示 Ours、Kang、Yeh 和统一贪心四种方法，以保持版面可读；这不等于定量 benchmark 只有四个方法。

~~~powershell
python scripts/visualize_v16_real_deployments.py --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt --real-samples-per-dataset 2 --selection-seed 20260909 --mse-tolerance 2.5e-5 --device auto --output-dir outputs/figures/v16_simplified_certified_k96/four_method_cases
~~~

每幅图包含原始参考折线、共同输入点、最终曲线、控制顶点和节点位置；标题给出输入 MSE、参考 MSE、节点数、network time 与完整方法时间。

## 10. 正式与诊断产物边界

`candidate_selection_v16_simplified_certified_k96.pt` 是当前正式训练的目标路径，不是文件名本身即可证明合格的随仓库结果。运行上述两种绘图前必须先执行 `inspect_v16_checkpoint.py` 并得到返回码 0；当前若尚未训练出合格主 `.pt`，就不能声称已经生成正式比较图或正式案例图。

为排查旧权重，可在两个绘图入口增加 `--allow-unqualified-diagnostic`。此时输入 benchmark JSON 本身也必须是允许诊断生成的报告；所有 PNG 和 JSON 会保留 `DIAGNOSTIC NOT FINAL` 标记，不得进入正式论文表格或结论。

## 11. 结果发布检查

1. 先检查 checkpoint qualification，不把 proposal 或 target_not_met 权重当正式模型；
2. 固定独立 test split、seed、manifest 和 checkpoint SHA-256；
3. 对所有方法使用同一输入、参数归一化和 MSE 定义；
4. 同时报告平均、P95、最大 MSE、通过率和节点数；
5. 区分 network time 与 full method time；
6. 失败样本不得删除，诊断回退不得伪装成一次性网络结果；
7. PNG 必须由保存的逐样本 JSON 重绘。

代码入口：[fit_v16_point_cloud.py](../scripts/fit_v16_point_cloud.py)、[benchmark_v16_datasets.py](../scripts/benchmark_v16_datasets.py)、[plot_v16_method_comparison.py](../scripts/plot_v16_method_comparison.py)、[visualize_v16_ours_cases.py](../scripts/visualize_v16_ours_cases.py)、[visualize_v16_real_deployments.py](../scripts/visualize_v16_real_deployments.py)。
