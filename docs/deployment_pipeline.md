# v16 部署、评估与可视化

当前主协议为 `Kc=56`、合成源内部节点 `K=4..56`（控制顶点 8～60）、
`MSE<=1e-4`、worst-source deployment pass `>=90%`。部署固定为一次网络
前向和一次标准 B 样条 refit；训练教师的组合搜索不进入部署。
合成训练/测试显式使用 `knot_min_span=0.01`，以支持 K=56 的 57 个 span。

## 1. 用户输入

输入是沿曲线方向排序的 CSV 或 TXT 点云，每行包含二维或三维坐标。脚本会：

1. 检查文件存在、维度一致且数值有限；
2. 按累计弦长重采样到模型要求的 192 个点；
3. 中心化并按尺度归一化；
4. 把归一化点云和 MSE 阈值送入网络。

无序点集不能直接使用；本项目不会在部署时求解点的拓扑顺序。

## 2. 固定深度部署

```text
normalized ordered points + epsilon
  -> forward_deployment() x 1
       GeometryEncoder
       ParameterHead: t0
       CandidateKnotHead: 56 ordered candidates
       adaptive beta + probability-mass Top-K: KeepMask
       selected-only parameter feedback and survivor relocation: t1, Udeploy
  -> endpoint-constrained cubic B-spline refit x 1
  -> fitted curve + control vertices + internal knots
```

KeepMask 使用曲线级动态阈值与概率质量：

\[
p_j=\sigma((s_j-\bar s)-\beta),\qquad
\hat K=\left\lceil\sum_jp_j+
\sigma_s\sqrt{\sum_jp_j(1-p_j)}+K_s\right\rceil.
\]

当前训练把安全储备从 `(sigma_s,K_s)=(0.20,2)` 退火到 `(0.03,0)`，部署
使用 checkpoint 记录的最终状态。筛选只执行一次 Top-K，不扫描阈值，也不按
最终拟合误差逐曲线回补。节点数是 `KeepMask.sum()`。

当前主模型还使用：Joint 初始保留比例 `30/56≈0.535714`（初始概率质量约
30，不是部署最终 K）、低计数教师逐一扫描
`K=4..16`、`synthetic-count-role=upper_bound`、合成 geometry-oracle 教师和
`one-shot-coverage-bins=0`。这些都是训练设置；不会增加部署分支。

部署没有 CountHead、BIC、Hard-Concrete、逐节点删除、beam search 或教师
搜索，因此给出统计意义的通过率，不承诺每条曲线必然满足阈值。

## 3. checkpoint 正式资格

正式入口会审计：

| 条件 | 当前要求 |
|---|---|
| objective | `candidate_selection_counterfactual_bspline_v16` |
| stage | joint |
| 候选容量 | `Kc=56` |
| 配置阈值 | `MSE=1e-4` |
| proposal/deployment target | 均不低于 0.90 |
| 实测 worst-source deployment pass | 不低于 0.90 |
| proposal_ready | true |
| allow-infeasible-proposals | false |

训练后先执行：

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --required-pass-rate 0.90 `
  --mse-tolerance 1e-4
```

返回码 0 才能用于正式表格和图片。排错时可以显式添加
`--allow-unqualified-diagnostic`；此时 JSON、Markdown 和 PNG 会标记
`DIAGNOSTIC NOT FINAL`，不得作为论文结果。

## 4. 点云部署

```powershell
python scripts/fit_v16_point_cloud.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --point-cloud data/my_curve.csv `
  --mse-tolerance 1e-4 `
  --output-dir outputs/fits/v16_mse1e-4/my_curve
```

输出：

| 文件 | 内容 |
|---|---|
| `fit.png` | 输入点、拟合曲线、控制多边形和曲线上的节点位置 |
| `report.json` | 参数、内部节点、控制顶点、MSE、计时、资格和输入信息 |

## 5. MSE、通过率和节点数

主误差是归一化点上的平均平方欧氏误差：

\[
\operatorname{MSE}=
\frac{1}{M}\sum_{i=0}^{M-1}\lVert C(t_i)-q_i\rVert_2^2.
\]

它不取平方根；`MSE<=1e-4` 等价于 `RMS<=0.01`。正式资格要求每个验证来源
分别统计通过率，且最差来源不低于 90%。总体通过率不能替代 worst-source。

真实数据还应报告 original-reference MSE，用原始密度参考折线检查 192 点
重采样之外的形状保真度。真实曲线没有节点真值，不报告节点 precision/recall。

`Kc=56` 只表示最多 56 个内部候选。三次开放样条的完整节点向量还包含 4 个
零和 4 个一，因此全保留时完整节点向量长度为 64，控制顶点数为 60。

## 6. 时间口径

| 指标 | 范围 |
|---|---|
| network time | 一次 `forward_deployment`；不含数据搬运、refit、绘图和 I/O |
| full deployment time | 网络前向、节点物化和一次最终 refit |
| baseline total time | 数值方法从参数化/初始化到最终统一 refit 的完整时间 |

CUDA 计时前后同步并先预热。RTX 3090 会加速 Ours 的训练和网络前向，但
Park、Liang、Dung、Kang、Luo 及 CPU `float64` refit 主要仍由 CPU 决定。
论文速度比较使用 full deployment time 对 baseline total time；network time
只作为网络延迟单列。

## 7. 六方法四指标正式比较

当前主表固定六种方法：Ours、Park–Lee、Liang、Dung–Tjahjowidodo、Kang、
Luo–Kang–Yang。所有方法处理同一批曲线、使用 56 个内部节点容量上限并交给
同一标准 refit。四项主指标为 MSE、阈值通过率、最终内部节点数和完整方法
时间。

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --output-dir outputs/comparisons/v16_mse1e-4_k56_six_methods `
  --method-set published `
  --samples-per-knot-count 5 `
  --min-knot-count 4 --max-knot-count 56 `
  --real-samples-per-dataset 20 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 `
  --max-internal-knots 56 --gradient-steps 12 `
  --paper-initial-knots 56 --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 --paper-relocation-iterations 12 `
  --liang-dense-knots 56 --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 `
  --torch-num-threads 4 --device cuda
```

中断后只能在实验指纹完全相同时追加 `--resume`。公开方法是按论文目标实现的
可审计 adaptation，不是作者官方代码；复现边界见
[公开方法复现](published_knot_methods_reproduction.md)。

绘制同一报告的 2×2 四指标图：

```powershell
python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_mse1e-4_k56_six_methods/comparison.json `
  --output-dir outputs/figures/v16_mse1e-4_k56_six_methods/metrics `
  --method-set published --reference --dpi 300
```

图片只读取实测 JSON，不重跑方法，也不得人工缩放或替换某个方法的误差。

合成 `source K=56` 与网络 `Kc=56` 恰好同时到达容量上限，但含义不同。该层
没有冗余候选余量，报告必须单列其 dense pass 与 deployment pass；任一项失败
都保留为失败，不能通过降低或平均化 90% worst-source 正式门槛处理。

## 8. 真实曲线六方法可视化

每条留出真实曲线生成一张 3×2 图，六个面板共享相同原始参考折线和输入点，
并显示最终拟合曲线、控制多边形、控制顶点、曲线上的内部节点、MSE、
PASS/FAIL、最终 K 和完整方法时间；Ours 额外标注 network time。

```powershell
python scripts/visualize_v16_real_deployments.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --output-dir outputs/figures/v16_mse1e-4_k56_six_methods/real_cases `
  --real-samples-per-dataset 2 --selection-seed 20260910 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 `
  --max-internal-knots 56 --gradient-steps 12 `
  --paper-initial-knots 56 --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 --paper-relocation-iterations 12 `
  --liang-dense-knots 56 --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 `
  --torch-num-threads 4 --device cuda --dpi 300
```

完整节点向量和控制顶点坐标写入 `deployment_visualizations.json`。失败样本必须
保留；不能只挑通过样本展示。

## 9. 一键 RTX 3090 流水线

以下命令按同一协议串行执行新训练、资格检查、六方法合成/真实数据比较、
四指标绘图和真实曲线六方法可视化：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_mse1e-4_3090.ps1
```

脚本默认拒绝覆盖同名产物，也不自动 resume。若 checkpoint 未通过
90% / `1e-4` 正式门槛，默认在资格检查处停止，不生成可误用的正式结果。

## 10. 历史消融边界

旧 `Kc=96 / MSE=2.5e-5 / 97%`、旧 K64、旧固定 0.5 KeepMask 和旧八方法图只可
明确标作历史消融或诊断。它们不是当前主协议，不能与本轮结果混表。Yeh 和
统一贪心仍可用 `--method-set all` 做附加控制，但当前正式六方法主表不包含
它们。

代码入口：[fit_v16_point_cloud.py](../scripts/fit_v16_point_cloud.py)、
[benchmark_v16_datasets.py](../scripts/benchmark_v16_datasets.py)、
[plot_v16_method_comparison.py](../scripts/plot_v16_method_comparison.py)、
[visualize_v16_real_deployments.py](../scripts/visualize_v16_real_deployments.py)。
