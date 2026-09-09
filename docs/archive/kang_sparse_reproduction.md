# 公开节点算法复现与公平对比说明

v15 多数据集比较使用新入口 `scripts/benchmark_v15_datasets.py`，同时记录 Ours 完整部署时间和纯网络时间。协议与运行指令见 [v15 多数据集比较](v15_multidataset_benchmark.md)。下文第 4–6 节保留 v14 历史实验口径及记录，不应与 v15 的完整耗时混用。

## 1. 当前纳入的方法

| 方法 | 核心机制 | 当前实现定位 | 网络 |
|---|---|---|---:|
| Ours | 一次性候选生成、KeepMask 与存活节点重定位 | checkpoint 的 `forward_deployment()` | 是 |
| Kang et al. 2015 | 导数跳跃稀疏化，再合并并重定位活跃节点 | 标量 CVX 原问题的二维 group-L1 ADMM 适配 | 否 |
| Yeh et al. 2020 | 高阶有限差分特征，按累计特征等质量放置节点 | 论文位置公式 + 明示的节点数升序扫描 | 否 |
| Uniform-Kmax gradient pruning | 最大节点集起步，删除并梯度更新剩余位置 | 非网络数值基线 | 否 |

Yeh 实现位于 `src/spline_fitting/evaluation/feature_cdf_knot_placement.py`。它原生支持二维有序点，补充了 Kang 原文只验证标量函数的局限。

另有两类相关工作可后续扩展：Kovács–Fekete 的 FOBA 初值加 variable projection 自由节点优化，以及 Yagishita–Gotoh 的 exact-penalty/GIST 节点选择。前者主要面向一维时间序列，后者需要给定节点数上界；两者尚未混入本轮数值结果。

## 2. Kang 复核与修正

Kang 原文求解标量三次 B 样条函数，本项目输入是二维参数曲线，所以这里只能称为二维适配：

1. 标量跳跃 L1 改为二维跳跃向量 group-L1；
2. CVX 约束求解改为 ADMM + 正则系数二分；
3. Algorithms 4–5 近似为活跃簇合并与有界二分重定位；
4. 尚未完整实现原文的重复节点判定。

旧比较还包含两个会人为增强 Kang 的非论文步骤：重定位失败后补回 active/dense 节点，以及端点约束超阈值后回退到 active/dense 集。现在 `feasibility_repair=False` 为默认值，正式比较也禁用端点 fallback；失败样本会如实计入通过率。历史增强版只能作为消融，不能标成 Kang 原方法。

标量复核：

```powershell
python scripts/reproduce_sparse_knot_paper.py `
  --output-dir outputs/paper/kang_sparse_reproduction_rechecked `
  --overwrite
```

| 算例 | 初始 K | active K | final K | final MSE | 原文参考 |
|---|---:|---:|---:|---:|---|
| Chebyshev T10，N=401 | 25 | 15 | 15 | `5.259e-5` | K=14，MSE=`3.4745e-5` |
| 已知 5 节点三次样条 | 41 | 10 | 5 | `9.066e-6` | 工作流验证 |

差异源于 ADMM/CVX、阈值读出和重复节点启发式不同，不能声称与论文数值一致。

## 3. Yeh 复现范围

三次 B 样条的 order 为 4，因此实现计算四阶有限差分，并采用

\[
f_i=\left\lVert q_i^{(4)}\right\rVert_2^{1/4}.
\]

梯形积分得到累计特征 (F_i)，按原文对 (F_i) 线性插值，再把节点放在等累计质量位置；默认启用论文的局部密度限制 (F_i-F_{i-1}\leq\Delta F)。

原文用特定 NURBS 数据回归目标 RMS 与 \(\Delta F\) 的关系。为避免把该数据先验带入本项目，这里从小到大扫描 K，并返回第一个满足公共 MSE 阈值的节点集。外层扫描属于公平比较包装，不是论文原回归公式。

## 4. 统一评测口径

- 同一批独立测试曲线、观测点与归一化坐标；
- 三次开放夹持 B 样条；非网络方法统一使用弦长参数；
- 最终均执行无正则、端点严格插值的标准 B 样条最小二乘；
- MSE 为 \(N^{-1}\sum_i\|C(t_i)-Q_i\|_2^2\)，不取平方根；
- 报告 MSE、阈值通过率、最终内部节点数和时间；
- Ours 的时间只含同步 batch-1 网络前向，其他方法为完整数值搜索。时间边界不对称，不能写成严格端到端加速比。

## 5. 当前初步结果

checkpoint 为 `outputs/candidate_pruning_one_shot_v14.pt`，source K=4–20 每档 1 条，阈值 `MSE=2.5e-5`。Gradient baseline 只用了 4 个位置优化步，因此这是实现复核，不是论文最终统计。

| 方法 | 平均 MSE | 通过率 | 平均 final K | 平均时间/曲线 |
|---|---:|---:|---:|---:|
| Ours learned | `2.508e-5` | 64.7% | 11.29 | 25.51 ms（仅网络） |
| Kang 2015 no-repair adaptation | `4.761e-5` | 76.5% | 6.82 | 3319.89 ms |
| Yeh 2020 feature-CDF + K scan | `1.451e-5` | 100% | 9.41 | 31.11 ms |
| Uniform-Kmax delete + gradient | `1.881e-5` | 100% | 6.47 | 12243.20 ms |

小样本不支持“Kang 所有指标都差于 Ours”。目前能成立的是：Ours 平均 MSE 优于 Kang，纯前向约快两个数量级；但该 checkpoint 的通过率低于 Kang，并保留更多节点。旧版 Kang+repair 在 340 条实验中为 100% 通过，正是非论文补回掩盖失败的证据。

结果位于 `outputs/comparisons/four_methods_v14_K4_20_n1/`。

## 6. 正式运行

```powershell
python scripts/compare_knot_methods.py `
  --checkpoint outputs/candidate_pruning_one_shot_v14.pt `
  --output-dir outputs/comparisons/four_methods_v14_K4_20_n20 `
  --samples-per-knot-count 20 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --max-internal-knots 28 `
  --gradient-steps 12 `
  --paper-initial-knots 40 `
  --paper-admm-iterations 400 `
  --paper-lambda-bisections 8 `
  --paper-relocation-iterations 8 `
  --mse-tolerance 2.5e-5 `
  --ours-mode learned `
  --sample-figures `
  --device auto `
  --network-warmups 10 `
  --network-repeats 50 `
  --dataset-seed 20000 `
  --dpi 300 `
  --overwrite
```

论文统计至少应使用每档 20 条并报告置信区间。若改用 `--ours-mode verified`，必须把验证/修复时间与网络时间分开列出。
