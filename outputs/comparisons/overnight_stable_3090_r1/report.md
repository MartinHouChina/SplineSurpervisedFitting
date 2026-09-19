# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；certified synthetic matched-knot MAE exceeds the formal limit 0.005；observed worst-source deployment pass rate 40.625% is below the required 90.000%；infeasible-proposal ablation was enabled

Checkpoint: `/home/feng/HouCode/SplineFitting_1070_overnight/outputs/checkpoints/overnight_stable_3090_r1.pt`（candidate_selection_counterfactual_bspline_v16，epoch 7）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 64, 'greedy_initial_and_yeh_max': 64, 'kang_dense_initial': 64, 'liang_dense_initial': 64, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 72, 'numerical_full_knot_vector_cap': 72}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。
数值基线协议：**threshold-safe adaptation**。Dung/Kang/Luo 在原生公共 refit 未达阈值时可执行有界修复；保留最佳拟合但不保证容量内必然可行。所有额外 refit 计入完整耗时；这不是原文算法步骤。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 42 | 85.7% | 3.370e-05 | 27.40 | 8.54 | 7.13 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 42 | 100.0% | 4.418e-05 | 39.19 | 56.06 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 42 | 100.0% | 4.641e-05 | 35.88 | 40.84 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 42 | 100.0% | 4.648e-05 | 26.88 | 417.19 | — |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 42 | 95.2% | 3.967e-05 | 36.21 | 811.33 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 42 | 97.6% | 4.542e-05 | 32.14 | 2786.19 | — |
| UJI | Ours v16 learned | 20 | 85.0% | 2.424e-05 | 19.55 | 8.41 | 7.09 |
| UJI | Park & Lee 2007 (DOM adaptation) | 20 | 100.0% | 3.388e-05 | 11.80 | 15.57 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 20 | 100.0% | 3.379e-05 | 8.95 | 9.01 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 100.0% | 3.105e-05 | 7.50 | 108.12 | — |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 100.0% | 2.143e-05 | 29.50 | 1049.69 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 100.0% | 2.722e-05 | 8.60 | 2928.34 | — |
| NaturalEarth | Ours v16 learned | 20 | 50.0% | 1.146e-04 | 26.15 | 8.51 | 7.10 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 20 | 95.0% | 5.027e-05 | 26.90 | 38.61 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 20 | 95.0% | 5.182e-05 | 25.55 | 28.53 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 95.0% | 4.935e-05 | 22.40 | 350.30 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 85.0% | 4.246e-05 | 36.80 | 803.32 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 85.0% | 5.783e-05 | 23.90 | 2619.09 | — |
| USGS | Ours v16 learned | 20 | 35.0% | 1.194e-04 | 30.80 | 8.57 | 7.10 |
| USGS | Park & Lee 2007 (DOM adaptation) | 20 | 85.0% | 4.976e-05 | 37.80 | 55.13 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 20 | 85.0% | 4.887e-05 | 34.85 | 41.67 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 85.0% | 4.715e-05 | 32.40 | 508.59 | — |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 85.0% | 4.152e-05 | 47.70 | 651.18 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 85.0% | 5.239e-05 | 35.45 | 2241.92 | — |
| IndustrialOffset | Ours v16 learned | 20 | 85.0% | 3.797e-05 | 17.55 | 8.35 | 7.08 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 20 | 100.0% | 3.687e-05 | 16.95 | 25.19 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 20 | 100.0% | 3.153e-05 | 11.10 | 11.15 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 100.0% | 3.076e-05 | 10.65 | 135.55 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 100.0% | 1.697e-05 | 28.25 | 1114.68 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 100.0% | 3.256e-05 | 11.45 | 3124.65 | — |

## 原生适配与最终结果审计

native 指修正后的仓库适配在公共可行性修复前的端点约束 refit，非作者原版复现。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 5.35 | 10.65 | 1.896e-03 | 3.076e-05 | 15.90 | 20/20 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 28.20 | 28.25 | 1.793e-05 | 1.697e-05 | 0.15 | 1/20 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 6.85 | 11.45 | 4.568e-04 | 3.256e-05 | 13.80 | 15/20 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 12.35 | 22.40 | 5.310e-04 | 4.935e-05 | 30.20 | 20/20 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 36.75 | 36.80 | 4.327e-05 | 4.246e-05 | 0.30 | 4/20 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 6.85 | 23.90 | 2.243e-03 | 5.783e-05 | 51.30 | 19/20 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 14.81 | 26.88 | 1.047e-03 | 4.648e-05 | 36.21 | 42/42 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 18.60 | 36.21 | 2.187e-03 | 3.967e-05 | 52.88 | 30/42 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 10.33 | 32.14 | 2.464e-03 | 4.542e-05 | 65.50 | 35/42 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 3.95 | 7.50 | 6.503e-04 | 3.105e-05 | 10.65 | 17/20 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 29.45 | 29.50 | 2.208e-05 | 2.143e-05 | 0.15 | 1/20 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 4.95 | 8.60 | 1.605e-03 | 2.722e-05 | 10.95 | 10/20 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 17.40 | 32.40 | 4.757e-04 | 4.715e-05 | 45.15 | 19/20 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 47.70 | 47.70 | 4.152e-05 | 4.152e-05 | 0.15 | 3/20 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 7.45 | 35.45 | 2.540e-03 | 5.239e-05 | 84.20 | 18/20 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 85.0% | 3.911e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 35.0% | 7.970e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 35.0% | 7.329e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 55.0% | 5.910e-05 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 55.0% | 6.426e-05 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 55.0% | 6.029e-05 |
| NaturalEarth | Ours v16 learned | 45.0% | 1.169e-04 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 75.0% | 5.244e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 60.0% | 3.064e-02 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 70.0% | 5.174e-05 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 85.0% | 4.466e-05 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 70.0% | 6.033e-05 |
| USGS | Ours v16 learned | 35.0% | 1.214e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 60.0% | 5.228e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 60.0% | 5.161e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 65.0% | 5.002e-05 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 80.0% | 4.367e-05 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 60.0% | 5.459e-05 |
| IndustrialOffset | Ours v16 learned | 85.0% | 3.790e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 95.0% | 3.740e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 95.0% | 3.185e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 3.104e-05 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 1.727e-05 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 100.0% | 3.288e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
