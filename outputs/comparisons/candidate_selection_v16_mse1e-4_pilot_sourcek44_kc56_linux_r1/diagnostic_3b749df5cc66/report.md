# v16-feasible-teacher 多数据集配对测试

**DIAGNOSTIC NOT FINAL — QUICK OR UNQUALIFIED RUN**

资格检查：checkpoint is not the current v16 adaptive-beta mass-TopK revision；formal proposal high-K stratum must start at K=40；formal synthetic source maximum must be 60 control points (56 internal knots)；formal v16 candidate-knot capacity must equal 72 internal knots；formal validation boundary must be the K=56 synthetic stratum；formal validation K=56 boundary audit must contain at least 32 synthetic samples

Checkpoint: `/home/feng/HouCode/SplineSurpervisedFitting/outputs/checkpoints/candidate_selection_v16_mse1e-4_pilot_sourcek44_kc56_linux_r1.pt`（candidate_selection_feasible_teacher_bspline_v16，epoch 70）。
统一阈值：MSE ≤ 1.000e-04；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16-feasible-teacher 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 56, 'source_max_internal_knots': 44, 'network_candidate_overhead_vs_source': 12, 'greedy_initial_and_yeh_max': 56, 'kang_dense_initial': 56, 'liang_dense_initial': 56, 'equal_initial_capacity': True, 'numerical_caps_match_source_max': False, 'formal_overcomplete_candidate_contract': False, 'main_comparison_capacity_valid': True, 'capacity_contract': 'equal_network_and_numerical_initial_capacity', 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 64, 'numerical_full_knot_vector_cap': 64}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16-feasible-teacher learned | 53 | 0.0% | 1.894e-03 | 32.64 | 8.45 | 6.82 |
| Synthetic | Ours v16 + disclosed numerical MSE repair | 53 | 77.4% | 2.243e-04 | 46.94 | 19.02 | — |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 53 | 86.8% | 8.900e-05 | 41.47 | 63.45 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 53 | 56.6% | 2.483e-04 | 44.32 | 57.98 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 53 | 69.8% | 1.364e-04 | 38.57 | 708.79 | — |
| Synthetic | Kang 2015 (threshold-safe ADMM adaptation) | 53 | 60.4% | 3.448e-04 | 43.19 | 456.85 | — |
| Synthetic | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 53 | 60.4% | 3.196e-04 | 42.26 | 2052.95 | — |
| UJI | Ours v16-feasible-teacher learned | 5 | 100.0% | 4.979e-05 | 19.60 | 8.25 | 6.81 |
| UJI | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 4.979e-05 | 19.60 | 8.47 | — |
| UJI | Park & Lee 2007 (DOM adaptation) | 5 | 100.0% | 7.724e-05 | 9.60 | 13.14 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 6.476e-05 | 7.80 | 8.83 | — |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 4.666e-05 | 6.60 | 111.73 | — |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 2.554e-05 | 30.00 | 949.50 | — |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 5.739e-05 | 7.00 | 2647.21 | — |
| NaturalEarth | Ours v16-feasible-teacher learned | 5 | 0.0% | 2.844e-04 | 21.20 | 8.29 | 6.83 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 8.704e-05 | 31.00 | 18.30 | — |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 5 | 100.0% | 7.966e-05 | 13.40 | 22.81 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 7.780e-05 | 14.00 | 15.44 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 7.135e-05 | 10.80 | 198.91 | — |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 4.983e-05 | 20.00 | 935.32 | — |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 7.795e-05 | 12.00 | 2691.20 | — |
| USGS | Ours v16-feasible-teacher learned | 5 | 20.0% | 2.032e-04 | 26.00 | 8.42 | 6.81 |
| USGS | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 8.247e-05 | 33.60 | 16.30 | — |
| USGS | Park & Lee 2007 (DOM adaptation) | 5 | 100.0% | 7.991e-05 | 25.20 | 35.48 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 9.160e-05 | 19.80 | 23.14 | — |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 7.921e-05 | 18.40 | 344.31 | — |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 3.586e-05 | 37.00 | 708.01 | — |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 7.336e-05 | 20.40 | 2785.99 | — |
| IndustrialOffset | Ours v16-feasible-teacher learned | 5 | 60.0% | 3.708e-04 | 20.60 | 8.25 | 6.81 |
| IndustrialOffset | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 7.582e-05 | 24.60 | 12.53 | — |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 5 | 100.0% | 7.179e-05 | 20.00 | 31.81 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 6.769e-05 | 7.80 | 8.71 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 6.393e-05 | 8.80 | 123.38 | — |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 5.558e-05 | 13.40 | 1033.28 | — |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 4.181e-05 | 8.60 | 2838.03 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16-feasible-teacher learned | 100.0% | 7.030e-05 |
| UJI | Ours v16 + disclosed numerical MSE repair | 100.0% | 7.030e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 1.514e-04 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 40.0% | 1.176e-04 |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 40.0% | 9.454e-05 |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 80.0% | 7.154e-05 |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 60.0% | 1.008e-04 |
| NaturalEarth | Ours v16-feasible-teacher learned | 0.0% | 2.855e-04 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 100.0% | 8.654e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 100.0% | 8.172e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 7.929e-05 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 7.309e-05 |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 5.129e-05 |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 7.914e-05 |
| USGS | Ours v16-feasible-teacher learned | 20.0% | 2.034e-04 |
| USGS | Ours v16 + disclosed numerical MSE repair | 100.0% | 8.179e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 100.0% | 8.101e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 80.0% | 9.322e-05 |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 60.0% | 8.067e-05 |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 3.659e-05 |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 60.0% | 7.472e-05 |
| IndustrialOffset | Ours v16-feasible-teacher learned | 60.0% | 3.755e-04 |
| IndustrialOffset | Ours v16 + disclosed numerical MSE repair | 100.0% | 7.623e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 100.0% | 7.277e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 80.0% | 6.844e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 6.475e-05 |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 5.638e-05 |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 4.250e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
