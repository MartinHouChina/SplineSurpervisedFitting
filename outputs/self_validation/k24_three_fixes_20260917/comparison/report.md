# v16-feasible-teacher 多数据集配对测试

**DIAGNOSTIC NOT FINAL — QUICK OR UNQUALIFIED RUN**

资格检查：checkpoint is not the current v16 adaptive-beta mass-TopK revision；formal proposal training must allocate exactly 50% of synthetic draws to the high-K stratum；formal proposal high-K stratum must start at K=40；synthetic data contract is not the certified K=4..56 source-subset minimality revision；formal synthetic source maximum must be 60 control points (56 internal knots)；formal v16 candidate-knot capacity must equal 72 internal knots；formal validation boundary must be the K=56 synthetic stratum；formal validation K=56 boundary audit must contain at least 32 synthetic samples

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\self_validation\v16_small_medium_k24_smoke_r1\model.pt`（candidate_selection_feasible_teacher_bspline_v16，epoch 5）。
统一阈值：MSE ≤ 1.000e-04；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16-feasible-teacher 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 24, 'source_max_internal_knots': 20, 'network_candidate_overhead_vs_source': 4, 'greedy_initial_and_yeh_max': 24, 'kang_dense_initial': 24, 'liang_dense_initial': 24, 'equal_initial_capacity': True, 'numerical_caps_match_source_max': False, 'formal_overcomplete_candidate_contract': False, 'main_comparison_capacity_valid': True, 'capacity_contract': 'equal_network_and_numerical_initial_capacity', 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 32, 'numerical_full_knot_vector_cap': 32}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce GTX 1070', 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 4, 'python': '3.13.9'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16-feasible-teacher learned | 17 | 11.8% | 1.716e-03 | 13.00 | 22.74 | 20.34 |
| Synthetic | Ours v16 + disclosed numerical MSE repair | 17 | 58.8% | 1.914e-04 | 20.47 | 37.00 | — |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 17 | 0.0% | 3.057e-04 | 24.00 | 96.32 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 17 | 58.8% | 1.838e-04 | 20.29 | 54.30 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 17 | 82.4% | 9.975e-05 | 17.53 | 416.22 | — |
| Synthetic | Kang 2015 (threshold-safe ADMM adaptation) | 17 | 58.8% | 1.781e-04 | 23.18 | 32.34 | — |
| Synthetic | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 17 | 58.8% | 1.774e-04 | 18.53 | 205.11 | — |
| UJI | Ours v16-feasible-teacher learned | 1 | 100.0% | 4.323e-05 | 13.00 | 23.70 | 18.12 |
| UJI | Ours v16 + disclosed numerical MSE repair | 1 | 100.0% | 4.323e-05 | 13.00 | 21.07 | — |
| UJI | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 7.495e-05 | 3.00 | 23.06 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 4.430e-05 | 4.00 | 11.56 | — |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 1 | 100.0% | 4.640e-05 | 4.00 | 139.46 | — |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 1 | 100.0% | 5.944e-06 | 22.00 | 47.45 | — |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 1 | 100.0% | 6.890e-05 | 3.00 | 136.38 | — |
| NaturalEarth | Ours v16-feasible-teacher learned | 1 | 0.0% | 4.101e-04 | 13.00 | 26.85 | 23.39 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 1 | 100.0% | 7.944e-05 | 22.00 | 34.21 | — |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 8.089e-05 | 20.00 | 99.12 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 9.655e-05 | 18.00 | 35.08 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 1 | 100.0% | 9.677e-05 | 14.00 | 416.20 | — |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 1 | 100.0% | 7.700e-05 | 24.00 | 16.52 | — |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 1 | 100.0% | 9.041e-05 | 20.00 | 220.12 | — |
| USGS | Ours v16-feasible-teacher learned | 1 | 100.0% | 5.273e-05 | 13.00 | 21.79 | 26.05 |
| USGS | Ours v16 + disclosed numerical MSE repair | 1 | 100.0% | 5.273e-05 | 13.00 | 20.51 | — |
| USGS | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 7.993e-05 | 8.00 | 24.43 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 8.767e-05 | 9.00 | 22.57 | — |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 1 | 100.0% | 9.791e-05 | 8.00 | 164.50 | — |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 1 | 100.0% | 3.640e-06 | 24.00 | 12.89 | — |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 1 | 100.0% | 5.670e-05 | 8.00 | 159.63 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16-feasible-teacher learned | 100.0% | 9.695e-05 |
| UJI | Ours v16 + disclosed numerical MSE repair | 100.0% | 9.695e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 1.137e-04 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 9.138e-05 |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 7.483e-05 |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 2.238e-05 |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 0.0% | 1.076e-04 |
| NaturalEarth | Ours v16-feasible-teacher learned | 0.0% | 4.161e-04 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 100.0% | 8.252e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 100.0% | 8.348e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 9.956e-05 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 9.987e-05 |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 8.012e-05 |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 9.366e-05 |
| USGS | Ours v16-feasible-teacher learned | 100.0% | 5.296e-05 |
| USGS | Ours v16 + disclosed numerical MSE repair | 100.0% | 5.296e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 100.0% | 8.020e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 8.798e-05 |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 9.831e-05 |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 3.643e-06 |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 5.695e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
