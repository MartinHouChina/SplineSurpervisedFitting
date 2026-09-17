# v16-feasible-teacher 多数据集配对测试

**DIAGNOSTIC NOT FINAL — QUICK OR UNQUALIFIED RUN**

资格检查：checkpoint is not the current v16 adaptive-beta mass-TopK revision；formal proposal training must allocate exactly 50% of synthetic draws to the high-K stratum；formal proposal high-K stratum must start at K=40；synthetic data contract is not the certified K=4..56 source-subset minimality revision；formal synthetic source maximum must be 60 control points (56 internal knots)；formal v16 candidate-knot capacity must equal 72 internal knots；formal validation boundary must be the K=56 synthetic stratum；formal validation K=56 boundary audit must contain at least 32 synthetic samples

Checkpoint: `/home/feng/HouCode/SplineSurpervisedFitting/outputs/checkpoints/v16_small_medium_k24_r1.pt`（candidate_selection_feasible_teacher_bspline_v16，epoch 48）。
统一阈值：MSE ≤ 1.000e-04；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16-feasible-teacher 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 24, 'source_max_internal_knots': 20, 'network_candidate_overhead_vs_source': 4, 'greedy_initial_and_yeh_max': 24, 'kang_dense_initial': 24, 'liang_dense_initial': 24, 'equal_initial_capacity': True, 'numerical_caps_match_source_max': False, 'formal_overcomplete_candidate_contract': False, 'main_comparison_capacity_valid': True, 'capacity_contract': 'equal_network_and_numerical_initial_capacity', 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 32, 'numerical_full_knot_vector_cap': 32}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16-feasible-teacher learned | 17 | 0.0% | 1.606e-03 | 14.71 | 8.72 | 7.35 |
| Synthetic | Ours v16 + disclosed numerical MSE repair | 17 | 88.2% | 7.386e-05 | 21.12 | 15.41 | — |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 17 | 17.6% | 3.706e-04 | 23.35 | 30.95 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 17 | 58.8% | 1.850e-04 | 20.47 | 22.80 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 17 | 88.2% | 9.208e-05 | 17.12 | 289.87 | — |
| Synthetic | Kang 2015 (threshold-safe ADMM adaptation) | 17 | 52.9% | 2.808e-04 | 21.53 | 290.72 | — |
| Synthetic | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 17 | 52.9% | 2.242e-04 | 19.29 | 1586.73 | — |
| UJI | Ours v16-feasible-teacher learned | 5 | 40.0% | 1.360e-04 | 15.60 | 8.68 | 7.36 |
| UJI | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 4.908e-05 | 17.40 | 12.35 | — |
| UJI | Park & Lee 2007 (DOM adaptation) | 5 | 100.0% | 7.724e-05 | 9.60 | 13.17 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 5.437e-05 | 7.80 | 8.16 | — |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 5.498e-05 | 6.40 | 108.70 | — |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 2.026e-05 | 19.60 | 556.64 | — |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 7.115e-05 | 7.60 | 2408.75 | — |
| NaturalEarth | Ours v16-feasible-teacher learned | 5 | 0.0% | 3.168e-04 | 16.60 | 8.71 | 7.37 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 6.125e-05 | 20.60 | 14.88 | — |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 5 | 100.0% | 7.966e-05 | 13.40 | 22.87 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 7.738e-05 | 13.40 | 14.15 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 8.250e-05 | 11.00 | 196.41 | — |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 5.038e-05 | 24.00 | 392.68 | — |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 8.756e-05 | 13.60 | 2079.82 | — |
| USGS | Ours v16-feasible-teacher learned | 5 | 0.0% | 6.412e-04 | 15.40 | 8.76 | 7.35 |
| USGS | Ours v16 + disclosed numerical MSE repair | 5 | 40.0% | 1.445e-04 | 23.40 | 16.37 | — |
| USGS | Park & Lee 2007 (DOM adaptation) | 5 | 40.0% | 2.024e-04 | 18.20 | 24.27 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 5 | 60.0% | 1.283e-04 | 18.20 | 20.11 | — |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 60.0% | 9.274e-05 | 17.00 | 337.06 | — |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 40.0% | 1.317e-04 | 22.40 | 246.91 | — |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 40.0% | 1.476e-04 | 17.20 | 1430.39 | — |
| IndustrialOffset | Ours v16-feasible-teacher learned | 5 | 100.0% | 3.148e-05 | 16.40 | 8.70 | 7.34 |
| IndustrialOffset | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 3.148e-05 | 16.40 | 8.91 | — |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 5 | 80.0% | 9.322e-05 | 13.80 | 20.47 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 8.203e-05 | 7.40 | 7.62 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 8.229e-05 | 8.00 | 117.89 | — |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 3.593e-05 | 21.20 | 558.41 | — |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 6.034e-05 | 8.40 | 2363.27 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16-feasible-teacher learned | 40.0% | 2.319e-04 |
| UJI | Ours v16 + disclosed numerical MSE repair | 60.0% | 7.684e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 1.514e-04 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 60.0% | 9.603e-05 |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 60.0% | 1.058e-04 |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 80.0% | 6.206e-05 |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 20.0% | 1.382e-04 |
| NaturalEarth | Ours v16-feasible-teacher learned | 0.0% | 3.190e-04 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 100.0% | 6.010e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 100.0% | 8.172e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 7.873e-05 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 8.423e-05 |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 5.211e-05 |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 8.943e-05 |
| USGS | Ours v16-feasible-teacher learned | 0.0% | 6.428e-04 |
| USGS | Ours v16 + disclosed numerical MSE repair | 40.0% | 1.433e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 40.0% | 2.045e-04 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 60.0% | 1.301e-04 |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 60.0% | 9.415e-05 |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 40.0% | 1.339e-04 |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 40.0% | 1.499e-04 |
| IndustrialOffset | Ours v16-feasible-teacher learned | 100.0% | 3.105e-05 |
| IndustrialOffset | Ours v16 + disclosed numerical MSE repair | 100.0% | 3.105e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 80.0% | 9.480e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 8.299e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 8.318e-05 |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 3.702e-05 |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 6.109e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
