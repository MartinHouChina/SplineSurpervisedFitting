# v16-feasible-teacher 多数据集配对测试

**DIAGNOSTIC NOT FINAL — QUICK OR UNQUALIFIED RUN**

资格检查：checkpoint is not the current v16 adaptive-beta mass-TopK revision；formal proposal training must allocate exactly 50% of synthetic draws to the high-K stratum；formal proposal high-K stratum must start at K=40；synthetic data contract is not the certified K=4..56 source-subset minimality revision；formal synthetic source maximum must be 60 control points (56 internal knots)；formal v16 candidate-knot capacity must equal 72 internal knots；formal validation boundary must be the K=56 synthetic stratum；formal validation K=56 boundary audit must contain at least 32 synthetic samples

Checkpoint: `/home/feng/HouCode/SplineSurpervisedFitting/outputs/checkpoints/candidate_selection_v16_mse5e-5_small_medium_sourcek20_kc24_p64_j64_linux_r1.pt`（candidate_selection_feasible_teacher_bspline_v16，epoch 125）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16-feasible-teacher 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 24, 'source_max_internal_knots': 20, 'network_candidate_overhead_vs_source': 4, 'greedy_initial_and_yeh_max': 24, 'kang_dense_initial': 24, 'liang_dense_initial': 24, 'equal_initial_capacity': True, 'numerical_caps_match_source_max': False, 'formal_overcomplete_candidate_contract': False, 'main_comparison_capacity_valid': True, 'capacity_contract': 'equal_network_and_numerical_initial_capacity', 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 32, 'numerical_full_knot_vector_cap': 32}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16-feasible-teacher learned | 17 | 0.0% | 1.273e-03 | 15.35 | 8.31 | 7.01 |
| Synthetic | Ours v16 + disclosed numerical MSE repair | 17 | 88.2% | 3.192e-05 | 21.71 | 14.83 | — |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 17 | 11.8% | 3.681e-04 | 23.59 | 29.62 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 17 | 41.2% | 1.691e-04 | 21.41 | 22.45 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 17 | 58.8% | 6.355e-05 | 19.59 | 337.66 | — |
| Synthetic | Kang 2015 (threshold-safe ADMM adaptation) | 17 | 35.3% | 2.736e-04 | 22.41 | 181.05 | — |
| Synthetic | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 17 | 41.2% | 2.124e-04 | 20.88 | 1282.36 | — |
| UJI | Ours v16-feasible-teacher learned | 5 | 60.0% | 4.539e-05 | 17.40 | 8.23 | 6.98 |
| UJI | Ours v16 + disclosed numerical MSE repair | 5 | 80.0% | 3.671e-05 | 18.80 | 11.02 | — |
| UJI | Park & Lee 2007 (DOM adaptation) | 5 | 100.0% | 4.047e-05 | 11.60 | 14.69 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 3.761e-05 | 8.60 | 8.51 | — |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 3.392e-05 | 7.20 | 118.28 | — |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 100.0% | 1.670e-05 | 19.00 | 527.46 | — |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 3.645e-05 | 10.00 | 2214.40 | — |
| NaturalEarth | Ours v16-feasible-teacher learned | 5 | 0.0% | 4.309e-04 | 17.40 | 8.28 | 7.00 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 5 | 80.0% | 4.566e-05 | 22.20 | 14.45 | — |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 5 | 80.0% | 4.502e-05 | 16.80 | 27.25 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 4.571e-05 | 16.20 | 16.35 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 4.041e-05 | 13.80 | 217.60 | — |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 40.0% | 5.038e-05 | 24.00 | 149.99 | — |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 60.0% | 4.932e-05 | 19.20 | 1259.15 | — |
| USGS | Ours v16-feasible-teacher learned | 5 | 20.0% | 5.514e-04 | 16.60 | 8.34 | 7.01 |
| USGS | Ours v16 + disclosed numerical MSE repair | 5 | 20.0% | 1.647e-04 | 23.20 | 14.94 | — |
| USGS | Park & Lee 2007 (DOM adaptation) | 5 | 40.0% | 1.883e-04 | 19.40 | 24.36 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 5 | 40.0% | 1.149e-04 | 18.60 | 19.32 | — |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 40.0% | 9.622e-05 | 18.40 | 378.50 | — |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 40.0% | 1.310e-04 | 22.60 | 220.91 | — |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 40.0% | 1.333e-04 | 18.60 | 1317.66 | — |
| IndustrialOffset | Ours v16-feasible-teacher learned | 5 | 60.0% | 5.441e-05 | 18.40 | 8.30 | 7.01 |
| IndustrialOffset | Ours v16 + disclosed numerical MSE repair | 5 | 100.0% | 1.833e-05 | 19.40 | 10.58 | — |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 5 | 80.0% | 6.113e-05 | 14.80 | 20.82 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 5 | 100.0% | 2.920e-05 | 10.00 | 9.87 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 5 | 100.0% | 2.273e-05 | 9.20 | 115.89 | — |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 5 | 80.0% | 3.657e-05 | 20.80 | 420.57 | — |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 5 | 100.0% | 3.999e-05 | 10.40 | 1957.65 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16-feasible-teacher learned | 60.0% | 5.416e-05 |
| UJI | Ours v16 + disclosed numerical MSE repair | 60.0% | 3.735e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 9.681e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 20.0% | 7.805e-05 |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 40.0% | 7.126e-05 |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 60.0% | 5.708e-05 |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 40.0% | 8.613e-05 |
| NaturalEarth | Ours v16-feasible-teacher learned | 0.0% | 4.331e-04 |
| NaturalEarth | Ours v16 + disclosed numerical MSE repair | 80.0% | 4.426e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 80.0% | 4.612e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 4.634e-05 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 4.152e-05 |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 40.0% | 5.211e-05 |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 40.0% | 5.083e-05 |
| USGS | Ours v16-feasible-teacher learned | 20.0% | 5.533e-04 |
| USGS | Ours v16 + disclosed numerical MSE repair | 20.0% | 1.634e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 40.0% | 1.905e-04 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 40.0% | 1.166e-04 |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 40.0% | 9.767e-05 |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 40.0% | 1.332e-04 |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 40.0% | 1.356e-04 |
| IndustrialOffset | Ours v16-feasible-teacher learned | 60.0% | 5.459e-05 |
| IndustrialOffset | Ours v16 + disclosed numerical MSE repair | 100.0% | 1.799e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 80.0% | 6.257e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 2.959e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 2.316e-05 |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 80.0% | 3.767e-05 |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 4.079e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
