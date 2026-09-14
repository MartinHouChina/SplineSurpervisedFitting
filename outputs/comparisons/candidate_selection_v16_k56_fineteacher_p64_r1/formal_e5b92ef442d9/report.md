# v16 多数据集配对测试

Checkpoint: `/home/feng/HouCode/SplineSurpervisedFitting/outputs/checkpoints/candidate_selection_v16_k56_fineteacher_p64_r1.pt`（candidate_selection_supervised_bspline_v16，epoch 122）。
统一阈值：MSE ≤ 1.000e-04；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 56, 'greedy_initial_and_yeh_max': 56, 'kang_dense_initial': 56, 'liang_dense_initial': 56, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 64, 'numerical_full_knot_vector_cap': 64}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 265 | 38.9% | 3.123e-04 | 37.60 | 8.38 | 6.75 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 265 | 86.8% | 9.115e-05 | 41.49 | 59.30 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 265 | 54.7% | 2.547e-04 | 44.31 | 53.48 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 265 | 68.7% | 1.455e-04 | 39.04 | 638.75 | — |
| Synthetic | Kang 2015 (threshold-safe ADMM adaptation) | 265 | 56.6% | 3.408e-04 | 42.85 | 1144.55 | — |
| Synthetic | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 265 | 56.6% | 3.274e-04 | 42.66 | 4804.39 | — |
| UJI | Ours v16 learned | 20 | 100.0% | 2.198e-05 | 29.85 | 7.65 | 6.22 |
| UJI | Park & Lee 2007 (DOM adaptation) | 20 | 100.0% | 6.871e-05 | 9.90 | 13.17 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 20 | 100.0% | 5.690e-05 | 7.75 | 7.86 | — |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 20 | 100.0% | 5.361e-05 | 6.70 | 94.19 | — |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 20 | 100.0% | 4.362e-05 | 21.70 | 2572.36 | — |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 20 | 100.0% | 4.941e-05 | 7.90 | 7266.52 | — |
| NaturalEarth | Ours v16 learned | 20 | 75.0% | 2.117e-04 | 27.05 | 7.64 | 6.19 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 20 | 95.0% | 9.391e-05 | 21.80 | 29.73 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 20 | 95.0% | 9.935e-05 | 20.05 | 21.81 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 20 | 95.0% | 8.703e-05 | 17.80 | 278.99 | — |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 20 | 90.0% | 8.516e-05 | 23.95 | 2335.17 | — |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 20 | 90.0% | 1.047e-04 | 19.60 | 6865.18 | — |
| USGS | Ours v16 learned | 20 | 50.0% | 1.984e-04 | 34.65 | 7.72 | 6.19 |
| USGS | Park & Lee 2007 (DOM adaptation) | 20 | 85.0% | 9.341e-05 | 31.40 | 43.45 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 20 | 85.0% | 9.017e-05 | 27.90 | 31.83 | — |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 20 | 95.0% | 8.526e-05 | 25.85 | 416.96 | — |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 20 | 85.0% | 6.536e-05 | 40.10 | 1781.95 | — |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 20 | 85.0% | 8.997e-05 | 27.95 | 5696.96 | — |
| IndustrialOffset | Ours v16 learned | 20 | 100.0% | 2.211e-05 | 27.40 | 7.60 | 6.20 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 20 | 100.0% | 7.158e-05 | 14.30 | 20.85 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 20 | 100.0% | 6.857e-05 | 9.25 | 9.38 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 20 | 100.0% | 5.832e-05 | 9.30 | 125.97 | — |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 20 | 100.0% | 4.004e-05 | 18.25 | 2770.97 | — |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 20 | 100.0% | 4.998e-05 | 10.30 | 8190.95 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 95.0% | 3.225e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 35.0% | 1.375e-04 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 50.0% | 1.159e-04 |
| UJI | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 60.0% | 1.068e-04 |
| UJI | Kang 2015 (threshold-safe ADMM adaptation) | 55.0% | 1.080e-04 |
| UJI | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 55.0% | 9.123e-05 |
| NaturalEarth | Ours v16 learned | 75.0% | 2.124e-04 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 80.0% | 9.663e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 65.0% | 1.024e-04 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 85.0% | 8.944e-05 |
| NaturalEarth | Kang 2015 (threshold-safe ADMM adaptation) | 90.0% | 8.796e-05 |
| NaturalEarth | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 75.0% | 1.075e-04 |
| USGS | Ours v16 learned | 50.0% | 1.992e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 85.0% | 9.633e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 70.0% | 2.753e-04 |
| USGS | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 65.0% | 8.805e-05 |
| USGS | Kang 2015 (threshold-safe ADMM adaptation) | 85.0% | 6.784e-05 |
| USGS | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 80.0% | 9.299e-05 |
| IndustrialOffset | Ours v16 learned | 100.0% | 2.154e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 90.0% | 7.236e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 95.0% | 6.911e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 100.0% | 5.880e-05 |
| IndustrialOffset | Kang 2015 (threshold-safe ADMM adaptation) | 100.0% | 4.057e-05 |
| IndustrialOffset | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 100.0% | 5.037e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
