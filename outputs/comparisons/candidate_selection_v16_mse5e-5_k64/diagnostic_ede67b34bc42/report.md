# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT**

资格检查：certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；certified synthetic matched-knot MAE exceeds the formal limit 0.005

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\checkpoints\candidate_selection_v16_mse5e-5_k64.pt`（candidate_selection_counterfactual_bspline_v16，epoch 62）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 64, 'greedy_initial_and_yeh_max': 64, 'kang_dense_initial': 64, 'liang_dense_initial': 64, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 72, 'numerical_full_knot_vector_cap': 72}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce GTX 1070', 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 1, 'python': '3.13.9'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 42 | 83.3% | 3.047e-05 | 24.19 | 15.32 | 12.15 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 42 | 100.0% | 4.418e-05 | 39.19 | 108.92 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 42 | 100.0% | 4.641e-05 | 35.88 | 73.85 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 42 | 0.0% | 1.047e-03 | 14.81 | 722.03 | — |
| Synthetic | Kang 2015 (ADMM adaptation) | 42 | 23.8% | 3.237e-03 | 5.29 | 1574.18 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 42 | 16.7% | 2.463e-03 | 10.33 | 5119.66 | — |
| Synthetic | Yeh 2020 (feature-CDF + K scan) | 42 | 100.0% | 4.599e-05 | 32.40 | 71.15 | — |
| Synthetic | Uniform Kmax greedy + gradient | 42 | 100.0% | 4.584e-05 | 22.10 | 123005.31 | — |
| UJI | Ours v16 learned | 8 | 87.5% | 2.465e-05 | 22.62 | 16.00 | 12.58 |
| UJI | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 3.834e-05 | 9.12 | 23.69 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 3.103e-05 | 7.50 | 16.59 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 8 | 25.0% | 6.258e-04 | 3.62 | 175.83 | — |
| UJI | Kang 2015 (ADMM adaptation) | 8 | 75.0% | 4.673e-05 | 6.12 | 2406.89 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 8 | 37.5% | 6.226e-04 | 3.75 | 5435.29 | — |
| UJI | Yeh 2020 (feature-CDF + K scan) | 8 | 100.0% | 3.660e-05 | 7.12 | 15.50 | — |
| UJI | Uniform Kmax greedy + gradient | 8 | 100.0% | 3.461e-05 | 4.50 | 138184.71 | — |
| NaturalEarth | Ours v16 learned | 8 | 37.5% | 9.760e-05 | 22.50 | 14.77 | 11.97 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 4.367e-05 | 24.88 | 75.77 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 4.714e-05 | 24.50 | 51.63 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 8 | 0.0% | 6.609e-04 | 11.00 | 526.98 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 8 | 12.5% | 6.741e-03 | 9.00 | 1781.41 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 8 | 0.0% | 2.256e-03 | 6.00 | 4944.83 | — |
| NaturalEarth | Yeh 2020 (feature-CDF + K scan) | 8 | 87.5% | 4.027e-05 | 24.50 | 52.97 | — |
| NaturalEarth | Uniform Kmax greedy + gradient | 8 | 100.0% | 4.219e-05 | 18.00 | 124345.49 | — |
| USGS | Ours v16 learned | 8 | 25.0% | 1.129e-04 | 32.50 | 15.28 | 12.41 |
| USGS | Park & Lee 2007 (DOM adaptation) | 8 | 87.5% | 5.025e-05 | 36.12 | 97.23 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 8 | 87.5% | 4.922e-05 | 33.75 | 71.39 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 8 | 0.0% | 4.758e-04 | 17.00 | 867.59 | — |
| USGS | Kang 2015 (ADMM adaptation) | 8 | 25.0% | 1.163e-02 | 2.88 | 1309.44 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 8 | 0.0% | 2.128e-03 | 5.75 | 4170.84 | — |
| USGS | Yeh 2020 (feature-CDF + K scan) | 8 | 87.5% | 4.451e-05 | 36.25 | 84.37 | — |
| USGS | Uniform Kmax greedy + gradient | 8 | 87.5% | 4.834e-05 | 28.62 | 102879.33 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 87.5% | 3.335e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 12.5% | 8.019e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 50.0% | 6.026e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 25.0% | 8.677e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 0.0% | 9.349e-05 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 37.5% | 7.385e-04 |
| UJI | Yeh 2020 (feature-CDF + K scan) | 0.0% | 7.995e-05 |
| UJI | Uniform Kmax greedy + gradient | 12.5% | 6.259e-05 |
| NaturalEarth | Ours v16 learned | 37.5% | 9.766e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 75.0% | 4.531e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 37.5% | 7.651e-02 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 6.710e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 12.5% | 6.808e-03 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 2.276e-03 |
| NaturalEarth | Yeh 2020 (feature-CDF + K scan) | 75.0% | 4.203e-05 |
| NaturalEarth | Uniform Kmax greedy + gradient | 62.5% | 4.393e-05 |
| USGS | Ours v16 learned | 25.0% | 1.103e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 75.0% | 5.181e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 62.5% | 5.137e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 4.793e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 25.0% | 1.168e-02 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 2.143e-03 |
| USGS | Yeh 2020 (feature-CDF + K scan) | 62.5% | 4.585e-05 |
| USGS | Uniform Kmax greedy + gradient | 75.0% | 4.960e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
