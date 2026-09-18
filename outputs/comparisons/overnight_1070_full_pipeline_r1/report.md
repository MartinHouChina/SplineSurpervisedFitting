# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6

Checkpoint: `/home/feng/HouCode/SplineFitting_1070_overnight/outputs/checkpoints/overnight_1070_full_pipeline_r1.pt`（candidate_selection_counterfactual_bspline_v16，epoch 63）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 64, 'greedy_initial_and_yeh_max': 64, 'kang_dense_initial': 64, 'liang_dense_initial': 64, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 72, 'numerical_full_knot_vector_cap': 72}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 42 | 95.2% | 2.510e-05 | 24.88 | 8.40 | 6.95 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 42 | 100.0% | 4.418e-05 | 39.19 | 58.99 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 42 | 100.0% | 4.641e-05 | 35.88 | 43.39 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 42 | 0.0% | 1.047e-03 | 14.81 | 395.25 | — |
| Synthetic | Kang 2015 (ADMM adaptation) | 42 | 23.8% | 3.237e-03 | 5.29 | 811.32 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 42 | 16.7% | 2.464e-03 | 10.33 | 2892.69 | — |
| UJI | Ours v16 learned | 8 | 87.5% | 2.694e-05 | 21.88 | 8.35 | 6.93 |
| UJI | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 3.834e-05 | 9.12 | 12.15 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 3.103e-05 | 7.50 | 8.01 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 8 | 25.0% | 6.258e-04 | 3.62 | 92.76 | — |
| UJI | Kang 2015 (ADMM adaptation) | 8 | 75.0% | 4.673e-05 | 6.12 | 1212.20 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 8 | 37.5% | 6.226e-04 | 3.75 | 3085.10 | — |
| NaturalEarth | Ours v16 learned | 8 | 25.0% | 1.068e-04 | 23.12 | 8.35 | 6.94 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 4.367e-05 | 24.88 | 39.44 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 4.714e-05 | 24.50 | 28.76 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 8 | 0.0% | 6.609e-04 | 11.00 | 295.41 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 8 | 12.5% | 6.741e-03 | 9.00 | 914.21 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 8 | 0.0% | 2.260e-03 | 6.00 | 2815.75 | — |
| USGS | Ours v16 learned | 8 | 37.5% | 7.223e-05 | 34.50 | 8.55 | 6.94 |
| USGS | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 4.656e-05 | 33.25 | 49.75 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 4.057e-05 | 29.50 | 36.16 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 8 | 12.5% | 2.618e-04 | 15.12 | 411.09 | — |
| USGS | Kang 2015 (ADMM adaptation) | 8 | 25.0% | 4.362e-03 | 3.50 | 783.17 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 8 | 25.0% | 7.746e-04 | 8.50 | 2609.17 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 75.0% | 3.561e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 12.5% | 8.019e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 50.0% | 6.026e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 25.0% | 8.677e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 0.0% | 9.349e-05 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 37.5% | 7.385e-04 |
| NaturalEarth | Ours v16 learned | 25.0% | 1.063e-04 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 75.0% | 4.531e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 37.5% | 7.652e-02 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 6.710e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 12.5% | 6.808e-03 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 2.280e-03 |
| USGS | Ours v16 learned | 37.5% | 7.043e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 62.5% | 4.764e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 87.5% | 4.160e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 12.5% | 2.649e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 25.0% | 4.388e-03 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 25.0% | 7.810e-04 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
