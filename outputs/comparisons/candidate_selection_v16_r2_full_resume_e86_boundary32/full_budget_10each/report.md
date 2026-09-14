# v16 多数据集配对测试

Checkpoint: `/home/feng/HouCode/SplineSurpervisedFitting/outputs/checkpoints/candidate_selection_v16_r2_full_resume_e86_boundary32.pt`（candidate_selection_counterfactual_bspline_v16，epoch 86）。
统一阈值：MSE ≤ 1.000e-04；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 56, 'greedy_initial_and_yeh_max': 56, 'kang_dense_initial': 56, 'liang_dense_initial': 56, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 64, 'numerical_full_knot_vector_cap': 64}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 10 | 100.0% | 3.963e-05 | 14.70 | 7.53 | 6.29 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 10 | 100.0% | 7.460e-05 | 24.80 | 30.57 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 8.141e-05 | 19.20 | 19.81 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 2.682e-03 | 7.30 | 155.05 | — |
| Synthetic | Kang 2015 (ADMM adaptation) | 10 | 60.0% | 9.300e-05 | 10.50 | 2730.18 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 60.0% | 9.058e-04 | 10.30 | 7990.91 | — |
| UJI | Ours v16 learned | 10 | 100.0% | 1.799e-05 | 7.30 | 7.46 | 6.28 |
| UJI | Park & Lee 2007 (DOM adaptation) | 10 | 100.0% | 5.620e-05 | 6.60 | 9.39 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 3.345e-05 | 5.80 | 5.99 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 60.0% | 4.469e-04 | 2.20 | 51.02 | — |
| UJI | Kang 2015 (ADMM adaptation) | 10 | 80.0% | 7.449e-05 | 3.30 | 1675.67 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 60.0% | 3.796e-04 | 3.60 | 7328.56 | — |
| NaturalEarth | Ours v16 learned | 10 | 90.0% | 7.029e-05 | 22.10 | 7.71 | 6.37 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 10 | 90.0% | 1.006e-04 | 19.60 | 27.88 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 10 | 90.0% | 1.058e-04 | 18.40 | 20.43 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 9.611e-04 | 10.00 | 240.18 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 10 | 70.0% | 5.888e-03 | 8.80 | 2452.54 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 10.0% | 3.124e-03 | 5.30 | 6674.46 | — |
| USGS | Ours v16 learned | 10 | 100.0% | 4.195e-05 | 28.30 | 7.94 | 6.46 |
| USGS | Park & Lee 2007 (DOM adaptation) | 10 | 100.0% | 7.581e-05 | 26.50 | 36.10 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 8.806e-05 | 20.00 | 22.57 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 5.063e-04 | 10.80 | 283.51 | — |
| USGS | Kang 2015 (ADMM adaptation) | 10 | 60.0% | 5.275e-03 | 7.40 | 2259.89 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 10.0% | 9.622e-04 | 7.30 | 6821.09 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 100.0% | 2.822e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 60.0% | 1.042e-04 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 70.0% | 6.703e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 50.0% | 5.525e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 50.0% | 1.327e-04 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 50.0% | 4.524e-04 |
| NaturalEarth | Ours v16 learned | 90.0% | 7.429e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 90.0% | 1.066e-04 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 70.0% | 1.115e-04 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 9.727e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 70.0% | 6.025e-03 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10.0% | 3.148e-03 |
| USGS | Ours v16 learned | 100.0% | 4.282e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 100.0% | 7.719e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 90.0% | 8.976e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 5.117e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 60.0% | 5.305e-03 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10.0% | 9.708e-04 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
