# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT**

资格检查：formal validation K=56 boundary audit must contain at least 32 synthetic samples

Checkpoint: `/home/feng/HouCode/SplineSurpervisedFitting/outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.pt`（candidate_selection_counterfactual_bspline_v16，epoch 70）。
统一阈值：MSE ≤ 1.000e-04；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 56, 'greedy_initial_and_yeh_max': 56, 'kang_dense_initial': 56, 'liang_dense_initial': 56, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 64, 'numerical_full_knot_vector_cap': 64}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 10 | 100.0% | 1.958e-05 | 17.40 | 7.71 | 6.42 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 10 | 100.0% | 5.819e-05 | 33.90 | 45.92 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 8.021e-05 | 19.70 | 19.99 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 2.621e-03 | 7.80 | 119.33 | — |
| Synthetic | Kang 2015 (ADMM adaptation) | 10 | 30.0% | 1.455e-03 | 6.90 | 243.79 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 30.0% | 9.388e-04 | 11.00 | 627.96 | — |
| UJI | Ours v16 learned | 10 | 100.0% | 1.756e-05 | 8.00 | 7.45 | 6.25 |
| UJI | Park & Lee 2007 (DOM adaptation) | 10 | 100.0% | 5.620e-05 | 6.60 | 9.57 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 3.230e-05 | 5.80 | 5.58 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 60.0% | 4.477e-04 | 2.20 | 37.68 | — |
| UJI | Kang 2015 (ADMM adaptation) | 10 | 50.0% | 3.672e-04 | 1.90 | 138.23 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 60.0% | 7.434e-04 | 2.70 | 584.59 | — |
| NaturalEarth | Ours v16 learned | 10 | 90.0% | 6.252e-05 | 23.90 | 7.73 | 6.24 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 10 | 90.0% | 1.006e-04 | 19.60 | 27.82 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 10 | 90.0% | 1.031e-04 | 18.50 | 19.98 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 9.579e-04 | 10.00 | 161.26 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 10 | 50.0% | 5.816e-03 | 8.30 | 238.52 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 0.0% | 2.668e-03 | 5.60 | 534.71 | — |
| USGS | Ours v16 learned | 10 | 100.0% | 3.208e-05 | 30.20 | 7.81 | 6.25 |
| USGS | Park & Lee 2007 (DOM adaptation) | 10 | 100.0% | 7.581e-05 | 26.50 | 35.66 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 8.764e-05 | 20.30 | 21.99 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 5.023e-04 | 10.80 | 185.18 | — |
| USGS | Kang 2015 (ADMM adaptation) | 10 | 50.0% | 5.508e-03 | 6.40 | 209.29 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 20.0% | 1.594e-03 | 7.90 | 556.63 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 100.0% | 3.092e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 60.0% | 1.042e-04 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 70.0% | 6.520e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 50.0% | 5.533e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 30.0% | 5.012e-04 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 50.0% | 8.262e-04 |
| NaturalEarth | Ours v16 learned | 90.0% | 6.669e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 90.0% | 1.066e-04 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 90.0% | 1.093e-04 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 9.695e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 50.0% | 5.952e-03 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 2.703e-03 |
| USGS | Ours v16 learned | 100.0% | 3.289e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 100.0% | 7.719e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 80.0% | 8.937e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 5.077e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 50.0% | 5.539e-03 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 20.0% | 1.605e-03 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
