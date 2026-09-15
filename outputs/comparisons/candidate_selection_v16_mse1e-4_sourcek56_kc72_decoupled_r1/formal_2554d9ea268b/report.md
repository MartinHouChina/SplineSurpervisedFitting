# v16 多数据集配对测试

Checkpoint: `/home/feng/HouCode/SplineSurpervisedFitting/outputs/checkpoints/candidate_selection_v16_mse1e-4_sourcek56_kc72_decoupled_r1.pt`（candidate_selection_supervised_bspline_v16，epoch 128）。
统一阈值：MSE ≤ 1.000e-04；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 72, 'source_max_internal_knots': 56, 'network_candidate_overhead_vs_source': 16, 'greedy_initial_and_yeh_max': 56, 'kang_dense_initial': 56, 'liang_dense_initial': 56, 'equal_initial_capacity': False, 'numerical_caps_match_source_max': True, 'formal_overcomplete_candidate_contract': True, 'main_comparison_capacity_valid': True, 'capacity_contract': 'formal_v16_kc72_source_and_baselines_kmax56', 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 80, 'numerical_full_knot_vector_cap': 64}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 23 | 8.7% | 1.798e-03 | 8.91 | 8.04 | 6.81 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 23 | 100.0% | 5.898e-05 | 29.48 | 40.58 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 23 | 100.0% | 7.285e-05 | 15.61 | 16.96 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (threshold-safe adaptation) | 23 | 100.0% | 7.951e-05 | 13.65 | 152.38 | — |
| Synthetic | Kang 2015 (threshold-safe ADMM adaptation) | 23 | 100.0% | 6.776e-05 | 8.00 | 2985.52 | — |
| Synthetic | Luo et al. 2022 (threshold-safe l-infinity,1 + DE adaptation) | 23 | 100.0% | 4.051e-05 | 10.83 | 8546.02 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
