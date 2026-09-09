# v15 多数据集配对测试

说明：本轮 Kang-inspired 适配在连续活跃节点簇合并时退化，应视为实现诊断，不能据此评价原论文性能。完整结论与复现限制见 [实测解释报告](../../../docs/archive/v15_multidataset_results_20260908.md)。以下数值保持原始测量不变。

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\checkpoints\candidate_pruning_one_shot_v15.pt`（candidate_pruning_deployment_aligned_feedback_v15，epoch 121）。
统一阈值：MSE ≤ 2.500e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化点，最终使用 CPU float64、端点插值、无正则标准 B 样条 refit。
完整耗时从归一化 CPU 输入开始，包含方法本身和 refit；不含数据加载、归一化和评价指标计算。网络时间另列。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce GTX 1070', 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 1, 'python': '3.13.9'}。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v15 learned | 34 | 47.1% | 3.907e-05 | 10.06 | 52.07 | 48.31 |
| Synthetic | Kang 2015 (ADMM adaptation) | 34 | 58.8% | 5.306e-05 | 6.74 | 2446.44 | — |
| Synthetic | Yeh 2020 (feature-CDF + K scan) | 34 | 100.0% | 1.580e-05 | 8.88 | 21.55 | — |
| Synthetic | Uniform Kmax greedy + gradient | 34 | 100.0% | 1.703e-05 | 6.56 | 22270.29 | — |
| UJI | Ours v15 learned | 20 | 70.0% | 6.703e-05 | 11.85 | 50.13 | 46.81 |
| UJI | Kang 2015 (ADMM adaptation) | 20 | 35.0% | 6.988e-03 | 4.10 | 1993.86 | — |
| UJI | Yeh 2020 (feature-CDF + K scan) | 20 | 100.0% | 1.868e-05 | 11.70 | 28.28 | — |
| UJI | Uniform Kmax greedy + gradient | 20 | 100.0% | 1.660e-05 | 7.20 | 21388.30 | — |
| NaturalEarth | Ours v15 learned | 20 | 0.0% | 1.306e-03 | 13.50 | 55.31 | 47.43 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 20 | 0.0% | 1.434e-02 | 1.00 | 691.92 | — |
| NaturalEarth | Yeh 2020 (feature-CDF + K scan) | 20 | 50.0% | 1.610e-04 | 23.65 | 57.22 | — |
| NaturalEarth | Uniform Kmax greedy + gradient | 20 | 65.0% | 1.272e-04 | 19.55 | 12157.52 | — |
| USGS | Ours v15 learned | 20 | 30.0% | 8.353e-04 | 14.35 | 53.44 | 48.79 |
| USGS | Kang 2015 (ADMM adaptation) | 20 | 10.0% | 1.107e-02 | 2.35 | 968.93 | — |
| USGS | Yeh 2020 (feature-CDF + K scan) | 20 | 45.0% | 1.414e-04 | 21.70 | 60.23 | — |
| USGS | Uniform Kmax greedy + gradient | 20 | 50.0% | 1.160e-04 | 19.90 | 10946.67 | — |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v15 learned | 30.0% | 1.269e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 10.0% | 7.611e-03 |
| UJI | Yeh 2020 (feature-CDF + K scan) | 10.0% | 5.432e-05 |
| UJI | Uniform Kmax greedy + gradient | 30.0% | 4.342e-05 |
| NaturalEarth | Ours v15 learned | 0.0% | 1.313e-03 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 0.0% | 1.442e-02 |
| NaturalEarth | Yeh 2020 (feature-CDF + K scan) | 40.0% | 1.651e-04 |
| NaturalEarth | Uniform Kmax greedy + gradient | 55.0% | 1.310e-04 |
| USGS | Ours v15 learned | 30.0% | 8.437e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 10.0% | 1.112e-02 |
| USGS | Yeh 2020 (feature-CDF + K scan) | 40.0% | 1.442e-04 |
| USGS | Uniform Kmax greedy + gradient | 50.0% | 1.184e-04 |

Kang 为 group-L1 ADMM 二维适配，非原文 CVX 完整复现；Yeh 为论文布点公式加递增 K 扫描包装。失败样本计入通过率分母；无有限解时 MSE 均值只含有解样本，failed 列另存。

小样本结果仅用于初步比较，真实曲线同一 writer/tile 的相关性使有效独立样本数低于曲线总数。原始记录、样本选择和全部配置保存在 comparison.json。

论文来源：[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
