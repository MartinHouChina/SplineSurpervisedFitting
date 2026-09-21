# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；observed worst-source deployment pass rate 64.400% is below the required 90.000%；observed worst-source dense proposal pass rate 84.375% is below the required 90.000%；proposal feasibility gate was not satisfied；infeasible-proposal ablation was enabled

Checkpoint: `/home/feng/HouCode/SplineFitting_1070_overnight/outputs/checkpoints/overnight_anchored_k32_3090_r1.pt`（candidate_selection_counterfactual_bspline_v16，epoch 5）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 32, 'greedy_initial_and_yeh_max': 32, 'kang_dense_initial': 32, 'liang_dense_initial': 32, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 40, 'numerical_full_knot_vector_cap': 40}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。
数值基线协议：**threshold-safe adaptation**。Dung/Kang/Luo 在原生公共 refit 未达阈值时可执行有界修复；保留最佳拟合但不保证容量内必然可行。所有额外 refit 计入完整耗时；这不是原文算法步骤。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 42 | 69.0% | 4.069e-05 | 27.38 | 8.97 | 7.57 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 42 | 26.2% | 1.361e-04 | 30.40 | 38.79 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 42 | 45.2% | 1.316e-04 | 27.43 | 28.69 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 42 | 66.7% | 5.386e-05 | 23.76 | 401.07 | — |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 42 | 47.6% | 9.951e-05 | 28.71 | 292.47 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 42 | 52.4% | 1.036e-04 | 25.71 | 1444.62 | — |
| UJI | Ours v16 learned | 20 | 90.0% | 1.925e-05 | 23.00 | 8.94 | 7.56 |
| UJI | Park & Lee 2007 (DOM adaptation) | 20 | 100.0% | 3.388e-05 | 11.80 | 15.60 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 20 | 100.0% | 3.380e-05 | 8.85 | 8.33 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 100.0% | 3.140e-05 | 7.35 | 104.46 | — |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 100.0% | 2.413e-05 | 18.50 | 772.07 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 100.0% | 2.857e-05 | 10.45 | 2510.11 | — |
| NaturalEarth | Ours v16 learned | 20 | 65.0% | 1.367e-04 | 28.90 | 8.98 | 7.55 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 20 | 80.0% | 1.531e-04 | 22.40 | 29.66 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 20 | 85.0% | 1.211e-04 | 20.95 | 21.15 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 85.0% | 1.281e-04 | 18.90 | 328.45 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 75.0% | 1.500e-04 | 31.40 | 337.18 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 75.0% | 1.350e-04 | 21.60 | 1802.13 | — |
| USGS | Ours v16 learned | 20 | 40.0% | 1.475e-04 | 28.90 | 8.98 | 7.56 |
| USGS | Park & Lee 2007 (DOM adaptation) | 20 | 40.0% | 2.392e-04 | 25.15 | 31.82 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 20 | 45.0% | 1.266e-04 | 24.25 | 25.36 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 45.0% | 1.368e-04 | 23.20 | 461.64 | — |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 40.0% | 1.573e-04 | 29.65 | 266.18 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 40.0% | 1.502e-04 | 23.55 | 1399.38 | — |
| IndustrialOffset | Ours v16 learned | 20 | 85.0% | 4.847e-05 | 21.55 | 8.95 | 7.56 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 20 | 90.0% | 4.410e-05 | 14.80 | 21.30 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 20 | 100.0% | 2.962e-05 | 11.30 | 10.81 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 100.0% | 3.499e-05 | 10.10 | 130.17 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 100.0% | 1.520e-05 | 22.40 | 729.06 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 100.0% | 3.136e-05 | 11.55 | 2658.33 | — |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 输入点 | 3.837e-04 | 8.345e-04 | 1.441e-03 | 1.671e-04 | 42/0 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 输入点 | 2.226e-03 | 7.418e-03 | 9.557e-03 | 5.498e-04 | 42/0 |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 7.789e-04 | 3.040e-03 | 3.483e-03 | 6.818e-04 | 42/0 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 2.389e-04 | 4.808e-04 | 7.084e-04 | 9.990e-05 | 42/0 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 6.664e-04 | 3.102e-03 | 3.492e-03 | 4.137e-04 | 42/0 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 6.965e-04 | 3.090e-03 | 3.492e-03 | 4.137e-04 | 42/0 |
| UJI | Ours v16 learned | 输入点 | 3.038e-04 | 8.623e-04 | 9.917e-04 | 6.404e-05 | 20/0 |
| UJI | Ours v16 learned | 原始参考点 | 3.926e-04 | 9.883e-04 | 1.426e-03 | 1.099e-04 | 20/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 输入点 | 6.084e-04 | 1.663e-03 | 1.863e-03 | 4.950e-05 | 20/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 7.946e-04 | 2.453e-03 | 2.537e-03 | 2.292e-04 | 20/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 3.932e-04 | 8.126e-04 | 8.139e-04 | 4.905e-05 | 20/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 4.943e-04 | 1.060e-03 | 1.278e-03 | 2.421e-04 | 20/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 2.594e-04 | 4.283e-04 | 6.960e-04 | 4.811e-05 | 20/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 3.427e-04 | 6.210e-04 | 9.891e-04 | 2.445e-04 | 20/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 5.365e-04 | 1.035e-03 | 1.341e-03 | 4.846e-05 | 20/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 7.123e-04 | 1.475e-03 | 1.475e-03 | 1.925e-04 | 20/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 4.142e-04 | 1.081e-03 | 1.394e-03 | 4.885e-05 | 20/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 5.706e-04 | 1.537e-03 | 1.596e-03 | 2.341e-04 | 20/0 |
| NaturalEarth | Ours v16 learned | 输入点 | 1.008e-03 | 3.721e-03 | 7.862e-03 | 1.491e-03 | 20/0 |
| NaturalEarth | Ours v16 learned | 原始参考点 | 1.095e-03 | 3.934e-03 | 8.327e-03 | 1.525e-03 | 20/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.338e-03 | 5.371e-03 | 5.918e-03 | 1.288e-03 | 20/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 1.518e-03 | 5.608e-03 | 6.826e-03 | 1.321e-03 | 20/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 6.826e-04 | 1.275e-03 | 5.167e-03 | 1.219e-03 | 20/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 2.127e-02 | 2.675e-02 | 4.085e-01 | 1.250e-03 | 20/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 8.219e-04 | 1.929e-03 | 7.056e-03 | 1.464e-03 | 20/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 1.088e-03 | 2.455e-03 | 8.798e-03 | 1.497e-03 | 20/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 1.371e-03 | 3.248e-03 | 1.319e-02 | 1.906e-03 | 20/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 1.667e-03 | 4.274e-03 | 1.404e-02 | 1.941e-03 | 20/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 1.026e-03 | 2.957e-03 | 5.821e-03 | 1.317e-03 | 20/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 1.364e-03 | 4.035e-03 | 7.831e-03 | 1.351e-03 | 20/0 |
| USGS | Ours v16 learned | 输入点 | 1.021e-03 | 3.562e-03 | 3.851e-03 | 5.716e-04 | 20/0 |
| USGS | Ours v16 learned | 原始参考点 | 1.150e-03 | 4.225e-03 | 4.516e-03 | 5.919e-04 | 20/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 输入点 | 2.300e-03 | 7.563e-03 | 1.064e-02 | 1.266e-03 | 20/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 2.522e-03 | 7.864e-03 | 1.219e-02 | 1.292e-03 | 20/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 6.673e-04 | 1.705e-03 | 3.139e-03 | 5.692e-04 | 20/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 9.101e-04 | 2.841e-03 | 5.269e-03 | 6.020e-04 | 20/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 8.928e-04 | 3.054e-03 | 5.090e-03 | 7.554e-04 | 20/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 1.160e-03 | 4.140e-03 | 7.064e-03 | 7.866e-04 | 20/0 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 1.204e-03 | 3.490e-03 | 5.090e-03 | 7.554e-04 | 20/0 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 1.468e-03 | 4.140e-03 | 7.064e-03 | 7.866e-04 | 20/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 1.103e-03 | 2.977e-03 | 3.564e-03 | 5.724e-04 | 20/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 1.326e-03 | 4.022e-03 | 4.706e-03 | 6.010e-04 | 20/0 |
| IndustrialOffset | Ours v16 learned | 输入点 | 4.295e-04 | 2.144e-03 | 2.160e-03 | 3.109e-04 | 20/0 |
| IndustrialOffset | Ours v16 learned | 原始参考点 | 4.804e-04 | 2.145e-03 | 2.166e-03 | 3.124e-04 | 20/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 输入点 | 5.640e-04 | 1.627e-03 | 1.674e-03 | 1.360e-04 | 20/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 6.683e-04 | 2.125e-03 | 2.245e-03 | 1.414e-04 | 20/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.640e-04 | 5.506e-04 | 6.362e-04 | 4.558e-05 | 20/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 3.078e-04 | 5.616e-04 | 6.365e-04 | 4.578e-05 | 20/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 2.721e-04 | 3.995e-04 | 6.722e-04 | 4.946e-05 | 20/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 3.444e-04 | 6.733e-04 | 7.231e-04 | 5.164e-05 | 20/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 2.245e-04 | 7.130e-04 | 1.102e-03 | 4.124e-05 | 20/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 3.079e-04 | 1.171e-03 | 1.222e-03 | 4.148e-05 | 20/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 3.772e-04 | 1.090e-03 | 1.105e-03 | 4.915e-05 | 20/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 1.574e+05 | 1.574e+05 | 3.149e+06 | 4.869e+03 | 20/0 |

## 原生适配与最终结果审计

native 指修正后的仓库适配在公共可行性修复前的端点约束 refit，非作者原版复现。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 5.35 | 10.10 | 1.896e-03 | 3.499e-05 | 14.25 | 20/20 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 22.40 | 22.40 | 1.520e-05 | 1.520e-05 | 0.00 | 0/20 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 4.95 | 11.55 | 2.881e-03 | 3.136e-05 | 19.80 | 16/20 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 12.20 | 18.90 | 5.483e-04 | 1.281e-04 | 20.25 | 20/20 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 31.40 | 31.40 | 1.500e-04 | 1.500e-04 | 0.25 | 5/20 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 3.10 | 21.60 | 4.206e-03 | 1.350e-04 | 55.75 | 20/20 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 14.81 | 23.76 | 1.047e-03 | 5.386e-05 | 27.19 | 42/42 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 26.05 | 28.71 | 4.746e-04 | 9.951e-05 | 8.29 | 27/42 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 4.21 | 25.71 | 4.285e-03 | 1.036e-04 | 64.52 | 39/42 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 3.95 | 7.35 | 6.503e-04 | 3.140e-05 | 10.20 | 17/20 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 18.45 | 18.50 | 2.760e-05 | 2.413e-05 | 0.15 | 1/20 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 4.00 | 10.45 | 3.563e-03 | 2.857e-05 | 19.35 | 12/20 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 17.20 | 23.20 | 4.990e-04 | 1.368e-04 | 18.55 | 19/20 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 29.65 | 29.65 | 1.573e-04 | 1.573e-04 | 0.60 | 12/20 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 4.10 | 23.55 | 3.394e-03 | 1.502e-04 | 58.95 | 18/20 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 75.0% | 3.668e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 35.0% | 7.970e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 35.0% | 7.641e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 45.0% | 6.371e-05 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 30.0% | 6.434e-05 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 45.0% | 6.835e-05 |
| NaturalEarth | Ours v16 learned | 70.0% | 1.389e-04 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 75.0% | 1.566e-04 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 70.0% | 1.802e-04 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 75.0% | 1.317e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 70.0% | 1.540e-04 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 50.0% | 1.388e-04 |
| USGS | Ours v16 learned | 40.0% | 1.491e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 40.0% | 2.435e-04 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 45.0% | 1.306e-04 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 45.0% | 1.406e-04 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 40.0% | 1.614e-04 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 40.0% | 1.541e-04 |
| IndustrialOffset | Ours v16 learned | 85.0% | 4.877e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 90.0% | 4.483e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 2.992e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 95.0% | 3.537e-05 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 1.553e-05 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 90.0% | 2.434e+02 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
