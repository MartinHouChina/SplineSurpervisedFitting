# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；certified synthetic matched-knot MAE exceeds the formal limit 0.005；observed worst-source deployment pass rate 9.375% is below the required 90.000%；observed worst-source dense proposal pass rate 9.375% is below the required 90.000%；proposal feasibility gate was not satisfied；infeasible-proposal ablation was enabled

Checkpoint: `/home/feng/HouCode/SplineFitting_1070_overnight/outputs/checkpoints/universal_m16_m32_3090_r1_m16.pt`（candidate_selection_counterfactual_bspline_v16，epoch 24）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 16, 'greedy_initial_and_yeh_max': 16, 'kang_dense_initial': 16, 'liang_dense_initial': 16, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 24, 'numerical_full_knot_vector_cap': 24}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。
数值基线协议：**threshold-safe adaptation**。Dung/Kang/Luo 在原生公共 refit 未达阈值时可执行有界修复；保留最佳拟合但不保证容量内必然可行。所有额外 refit 计入完整耗时；这不是原文算法步骤。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 42 | 16.7% | 1.647e-03 | 15.33 | 14.49 | 13.25 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 42 | 2.4% | 1.351e-03 | 16.00 | 19.10 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 42 | 11.9% | 1.185e-03 | 15.74 | 14.96 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 42 | 21.4% | 1.448e-03 | 15.17 | 344.03 | — |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 42 | 16.7% | 1.814e-03 | 16.00 | 70.00 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 42 | 21.4% | 1.650e-03 | 15.38 | 784.05 | — |
| UJI | Ours v16 learned | 20 | 70.0% | 5.503e-05 | 14.30 | 14.46 | 13.23 |
| UJI | Park & Lee 2007 (DOM adaptation) | 20 | 75.0% | 8.753e-05 | 9.20 | 12.22 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 20 | 85.0% | 3.957e-05 | 8.35 | 7.58 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 95.0% | 3.483e-05 | 7.20 | 105.41 | — |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 75.0% | 5.889e-05 | 11.70 | 444.43 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 75.0% | 5.501e-05 | 8.45 | 1878.40 | — |
| NaturalEarth | Ours v16 learned | 20 | 10.0% | 6.735e-04 | 15.70 | 14.49 | 13.25 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 20 | 25.0% | 7.102e-04 | 15.50 | 20.30 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 20 | 35.0% | 3.761e-04 | 15.15 | 14.35 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 50.0% | 4.276e-04 | 14.55 | 289.47 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 0.0% | 4.807e-04 | 16.00 | 5.60 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 20.0% | 4.458e-04 | 15.65 | 566.19 | — |
| USGS | Ours v16 learned | 20 | 15.0% | 8.021e-04 | 15.45 | 14.49 | 13.24 |
| USGS | Park & Lee 2007 (DOM adaptation) | 20 | 15.0% | 9.379e-04 | 14.35 | 17.33 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 20 | 25.0% | 5.713e-04 | 14.35 | 13.58 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 40.0% | 5.826e-04 | 13.80 | 352.41 | — |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 25.0% | 6.360e-04 | 15.00 | 154.81 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 35.0% | 6.020e-04 | 14.15 | 919.71 | — |
| IndustrialOffset | Ours v16 learned | 20 | 75.0% | 9.517e-05 | 14.10 | 14.47 | 13.26 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 20 | 70.0% | 2.820e-04 | 12.15 | 17.83 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 20 | 80.0% | 8.008e-05 | 9.55 | 8.75 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 20 | 85.0% | 6.746e-05 | 9.40 | 129.48 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 20 | 55.0% | 1.643e-04 | 15.00 | 301.23 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 20 | 65.0% | 1.555e-04 | 11.35 | 1487.40 | — |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 输入点 | 6.065e-03 | 1.338e-02 | 1.943e-02 | 4.169e-03 | 42/0 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 输入点 | 9.880e-03 | 2.063e-02 | 2.351e-02 | 3.407e-03 | 42/0 |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 5.694e-03 | 1.155e-02 | 1.247e-02 | 3.206e-03 | 42/0 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 6.345e-03 | 1.568e-02 | 1.998e-02 | 4.071e-03 | 42/0 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 6.951e-03 | 1.515e-02 | 1.710e-02 | 4.136e-03 | 42/0 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 6.623e-03 | 1.494e-02 | 1.927e-02 | 4.049e-03 | 42/0 |
| UJI | Ours v16 learned | 输入点 | 6.103e-04 | 1.873e-03 | 2.743e-03 | 4.441e-04 | 20/0 |
| UJI | Ours v16 learned | 原始参考点 | 6.982e-04 | 2.462e-03 | 3.547e-03 | 3.893e-04 | 20/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.568e-03 | 7.087e-03 | 7.567e-03 | 4.088e-04 | 20/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 1.865e-03 | 7.916e-03 | 8.609e-03 | 6.697e-04 | 20/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 4.145e-04 | 1.055e-03 | 1.148e-03 | 1.392e-04 | 20/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 5.141e-04 | 1.243e-03 | 1.285e-03 | 3.397e-04 | 20/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 3.480e-04 | 9.802e-04 | 9.858e-04 | 6.233e-05 | 20/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 4.768e-04 | 1.120e-03 | 1.620e-03 | 3.092e-04 | 20/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 1.046e-03 | 2.823e-03 | 4.866e-03 | 2.582e-04 | 20/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 1.321e-03 | 4.036e-03 | 5.283e-03 | 5.254e-04 | 20/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 8.870e-04 | 2.794e-03 | 4.265e-03 | 2.408e-04 | 20/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 1.143e-03 | 3.996e-03 | 4.467e-03 | 4.358e-04 | 20/0 |
| NaturalEarth | Ours v16 learned | 输入点 | 3.643e-03 | 1.245e-02 | 3.001e-02 | 6.875e-03 | 20/0 |
| NaturalEarth | Ours v16 learned | 原始参考点 | 4.092e-03 | 1.280e-02 | 3.357e-02 | 6.909e-03 | 20/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 输入点 | 5.395e-03 | 3.014e-02 | 3.759e-02 | 5.963e-03 | 20/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 5.901e-03 | 3.364e-02 | 4.099e-02 | 6.029e-03 | 20/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 1.980e-03 | 7.370e-03 | 1.643e-02 | 4.353e-03 | 20/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 2.305e-03 | 8.238e-03 | 1.922e-02 | 4.406e-03 | 20/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 3.274e-03 | 9.805e-03 | 3.838e-02 | 4.868e-03 | 20/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 3.509e-03 | 1.069e-02 | 3.874e-02 | 4.921e-03 | 20/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 4.179e-03 | 9.805e-03 | 3.838e-02 | 4.868e-03 | 20/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 4.553e-03 | 1.069e-02 | 3.874e-02 | 4.921e-03 | 20/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 3.781e-03 | 9.351e-03 | 3.843e-02 | 4.717e-03 | 20/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 4.002e-03 | 9.799e-03 | 3.878e-02 | 4.761e-03 | 20/0 |
| USGS | Ours v16 learned | 输入点 | 4.593e-03 | 1.274e-02 | 1.809e-02 | 3.012e-03 | 20/0 |
| USGS | Ours v16 learned | 原始参考点 | 4.936e-03 | 1.421e-02 | 1.975e-02 | 3.054e-03 | 20/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 输入点 | 6.266e-03 | 2.157e-02 | 2.373e-02 | 4.083e-03 | 20/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 6.561e-03 | 2.209e-02 | 2.625e-02 | 4.140e-03 | 20/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.868e-03 | 9.252e-03 | 1.119e-02 | 2.844e-03 | 20/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 4.703e-02 | 5.848e-02 | 8.755e-01 | 2.908e-03 | 20/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 3.688e-03 | 1.251e-02 | 1.430e-02 | 2.261e-03 | 20/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 4.088e-03 | 1.434e-02 | 1.496e-02 | 2.314e-03 | 20/0 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 3.982e-03 | 1.233e-02 | 1.264e-02 | 2.292e-03 | 20/0 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 4.329e-03 | 1.368e-02 | 1.496e-02 | 2.324e-03 | 20/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 3.652e-03 | 1.248e-02 | 1.264e-02 | 2.292e-03 | 20/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 3.961e-03 | 1.281e-02 | 1.313e-02 | 2.324e-03 | 20/0 |
| IndustrialOffset | Ours v16 learned | 输入点 | 6.405e-04 | 2.757e-03 | 3.434e-03 | 1.058e-03 | 20/0 |
| IndustrialOffset | Ours v16 learned | 原始参考点 | 7.247e-04 | 2.758e-03 | 3.466e-03 | 1.061e-03 | 20/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.819e-03 | 1.074e-02 | 1.089e-02 | 3.546e-03 | 20/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 1.991e-03 | 1.084e-02 | 1.276e-02 | 3.560e-03 | 20/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 4.682e-04 | 1.039e-03 | 2.876e-03 | 6.786e-04 | 20/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 5.064e-04 | 1.040e-03 | 2.876e-03 | 6.823e-04 | 20/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 5.930e-04 | 9.036e-04 | 5.299e-03 | 5.920e-04 | 20/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 6.533e-04 | 9.534e-04 | 5.327e-03 | 5.951e-04 | 20/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 1.601e-03 | 5.819e-03 | 7.779e-03 | 1.168e-03 | 20/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 1.763e-03 | 6.930e-03 | 7.966e-03 | 1.174e-03 | 20/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 1.566e-03 | 5.436e-03 | 7.777e-03 | 9.609e-04 | 20/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 1.758e-03 | 6.530e-03 | 7.961e-03 | 9.658e-04 | 20/0 |

## 原生适配与最终结果审计

native 指修正后的仓库适配在公共可行性修复前的端点约束 refit，非作者原版复现。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 5.35 | 9.40 | 1.896e-03 | 6.746e-05 | 12.30 | 20/20 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 15.00 | 15.00 | 1.643e-04 | 1.643e-04 | 0.45 | 9/20 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1.90 | 11.35 | 5.145e-03 | 1.555e-04 | 28.65 | 20/20 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 10.30 | 14.55 | 9.173e-04 | 4.276e-04 | 13.25 | 20/20 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 16.00 | 16.00 | 4.807e-04 | 4.807e-04 | 1.00 | 20/20 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1.75 | 15.65 | 1.014e-02 | 4.458e-04 | 42.15 | 20/20 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 12.74 | 15.17 | 2.187e-03 | 1.448e-03 | 8.07 | 42/42 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 16.00 | 16.00 | 1.814e-03 | 1.814e-03 | 0.83 | 35/42 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 2.24 | 15.38 | 4.897e-03 | 1.650e-03 | 39.76 | 42/42 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 3.95 | 7.20 | 6.503e-04 | 3.483e-05 | 9.80 | 17/20 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 11.65 | 11.70 | 6.171e-05 | 5.889e-05 | 0.40 | 6/20 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 2.20 | 8.45 | 4.127e-03 | 5.501e-05 | 19.00 | 16/20 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 12.10 | 13.80 | 8.835e-04 | 5.826e-04 | 5.70 | 19/20 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 14.95 | 15.00 | 6.375e-04 | 6.360e-04 | 0.90 | 16/20 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1.90 | 14.15 | 1.074e-02 | 6.020e-04 | 37.10 | 19/20 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 65.0% | 6.299e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 30.0% | 1.685e-04 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 30.0% | 8.651e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 35.0% | 7.405e-05 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 30.0% | 1.355e-04 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 30.0% | 1.222e-04 |
| NaturalEarth | Ours v16 learned | 10.0% | 6.754e-04 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 25.0% | 7.165e-04 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 35.0% | 3.822e-04 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 45.0% | 4.342e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 0.0% | 4.882e-04 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 15.0% | 4.523e-04 |
| USGS | Ours v16 learned | 15.0% | 8.064e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 15.0% | 9.468e-04 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 25.0% | 6.739e-04 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 35.0% | 5.902e-04 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 25.0% | 6.441e-04 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 35.0% | 6.097e-04 |
| IndustrialOffset | Ours v16 learned | 75.0% | 9.549e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 70.0% | 2.840e-04 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 80.0% | 8.063e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 80.0% | 6.796e-05 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 55.0% | 1.662e-04 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 60.0% | 1.572e-04 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
