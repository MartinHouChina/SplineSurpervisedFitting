# v16 多数据集配对测试

Per-case measured geometry: each measurements.jsonl row links a hash-verified JSON + NPZ artifact under geometry/. These include parameters, full/internal knots, control vertices, dense curves, pointwise residuals and original-coordinate transforms. Failures/unavailable outputs remain absent; export and plots do not rerun fitting. NumPy loading uses allow_pickle=False.

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；observed worst-source deployment pass rate 36.000% is below the required 90.000%；observed worst-source dense proposal pass rate 78.000% is below the required 90.000%；proposal feasibility gate was not satisfied；infeasible-proposal ablation was enabled

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\downloads\server_models_20260922_215418\paper_coupled_clean_3090_r3_m32.pt`（candidate_selection_counterfactual_bspline_v16，epoch 13）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点。新协议直接评估实际返回的曲线：Dung/Kang 保留算法内无端点强制约束的最小二乘解，不再附加公共端点 refit；Ours 使用自身部署 refit，其余对照保留当前求解路径，详见结果中的逐方法协议。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 32, 'greedy_initial_and_yeh_max': 32, 'kang_dense_initial': 32, 'liang_dense_initial': 32, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 40, 'numerical_full_knot_vector_cap': 40}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce GTX 1070', 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 4, 'python': '3.13.9'}。
数值基线协议：未启用或历史报告未记录公共可行性修复；不将结果标为 threshold-safe adaptation。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 21 | 66.7% | 5.642e-05 | 26.14 | 88.00 | 65.16 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 21 | 33.3% | 1.460e-04 | 30.38 | 151.88 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 21 | 38.1% | 1.206e-04 | 27.52 | 129.10 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 21 | 0.0% | 8.821e-04 | 14.81 | 1301.24 | — |
| Synthetic | Kang 2015 (ADMM adaptation) | 21 | 0.0% | 4.706e-03 | 1.00 | 1094.22 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 21 | 0.0% | 6.275e-03 | 3.38 | 3460.53 | — |
| UJI | Ours v16 learned | 10 | 90.0% | 2.058e-05 | 17.80 | 71.96 | 53.61 |
| UJI | Park & Lee 2007 (DOM adaptation) | 10 | 100.0% | 2.932e-05 | 5.70 | 26.03 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 2.640e-05 | 5.90 | 30.06 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 50.0% | 2.633e-04 | 3.00 | 192.20 | — |
| UJI | Kang 2015 (ADMM adaptation) | 10 | 50.0% | 4.669e-03 | 0.80 | 2541.33 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 70.0% | 5.770e-03 | 3.10 | 7326.51 | — |
| NaturalEarth | Ours v16 learned | 10 | 40.0% | 9.265e-04 | 26.90 | 77.98 | 57.02 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 10 | 60.0% | 2.332e-04 | 23.00 | 113.06 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 10 | 70.0% | 1.669e-04 | 22.00 | 77.25 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 5.077e-04 | 13.33 | 815.05 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 10 | 0.0% | 2.634e-02 | 1.30 | 1091.50 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 0.0% | 9.013e-03 | 3.60 | 4544.20 | — |
| USGS | Ours v16 learned | 10 | 50.0% | 7.623e-05 | 26.10 | 80.11 | 63.78 |
| USGS | Park & Lee 2007 (DOM adaptation) | 10 | 50.0% | 1.001e-04 | 21.00 | 93.34 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 10 | 60.0% | 6.235e-05 | 20.40 | 65.80 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 4.015e-04 | 11.80 | 965.05 | — |
| USGS | Kang 2015 (ADMM adaptation) | 10 | 0.0% | 7.784e-03 | 1.30 | 1663.90 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 10.0% | 2.874e-03 | 3.80 | 5465.57 | — |
| IndustrialOffset | Ours v16 learned | 10 | 80.0% | 6.608e-05 | 19.60 | 70.93 | 52.73 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 10 | 90.0% | 4.532e-05 | 14.20 | 63.94 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 10 | 100.0% | 3.582e-05 | 11.70 | 31.97 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 1.613e-03 | 5.70 | 322.44 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 10 | 0.0% | 5.447e-03 | 1.40 | 2563.99 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 20.0% | 2.309e-03 | 5.50 | 7338.27 | — |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 输入点 | 9.349e-04 | 2.016e-03 | 5.954e-03 | 2.138e-04 | 21/0 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 输入点 | 3.588e-03 | 1.025e-02 | 1.041e-02 | 3.415e-04 | 21/0 |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 7.719e-04 | 3.473e-03 | 3.710e-03 | 5.128e-04 | 21/0 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 7.949e-03 | 1.296e-02 | 1.300e-02 | 2.238e-03 | 21/0 |
| Synthetic | Kang 2015 (ADMM adaptation) | 输入点 | 3.529e-02 | 6.748e-02 | 8.362e-02 | 5.836e-03 | 21/0 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 3.550e-02 | 7.227e-02 | 9.332e-02 | 1.287e-02 | 21/0 |
| UJI | Ours v16 learned | 输入点 | 3.759e-04 | 1.245e-03 | 1.553e-03 | 1.099e-04 | 10/0 |
| UJI | Ours v16 learned | 原始参考点 | 3.864e-04 | 1.238e-03 | 1.497e-03 | 1.506e-04 | 10/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 输入点 | 5.069e-04 | 1.346e-03 | 1.483e-03 | 4.728e-05 | 10/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 6.299e-04 | 1.791e-03 | 2.240e-03 | 1.285e-04 | 10/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.729e-04 | 6.256e-04 | 7.872e-04 | 4.763e-05 | 10/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 3.231e-04 | 6.241e-04 | 6.384e-04 | 1.145e-04 | 10/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 2.900e-03 | 1.022e-02 | 1.037e-02 | 1.383e-03 | 10/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 3.169e-03 | 1.138e-02 | 1.209e-02 | 1.677e-03 | 10/0 |
| UJI | Kang 2015 (ADMM adaptation) | 输入点 | 2.556e-02 | 1.312e-01 | 2.116e-01 | 4.203e-02 | 10/0 |
| UJI | Kang 2015 (ADMM adaptation) | 原始参考点 | 2.573e-02 | 1.315e-01 | 2.116e-01 | 4.899e-02 | 10/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 1.884e-02 | 1.014e-01 | 1.769e-01 | 5.624e-02 | 10/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 1.912e-02 | 1.028e-01 | 1.789e-01 | 5.796e-02 | 10/0 |
| NaturalEarth | Ours v16 learned | 输入点 | 1.020e-02 | 5.206e-02 | 9.228e-02 | 7.872e-03 | 10/0 |
| NaturalEarth | Ours v16 learned | 原始参考点 | 1.058e-02 | 5.337e-02 | 9.328e-02 | 7.934e-03 | 10/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.939e-03 | 6.369e-03 | 6.928e-03 | 7.790e-04 | 10/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 2.222e-03 | 6.442e-03 | 7.012e-03 | 8.290e-04 | 10/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 1.115e-03 | 2.828e-03 | 3.049e-03 | 5.099e-04 | 10/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 1.356e-03 | 3.721e-03 | 3.930e-03 | 5.619e-04 | 10/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 5.145e-03 | 8.909e-03 | 9.197e-03 | 8.357e-04 | 9/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 5.486e-03 | 9.287e-03 | 9.336e-03 | 8.423e-04 | 9/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 输入点 | 1.209e-01 | 2.728e-01 | 2.763e-01 | 7.022e-02 | 10/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 原始参考点 | 1.225e-01 | 2.763e-01 | 2.764e-01 | 7.031e-02 | 10/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 3.388e-02 | 1.028e-01 | 1.280e-01 | 2.677e-02 | 10/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 3.604e-02 | 1.086e-01 | 1.346e-01 | 2.700e-02 | 10/0 |
| USGS | Ours v16 learned | 输入点 | 5.432e-04 | 1.354e-03 | 1.467e-03 | 2.634e-04 | 10/0 |
| USGS | Ours v16 learned | 原始参考点 | 6.389e-04 | 1.672e-03 | 1.778e-03 | 2.646e-04 | 10/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.058e-03 | 2.578e-03 | 2.592e-03 | 2.843e-04 | 10/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 1.186e-03 | 2.793e-03 | 2.972e-03 | 2.868e-04 | 10/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 3.549e-04 | 6.723e-04 | 6.931e-04 | 1.712e-04 | 10/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 4.444e-04 | 9.522e-04 | 1.193e-03 | 1.722e-04 | 10/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 2.724e-03 | 3.896e-03 | 4.289e-03 | 5.353e-04 | 10/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 2.923e-03 | 4.517e-03 | 5.221e-03 | 5.344e-04 | 10/0 |
| USGS | Kang 2015 (ADMM adaptation) | 输入点 | 4.165e-02 | 1.204e-01 | 1.552e-01 | 2.966e-02 | 10/0 |
| USGS | Kang 2015 (ADMM adaptation) | 原始参考点 | 4.210e-02 | 1.213e-01 | 1.568e-01 | 2.976e-02 | 10/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 1.191e-02 | 3.742e-02 | 4.036e-02 | 9.584e-03 | 10/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 1.238e-02 | 3.838e-02 | 4.186e-02 | 9.642e-03 | 10/0 |
| IndustrialOffset | Ours v16 learned | 输入点 | 4.733e-04 | 1.901e-03 | 2.158e-03 | 2.878e-04 | 10/0 |
| IndustrialOffset | Ours v16 learned | 原始参考点 | 4.978e-04 | 1.902e-03 | 2.159e-03 | 2.892e-04 | 10/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 输入点 | 4.472e-04 | 1.272e-03 | 1.674e-03 | 1.360e-04 | 10/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 5.172e-04 | 1.635e-03 | 2.245e-03 | 1.414e-04 | 10/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.451e-04 | 4.452e-04 | 4.895e-04 | 4.773e-05 | 10/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 2.784e-04 | 5.184e-04 | 5.221e-04 | 4.794e-05 | 10/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 9.817e-03 | 2.663e-02 | 3.101e-02 | 2.943e-03 | 10/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 1.011e-02 | 2.691e-02 | 3.129e-02 | 2.954e-03 | 10/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 输入点 | 2.635e-02 | 6.193e-02 | 7.059e-02 | 1.772e-02 | 10/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 原始参考点 | 2.647e-02 | 6.223e-02 | 7.082e-02 | 1.772e-02 | 10/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 1.104e-02 | 4.701e-02 | 6.445e-02 | 1.153e-02 | 10/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 1.110e-02 | 4.722e-02 | 6.447e-02 | 1.163e-02 | 10/0 |

## 原生适配与最终结果审计

native 字段指仓库实现执行公共可行性修复前的拟合结果，非作者原版复现；具体端点约束由当次保存的逐方法协议决定。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 5.70 | 5.70 | 1.613e-03 | 1.613e-03 | 0.00 | 0/10 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 1.40 | 1.40 | 5.447e-03 | 5.447e-03 | 0.00 | 0/10 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 5.50 | 5.50 | 2.309e-03 | 2.309e-03 | 0.00 | 0/10 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 13.33 | 13.33 | 5.077e-04 | 5.077e-04 | 0.00 | 0/10 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 1.30 | 1.30 | 2.634e-02 | 2.634e-02 | 0.00 | 0/10 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 3.60 | 3.60 | 9.013e-03 | 9.013e-03 | 0.00 | 0/10 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 14.81 | 14.81 | 8.821e-04 | 8.821e-04 | 0.00 | 0/21 |
| Synthetic | Kang 2015 (ADMM adaptation) | 1.00 | 1.00 | 4.706e-03 | 4.706e-03 | 0.00 | 0/21 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 3.38 | 3.38 | 6.275e-03 | 6.275e-03 | 0.00 | 0/21 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 3.00 | 3.00 | 2.633e-04 | 2.633e-04 | 0.00 | 0/10 |
| UJI | Kang 2015 (ADMM adaptation) | 0.80 | 0.80 | 4.669e-03 | 4.669e-03 | 0.00 | 0/10 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 3.10 | 3.10 | 5.770e-03 | 5.770e-03 | 0.00 | 0/10 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 11.80 | 11.80 | 4.015e-04 | 4.015e-04 | 0.00 | 0/10 |
| USGS | Kang 2015 (ADMM adaptation) | 1.30 | 1.30 | 7.784e-03 | 7.784e-03 | 0.00 | 0/10 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 3.80 | 3.80 | 2.874e-03 | 2.874e-03 | 0.00 | 0/10 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 60.0% | 3.875e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 20.0% | 6.960e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 40.0% | 5.411e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 30.0% | 4.241e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 10.0% | 5.578e-03 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 30.0% | 6.017e-03 |
| NaturalEarth | Ours v16 learned | 40.0% | 9.337e-04 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 60.0% | 2.409e-04 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 50.0% | 1.741e-04 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 5.123e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 0.0% | 2.642e-02 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 9.078e-03 |
| USGS | Ours v16 learned | 50.0% | 7.614e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 50.0% | 1.012e-04 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 60.0% | 6.301e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 4.013e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 0.0% | 7.770e-03 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10.0% | 2.890e-03 |
| IndustrialOffset | Ours v16 learned | 80.0% | 6.645e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 90.0% | 4.606e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 3.613e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 1.611e-03 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 0.0% | 5.433e-03 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 20.0% | 2.325e-03 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
