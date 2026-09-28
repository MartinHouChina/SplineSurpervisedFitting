# v16 多数据集配对测试

Per-case measured geometry: each measurements.jsonl row links a hash-verified JSON + NPZ artifact under geometry/. These include parameters, full/internal knots, control vertices, dense curves, pointwise residuals and original-coordinate transforms. Failures/unavailable outputs remain absent; export and plots do not rerun fitting. NumPy loading uses allow_pickle=False.

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；observed worst-source deployment pass rate 13.000% is below the required 90.000%；observed worst-source dense proposal pass rate 15.000% is below the required 90.000%；proposal feasibility gate was not satisfied；infeasible-proposal ablation was enabled

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\downloads\server_models_20260922_215418\paper_coupled_clean_3090_r2_m16.pt`（candidate_selection_counterfactual_bspline_v16，epoch 13）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点。新协议直接评估实际返回的曲线：Dung/Kang 保留算法内无端点强制约束的最小二乘解，不再附加公共端点 refit；Ours 使用自身部署 refit，其余对照保留当前求解路径，详见结果中的逐方法协议。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 16, 'greedy_initial_and_yeh_max': 16, 'kang_dense_initial': 16, 'liang_dense_initial': 16, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 24, 'numerical_full_knot_vector_cap': 24}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce GTX 1070', 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 4, 'python': '3.13.9'}。
数值基线协议：未启用或历史报告未记录公共可行性修复；不将结果标为 threshold-safe adaptation。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 21 | 38.1% | 1.628e-03 | 15.43 | 68.32 | 51.28 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 21 | 4.8% | 1.297e-03 | 16.00 | 59.30 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 21 | 14.3% | 1.269e-03 | 15.43 | 46.86 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 21 | 0.0% | 1.159e-03 | 10.33 | 570.39 | — |
| Synthetic | Kang 2015 (ADMM adaptation) | 21 | 0.0% | 4.698e-03 | 1.00 | 312.37 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 21 | 0.0% | 5.742e-03 | 1.95 | 1866.61 | — |
| UJI | Ours v16 learned | 10 | 90.0% | 1.779e-05 | 13.10 | 73.16 | 50.50 |
| UJI | Park & Lee 2007 (DOM adaptation) | 10 | 90.0% | 6.114e-05 | 4.50 | 23.66 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 10 | 90.0% | 2.693e-05 | 5.70 | 18.10 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 50.0% | 2.633e-04 | 3.00 | 208.66 | — |
| UJI | Kang 2015 (ADMM adaptation) | 10 | 50.0% | 4.681e-03 | 0.80 | 1498.98 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 40.0% | 5.941e-03 | 1.90 | 5829.84 | — |
| NaturalEarth | Ours v16 learned | 10 | 10.0% | 1.340e-03 | 15.70 | 71.69 | 52.69 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 10 | 30.0% | 1.740e-03 | 14.90 | 70.33 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 10 | 30.0% | 6.528e-04 | 14.40 | 54.27 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 4.456e-04 | 9.00 | 608.58 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 10 | 0.0% | 2.180e-02 | 1.40 | 341.83 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 0.0% | 2.003e-02 | 2.40 | 2249.71 | — |
| USGS | Ours v16 learned | 10 | 50.0% | 3.161e-04 | 15.50 | 75.24 | 50.51 |
| USGS | Park & Lee 2007 (DOM adaptation) | 10 | 40.0% | 5.260e-04 | 12.90 | 56.28 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 10 | 50.0% | 2.324e-04 | 12.40 | 38.05 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 4.215e-04 | 7.57 | 506.57 | — |
| USGS | Kang 2015 (ADMM adaptation) | 10 | 0.0% | 7.765e-03 | 1.30 | 852.42 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 0.0% | 9.015e-03 | 1.50 | 3261.29 | — |
| IndustrialOffset | Ours v16 learned | 10 | 80.0% | 1.332e-04 | 14.50 | 74.52 | 51.81 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 10 | 70.0% | 1.618e-04 | 11.80 | 61.93 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 10 | 60.0% | 1.051e-04 | 10.40 | 34.11 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 10 | 0.0% | 1.613e-03 | 5.70 | 367.54 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 10 | 0.0% | 5.313e-03 | 1.50 | 1138.87 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 10 | 0.0% | 4.804e-03 | 2.00 | 4090.81 | — |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 输入点 | 7.391e-03 | 1.760e-02 | 1.965e-02 | 3.969e-03 | 21/0 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.016e-02 | 1.689e-02 | 1.993e-02 | 3.252e-03 | 21/0 |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 5.560e-03 | 1.114e-02 | 1.153e-02 | 3.032e-03 | 21/0 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 7.734e-03 | 1.143e-02 | 1.296e-02 | 2.238e-03 | 12/0 |
| Synthetic | Kang 2015 (ADMM adaptation) | 输入点 | 3.512e-02 | 6.645e-02 | 8.343e-02 | 5.794e-03 | 21/0 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 2.934e-02 | 6.830e-02 | 7.227e-02 | 1.122e-02 | 21/0 |
| UJI | Ours v16 learned | 输入点 | 2.032e-04 | 7.564e-04 | 1.235e-03 | 1.126e-04 | 10/0 |
| UJI | Ours v16 learned | 原始参考点 | 2.569e-04 | 1.091e-03 | 1.752e-03 | 1.292e-04 | 10/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 输入点 | 5.962e-04 | 1.837e-03 | 2.376e-03 | 3.546e-04 | 10/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 7.510e-04 | 2.457e-03 | 3.451e-03 | 3.194e-04 | 10/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.903e-04 | 5.332e-04 | 5.639e-04 | 5.644e-05 | 10/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 3.338e-04 | 5.721e-04 | 6.113e-04 | 1.182e-04 | 10/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 2.900e-03 | 1.022e-02 | 1.037e-02 | 1.383e-03 | 10/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 3.169e-03 | 1.138e-02 | 1.209e-02 | 1.677e-03 | 10/0 |
| UJI | Kang 2015 (ADMM adaptation) | 输入点 | 2.571e-02 | 1.315e-01 | 2.119e-01 | 4.202e-02 | 10/0 |
| UJI | Kang 2015 (ADMM adaptation) | 原始参考点 | 2.588e-02 | 1.318e-01 | 2.119e-01 | 4.898e-02 | 10/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 2.046e-02 | 1.076e-01 | 1.769e-01 | 5.624e-02 | 10/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 2.080e-02 | 1.090e-01 | 1.789e-01 | 5.796e-02 | 10/0 |
| NaturalEarth | Ours v16 learned | 输入点 | 8.494e-03 | 3.419e-02 | 4.284e-02 | 6.995e-03 | 10/0 |
| NaturalEarth | Ours v16 learned | 原始参考点 | 9.090e-03 | 3.610e-02 | 4.628e-02 | 7.139e-03 | 10/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.551e-02 | 6.330e-02 | 8.545e-02 | 8.829e-03 | 10/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 1.643e-02 | 6.696e-02 | 9.082e-02 | 8.852e-03 | 10/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 4.170e-03 | 1.380e-02 | 1.786e-02 | 2.997e-03 | 10/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 4.538e-03 | 1.493e-02 | 1.801e-02 | 2.980e-03 | 10/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 5.037e-03 | 8.981e-03 | 9.197e-03 | 7.873e-04 | 7/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 5.317e-03 | 9.299e-03 | 9.336e-03 | 7.988e-04 | 7/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 输入点 | 1.079e-01 | 2.738e-01 | 2.776e-01 | 5.984e-02 | 10/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 原始参考点 | 1.097e-01 | 2.773e-01 | 2.777e-01 | 6.012e-02 | 10/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 6.945e-02 | 2.154e-01 | 2.396e-01 | 6.082e-02 | 10/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 7.141e-02 | 2.195e-01 | 2.469e-01 | 6.130e-02 | 10/0 |
| USGS | Ours v16 learned | 输入点 | 1.801e-03 | 5.941e-03 | 8.599e-03 | 1.388e-03 | 10/0 |
| USGS | Ours v16 learned | 原始参考点 | 1.889e-03 | 6.128e-03 | 8.826e-03 | 1.395e-03 | 10/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 输入点 | 4.196e-03 | 1.224e-02 | 1.554e-02 | 1.766e-03 | 10/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 4.504e-03 | 1.291e-02 | 1.658e-02 | 1.775e-03 | 10/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 1.393e-03 | 3.605e-03 | 3.657e-03 | 6.750e-04 | 10/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 1.615e-03 | 4.199e-03 | 4.646e-03 | 6.790e-04 | 10/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 2.555e-03 | 3.383e-03 | 3.416e-03 | 5.353e-04 | 7/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 2.653e-03 | 3.552e-03 | 3.656e-03 | 5.344e-04 | 7/0 |
| USGS | Kang 2015 (ADMM adaptation) | 输入点 | 4.156e-02 | 1.204e-01 | 1.550e-01 | 2.964e-02 | 10/0 |
| USGS | Kang 2015 (ADMM adaptation) | 原始参考点 | 4.200e-02 | 1.212e-01 | 1.566e-01 | 2.974e-02 | 10/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 3.998e-02 | 1.273e-01 | 1.327e-01 | 2.782e-02 | 10/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 4.063e-02 | 1.291e-01 | 1.346e-01 | 2.791e-02 | 10/0 |
| IndustrialOffset | Ours v16 learned | 输入点 | 5.880e-04 | 2.518e-03 | 4.193e-03 | 1.119e-03 | 10/0 |
| IndustrialOffset | Ours v16 learned | 原始参考点 | 6.413e-04 | 2.639e-03 | 4.213e-03 | 1.122e-03 | 10/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 输入点 | 1.873e-03 | 8.303e-03 | 1.089e-02 | 6.662e-04 | 10/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 2.076e-03 | 9.346e-03 | 1.276e-02 | 6.794e-04 | 10/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 5.411e-04 | 1.605e-03 | 2.406e-03 | 5.752e-04 | 10/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 5.494e-04 | 1.618e-03 | 2.426e-03 | 5.783e-04 | 10/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 9.817e-03 | 2.663e-02 | 3.101e-02 | 2.943e-03 | 10/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 1.011e-02 | 2.691e-02 | 3.129e-02 | 2.954e-03 | 10/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 输入点 | 2.746e-02 | 6.181e-02 | 7.046e-02 | 1.772e-02 | 10/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 原始参考点 | 2.757e-02 | 6.212e-02 | 7.069e-02 | 1.772e-02 | 10/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 1.895e-02 | 5.351e-02 | 7.414e-02 | 2.039e-02 | 10/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 1.916e-02 | 5.422e-02 | 7.499e-02 | 2.052e-02 | 10/0 |

## 原生适配与最终结果审计

native 字段指仓库实现执行公共可行性修复前的拟合结果，非作者原版复现；具体端点约束由当次保存的逐方法协议决定。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 5.70 | 5.70 | 1.613e-03 | 1.613e-03 | 0.00 | 0/10 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 1.50 | 1.50 | 5.313e-03 | 5.313e-03 | 0.00 | 0/10 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 2.00 | 2.00 | 4.804e-03 | 4.804e-03 | 0.00 | 0/10 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 9.00 | 9.00 | 4.456e-04 | 4.456e-04 | 0.00 | 0/10 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 1.40 | 1.40 | 2.180e-02 | 2.180e-02 | 0.00 | 0/10 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 2.40 | 2.40 | 2.003e-02 | 2.003e-02 | 0.00 | 0/10 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) | 10.33 | 10.33 | 1.159e-03 | 1.159e-03 | 0.00 | 0/21 |
| Synthetic | Kang 2015 (ADMM adaptation) | 1.00 | 1.00 | 4.698e-03 | 4.698e-03 | 0.00 | 0/21 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 1.95 | 1.95 | 5.742e-03 | 5.742e-03 | 0.00 | 0/21 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 3.00 | 3.00 | 2.633e-04 | 2.633e-04 | 0.00 | 0/10 |
| UJI | Kang 2015 (ADMM adaptation) | 0.80 | 0.80 | 4.681e-03 | 4.681e-03 | 0.00 | 0/10 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 1.90 | 1.90 | 5.941e-03 | 5.941e-03 | 0.00 | 0/10 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 7.57 | 7.57 | 4.215e-04 | 4.215e-04 | 0.00 | 0/10 |
| USGS | Kang 2015 (ADMM adaptation) | 1.30 | 1.30 | 7.765e-03 | 7.765e-03 | 0.00 | 0/10 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 1.50 | 1.50 | 9.015e-03 | 9.015e-03 | 0.00 | 0/10 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 90.0% | 2.373e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 20.0% | 9.597e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 30.0% | 5.978e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 30.0% | 4.241e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 10.0% | 5.600e-03 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 30.0% | 6.224e-03 |
| NaturalEarth | Ours v16 learned | 10.0% | 1.357e-03 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 30.0% | 1.746e-03 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 30.0% | 6.558e-04 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 4.483e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 0.0% | 2.185e-02 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 2.022e-02 |
| USGS | Ours v16 learned | 50.0% | 3.172e-04 |
| USGS | Park & Lee 2007 (DOM adaptation) | 40.0% | 5.298e-04 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 50.0% | 2.343e-04 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 4.194e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 0.0% | 7.751e-03 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 9.058e-03 |
| IndustrialOffset | Ours v16 learned | 80.0% | 1.336e-04 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 70.0% | 1.636e-04 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 60.0% | 1.057e-04 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 1.611e-03 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 0.0% | 5.277e-03 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 4.832e-03 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
