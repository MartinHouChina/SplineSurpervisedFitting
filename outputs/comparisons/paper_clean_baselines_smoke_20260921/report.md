# v16 多数据集配对测试

Per-case measured geometry: each measurements.jsonl row links a hash-verified JSON + NPZ artifact under geometry/. These include parameters, full/internal knots, control vertices, dense curves, pointwise residuals and original-coordinate transforms. Failures/unavailable outputs remain absent; export and plots do not rerun fitting. NumPy loading uses allow_pickle=False.

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；observed worst-source deployment pass rate 53.125% is below the required 90.000%；observed worst-source dense proposal pass rate 81.250% is below the required 90.000%；proposal feasibility gate was not satisfied；infeasible-proposal ablation was enabled；Forced diagnostic: quick or reduced benchmark protocol

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\checkpoints\universal_m16_m32_3090_r1_m32.pt`（candidate_selection_counterfactual_bspline_v16，epoch 5）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点。新协议直接评估实际返回的曲线：Dung/Kang 保留算法内无端点强制约束的最小二乘解，不再附加公共端点 refit；Ours 使用自身部署 refit，其余对照保留当前求解路径，详见结果中的逐方法协议。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 32, 'greedy_initial_and_yeh_max': 32, 'kang_dense_initial': 32, 'liang_dense_initial': 32, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 40, 'numerical_full_knot_vector_cap': 40}。
设备：{'device': 'cpu', 'gpu': None, 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 2, 'python': '3.13.9'}。
数值基线协议：未启用或历史报告未记录公共可行性修复；不将结果标为 threshold-safe adaptation。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| UJI | Ours v16 learned | 1 | 100.0% | 7.857e-06 | 20.00 | 24.75 | 23.98 |
| UJI | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 2.697e-05 | 5.00 | 13.79 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 3.931e-05 | 4.00 | 10.93 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 1 | 0.0% | 1.571e-04 | 3.00 | 132.85 | — |
| UJI | Kang 2015 (ADMM adaptation) | 1 | 0.0% | 2.388e-04 | 1.00 | 2662.75 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 1 | 0.0% | 1.112e-04 | 2.00 | 6655.31 | — |
| NaturalEarth | Ours v16 learned | 1 | 0.0% | 7.656e-05 | 27.00 | 28.51 | 21.59 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 3.522e-05 | 25.00 | 119.50 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 4.282e-05 | 24.00 | 57.77 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 1 | 0.0% | 5.963e-04 | 9.00 | 556.97 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 1 | 0.0% | 2.384e-02 | 1.00 | 1278.60 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 1 | 0.0% | 1.967e-03 | 7.00 | 3055.40 | — |
| USGS | Ours v16 learned | 1 | 100.0% | 2.162e-05 | 28.00 | 22.17 | 25.31 |
| USGS | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 4.076e-05 | 9.00 | 29.00 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 3.390e-05 | 11.00 | 28.02 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 1 | 0.0% | 8.049e-04 | 5.00 | 263.83 | — |
| USGS | Kang 2015 (ADMM adaptation) | 1 | 0.0% | 3.234e-03 | 2.00 | 2663.43 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 1 | 0.0% | 3.495e-04 | 5.00 | 6254.44 | — |
| IndustrialOffset | Ours v16 learned | 1 | 100.0% | 4.219e-06 | 19.00 | 29.58 | 28.10 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 2.382e-05 | 10.00 | 40.40 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 2.198e-05 | 7.00 | 18.56 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 1 | 0.0% | 2.910e-03 | 4.00 | 177.63 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 1 | 0.0% | 5.996e-03 | 1.00 | 2144.20 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 1 | 0.0% | 5.914e-04 | 3.00 | 6965.20 | — |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| UJI | Ours v16 learned | 输入点 | 1.514e-04 | 1.514e-04 | 1.514e-04 | 7.857e-06 | 1/0 |
| UJI | Ours v16 learned | 原始参考点 | 2.733e-04 | 2.733e-04 | 2.733e-04 | 2.326e-05 | 1/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 输入点 | 3.100e-04 | 3.100e-04 | 3.100e-04 | 2.697e-05 | 1/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 3.259e-04 | 3.259e-04 | 3.259e-04 | 5.456e-05 | 1/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 8.125e-04 | 8.125e-04 | 8.125e-04 | 3.931e-05 | 1/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 1.049e-03 | 1.049e-03 | 1.049e-03 | 8.361e-05 | 1/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 2.336e-03 | 2.336e-03 | 2.336e-03 | 1.571e-04 | 1/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 2.664e-03 | 2.664e-03 | 2.664e-03 | 2.909e-04 | 1/0 |
| UJI | Kang 2015 (ADMM adaptation) | 输入点 | 2.896e-03 | 2.896e-03 | 2.896e-03 | 2.388e-04 | 1/0 |
| UJI | Kang 2015 (ADMM adaptation) | 原始参考点 | 3.306e-03 | 3.306e-03 | 3.306e-03 | 4.265e-04 | 1/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 6.224e-04 | 6.224e-04 | 6.224e-04 | 1.112e-04 | 1/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 6.758e-04 | 6.758e-04 | 6.758e-04 | 1.516e-04 | 1/0 |
| NaturalEarth | Ours v16 learned | 输入点 | 7.231e-04 | 7.231e-04 | 7.231e-04 | 7.656e-05 | 1/0 |
| NaturalEarth | Ours v16 learned | 原始参考点 | 7.233e-04 | 7.233e-04 | 7.233e-04 | 7.513e-05 | 1/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 输入点 | 8.084e-04 | 8.084e-04 | 8.084e-04 | 3.522e-05 | 1/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 9.227e-04 | 9.227e-04 | 9.227e-04 | 3.656e-05 | 1/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 4.358e-04 | 4.358e-04 | 4.358e-04 | 4.282e-05 | 1/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 5.595e-04 | 5.595e-04 | 5.595e-04 | 4.423e-05 | 1/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 5.210e-03 | 5.210e-03 | 5.210e-03 | 5.963e-04 | 1/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 5.634e-03 | 5.634e-03 | 5.634e-03 | 6.002e-04 | 1/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 输入点 | 1.050e-01 | 1.050e-01 | 1.050e-01 | 2.384e-02 | 1/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 原始参考点 | 1.050e-01 | 1.050e-01 | 1.050e-01 | 2.373e-02 | 1/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 7.232e-03 | 7.232e-03 | 7.232e-03 | 1.967e-03 | 1/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 7.863e-03 | 7.863e-03 | 7.863e-03 | 1.981e-03 | 1/0 |
| USGS | Ours v16 learned | 输入点 | 2.049e-04 | 2.049e-04 | 2.049e-04 | 2.162e-05 | 1/0 |
| USGS | Ours v16 learned | 原始参考点 | 2.041e-04 | 2.041e-04 | 2.041e-04 | 2.044e-05 | 1/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 输入点 | 3.360e-04 | 3.360e-04 | 3.360e-04 | 4.076e-05 | 1/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 3.576e-04 | 3.576e-04 | 3.576e-04 | 4.093e-05 | 1/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.399e-04 | 2.399e-04 | 2.399e-04 | 3.390e-05 | 1/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 2.765e-04 | 2.765e-04 | 2.765e-04 | 3.405e-05 | 1/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 3.062e-03 | 3.062e-03 | 3.062e-03 | 8.049e-04 | 1/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 3.089e-03 | 3.089e-03 | 3.089e-03 | 7.979e-04 | 1/0 |
| USGS | Kang 2015 (ADMM adaptation) | 输入点 | 8.471e-03 | 8.471e-03 | 8.471e-03 | 3.234e-03 | 1/0 |
| USGS | Kang 2015 (ADMM adaptation) | 原始参考点 | 8.483e-03 | 8.483e-03 | 8.483e-03 | 3.229e-03 | 1/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 1.213e-03 | 1.213e-03 | 1.213e-03 | 3.495e-04 | 1/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 1.316e-03 | 1.316e-03 | 1.316e-03 | 3.509e-04 | 1/0 |
| IndustrialOffset | Ours v16 learned | 输入点 | 4.904e-05 | 4.904e-05 | 4.904e-05 | 4.219e-06 | 1/0 |
| IndustrialOffset | Ours v16 learned | 原始参考点 | 8.819e-05 | 8.819e-05 | 8.819e-05 | 4.274e-06 | 1/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 输入点 | 5.911e-04 | 5.911e-04 | 5.911e-04 | 2.382e-05 | 1/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 6.052e-04 | 6.052e-04 | 6.052e-04 | 2.418e-05 | 1/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 4.144e-04 | 4.144e-04 | 4.144e-04 | 2.198e-05 | 1/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 4.161e-04 | 4.161e-04 | 4.161e-04 | 2.220e-05 | 1/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 输入点 | 2.667e-02 | 2.667e-02 | 2.667e-02 | 2.910e-03 | 1/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 原始参考点 | 2.714e-02 | 2.714e-02 | 2.714e-02 | 2.918e-03 | 1/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 输入点 | 4.589e-02 | 4.589e-02 | 4.589e-02 | 5.996e-03 | 1/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 原始参考点 | 4.677e-02 | 4.677e-02 | 4.677e-02 | 5.998e-03 | 1/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 输入点 | 2.740e-03 | 2.740e-03 | 2.740e-03 | 5.914e-04 | 1/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 原始参考点 | 2.757e-03 | 2.757e-03 | 2.757e-03 | 5.945e-04 | 1/0 |

## 原生适配与最终结果审计

native 字段指仓库实现执行公共可行性修复前的拟合结果，非作者原版复现；具体端点约束由当次保存的逐方法协议决定。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 4.00 | 4.00 | 2.910e-03 | 2.910e-03 | 0.00 | 0/1 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 1.00 | 1.00 | 5.996e-03 | 5.996e-03 | 0.00 | 0/1 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 3.00 | 3.00 | 5.914e-04 | 5.914e-04 | 0.00 | 0/1 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 9.00 | 9.00 | 5.963e-04 | 5.963e-04 | 0.00 | 0/1 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 1.00 | 1.00 | 2.384e-02 | 2.384e-02 | 0.00 | 0/1 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 7.00 | 7.00 | 1.967e-03 | 1.967e-03 | 0.00 | 0/1 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 3.00 | 3.00 | 1.571e-04 | 1.571e-04 | 0.00 | 0/1 |
| UJI | Kang 2015 (ADMM adaptation) | 1.00 | 1.00 | 2.388e-04 | 2.388e-04 | 0.00 | 0/1 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 2.00 | 2.00 | 1.112e-04 | 1.112e-04 | 0.00 | 0/1 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 5.00 | 5.00 | 8.049e-04 | 8.049e-04 | 0.00 | 0/1 |
| USGS | Kang 2015 (ADMM adaptation) | 2.00 | 2.00 | 3.234e-03 | 3.234e-03 | 0.00 | 0/1 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 5.00 | 5.00 | 3.495e-04 | 3.495e-04 | 0.00 | 0/1 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 100.0% | 2.326e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 5.456e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 0.0% | 8.361e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 2.909e-04 |
| UJI | Kang 2015 (ADMM adaptation) | 0.0% | 4.265e-04 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 1.516e-04 |
| NaturalEarth | Ours v16 learned | 0.0% | 7.513e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 100.0% | 3.656e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 4.423e-05 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 6.002e-04 |
| NaturalEarth | Kang 2015 (ADMM adaptation) | 0.0% | 2.373e-02 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 1.981e-03 |
| USGS | Ours v16 learned | 100.0% | 2.044e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 100.0% | 4.093e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 3.405e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 7.979e-04 |
| USGS | Kang 2015 (ADMM adaptation) | 0.0% | 3.229e-03 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 3.509e-04 |
| IndustrialOffset | Ours v16 learned | 100.0% | 4.274e-06 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 100.0% | 2.418e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 2.220e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) | 0.0% | 2.918e-03 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) | 0.0% | 5.998e-03 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) | 0.0% | 5.945e-04 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
