# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；certified synthetic matched-knot MAE exceeds the formal limit 0.005；observed worst-source deployment pass rate 0.000% is below the required 90.000%；infeasible-proposal ablation was enabled；Forced diagnostic: quick or reduced benchmark protocol

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\checkpoints\overnight_stable_k32_pipeline_smoke_20260919.pt`（candidate_selection_counterfactual_bspline_v16，epoch 2）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 32, 'greedy_initial_and_yeh_max': 32, 'kang_dense_initial': 32, 'liang_dense_initial': 32, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 40, 'numerical_full_knot_vector_cap': 40}。
设备：{'device': 'cpu', 'gpu': None, 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 4, 'python': '3.13.9'}。
数值基线协议：**threshold-safe adaptation**。Dung/Kang/Luo 在原生公共 refit 未达阈值时可执行有界修复；保留最佳拟合但不保证容量内必然可行。所有额外 refit 计入完整耗时；这不是原文算法步骤。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 21 | 0.0% | 1.373e-03 | 16.00 | 17.92 | 16.48 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 21 | 28.6% | 1.361e-04 | 30.29 | 124.01 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 21 | 42.9% | 1.254e-04 | 27.29 | 85.69 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 21 | 66.7% | 5.474e-05 | 23.67 | 1052.69 | — |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 21 | 52.4% | 9.278e-05 | 30.19 | 312.04 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 21 | 52.4% | 9.768e-05 | 25.86 | 1395.08 | — |
| UJI | Ours v16 learned | 1 | 0.0% | 5.467e-05 | 12.00 | 18.00 | 11.34 |
| UJI | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 2.697e-05 | 5.00 | 19.65 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 3.931e-05 | 4.00 | 10.87 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.490e-05 | 4.00 | 230.43 | — |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 2.922e-05 | 6.00 | 1096.48 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 3.124e-05 | 5.00 | 1863.74 | — |
| NaturalEarth | Ours v16 learned | 1 | 0.0% | 4.246e-04 | 13.00 | 13.59 | 9.86 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 3.522e-05 | 25.00 | 129.73 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 4.282e-05 | 24.00 | 57.97 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.358e-05 | 20.00 | 683.91 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.969e-05 | 32.00 | 343.95 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.702e-05 | 23.00 | 1740.92 | — |
| USGS | Ours v16 learned | 1 | 0.0% | 7.221e-05 | 14.00 | 15.44 | 9.73 |
| USGS | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 4.076e-05 | 9.00 | 25.46 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 3.390e-05 | 11.00 | 27.17 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 2.516e-05 | 9.00 | 370.88 | — |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 3.833e-06 | 23.00 | 685.89 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 2.605e-05 | 10.00 | 2122.49 | — |
| IndustrialOffset | Ours v16 learned | 1 | 100.0% | 2.158e-05 | 10.00 | 16.00 | 8.83 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 2.382e-05 | 10.00 | 55.15 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 2.198e-05 | 7.00 | 18.65 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 1.529e-05 | 6.00 | 212.58 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 5.848e-06 | 32.00 | 306.07 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 2.137e-05 | 6.00 | 2079.31 | — |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 输入点 | 7.874e-03 | 1.139e-02 | 1.294e-02 | 2.294e-03 | 21/0 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 输入点 | 2.134e-03 | 6.425e-03 | 9.557e-03 | 5.498e-04 | 21/0 |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 8.322e-04 | 3.235e-03 | 3.483e-03 | 5.550e-04 | 21/0 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 2.513e-04 | 4.818e-04 | 7.084e-04 | 9.990e-05 | 21/0 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 6.642e-04 | 3.135e-03 | 3.388e-03 | 3.987e-04 | 21/0 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 7.066e-04 | 3.129e-03 | 3.352e-03 | 3.601e-04 | 21/0 |
| UJI | Ours v16 learned | 输入点 | 9.560e-04 | 9.560e-04 | 9.560e-04 | 5.467e-05 | 1/0 |
| UJI | Ours v16 learned | 原始参考点 | 1.162e-03 | 1.162e-03 | 1.162e-03 | 1.021e-04 | 1/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 输入点 | 3.100e-04 | 3.100e-04 | 3.100e-04 | 2.697e-05 | 1/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 3.259e-04 | 3.259e-04 | 3.259e-04 | 5.456e-05 | 1/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 8.125e-04 | 8.125e-04 | 8.125e-04 | 3.931e-05 | 1/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 1.049e-03 | 1.049e-03 | 1.049e-03 | 8.361e-05 | 1/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 3.700e-04 | 3.700e-04 | 3.700e-04 | 4.490e-05 | 1/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 3.235e-04 | 3.235e-04 | 3.235e-04 | 7.394e-05 | 1/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 6.198e-04 | 6.198e-04 | 6.198e-04 | 2.922e-05 | 1/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 8.364e-04 | 8.364e-04 | 8.364e-04 | 7.107e-05 | 1/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 2.922e-04 | 2.922e-04 | 2.922e-04 | 3.124e-05 | 1/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 4.205e-04 | 4.205e-04 | 4.205e-04 | 5.407e-05 | 1/0 |
| NaturalEarth | Ours v16 learned | 输入点 | 2.221e-03 | 2.221e-03 | 2.221e-03 | 4.246e-04 | 1/0 |
| NaturalEarth | Ours v16 learned | 原始参考点 | 2.702e-03 | 2.702e-03 | 2.702e-03 | 4.262e-04 | 1/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 输入点 | 8.084e-04 | 8.084e-04 | 8.084e-04 | 3.522e-05 | 1/0 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 9.227e-04 | 9.227e-04 | 9.227e-04 | 3.656e-05 | 1/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 4.358e-04 | 4.358e-04 | 4.358e-04 | 4.282e-05 | 1/0 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 5.595e-04 | 5.595e-04 | 5.595e-04 | 4.423e-05 | 1/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 4.382e-04 | 4.382e-04 | 4.382e-04 | 4.358e-05 | 1/0 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 7.676e-04 | 7.676e-04 | 7.676e-04 | 4.567e-05 | 1/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 5.702e-04 | 5.702e-04 | 5.702e-04 | 4.969e-05 | 1/0 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 9.140e-04 | 9.140e-04 | 9.140e-04 | 5.222e-05 | 1/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 4.477e-04 | 4.477e-04 | 4.477e-04 | 4.702e-05 | 1/0 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 8.464e-04 | 8.464e-04 | 8.464e-04 | 4.926e-05 | 1/0 |
| USGS | Ours v16 learned | 输入点 | 5.268e-04 | 5.268e-04 | 5.268e-04 | 7.221e-05 | 1/0 |
| USGS | Ours v16 learned | 原始参考点 | 5.249e-04 | 5.249e-04 | 5.249e-04 | 7.061e-05 | 1/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 输入点 | 3.360e-04 | 3.360e-04 | 3.360e-04 | 4.076e-05 | 1/0 |
| USGS | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 3.576e-04 | 3.576e-04 | 3.576e-04 | 4.093e-05 | 1/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.399e-04 | 2.399e-04 | 2.399e-04 | 3.390e-05 | 1/0 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 2.765e-04 | 2.765e-04 | 2.765e-04 | 3.405e-05 | 1/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 1.469e-04 | 1.469e-04 | 1.469e-04 | 2.516e-05 | 1/0 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 1.740e-04 | 1.740e-04 | 1.740e-04 | 2.532e-05 | 1/0 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 3.304e-05 | 3.304e-05 | 3.304e-05 | 3.833e-06 | 1/0 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 4.710e-05 | 4.710e-05 | 4.710e-05 | 3.864e-06 | 1/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 2.002e-04 | 2.002e-04 | 2.002e-04 | 2.605e-05 | 1/0 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 2.290e-04 | 2.290e-04 | 2.290e-04 | 2.614e-05 | 1/0 |
| IndustrialOffset | Ours v16 learned | 输入点 | 2.107e-04 | 2.107e-04 | 2.107e-04 | 2.158e-05 | 1/0 |
| IndustrialOffset | Ours v16 learned | 原始参考点 | 2.340e-04 | 2.340e-04 | 2.340e-04 | 2.181e-05 | 1/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 输入点 | 5.911e-04 | 5.911e-04 | 5.911e-04 | 2.382e-05 | 1/0 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 6.052e-04 | 6.052e-04 | 6.052e-04 | 2.418e-05 | 1/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 4.144e-04 | 4.144e-04 | 4.144e-04 | 2.198e-05 | 1/0 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 4.161e-04 | 4.161e-04 | 4.161e-04 | 2.220e-05 | 1/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 2.707e-04 | 2.707e-04 | 2.707e-04 | 1.529e-05 | 1/0 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 4.110e-04 | 4.110e-04 | 4.110e-04 | 1.566e-05 | 1/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 1.775e-04 | 1.775e-04 | 1.775e-04 | 5.848e-06 | 1/0 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 3.250e-04 | 3.250e-04 | 3.250e-04 | 6.122e-06 | 1/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 5.753e-04 | 5.753e-04 | 5.753e-04 | 2.137e-05 | 1/0 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 7.147e-04 | 7.147e-04 | 7.147e-04 | 2.180e-05 | 1/0 |

## 原生适配与最终结果审计

native 指修正后的仓库适配在公共可行性修复前的端点约束 refit，非作者原版复现。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 4.00 | 6.00 | 2.991e-03 | 1.529e-05 | 6.00 | 1/1 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 32.00 | 32.00 | 5.848e-06 | 5.848e-06 | 0.00 | 0/1 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 3.00 | 6.00 | 5.712e-04 | 2.137e-05 | 9.00 | 1/1 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 9.00 | 20.00 | 6.103e-04 | 4.358e-05 | 33.00 | 1/1 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 32.00 | 32.00 | 4.969e-05 | 4.969e-05 | 0.00 | 0/1 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 7.00 | 23.00 | 1.735e-03 | 4.702e-05 | 48.00 | 1/1 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 14.76 | 23.67 | 1.135e-03 | 5.474e-05 | 27.05 | 21/21 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 26.48 | 30.19 | 6.353e-04 | 9.278e-05 | 11.24 | 12/21 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 4.48 | 25.86 | 4.373e-03 | 9.768e-05 | 64.14 | 20/21 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 3.00 | 4.00 | 1.679e-04 | 4.490e-05 | 3.00 | 1/1 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 6.00 | 6.00 | 2.922e-05 | 2.922e-05 | 0.00 | 0/1 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 2.00 | 5.00 | 1.092e-04 | 3.124e-05 | 9.00 | 1/1 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 5.00 | 9.00 | 8.911e-04 | 2.516e-05 | 12.00 | 1/1 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 23.00 | 23.00 | 3.833e-06 | 3.833e-06 | 0.00 | 0/1 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 4.00 | 10.00 | 1.920e-03 | 2.605e-05 | 18.00 | 1/1 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 0.0% | 1.021e-04 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 5.456e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 0.0% | 8.361e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 0.0% | 7.394e-05 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 0.0% | 7.107e-05 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 0.0% | 5.407e-05 |
| NaturalEarth | Ours v16 learned | 0.0% | 4.262e-04 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 100.0% | 3.656e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 4.423e-05 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 4.567e-05 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 0.0% | 5.222e-05 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 100.0% | 4.926e-05 |
| USGS | Ours v16 learned | 0.0% | 7.061e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 100.0% | 4.093e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 3.405e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 2.532e-05 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 3.864e-06 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 100.0% | 2.614e-05 |
| IndustrialOffset | Ours v16 learned | 100.0% | 2.181e-05 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 100.0% | 2.418e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 2.220e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 1.566e-05 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 6.122e-06 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 100.0% | 2.180e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
