# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；certified synthetic matched-knot MAE exceeds the formal limit 0.005；observed worst-source deployment pass rate 46.875% is below the required 90.000%；infeasible-proposal ablation was enabled；Forced diagnostic: quick or reduced benchmark protocol

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\checkpoints\overnight_compact_3090_r1.pt`（candidate_selection_counterfactual_bspline_v16，epoch 7）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 64, 'greedy_initial_and_yeh_max': 64, 'kang_dense_initial': 64, 'liang_dense_initial': 64, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 72, 'numerical_full_knot_vector_cap': 72}。
设备：{'device': 'cpu', 'gpu': None, 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 4, 'python': '3.13.9'}。
数值基线协议：**threshold-safe adaptation**。Dung/Kang/Luo 在原生公共 refit 未达阈值时可执行有界修复；保留最佳拟合但不保证容量内必然可行。所有额外 refit 计入完整耗时；这不是原文算法步骤。

数据来源：UJI: External held-out observations; no ground-truth B-spline knots.；NaturalEarth: External held-out observations; no ground-truth B-spline knots.；USGS: External held-out observations; no ground-truth B-spline knots.；IndustrialOffset: Procedurally generated CAD-style offset curves; not measured industrial data; no ground-truth B-spline knots.

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 21 | 85.7% | 3.332e-05 | 27.95 | 18.48 | 15.51 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 21 | 100.0% | 4.402e-05 | 38.86 | 165.53 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 21 | 100.0% | 4.670e-05 | 35.24 | 109.89 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 21 | 100.0% | 4.763e-05 | 26.57 | 1072.89 | — |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 21 | 100.0% | 4.393e-05 | 34.33 | 906.90 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 21 | 100.0% | 4.687e-05 | 32.57 | 2593.94 | — |
| UJI | Ours v16 learned | 1 | 100.0% | 6.444e-06 | 19.00 | 19.80 | 18.88 |
| UJI | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 2.697e-05 | 5.00 | 31.91 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 4.132e-05 | 4.00 | 11.87 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 2.864e-05 | 5.00 | 244.27 | — |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 3.678e-06 | 53.00 | 963.17 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.204e-05 | 5.00 | 2318.52 | — |
| NaturalEarth | Ours v16 learned | 1 | 0.0% | 9.130e-05 | 21.00 | 14.09 | 12.08 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 3.522e-05 | 25.00 | 128.21 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 4.811e-05 | 24.00 | 71.05 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.775e-05 | 19.00 | 857.45 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.420e-06 | 64.00 | 331.19 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.332e-05 | 21.00 | 2253.02 | — |
| USGS | Ours v16 learned | 1 | 100.0% | 2.983e-05 | 25.00 | 18.74 | 16.16 |
| USGS | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 4.076e-05 | 9.00 | 31.27 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 4.660e-05 | 11.00 | 30.17 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 2.783e-05 | 9.00 | 328.56 | — |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 5.324e-06 | 24.00 | 888.80 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 4.253e-05 | 8.00 | 2252.13 | — |
| IndustrialOffset | Ours v16 learned | 1 | 100.0% | 8.055e-06 | 20.00 | 12.38 | 13.88 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 1 | 100.0% | 2.382e-05 | 10.00 | 49.74 | — |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 1 | 100.0% | 2.352e-05 | 7.00 | 21.24 | — |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 100.0% | 1.383e-05 | 6.00 | 224.06 | — |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 100.0% | 3.830e-05 | 10.00 | 1146.07 | — |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 100.0% | 3.922e-05 | 7.00 | 2239.72 | — |

## 原生适配与最终结果审计

native 指修正后的仓库适配在公共可行性修复前的端点约束 refit，非作者原版复现。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 4.00 | 6.00 | 2.991e-03 | 1.383e-05 | 6.00 | 1/1 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 10.00 | 10.00 | 3.830e-05 | 3.830e-05 | 0.00 | 0/1 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 2.00 | 7.00 | 2.579e-03 | 3.922e-05 | 15.00 | 1/1 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 9.00 | 19.00 | 6.103e-04 | 4.775e-05 | 30.00 | 1/1 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 64.00 | 64.00 | 4.420e-06 | 4.420e-06 | 0.00 | 0/1 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 15.00 | 21.00 | 1.221e-04 | 4.332e-05 | 18.00 | 1/1 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 14.76 | 26.57 | 1.135e-03 | 4.763e-05 | 35.43 | 21/21 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 14.38 | 34.33 | 2.748e-03 | 4.393e-05 | 59.86 | 18/21 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 10.43 | 32.57 | 2.407e-03 | 4.687e-05 | 66.43 | 20/21 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 3.00 | 5.00 | 1.679e-04 | 2.864e-05 | 6.00 | 1/1 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 53.00 | 53.00 | 3.678e-06 | 3.678e-06 | 0.00 | 0/1 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 2.00 | 5.00 | 9.685e-05 | 4.204e-05 | 9.00 | 1/1 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 5.00 | 9.00 | 8.911e-04 | 2.783e-05 | 12.00 | 1/1 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 24.00 | 24.00 | 5.324e-06 | 5.324e-06 | 0.00 | 0/1 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 8.00 | 8.00 | 4.253e-05 | 4.253e-05 | 0.00 | 0/1 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 100.0% | 1.865e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 5.456e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 0.0% | 8.669e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 4.383e-05 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 1.554e-05 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 0.0% | 6.042e-05 |
| NaturalEarth | Ours v16 learned | 0.0% | 9.121e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 100.0% | 3.656e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 4.990e-05 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 4.993e-05 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 5.327e-06 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 100.0% | 4.515e-05 |
| USGS | Ours v16 learned | 100.0% | 2.806e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 100.0% | 4.093e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 4.676e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 2.803e-05 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 5.364e-06 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 100.0% | 4.273e-05 |
| IndustrialOffset | Ours v16 learned | 100.0% | 8.223e-06 |
| IndustrialOffset | Park & Lee 2007 (DOM adaptation) | 100.0% | 2.418e-05 |
| IndustrialOffset | Liang et al. 2017 (feature-IKI adaptation) | 100.0% | 2.365e-05 |
| IndustrialOffset | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 100.0% | 1.412e-05 |
| IndustrialOffset | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 3.895e-05 |
| IndustrialOffset | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 100.0% | 3.958e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
