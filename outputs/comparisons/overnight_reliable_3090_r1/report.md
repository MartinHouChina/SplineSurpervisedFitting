# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：joint simplification curriculum was not mature when saved；checkpoint was not evaluated at the final safety sigma；checkpoint was not evaluated at the final safety-knot reserve；certified synthetic count MAE exceeds the formal limit 2；certified synthetic knot-match F1 is below the formal minimum 0.6；certified synthetic matched-knot MAE exceeds the formal limit 0.005；observed worst-source deployment pass rate 65.625% is below the required 90.000%；infeasible-proposal ablation was enabled

Checkpoint: `/home/feng/HouCode/SplineFitting_1070_overnight/outputs/checkpoints/overnight_reliable_3090_r1.pt`（candidate_selection_counterfactual_bspline_v16，epoch 5）。
统一阈值：MSE ≤ 5.000e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 64, 'greedy_initial_and_yeh_max': 64, 'kang_dense_initial': 64, 'liang_dense_initial': 64, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 72, 'numerical_full_knot_vector_cap': 72}。
设备：{'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3090', 'cpu': 'x86_64', 'torch': '2.11.0+cu126', 'threads': 4, 'python': '3.11.5'}。
数值基线协议：**threshold-safe adaptation**。Dung/Kang/Luo 在原生公共 refit 未达阈值时可执行有界修复；保留最佳拟合但不保证容量内必然可行。所有额外 refit 计入完整耗时；这不是原文算法步骤。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 42 | 100.0% | 1.893e-05 | 31.17 | 8.84 | 7.34 |
| Synthetic | Park & Lee 2007 (DOM adaptation) | 42 | 100.0% | 4.418e-05 | 39.19 | 57.14 | — |
| Synthetic | Liang et al. 2017 (feature-IKI adaptation) | 42 | 100.0% | 4.641e-05 | 35.88 | 41.71 | — |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 42 | 100.0% | 4.648e-05 | 26.88 | 425.08 | — |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 42 | 95.2% | 3.967e-05 | 36.21 | 825.53 | — |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 42 | 97.6% | 4.542e-05 | 32.14 | 2829.41 | — |
| UJI | Ours v16 learned | 8 | 100.0% | 7.046e-06 | 27.50 | 8.83 | 7.37 |
| UJI | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 3.834e-05 | 9.12 | 11.65 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 3.103e-05 | 7.50 | 7.69 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 8 | 100.0% | 3.258e-05 | 6.00 | 99.25 | — |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 8 | 100.0% | 1.739e-05 | 34.50 | 1107.44 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 8 | 100.0% | 2.868e-05 | 6.88 | 2938.33 | — |
| NaturalEarth | Ours v16 learned | 8 | 75.0% | 4.876e-05 | 29.12 | 8.78 | 7.31 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 4.367e-05 | 24.88 | 37.91 | — |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 4.714e-05 | 24.50 | 27.42 | — |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 8 | 100.0% | 4.454e-05 | 20.88 | 319.08 | — |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 8 | 87.5% | 1.993e-05 | 40.88 | 781.66 | — |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 8 | 87.5% | 4.342e-05 | 22.50 | 2711.59 | — |
| USGS | Ours v16 learned | 8 | 50.0% | 4.159e-05 | 38.25 | 8.88 | 7.23 |
| USGS | Park & Lee 2007 (DOM adaptation) | 8 | 100.0% | 4.656e-05 | 33.25 | 47.59 | — |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 8 | 100.0% | 4.057e-05 | 29.50 | 34.43 | — |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 8 | 100.0% | 4.066e-05 | 27.38 | 439.36 | — |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 8 | 100.0% | 2.604e-05 | 50.50 | 721.00 | — |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 8 | 100.0% | 4.001e-05 | 30.75 | 2528.83 | — |

## 原生适配与最终结果审计

native 指修正后的仓库适配在公共可行性修复前的端点约束 refit，非作者原版复现。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 11.00 | 20.88 | 6.609e-04 | 4.454e-05 | 29.62 | 8/8 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 40.88 | 40.88 | 1.993e-05 | 1.993e-05 | 0.12 | 1/8 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 6.00 | 22.50 | 2.260e-03 | 4.342e-05 | 49.62 | 8/8 |
| Synthetic | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 14.81 | 26.88 | 1.047e-03 | 4.648e-05 | 36.21 | 42/42 |
| Synthetic | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 18.60 | 36.21 | 2.187e-03 | 3.967e-05 | 52.88 | 30/42 |
| Synthetic | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 10.33 | 32.14 | 2.464e-03 | 4.542e-05 | 65.50 | 35/42 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 3.62 | 6.00 | 6.258e-04 | 3.258e-05 | 7.12 | 6/8 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 34.50 | 34.50 | 1.739e-05 | 1.739e-05 | 0.00 | 0/8 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 3.75 | 6.88 | 6.226e-04 | 2.868e-05 | 9.38 | 5/8 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 15.12 | 27.38 | 2.618e-04 | 4.066e-05 | 36.75 | 7/8 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 50.50 | 50.50 | 2.604e-05 | 2.604e-05 | 0.00 | 0/8 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 8.50 | 30.75 | 7.746e-04 | 4.001e-05 | 66.75 | 6/8 |

真实曲线原始参考点测试：仅在 192 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 100.0% | 1.228e-05 |
| UJI | Park & Lee 2007 (DOM adaptation) | 12.5% | 8.019e-05 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 50.0% | 6.026e-05 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 62.5% | 5.765e-05 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 62.5% | 4.411e-05 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 62.5% | 4.959e-05 |
| NaturalEarth | Ours v16 learned | 62.5% | 5.010e-05 |
| NaturalEarth | Park & Lee 2007 (DOM adaptation) | 75.0% | 4.531e-05 |
| NaturalEarth | Liang et al. 2017 (feature-IKI adaptation) | 37.5% | 7.652e-02 |
| NaturalEarth | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 50.0% | 4.653e-05 |
| NaturalEarth | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 87.5% | 2.143e-05 |
| NaturalEarth | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 75.0% | 4.520e-05 |
| USGS | Ours v16 learned | 50.0% | 4.200e-05 |
| USGS | Park & Lee 2007 (DOM adaptation) | 62.5% | 4.764e-05 |
| USGS | Liang et al. 2017 (feature-IKI adaptation) | 87.5% | 4.160e-05 |
| USGS | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 75.0% | 4.210e-05 |
| USGS | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 100.0% | 2.692e-05 |
| USGS | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 75.0% | 4.106e-05 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
