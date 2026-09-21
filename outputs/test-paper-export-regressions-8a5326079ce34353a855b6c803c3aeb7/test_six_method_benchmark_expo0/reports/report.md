# v16 多数据集配对测试

Per-case measured geometry: each measurements.jsonl row links a hash-verified JSON + NPZ artifact under geometry/. These include parameters, full/internal knots, control vertices, dense curves, pointwise residuals and original-coordinate transforms. Failures/unavailable outputs remain absent; export and plots do not rerun fitting. NumPy loading uses allow_pickle=False.

Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\test-paper-export-regressions-8a5326079ce34353a855b6c803c3aeb7\test_six_method_benchmark_expo0\checkpoint.pt`（candidate_selection_counterfactual_bspline_v16，epoch 1）。
统一阈值：MSE ≤ 2.500e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：{'network_candidates': 32, 'greedy_initial_and_yeh_max': 32, 'kang_dense_initial': 32, 'liang_dense_initial': 32, 'equal_initial_capacity': True, 'degree': 3, 'clamped_endpoint_entries': 8, 'network_full_knot_vector_size_at_all_keep': 40, 'numerical_full_knot_vector_cap': 40}。
设备：{'device': 'cpu', 'gpu': None, 'cpu': 'Intel64 Family 6 Model 94 Stepping 3, GenuineIntel', 'torch': '2.12.0+cu126', 'threads': 4, 'python': '3.13.9'}。
数值基线协议：**threshold-safe adaptation**。Dung/Kang/Luo 在原生公共 refit 未达阈值时可执行有界修复；保留最佳拟合但不保证容量内必然可行。所有额外 refit 计入完整耗时；这不是原文算法步骤。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| UJI | Ours v16 learned | 1 | 0.0% | 8.095e-03 | 0.00 | 3.00 | 1.00 |
| UJI | Park & Lee 2007 (DOM adaptation) | 1 | 0.0% | 8.095e-03 | 0.00 | 4.00 | — |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 1 | 0.0% | 8.095e-03 | 0.00 | 4.00 | — |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 1 | 0.0% | 8.095e-03 | 0.00 | 4.00 | — |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 1 | 0.0% | 8.095e-03 | 0.00 | 4.00 | — |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 1 | 0.0% | 8.095e-03 | 0.00 | 4.00 | — |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| UJI | Ours v16 learned | 输入点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Ours v16 learned | 原始参考点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 输入点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Park & Lee 2007 (DOM adaptation) | 原始参考点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 输入点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 原始参考点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 输入点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 原始参考点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 输入点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 原始参考点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 输入点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 原始参考点 | 2.066e-02 | 2.066e-02 | 2.066e-02 | 8.095e-03 | 1/0 |

## 原生适配与最终结果审计

native 指修正后的仓库适配在公共可行性修复前的端点约束 refit，非作者原版复现。下表均值仅对有记录值计算；逐样本 native/final K、MSE、修复动作、额外 refit 次数和完整耗时见 `native_baseline_summary.csv`（旧记录缺少原生信息时留空，不补造）。

| 数据集 | 方法 | native K | final K | native MSE | final MSE | 平均额外 refit | 使用修复的样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | — | 0.00 | — | 8.095e-03 | — | 0/1 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | — | 0.00 | — | 8.095e-03 | — | 0/1 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | — | 0.00 | — | 8.095e-03 | — | 0/1 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 21 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|
| UJI | Ours v16 learned | 0.0% | 8.095e-03 |
| UJI | Park & Lee 2007 (DOM adaptation) | 0.0% | 8.095e-03 |
| UJI | Liang et al. 2017 (feature-IKI adaptation) | 0.0% | 8.095e-03 |
| UJI | Dung & Tjahjowidodo 2017 (serial adaptation) [threshold-safe adaptation] | 0.0% | 8.095e-03 |
| UJI | Kang 2015 (ADMM adaptation) [threshold-safe adaptation] | 0.0% | 8.095e-03 |
| UJI | Luo et al. 2022 (l-infinity,1 + DE adaptation) [threshold-safe adaptation] | 0.0% | 8.095e-03 |

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
