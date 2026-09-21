# v16 多数据集配对测试

**DIAGNOSTIC NOT FINAL**

诊断原因：configured target below 97%

Checkpoint: `v16.pt`（candidate_selection_counterfactual_bspline_v16，epoch 1）。
统一阈值：MSE ≤ 2.500e-05；MSE = mean_i ||C(t_i)-Q_i||²，不开方。
新增最大拟合误差 MaxSqErr = max_i ||C(t_i)-Q_i||²，同样不开方；它衡量单条曲线最差采样点，与一组曲线中最大的 MSE 不同。输入点和原始参考点分别计算，沿用同一参数映射；这是离散对应点误差，不是 Hausdorff 距离或连续曲线最大误差保证。通过率仍由 MSE 阈值判断。
所有方法接收相同归一化有序点，并以 CPU float64、端点插值、无正则标准 B 样条最小二乘作为最终报告拟合。
参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，Ours v16 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 fixed-chord ablation。
完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。
节点容量（分别列出，不隐含相等）：见 experiment.json configuration。
设备：{'device': 'cpu'}。
数值基线协议：未启用或历史报告未记录公共可行性修复；不将结果标为 threshold-safe adaptation。

| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 1 | 100.0% | 1.000e-06 | 2.00 | 3.00 | 1.00 |

## 最大拟合误差（平方欧氏距离，不开方）

先对每条曲线取点误差最大值，再报告其均值、P95 和全组最大值。最差曲线 MSE 单独列出，不与 MaxSqErr 混用。旧结果缺少逐点残差时不从 MSE 推算：只要有成功样本缺失该指标，相应峰值统计就记为 N/A；失败样本数见 failed 字段。

| 数据集 | 方法 | 点集 | MaxSqErr 均值 | MaxSqErr P95 | MaxSqErr 最大值 | 最差曲线 MSE | 峰值有效/缺失样本数 |
|---|---|---|---:|---:|---:|---:|---:|
| Synthetic | Ours v16 learned | 输入点 | N/A | N/A | N/A | 1.000e-06 | 0/1 |

外部曲线原始参考点测试（包括程序生成工业等距线）：仅在 24 个重采样输入点上拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，这不产生新的独立观测。通过率由参考点 MSE 单独判定。

| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |
|---|---|---:|---:|

复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码逐位一致。
失败样本计入通过率分母和耗时均值；MSE 与保留内部节点 K 均值只含有有限解的样本（不要求达标），failed 列单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。

论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。
