# v15：论文方法、多数据集配对比较

入口为 `scripts/benchmark_v15_datasets.py`。当前入口支持 Ours 与 7 个公开／数值基线在相同输入上配对测试，保存完整拟合时间、单独的网络时间、MSE、通过率与最终内部节点数。Park、Liang、Dung、Luo 的新增适配范围与正式命令见 [论文节点方法复现协议](../published_knot_methods_reproduction.md)；下述 2026-09-08 历史结果仍只包含当时的四个方法。

已完成的 94 条曲线配对结果见 [2026-09-08 实测报告](v15_multidataset_results_20260908.md)，包含完整统计、两张 PNG 与 Kang 适配退化分析。

| 方法 | 实现与范围 | 参数化 |
|---|---|---|
| Ours v15 learned | v15 一次性 KeepMask、位置与参数预测，随后标准 B 样条 refit | 网络预测 |
| Kang 2015 adaptation | group-L1 ADMM、活跃簇合并及位置更新；二维适配，未完整复现原文 CVX 与重复节点规则 | 弦长 |
| Yeh 2020 adaptation | 高阶差分特征累计分布布点，加节点数递增扫描；扫描替代原文特定数据集误差回归 | 弦长 |
| Uniform Kmax greedy + gradient | 独立均匀节点集起步，贪心删除、自动微分更新位置，再检验误差 | 弦长 |

论文来源：[Kang et al., CAD 2015](https://doi.org/10.1016/j.cad.2014.08.022)、[Yeh et al., CAD 2020](https://doi.org/10.1016/j.cad.2020.102905)。复现差异见 [论文复现范围](kang_sparse_reproduction.md)。比较结果只评价本仓库这些具体实现。

**本轮发现的 Kang 适配限制：** Natural Earth 的 20 条记录全部出现 `active=40 → cluster_sizes=[40] → final K=1`。当前实现按初始间距分簇，再无条件用一个代表替换每簇，缺少完整的误差约束合并/多代表规则。因此其结果必须标为实现退化诊断，不能用来宣称原版 Kang 论文差于 Ours。`status=ok` 仅表示算出了有限数值，是否可行仍由 `fit_pass` 判定。本轮不会为了提高或降低某方法排名而改变这些原始记录。

## 数据与指标

- 合成数据：读取 checkpoint 的生成配置，在独立 seed=20000 的样本中按源内部节点数 K=4–20 分层抽取。选中样本的逐样本 seed 不得落入训练/验证 seed 区间。源 K 用于分层，预测器不读取节点真值。本轮权重对应 `certified_minimal_source=False`、`canonical_knot_tolerance=0.005`，所以源 K 不是“达到阈值所需最少节点数”的证明；报告同时保留源 K 和 canonical K，不能把两者混称最优 K。
- UJI Pen v2：官方 test 划分，6,093 条 stroke，20 位 writer；分组随机抽样先覆盖各 writer。
- Natural Earth 10m coastline v5.1.2：test 共 533 个曲线窗口，102 个地理 tile。10m 表示 1:10,000,000 地图比例尺。
- USGS 等高线：使用 `large_scale` 目录，test 共 201 个窗口，3 个地理 tile。
- 真实数据只使用 test split，并检查 manifest 内的 writer/tile 没有跨 train/val/test 泄漏。全部没有节点真值，不计算真实数据节点 precision/recall。

所有方法使用相同归一化的 192 点输入；最终是 CPU float64、严格端点插值、无正则的三次 B 样条最小二乘。MSE 定义为 `mean_i ||C(t_i)-Q_i||²`，不取平方根，也不除以坐标维数。主通过率为每条曲线 MSE 不超过 `2.5e-5` 的比例。另报 P95 MSE。

真实数据另评估原始参考点 MSE：控制顶点只在共同重采样输入网格上求解；用原始曲线弦长采样位置将各自参数映射回参考点。原始点使用输入的同一中心和尺度。UJI 原始测试点通常少于 192（中位数 26），上采样不构成新的独立观测；海岸线和等高线参考数据均为 768 点。输入通过率与参考点通过率分开报告。

完整时间从归一化 CPU 输入开始，到最终 B 样条 refit 完成；包括 Ours 的 CPU/GPU 传输。文件加载、共同预处理与评价指标计算不计时。Ours 额外报告同步 batch-1 网络前向时间，输入预先放在设备上；这个时间不能与传统方法完整耗时直接用作端到端加速比。

数值后端全局预热；传统方法每条曲线完整运行一次。Ours 每曲线分别记录 10 次网络前向、3 次完整部署的中位数；汇总表是这些逐曲线时间的均值。设备、线程数、原始重复计时、样本 IDs、代码与权重哈希均保存。计时期间避免其他大规模计算任务。

## 本轮运行

```powershell
python scripts/benchmark_v15_datasets.py `
  --checkpoint outputs/checkpoints/candidate_pruning_one_shot_v15.pt `
  --output-dir outputs/comparisons/v15_multidata_K4_20_n2_real20 `
  --samples-per-knot-count 2 `
  --real-samples-per-dataset 20 `
  --max-internal-knots 28 `
  --gradient-steps 12 `
  --paper-initial-knots 40 `
  --paper-admm-iterations 400 `
  --paper-lambda-bisections 8 `
  --paper-relocation-iterations 8 `
  --mse-tolerance 2.5e-5 `
  --torch-num-threads 1 `
  --device auto
```

以上为 34 条合成 + 60 条真实曲线的初步比较，不能表述为完整 test 集结果。论文扩充可改成 `--samples-per-knot-count 20 --real-samples-per-dataset 100` 并换输出目录，得到 340+300 条配对测试。传统搜索较慢，应为它预留充足时间。

只测合成添加 `--skip-real`；只测真实添加 `--skip-synthetic`。替换 manifest 使用可重复参数：

```powershell
--manifest UJI=data/splits/uji_pen_v2.jsonl
--manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl
--manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl
```

程序每个方法结束即写 `measurements.jsonl`，每条曲线结束更新 `comparison.json`、`measurements.csv`、`summary.csv` 和 `report.md`。中断后原指令加 `--resume`；只有配置、数据、代码和权重指纹一致才会续跑。计算失败样本保留错误记录并计入通过率分母；失败不是删掉困难样本的理由。

测试结束后，从真实记录生成三指标对比图，不重新执行拟合：

```powershell
python scripts/plot_v15_dataset_benchmark.py `
  --input outputs/comparisons/v15_multidata_K4_20_n2_real20/comparison.json
```

输出 `comparison_three_metrics.png`（输入点 MSE、通过率、完整耗时）和 `comparison_reference_metrics.png`（真实数据原始参考点 MSE、通过率、完整耗时）。纯网络耗时独立写在图注及数据表中，避免与完整搜索时间混淆。

合成每档仅 2 条，USGS test 仅 3 个 tile，当前统计独立性有限。应同时看通过率、平均 MSE、P95、节点数及完整时间，不能单凭均值或某张示例图推断方法全面占优。
