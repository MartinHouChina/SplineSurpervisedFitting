# 论文图表：最终部署与冗余节点删除

## 已生成的五类图表

目录：`outputs/figures/postprune_paper_20260923/`。PNG 均为300 dpi。

| 内容 | 文件 | 版式 |
|---|---|---|
| 当前方法框架 | `framework.png` | 网络预测与数值修复、删除的实际数据流 |
| 六方法案例对比 | `six_methods_6x5.png` | 六列方法 × 五行数据来源；每行使用同一条曲线 |
| Ours 案例综述 | `ours_5x5.png` | 每个来源五例，共25例 |
| 四组核心指标 | `core_metrics.png` | 节点数、双阈值通过率、误差、总耗时；误差细分 MSE 与最大平方误差 |
| 分数据集指标 | `core_metrics_by_dataset.png` | 五个来源分别统计，不以展示案例代替全部数据 |
| 实验参数 | `experiment_parameters.png` | 实际检查点、训练配置、测试配置和对照方法预算 |

案例图不标具体误差、时间、节点数量或 PASS/MISS，不加水印或额外方法脚注。六方法名称仅在列头出现一次，Ours 图不重复方法名称；每图仅有一个统一图例。红色叉号表示曲线上的内部节点，下方红色短线表示参数域内的节点分布；橙色方块与虚线表示控制顶点与控制多边形。所有坐标来自实际保存的拟合结果，没有移动节点、美化曲线或重新计算拟合。

## 模型与数据来源

- 使用 `outputs/checkpoints/overnight_stable_k32_3090_r1.pt`，保存于 epoch 14；不是后来 M16/M32 coupled 模型。参数表中的12轮 Proposal＋20轮 Joint 是配置训练计划，不是保存权重已经运行的轮数。
- 实测报告：`outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json`。
- 61条曲线、6种方法，保留全部366项最终记录：Synthetic 21条（原始内部 K=4…24），UJI、Natural Earth、USGS、IndustrialOffset 各10条。IndustrialOffset 是工业模型生成的等距线，不是实地测量数据。
- 内部节点上限32。三次开区间夹持样条两端各重复4次，因此完整节点向量长度至多40，控制顶点至多36。
- 框架图已经核对当前权重的真实结构：没有将较新模型的重复交互层画入旧模型。
- 此次 Ours 结果包含既有公共插入修复和新增冗余节点删除。其他五种方法沿用仓库适配实现及既有公共插入修复。本次绘图没有修改任何方法或重新挑选模型。

## 数据保存与统计口径

- `case_panels_manifest.json`：55个面板的样本编号、来源、方法、完整指标、文件哈希、显示设置和选择规则。
- `case_panels_data/*.npz`：51份去重后的样本/方法几何数据，包含实际保存的输入、参数、节点向量、控制顶点、密集曲线与残差等数组。面板与文件的映射在 manifest 中。
- `core_metrics.json`、`core_metrics.md`：全部61条测试曲线的总体及分来源统计，包括每种方法有效返回数量和失败数量。
- `experiment_parameters.json`、`experiment_parameters.md`：完整参数、检查点哈希、配置来源及对照方法设置。
- `framework_metadata.json`：绘制框架图时验证的实际配置与模型哈希。
- 原始实测报告及 `geometry/`、`raw/`、`before_pruning/` 保留最终、原始与删除前的完整几何；没有覆盖旧实验。

MSE 为归一化输入采样点上的平均平方欧氏距离；最大平方误差为所有有限返回曲线的所有输入点中的最大平方欧氏误差。两者均不取平方根。双阈值为 MSE≤5e-5 且 MaxSE≤5e-4；这不是连续曲线或 Hausdorff 误差保证。误差及节点均值包含有限但超阈值的曲线，失败计入通过率分母；表中 `Valid / all` 是有限返回数/请求数，不是通过数。

总耗时使用此次完整重测记录，包含原方法、公共插入和 Ours 删除，不以 network-only 时间替代。Ours 总体平均内部 K=16.64、双阈值通过48/61、总耗时625.87 ms；这是本地 GTX 1070 与CPU数值求解混合环境下的探索性计时，不据此宣称严格硬件加速比。

六方法图采用固定种子20260922，每个来源随机抽取一条，不依据结果好坏选图。Ours 图为定性的精选案例：优先双阈值达标、兼顾不同来源分组，并记录完整排序规则。当前25个展示案例均达标，但不能据此推断总体通过率为100%。

## 一条命令重新绘制

在现有项目环境运行，无需重新训练，也不会重新运行六种拟合算法：

```bash
python scripts/render_postprune_paper.py --report outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json --checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt --output-dir outputs/figures/postprune_paper_r2 --dpi 300
```

Windows PowerShell 与 Linux Bash 均可直接运行上述单行命令。输出目录必须为新目录或空目录，避免覆盖已有论文图片。模型只用于核对框架和参数，必须与实测报告中的权重哈希一致。

## 交付包

`outputs/delivery/postprune_paper_figures_20260923.zip`：

- `figures/`：所有PNG及对应JSON/Markdown/面板NPZ。
- `data/`：完整61×6重测报告和全部几何数据，保留相对路径及原哈希。
- `replot_scripts/`：此次新增的五个绘图/导出入口，需配合现有项目依赖运行。
- `README.md`：本说明；方法细节与统计口径留在文档中，不叠加到案例图。

解压后可直接使用PNG；读取数据时以 `data/comparison.json` 及其几何引用为准。元数据中原运行的绝对路径仅作来源记录，不代表另一台电脑必须具有同一盘符。

## USGS 展示案例替换版

另存于 `outputs/figures/postprune_paper_usgs_alternative_20260923/`，不覆盖上述原图与交付包。

- 仅将六方法图中的 USGS 行替换为 `usgs_tnm_contours_large_scale_08cddbc994fc_w000_08cddbc9`；六种方法使用相同输入，其余四行保持原样。
- 这是结果已知后选取的说明性案例，不再把 USGS 行称作随机样例。选择方式保存在 `case_panels_manifest.json`；61条测试曲线的总体统计没有更改。
- 六种方法均满足双阈值。Ours 内部节点13个，其余方法依次为 Park 18、Liang 16、Dung 14、Kang 14、Luo 14。Ours 的最大平方误差最低，但不是 MSE 最低。
- `six_methods_case_values.md` 直接列出新版图中30个面板的内部节点数、MSE、最大平方误差、时间与达标情况。无需手动展开较大的JSON即可查看。
- 原图30个面板的数据仍在原目录的 `case_panels_manifest.json`；完整366项测试在实测报告 `measurements` 中，字段分别为 `final_k`、`mse`、`max_squared_error`、`total_ms`。`raw_measurements` 是处理前结果，勿与最终值混用。

复现替换版（使用新的输出目录）：

```bash
python scripts/plot_postprune_case_panels.py --report outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json --output-dir outputs/figures/postprune_paper_usgs_alternative_r2 --comparison-sample USGS=usgs_tnm_contours_large_scale_08cddbc994fc_w000_08cddbc9
```
