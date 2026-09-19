# 文档索引

[Overnight Stable：可达教师与全部外部数据集评测](overnight_stable.md)：Compact 实验的配对审计、轨迹监督改进、工业等距线及四个外部来源的六方法测试/案例、上传与 Linux 一条龙。

[Overnight Compact：可行性约束下的紧凑节点实验](overnight_compact.md)：新增 `--compact-selection`，24-epoch 显式 best 权重 warm-start、四项训练改进、更新包上传与 Linux 一条龙；不增加部署搜索，不保证全局最简或达标。

[Overnight Reliable：短程改进与独立真实验证](overnight_reliable.md)：推荐的 `--reliable-selection` 入口；参数 trust gate、几何教师、32-epoch warm-start、六方法四指标与案例图，以及修复前后对照口径。

[1070 历史架构的 Linux 一键训练与六方法比较](overnight_1070_linux.md)：独立 warm-start 新实验、四指标图、Ours 和六方法真实案例；与主线离线 teacher 流程分开使用。

[Overnight Plus 增强训练与全模型 warm-start](overnight_plus.md)：可选 `--enhanced-selection`，保留历史架构，增强在线 teacher、边界排序和 Joint 学习率；含服务器一条龙与续跑命令。

## 当前 v16 主线

按以下顺序阅读即可：

1. [算法与数据流](architecture.md)
2. [数学定义](math_formulation.md)
3. [训练流程](training_pipeline.md)
4. [部署流程](deployment_pipeline.md)
5. [合成曲线最简性证书](synthetic_data_minimality_report.md)
6. [真实数据集与拆分](real_world_datasets.md)
7. [公开论文方法适配与公平比较](published_knot_methods_reproduction.md)
8. [v16 完整说明](v16_counterfactual_subset.md)
9. [代码与产物索引](file_guide.md)
10. [测试和 checkpoint 验证状态](v16_verification.md)

PPT 展示可直接使用 [演示速查](presentation_demo.md) 和
[v16 流程图](figures/v16_pipeline.png)。UJI、Natural Earth 与 USGS 的数据准备分别见
[UJI 接入](uji_pen_integration.md)和[地理数据接入](geospatial_real_world_data.md)。

## 历史资料

v7–v15 的旧命令、实验记录和失败分析已集中到 [archive](archive/README.md)。历史文档不再
作为当前命令入口，但保留用于结果追溯。
