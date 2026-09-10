# 文档索引

当前主协议统一为 `Kc=56` 个内部候选、完整三次开放节点向量最多 64 项、
控制顶点最多 60 个、合成 source `K=4..56`（控制顶点 8～60）、`MSE<=1e-4` 和 worst-source
deployment pass `>=90%`。旧 K64/K96 结果只作为历史容量消融。
正式简化合同为 `ranked_prefix_ordered_proposal_high_k_adaptive_complexity_v2`：
前 32/96 个 epoch 的 Proposal synthetic draws 有 50% 来自 `K>=40`，并行使用
directed coverage 与 monotone one-to-one assignment；后 64 个 Joint epoch 恢复
K=4..56 原始分布。

source K 和候选 Kc 虽然都可达到 56，但语义不同；`source K=56` 是生成复杂度
边界，`Kc=56` 是候选容量边界。该层没有冗余候选，必须单独报告 dense 与
deployment pass，且不得为其放宽 90% 资格门槛。当前合成数据显式使用
`--knot-min-span 0.01`；底层通用 synthetic 生成器的旧 0.02 默认只兼容历史范围。

正式训练显式使用 `--synthetic-boundary-val-size 32`：这 32 条 K=56 边界样本
包含在 `--val-size` 指定的合成验证总数内，不额外增加验证集大小。资格审计
单独以 `n>=32` 报告该层 dense/deployment pass。checkpoint 数据合同已升级为
`source_subset_threshold_minimal_k4_56_span001_v2`；旧 K=4..24 checkpoint 与
该合同不兼容，正式入口会拒绝，不能通过改名或 resume 复用资格。

## 当前 v16 主线

按以下顺序阅读即可：

1. [算法与数据流](architecture.md)
2. [数学定义](math_formulation.md)
3. [训练流程](training_pipeline.md)
4. [部署流程](deployment_pipeline.md)
5. [合成曲线最简性证书](synthetic_data_minimality_report.md)
6. [真实数据集与拆分](real_world_datasets.md)
7. [公开论文方法适配与公平比较](published_knot_methods_reproduction.md)
8. [Kang/Luo 在 MSE=1e-4 下的统一复查](kang_luo_mse1e-4_audit.md)
9. [v16 完整说明](v16_counterfactual_subset.md)
10. [代码与产物索引](file_guide.md)
11. [测试和 checkpoint 验证状态](v16_verification.md)

RTX 3090 的新训练、资格检查、六方法比较和真实曲线作图由
[`run_v16_mse1e-4_3090.ps1`](../scripts/run_v16_mse1e-4_3090.ps1) 串行执行；
目标产物使用 `k56_ordered_highk` 路径，避免与旧失败 K56 运行冲突。

PPT 展示可直接使用 [演示速查](presentation_demo.md) 和
[v16 流程图](figures/v16_pipeline.png)。UJI、Natural Earth 与 USGS 的数据准备分别见
[UJI 接入](uji_pen_integration.md)和[地理数据接入](geospatial_real_world_data.md)。

## 历史资料

v7–v15 的旧命令、实验记录和失败分析已集中到 [archive](archive/README.md)。历史文档不再
作为当前命令入口，但保留用于结果追溯。
