# 文档索引

当前正式主线是 supervised-only v16：`Kc=56`，certified Synthetic source `K=4..56`，`MSE<=1e-4`。训练只使用带真参数、真节点和真 K 的认证合成曲线；UJI、Natural Earth、USGS 仅用于 validation/test。Joint 通过有序一一匹配直接监督 KeepMask、计数与存活节点重定位，不运行在线 Teacher。

建议阅读顺序：

1. [算法与张量流](architecture.md)
2. [数学定义](math_formulation.md)
3. [训练流程](training_pipeline.md)
4. [部署、六方法表与作图](deployment_pipeline.md)
5. [合成曲线最简性证书](synthetic_data_minimality_report.md)
6. [真实数据集](real_world_datasets.md)
7. [论文方法 adaptation 与公平协议](published_knot_methods_reproduction.md)
8. [v16 完整说明](v16_counterfactual_subset.md)
9. [验证清单](v16_verification.md)
10. [代码与产物索引](file_guide.md)
11. [PPT 展示提纲](presentation_demo.md)
12. [版本演进](pruning_redesign.md)

一条龙入口为 [`run_v16_mse1e-4_3090.ps1`](../scripts/run_v16_mse1e-4_3090.ps1)。成功运行后应得到 checkpoint、六方法×四数据集表、input/reference 两张 2×2 图和真实六方法案例图。文档仅说明协议与预期产物，不代表这些结果已经在当前机器上生成。

`archive/` 只保存历史版本说明。旧 counterfactual/Teacher、K64/K96 或其他阈值实验不得与当前 supervised K56 主结果混写。
