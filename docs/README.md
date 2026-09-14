# 文档索引

当前正式主线是 supervised-only v16：`Kc=56`，certified Synthetic source `K=4..56`，`MSE<=1e-4`。训练只使用带真参数、真节点、真 K 和逐节点删除 MSE 的认证合成曲线；UJI、Natural Earth、USGS 与 IndustrialOffset 仅用于 validation/test。Joint 通过有序一一匹配直接监督 KeepMask、计数与存活节点重定位，并用最简性证书的 single-deletion MSE 细化关键节点权重。在线 prefix/counterfactual self-teacher 仍禁用。

工业型线等距线外部验证集的生成、几何有效性规则与运行命令见 [工业模型等距线数据集](industrial_offset_dataset.md)。它是 CAD 驱动半合成数据，不参与训练。

建议阅读顺序：

1. [算法与张量流](architecture.md)
2. [数学定义](math_formulation.md)
3. [训练流程](training_pipeline.md)
4. [细粒度合成教师与新增监督](v16_fine_grained_supervision.md)
5. [部署、六方法表与作图](deployment_pipeline.md)
6. [合成曲线最简性证书](synthetic_data_minimality_report.md)
7. [真实数据集](real_world_datasets.md)
8. [论文方法 adaptation 与公平协议](published_knot_methods_reproduction.md)
9. [v16 完整说明](v16_counterfactual_subset.md)
10. [验证清单](v16_verification.md)
11. [代码与产物索引](file_guide.md)
12. [PPT 展示提纲](presentation_demo.md)
13. [版本演进](pruning_redesign.md)

一条龙入口为 [`run_v16_mse1e-4_3090.ps1`](../scripts/run_v16_mse1e-4_3090.ps1)。成功运行后应得到 checkpoint、六方法×五数据集表、input/reference 两张 2×2 图和外部数据六方法案例图。文档仅说明协议与预期产物，不代表这些结果已经在当前机器上生成。

`archive/` 只保存历史版本说明。旧 counterfactual/Teacher、K64/K96 或其他阈值实验不得与当前 supervised K56 主结果混写。

补充说明：[Dung/Kang/Luo 坍缩诊断与公共 MSE 可行性保护层](published_baseline_collapse_safeguard.md)。正式表格必须把保护后的实现标注为 `threshold-safe adaptation`，并保留原生阶段失败诊断。
