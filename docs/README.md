# 文档索引

当前 3090 试验主线是 fixed-Proposal + offline feasible-subset Teacher v16，工程阈值为 `MSE<=1e-4`。已拉取的 [Kc=56 快速诊断档](v16_kc56_fast_pilot.md) 显示节点数减少却伴随一次性通过率下降、高 K 候选不可行；新的 [KeepMask 交换监督与高 K 诊断档](v16_keep_swap_highk.md) 恢复源 K=4..56、Kc=72，并单独处理两种故障，效果仍待训练验证。训练只使用认证合成曲线；UJI、Natural Earth、USGS、IndustrialOffset 只用于留出验证/测试。旧 supervised-only Joint 的 `K*=targetK` 可能在预测候选域不可行，不应作为当前正式结论。

工业型线等距线外部验证集的生成、几何有效性规则与运行命令见 [工业模型等距线数据集](industrial_offset_dataset.md)。它是 CAD 驱动半合成数据，不参与训练。

建议阅读顺序：

1. [当前 v16 可行子集教师：数据流、训练、部署与 Linux 指令](v16_feasible_teacher_workflow.md)
2. [KeepMask 交换监督与高 K 分层诊断：新 Linux 一条龙](v16_keep_swap_highk.md)
   [历史 r2 同口径回退与短训验证](v16_r2_rollback_fast.md)是独立诊断，不替换当前主线。
3. [Kc=56 快速诊断：短流程、超范围压力测试与命令](v16_kc56_fast_pilot.md)
4. [算法与张量流](architecture.md)
5. [数学定义](math_formulation.md)
6. [旧 supervised-only 训练流程（历史对照）](training_pipeline.md)
7. [细粒度合成教师与新增监督](v16_fine_grained_supervision.md)
8. [部署、六方法表与作图](deployment_pipeline.md)
9. [合成曲线最简性证书](synthetic_data_minimality_report.md)
10. [真实数据集](real_world_datasets.md)
11. [论文方法 adaptation 与公平协议](published_knot_methods_reproduction.md)
12. [v16 历史 counterfactual 说明](v16_counterfactual_subset.md)
13. [验证清单](v16_verification.md)
14. [代码与产物索引](file_guide.md)
15. [PPT 展示提纲](presentation_demo.md)
16. [版本演进](pruning_redesign.md)

Linux 一条龙入口为 [`run_v16_mse1e-4_3090.sh`](../scripts/run_v16_mse1e-4_3090.sh)。它启用离线可行子集教师和固定 Joint 样本；输出 checkpoint、教师缓存、六方法及单列数值修复的测试记录、原六方法四指标图和外部案例图。`quick` 是诊断采样，不是论文结果；文档说明协议，不代表新 checkpoint 已经训练或达标。

`archive/` 只保存历史版本说明。旧 counterfactual、supervised-only、K64/K96、旧 Kc72 与 Kc56 结果不得和新 KeepMask 交换监督诊断混写；各配置必须分别报告。`.proposal.pt` 只供 Joint 初始化；部署使用最佳成熟 Joint `.pt`，`.last.pt` 仅用于恢复。

补充说明：[Dung/Kang/Luo 坍缩诊断与公共 MSE 可行性保护层](published_baseline_collapse_safeguard.md)。正式表格必须把保护后的实现标注为 `threshold-safe adaptation`，并保留原生阶段失败诊断。
