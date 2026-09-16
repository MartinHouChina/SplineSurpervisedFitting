# 文档索引

当前 3090 试验主线是 fixed-Proposal + offline feasible-subset Teacher v16：原有配置为认证合成源节点 `K=4..56`、候选容量 `Kc=72`；新增 [Kc=56 快速诊断档](v16_kc56_fast_pilot.md) 先用训练源 `K=4..44` 验证筛选机制，两者工程阈值均为 `MSE<=1e-4`。真参数/真节点监督候选生成；固定 Proposal 在预测候选与参数域上离线建立可行 KeepMask/K，随后 Joint 学习一次性筛选与存活节点重定位。真实数据 UJI、Natural Earth、USGS、IndustrialOffset 只用于留出验证/测试。旧 supervised-only Joint 的 `K*=targetK` 可能在预测候选域不可行，不应作为当前正式结论。

工业型线等距线外部验证集的生成、几何有效性规则与运行命令见 [工业模型等距线数据集](industrial_offset_dataset.md)。它是 CAD 驱动半合成数据，不参与训练。

建议阅读顺序：

1. [当前 v16 可行子集教师：数据流、训练、部署与 Linux 指令](v16_feasible_teacher_workflow.md)
2. [Kc=56 快速诊断：短流程、超范围压力测试与命令](v16_kc56_fast_pilot.md)
3. [算法与张量流](architecture.md)
4. [数学定义](math_formulation.md)
5. [旧 supervised-only 训练流程（历史对照）](training_pipeline.md)
6. [细粒度合成教师与新增监督](v16_fine_grained_supervision.md)
7. [部署、六方法表与作图](deployment_pipeline.md)
8. [合成曲线最简性证书](synthetic_data_minimality_report.md)
9. [真实数据集](real_world_datasets.md)
10. [论文方法 adaptation 与公平协议](published_knot_methods_reproduction.md)
11. [v16 历史 counterfactual 说明](v16_counterfactual_subset.md)
12. [验证清单](v16_verification.md)
13. [代码与产物索引](file_guide.md)
14. [PPT 展示提纲](presentation_demo.md)
15. [版本演进](pruning_redesign.md)

Linux 一条龙入口为 [`run_v16_mse1e-4_3090.sh`](../scripts/run_v16_mse1e-4_3090.sh)。它启用离线可行子集教师和固定 Joint 样本；输出 checkpoint、教师缓存、六方法及单列数值修复的测试记录、原六方法四指标图和外部案例图。`quick` 是诊断采样，不是论文结果；文档说明协议，不代表新 checkpoint 已经训练或达标。

`archive/` 只保存历史版本说明。旧 counterfactual、supervised-only、K64/K96 和过去的 Kc56 结果不得与当前 Kc72 实验或新 Kc56 诊断档混写；两套当前配置也应分别报告。`.proposal.pt` 只供 Joint 初始化；部署使用最佳成熟 Joint `.pt`，`.last.pt` 仅用于恢复。

补充说明：[Dung/Kang/Luo 坍缩诊断与公共 MSE 可行性保护层](published_baseline_collapse_safeguard.md)。正式表格必须把保护后的实现标注为 `threshold-safe adaptation`，并保留原生阶段失败诊断。
