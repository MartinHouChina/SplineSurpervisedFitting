# 文档索引

历史 Kc56 实测：[本地短训、筛选与重定位验证](v16_local_self_validation.md)。此前 Keep 状态恢复档尚未证明有效简化；该报告用于失败定位，不代表当前 K48、`5e-5`、128 代配置的实训结果。

历史代码联调：[K24 检查与三项流程修复记录](v16_k24_code_audit.md)，涵盖四指标图读取、数值失败统计和分阶段续跑；这些旧配置检查不是 K48 实训证据，流程修复也不等于模型质量已经达标。

当前新实验为 [中小 K / K48 独立配置](v16_small_medium_k48.md)：源内部 K=4..20 不变，Kc 从 24 提高到 48，完整节点向量最多 56 项、控制顶点最多 52 个，`MSE<=5e-5`，从零训练 Proposal 64 + Joint 64 = 128 代（Joint 已含 4 代 warmup）；合成训练/验证 600/160、每个真实来源验证最多 20、192 点、batch 32。保留离线可行 Teacher 和 Keep 状态交互，不混合高 K 训练；六方法统一内部节点预算 48。训练为 100% 合成，真实数据仅用于验证/测试。初始 Keep 比例 0.5 对应概率质量 24，不等于固定保留 24 个节点。[旧 K24 配置与检查](v16_small_medium_k24.md)原样保留，不代表 K48、128 代配置已完成实训或证明最终简化能力。

K48 默认运行名为 `candidate_selection_v16_mse5e-5_small_medium_sourcek20_kc48_p64_j64_linux_r1`；推荐确认名称未被使用，并显式传入 `--mse-tolerance 5e-5 --epochs 128 --proposal-epochs 64`。独立范围为 `small_medium_k48`，认证合同为 `certified_source_subset_threshold_minimal_k4_20_kc48_span001_v1`；K48 须新建实验，不得复用旧 K24 或旧阈值数据/Teacher 缓存及 checkpoint。历史共享入口默认阈值仍为 `1e-4`。

此前 [Keep 状态交互恢复档](v16_keep_state_recovery.md)使用完整 r2 热启动、Kc=56、源 K=4..56。缩小规模的本地验证未通过，不能作为有效改良发布；其中旧 r2 的真实数据预训练来源仍保留。Kc72 交换监督、Kc56 pilot 和仅 Proposal 热启动的 r2 短训也是独立历史诊断。

工业型线等距线外部验证集的生成、几何有效性规则与运行命令见 [工业模型等距线数据集](industrial_offset_dataset.md)。它是 CAD 驱动半合成数据，不参与训练。

建议阅读顺序：

先阅读 [K48 范围、训练和 3090 命令](v16_small_medium_k48.md)，再查阅 [历史 K24 配置与检查](v16_small_medium_k24.md)、[此前恢复实验的实测失败定位](v16_local_self_validation.md)及以下资料。

1. [v16 可行子集教师：数据流、训练、部署与历史 Linux 指令](v16_feasible_teacher_workflow.md)
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

当前 Linux 一条龙入口为 [`run_v16_small_medium_k48_3090.sh`](../scripts/run_v16_small_medium_k48_3090.sh)，它调用共享脚本的 `--small-medium-k48` 并启用独立 Kc48 档。输出 checkpoint、教师缓存、六方法及单列数值修复的测试记录、原六方法四指标图和外部案例图。该新范围按诊断协议运行，不能套用旧 K=4..56 的正式资格；文档说明协议，不代表新 checkpoint 已经充分训练或达标。旧 K24 入口和历史 1070 配置保留，不随 K48 切换。

`archive/` 只保存历史版本说明。旧 counterfactual、supervised-only、K64/K96、Kc72、Kc56 与 K24 结果不得和当前 K48 诊断混写；各配置必须分别报告。`.proposal.pt` 只供 Joint 初始化；部署使用最佳成熟 Joint `.pt`，`.last.pt` 仅用于恢复。

补充说明：[Dung/Kang/Luo 坍缩诊断与公共 MSE 可行性保护层](published_baseline_collapse_safeguard.md)。正式表格必须把保护后的实现标注为 `threshold-safe adaptation`，并保留原生阶段失败诊断。
