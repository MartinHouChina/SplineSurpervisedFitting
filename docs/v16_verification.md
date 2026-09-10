# v16：MSE=1e-4 代码验证与 checkpoint 状态

更新日期：2026-09-10。当前主协议是 `Kc=56`（完整三次开放节点向量 64 项、
控制顶点 60 个）、合成 source internal `K=4..56`（控制顶点 8～60）、`MSE <= 1e-4`、最差
数据源 deployment pass 至少 90%；简化合同为
`ranked_prefix_ordered_proposal_high_k_adaptive_complexity_v2`。

## 1. 当前状态

目标 checkpoint：

~~~text
outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk.pt
~~~

该文件当前尚不存在，必须在 3090 上完成新训练并通过资格检查后才能正式汇报。已有
`candidate_selection_v16_mse5e-5_k64.proposal.pt` 只用于 warm start Encoder、
ParameterHead 和 CandidateHead：65 个旧 interval query 沿参数域插值为 57 个，
其他形状兼容的 proposal 张量迁移。Selector、联合重定位头及优化器重新训练。

旧 K64、Kc=96/2.5e-5、旧 5e-5 checkpoint 和任何 diagnostic checkpoint 都不是本轮
1e-4 结果，不能改名或混入正式表格。

## 2. 本轮机制检查

| 模块 | 已验证内容 |
|---|---|
| Candidate network | Kc=56 稠密候选、严格有序参数/候选、局部 cross-attention；Proposal 使用 coverage + monotone one-to-one assignment |
| Selector | 曲线级自适应 beta、一次性 mass-TopK、Joint 初始 keep fraction `30/56≈0.535714`（质量约 30，不是最终 K）、无固定 0.5 mask |
| Teacher | 低 K=4..16 逐计数扫描、粗到细前缀搜索、反事实集合、合成 geometry-oracle 候选组合 |
| Count semantics | certified source K 在可重定位条件下作为上界，不把更小的可行教师强拉回 source K |
| Relocation/refit | 仅存活节点交互重定位、一次标准三次 B 样条 refit、MSE 不开方 |
| Mixed data | Synthetic K=4..56 使用显式 `knot_min_span=0.01`；Proposal synthetic 50% 来自 K>=40，Joint 恢复原分布；UJI、Natural Earth、USGS 训练/验证分离和真实无标签路由 |
| Published baselines | Park、Liang、Dung、Kang、Luo 的统一容量、refit、MSE 和计时接口 |
| Visualization | 六方法 3x2 真实曲线图，含输入/参考、拟合、控制顶点、节点、K、MSE 和时间 |
| Safety | 不覆盖已有实验；未合格 checkpoint 默认停止，diagnostic 输出隔离并带水印 |

本轮定向测试：

~~~text
209 passed in 41.04s
compileall: passed
Ruff（忽略仓库既有的 sys.path/E402 与单行 E702）：passed
PowerShell one-click dry-run: passed
git diff --check: passed
~~~

测试覆盖本轮修改的网络、损失、训练恢复、Kang/Luo、六方法 benchmark、真实案例图和
3090 串行入口。测试通过只说明代码链路成立，不等于新模型已经达到目标数值。

## 3. 正式资格

训练完成后执行：

~~~powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk.pt `
  --required-pass-rate 0.90 `
  --mse-tolerance 1e-4
~~~

返回码 0 才能生成正式 benchmark 和图片。守卫至少核查：

- `stage=joint`，proposal 和 simplification 课程均已成熟；
- objective、`ranked_prefix_ordered_proposal_high_k_adaptive_complexity_v2` 简化合同、架构和一次性 mass-TopK 部署合同一致；
- Synthetic/三个真实验证来源中最差 dense 与 deployment pass 均不低于 90%；
- 最终安全储备已经退火到配置值，且不是 all-keep 退化解；
- Synthetic count MAE、节点匹配 F1 和 matched MAE 达到正式门槛；
- `source K=56` 容量边界层的 dense/deployment pass 单独报告且达到既定资格；
- 阈值确实为 `1e-4`，失败样本未被移出统计。

返回码 2 表示只能排错。需要观察失败案例时，显式使用
`--allow-unqualified-diagnostic`；输出不得作为论文或汇报结果。

## 4. 一键验证

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_mse1e-4_3090.ps1
~~~

该脚本依次运行：新训练、资格检查、Synthetic 与三类真实数据上的六方法四指标
benchmark、2x2 指标图、六方法真实案例图。若资格不通过，正式模式会在 benchmark 前
停止。详细参数与分步命令见[训练流程](training_pipeline.md)。

## 5. 结果解释边界

当前可以声明代码和实验协议已经接通；不能提前声明：

- 新模型已经达到 90% 通过率或平均节点数目标；
- Ours 已在六方法正式统计中占优；
- teacher 找到了连续节点空间的全局最少解；
- 单例 smoke、旧 checkpoint 或 diagnostic 图片等同正式结果。

Kang/Luo 的实现修正与简单/复杂曲线失效位置见
[MSE=1e-4 统一复查](kang_luo_mse1e-4_audit.md)。
