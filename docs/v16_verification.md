# v16 当前代码验证与历史 checkpoint 审计

更新日期：2026-09-09。本文区分当前“ranked-prefix + certified-cardinality + adaptive-complexity”代码机制、旧 checkpoint 状态和尚未完成的正式泛化实验；旧权重不能被重新命名成当前架构结果。

## 1. 代码验证

本次简化增强修改前的全仓库基线为：

~~~text
592 passed
~~~

该数字不能自动作为本次改动后的测试结论。本次需用下方命令重新复查；正式汇报应记录实际最新结果，不应继续复制旧计数。覆盖范围新增 ranked-prefix 搜索、真计数/真参数/真节点标签路由、容量无关计数损失、安全储备回退和简化成熟度守卫。

覆盖：

| 模块 | 已验证内容 |
|---|---|
| v16 network | 空/全保留集合、selected-only KV、coverage 空分区、参数与节点严格有序、跨删除区间重定位、有限梯度 |
| subset objective | 标准 B 样条可微 refit、粗到细 ranked-prefix 边界、Bernoulli score-function、真计数/位置监督 |
| two-stage training | proposal gate、joint 切换、best/last/proposal 保存和严格恢复 |
| mixed data | 来源平衡、writer/地理 group 防泄漏、manifest 指纹、certified Synthetic 标签和真实样本无效 mask |
| deployment | 点云 PNG/JSON、网络/完整时间、Ours 单例结构图、多案例总览和真实曲线四宫格 |
| benchmark | Ours + Park/Liang/Dung/Kang/Luo/Yeh/uniform greedy 七基线及只读 2×2 指标重绘 |
| qualification | proposal、旧简化合同、未成熟课程、非最终安全储备、缺失 Synthetic count/knot 指标和低于 90% 均默认拒绝 |
| diagnostic mode | 终端、JSON、Markdown 和 PNG 强制标记 DIAGNOSTIC NOT FINAL |

复查：

~~~powershell
$env:PYTHONPATH = 'src'
python -B -m pytest tests -q
~~~

复查本轮 v16 训练、资格、部署和比较入口：

~~~powershell
$env:PYTHONPATH = 'src'
python -B -m pytest -p no:cacheprovider tests/test_v16_subset_loss.py tests/test_v16_network.py tests/test_train_v16.py tests/test_v16_checkpoint_integration.py tests/test_inspect_v16_checkpoint.py tests/test_benchmark_v15_datasets.py tests/test_plot_v16_method_comparison.py tests/test_visualize_v16_real_deployments.py tests/test_fit_v16_point_cloud.py -q
~~~

小规模 smoke 已验证组合学习、混合真实数据加载、benchmark 和点云可视化链路。此类产物只写入 `outputs/tmp/`，可随时再生成，不得作为论文证据。

### 1.1 正式绘图入口

方法对比图从 benchmark 的 `comparison.json` 只读生成；默认展示 Ours、Park、Liang、Dung、Kang 和 Luo，可用 `--method-set all` 增加 Yeh 与统一贪心：

~~~powershell
python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_certified_k96_published/comparison.json `
  --output-dir outputs/figures/v16_certified_k96/method_comparison `
  --method-set all `
  --dpi 300
~~~

Ours 单例和总览图从合格 checkpoint 直接部署留出真实测试曲线：

~~~powershell
python scripts/visualize_v16_ours_cases.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --real-samples-per-dataset 2 `
  --selection-seed 20260909 `
  --mse-tolerance 2.5e-5 `
  --device cuda `
  --dpi 300 `
  --output-dir outputs/figures/v16_certified_k96/ours_cases
~~~

前者的四个子图是 MSE、通过率、最终内部节点数和完整算法时间；后者的单例图明确标注采样点、拟合曲线、控制多边形、控制顶点、曲线节点及参数域节点条，并写出 `ours_cases_overview.png` 与 `deployment_visualizations.json`。

当前文档中的 `candidate_selection_v16_simplified_certified_k96.pt` 仍是按新简化合同训练的目标路径。只有它实际存在且 `inspect_v16_checkpoint.py` 返回 0 后，才能生成或引用正式图。本节不声称正式图已经生成；未合格输入只能增加 `--allow-unqualified-diagnostic` 进行带水印排错。

### 1.2 本轮四项简化增强

1. 训练教师改为 coarse-to-fine approximate minimum feasible ranked-prefix
   搜索：所有曲线先检查低 K 二次加密网格，certified Synthetic 额外检查
   真 K，再做边界与局部细化。它不是全局最优证明，且部署仍是一次前向、
   一次 Top-K、一次 refit。
2. certified Synthetic 提供真参数、真内部节点和真计数；真实数据不伪造
   标签。节点先可微 warp 到真参数域，再用 detached 一维单调最大基数
   一一匹配确定配对、SmoothL1 更新预测位置；未匹配数由 count 监督处理。
   真计数项在部署不可行时只纠正 under-count，可行后再对称贴近真 K，并
   额外抑制 over-count。
3. 教师计数改为容量无关的 `log1p` 损失；复杂度由 validation pass feedback
   multiplier 调整：90%～92% 以 0.5 倍速度继续简化、≥92% 全速，<90%
   以 2 倍速度回滚；不能称为严格 Lagrangian/primal-dual。
4. 选择安全储备从 `2+0.25σ` 退火到 `0+0.05σ`，并随上述 controller
   可逆变化；coverage 空分区不会再覆盖先前有效 anchor。

对 source K=4～20 的 certified Synthetic，目标平均预测 K 为 10～14，中心
接近真值均值约 12。该目标必须在新训练上实测，不构成无训练保证。

## 2. 历史 Kc=64 固定阈值 checkpoint 审计

当前存在：

~~~text
outputs/checkpoints/candidate_selection_v16.proposal.pt
outputs/checkpoints/candidate_selection_v16.last.pt
outputs/checkpoints/candidate_selection_v16.history.json
~~~

不存在：

~~~text
outputs/checkpoints/candidate_selection_v16.pt
~~~

原 97% 实验已结束。以下是 2026-09-09 11:41 的最佳 proposal 固定快照：

| 指标 | 数值 |
|---|---:|
| best proposal epoch / stage | 59 / proposal |
| overall dense MSE | 1.5519772e-6 |
| overall dense pass | 98.923% |
| overall deployment pass | 98.846% |
| Synthetic dense / deployment pass | 100% / 100% |
| UJI dense / deployment pass | 100% / 100% |
| Natural Earth dense / deployment pass | 93% / 92% |
| USGS dense / deployment pass | 93% / 93% |
| worst-source dense pass | 93% |
| worst-source deployment pass | 92% |
| proposal target | 97% |

最终 last=epoch 80 proposal，worst-source dense/deployment=92%/92%；随后在进入 joint 前因原 97% gate 未满足而停止。因此该 best-proposal 快照：

~~~text
proposal_ready = false
formal_reporting_eligible = false
~~~

总体 dense 98.923% 不能替代最差来源 deployment 92%。该快照尚未进入 selector 的有效 joint 训练，不能据此评价最终节点数或一次性筛选效果。训练会继续更新 checkpoint，结束后必须重新审计。

## 3. 正式资格守卫

共享 assess_v16_checkpoint 默认使用 90% 工程协议并重新核验原始字段，不直接信任 checkpoint_quality 字符串。正式资格需要：

- stage=joint；
- objective_version=candidate_selection_counterfactual_bspline_v16；
- architecture_revision=v16_ranked_prefix_count_coupled_mass_topk；
- model/deployment policy=mass_topk 且 adaptive keep threshold=true；
- simplification_contract=ranked_prefix_certified_cardinality_adaptive_complexity_v1；
- simplification_ready=true，即 Joint 成熟期已完成；
- checkpoint 实测 safety sigma/knots 等于训练配置的最终储备，且正式储备要求 `knots=0`、`sigma<=0.05`；
- synthetic_data_contract=source_subset_threshold_minimal_v1 且 certified-minimal source=true；
- Synthetic `count MAE<=2.0`、`knot-match F1@0.01>=0.60`、`matched-knot MAE<=0.005`；
- `count`、真计数、over-count、complexity、真参数、候选覆盖和存活节点位置损失均为正权重；
- 验证集平均保留数必须小于候选容量，拒绝全保留退化解；
- configured proposal/deployment target 均不低于 0.90；
- observed worst-source dense proposal 与 deployment pass 均不低于 0.90；
- proposal_ready=true；
- allow-infeasible-proposals=false；
- 正式 MSE 阈值一致；
- 保存的旧布尔/字符串与实测结果一致。

fit、benchmark 和 visualization 默认拒绝未达标 checkpoint。--allow-unqualified-diagnostic 只用于排查，并强制写入诊断水印。

训练结束后先执行：

~~~powershell
python scripts/inspect_v16_checkpoint.py --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt --required-pass-rate 0.90 --mse-tolerance 2.5e-5
~~~

返回码 0 表示可进入正式部署；返回码 2 表示只能作为诊断权重，终端会列出未通过的条件。任何在本次 ranked-prefix/count-coupled 简化合同接线前启动或保存的权重都会因缺少新架构、简化成熟度、最终安全储备或 Synthetic 真值指标而返回 2；可把形状兼容的 proposal 参数作为 warm-start，但正式结果必须按新合同重新训练。

## 4. 恢复验证

resume 语义已经由测试固定：

- epochs 是总轮数；
- proposal-epochs 是绝对阶段边界；
- last 仍为 proposal 时，proposal-epochs 只能保持或增加；
- 可修改设备、worker、线程和日志频率；
- 数据规模、控制顶点范围、Kc、阈值、目标、真实比例、manifest 和 output 必须保持一致；
- 进入 joint 后不能通过增加 proposal-epochs 回退阶段。

旧实验配置和恢复约束见 [训练流程](training_pipeline.md)。它采用 batch=8、train=4000、val=1000 和 97% target；已经结束，也不能通过 resume 改成快速配置。

## 5. 清理验证

本轮删除的只有可再生成项目：

- Python bytecode 和 __pycache__；
- pytest、Ruff 和 Codex 临时缓存；
- outputs/smoke 与 outputs/v16_smoke；
- 名称明确包含 smoke 的旧 checkpoint、比较图和恢复测试产物。

以下内容未删除：

- 当前 v16 proposal/last/history；
- v15 正式 checkpoint 链；
- 正式多数据集结果；
- 历史 checkpoint archive；
- 真实数据；
- Kang 论文的唯一工作区副本；
- recoverable_cleanup 历史归档。

## 6. 尚未完成的实验

1. 从相同设置分别训练当前自适应 `Kc=64` 与 `Kc=96`，完成主容量消融；
2. 验证 Kc=96 的 Natural Earth 和 USGS dense proposal 是否均稳定超过 92% 安全线；
3. 新简化合同下合格 joint checkpoint 的独立 K=4–20 测试，同时验收平均 K=10～14、count MAE/bias 和 knot-match F1；
4. 八方法同内部容量下的 MSE、通过率、节点数和时间比较；
5. 用正式 benchmark JSON 生成 Ours/Kang 等方法的 2×2 对比图，并从同一合格权重生成 Ours 留出真实案例单图和总览；
6. 若 Kc=96 仍受候选上限限制，再附加 Kc=128 容量诊断，不作为默认主模型；
7. 单独改变真实样本比例的域泛化消融，不能和容量变化合并归因。

不同 Kc 不能互相 resume。`--init-checkpoint` 只能迁移形状兼容的 encoder、ParameterHead 和 CandidateKnotHead 张量；新的 adaptive-beta Selector 和 subset decoder 必须重新训练。

## 7. 结论边界

当前可以声明：

- v16 的代码路径、组合学习机制、资格守卫和实验入口通过自动测试；
- 当前代码已接入自适应 beta、概率质量 Top-K、联合参数/节点重定位、粗到细 ranked-prefix 教师和 certified Synthetic 真值监督；本轮实际测试结果应以第 1 节命令的最新输出为准；
- 历史 Kc=64 固定阈值 proposal 在地理来源上达到过约 92% dense feasibility，但它不能证明当前 Selector 的泛化效果；
- Kc=96 是 90% 工程目标的推荐主容量，Kc=64 是轻量消融，Kc=128 仅在实测仍受容量限制时追加。

当前不能声明：

- 原 97% 实验已达到 97%，或 fast90 在尚未完成 joint 审计时已经合格；
- v16 selector 已优于 v15 或 Greedy；
- Kc=96 或 Kc=128 已经改善正式独立测试结果；
- 新模型平均 K 已达到 10～14，或节点预测已贴近真值；这些都要由重新训练后的 count MAE/bias 和 knot-match 指标确认；
- 得到全局最少节点；
- smoke 或诊断图是正式论文结果。
