# 文件索引

## 1. 当前 v16 主线

objective_version：

~~~text
candidate_selection_counterfactual_bspline_v16
~~~

| 文件 | 作用 |
|---|---|
| [README](../README.md) | 当前状态、主命令和目录入口 |
| [文档总索引](README.md) | 当前文档阅读顺序和历史归档入口 |
| [v16 主说明](v16_counterfactual_subset.md) | 完整算法、训练、状态和运行命令 |
| [模型架构](architecture.md) | 模块职责、张量形状和数据流 |
| [训练流程](training_pipeline.md) | 数据混合、proposal/joint 两阶段和恢复规则 |
| [部署流程](deployment_pipeline.md) | 点云部署、资格检查、指标和时间口径 |
| [数学定义](math_formulation.md) | 参数、候选、组合代价与训练目标 |
| [PPT 速查](presentation_demo.md) | 十页展示顺序和禁止误述项 |
| [验证记录](v16_verification.md) | 测试、当前 checkpoint 状态和待完成实验 |
| [Kang/Luo 1e-4 复查](kang_luo_mse1e-4_audit.md) | 统一阈值、修正项、分阶段失效诊断和正式预算 |
| [流程图 SVG](figures/v16_pipeline.svg) / [PNG](figures/v16_pipeline.png) | 同一 v16 流程图的矢量版和 1600×760 位图 |

## 2. v16 代码

### 模型与损失

| 文件 | 作用 |
|---|---|
| src/spline_fitting/models/v16_network.py | GeometryEncoder、初始 ParameterHead、稠密候选、上下文筛选和 selected-only 联合解码 |
| src/spline_fitting/models/candidate_knot_head.py | 锚定 interval query、局部 Gaussian cross-attention 和有序候选 |
| src/spline_fitting/losses/v16_subset_loss.py | 可微标准 refit、Bernoulli 策略梯度、反事实集合及在线蒸馏 |
| src/spline_fitting/losses/deployment_bspline_loss.py | 训练期可微标准 B 样条 refit |
| src/spline_fitting/evaluation/bspline_inference.py | 部署期标准 B 样条控制顶点求解 |
| src/spline_fitting/checkpointing.py | v16 构建及 90% 工程资格审计；MSE 阈值由检查命令显式指定 |

### 数据

| 文件 | 作用 |
|---|---|
| src/spline_fitting/data/v16_mixed.py | 合成/真实混合抽样、来源平衡和分组验证 |
| src/spline_fitting/data/synthetic.py | 合成开放三次 B 样条、干净标签及 source-subset 最简性证书 |
| src/spline_fitting/data/real_world.py | 真实 manifest、点文件读取和参考曲线 |
| src/spline_fitting/data/point_cloud_io.py | 用户 CSV/TXT 点云读取、重采样和归一化 |
| src/spline_fitting/data/uji_pen.py | UJI v2 解析及 writer-disjoint 划分 |

真实数据位于 data/raw、data/processed 和 data/splits；这些目录被 Git 忽略，但正式实验不得删除或原位替换。

### 入口脚本

| 文件 | 作用 |
|---|---|
| scripts/train_v16.py | 两阶段训练、worst-source gate、续训和 checkpoint qualification |
| scripts/run_v16_mse1e-4_3090.ps1 | 当前 Kc=56（完整节点向量 64 项）、source K=4..56（控制顶点 8..60）、`knot_min_span=0.01`、MSE=1e-4 的 3090 串行入口；训练、资格检查、六方法四指标及真实案例图 |
| scripts/run_v16_overnight_12h.ps1 | 历史 Kc=64、MSE=5e-5 无人值守入口；只用于旧消融追溯，不是当前命令 |
| scripts/inspect_v16_checkpoint.py | 训练后统一资格检查；合格返回 0，不合格返回 2 |
| scripts/fit_v16_point_cloud.py | 单条用户点云部署，输出 PNG/JSON |
| scripts/benchmark_v16_datasets.py | v16 合成和三个真实数据集的六方法主表；`--method-set all` 可运行附加控制 |
| scripts/benchmark_v15_datasets.py | benchmark 的共享执行内核；v16 入口依赖它，不能删除 |
| scripts/plot_v16_method_comparison.py | 从正式 v16 benchmark JSON 生成 Ours/Park/Liang/Dung/Kang/Luo 的 2×2 MSE、通过率、节点数和时间图；`--method-set all` 增加 Yeh/贪心 |
| scripts/plot_v15_dataset_benchmark.py | v15/v16 兼容的旧通用指标绘图入口；保留给历史报告 |
| scripts/visualize_v16_ours_cases.py | Ours 留出真实曲线单例结构图和多案例总览；标注采样点、曲线、控制顶点及节点参数条 |
| scripts/visualize_v16_real_deployments.py | Ours/Park/Liang/Dung/Kang/Luo 的真实曲线 3×2 对比图；逐面板展示输入/参考、拟合、控制多边形与顶点、节点、MSE、K 和完整耗时，Ours 另列网络耗时 |
| scripts/smoke_v16_subset_learning.py | 最小组合学习机制检查；输出写入临时目录 |

## 3. 公开方法适配

| 文件 | 方法 |
|---|---|
| src/spline_fitting/evaluation/park_dominant_point.py | Park–Lee dominant-point 自适应细分 |
| src/spline_fitting/evaluation/liang_feature_iki.py | Liang feature-integral + IKI |
| src/spline_fitting/evaluation/dung_direct_knot.py | Dung–Tjahjowidodo direct/simple-knot |
| src/spline_fitting/evaluation/luo_linf_de.py | Luo–Kang–Yang l_inf,1 + differential evolution |
| src/spline_fitting/evaluation/sparse_knot_paper.py | Kang 稀疏优化适配 |
| src/spline_fitting/evaluation/feature_cdf_knot_placement.py | Yeh 特征密度适配 |
| src/spline_fitting/evaluation/gradient_knot_pruning.py | 统一均匀候选贪心删除与位置更新 |
| src/spline_fitting/evaluation/published_baselines.py | 七个基线的统一 refit、MSE、计时和诊断接口 |

复现边界与正式命令见[公开方法复现协议](published_knot_methods_reproduction.md)；旧 Kang 单独实验记录保存在[历史归档](archive/kang_sparse_reproduction.md)。

## 4. 当前 checkpoint

同一实验的四类文件不是重复副本：主 `.pt` 是最佳 Joint 模型；`.proposal.pt` 是最佳候选生成器；`.last.pt` 保存最新优化器和随机状态；`.history.json` 保存逐轮记录。训练进行期间不得移动或删除其中任何一个。

检查命令：

~~~powershell
python scripts/inspect_v16_checkpoint.py --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt --required-pass-rate 0.90 --mse-tolerance 1e-4
~~~

`candidate_selection_v16_mse1e-4_k56.pt` 是本轮待训练的正式目标，不保证当前工作区已经存在或合格；文件名不能代替上述资格审计。旧 K64/K96 仅保留为历史容量或阈值消融。合成最简数据合同和旧 proposal 迁移规则见[训练流程](training_pipeline.md)及[合成曲线最简性报告](synthetic_data_minimality_report.md)。旧 K64 proposal 只可用新 output 配合 `--init-checkpoint` warm start：65 个 interval query 沿参数域插值为 57 个，其他形状兼容的 proposal 张量迁移，Selector、联合解码器和优化器新训；不能 `--resume` 成当前 K56 合同。未合格权重的图只允许通过 `--allow-unqualified-diagnostic` 生成，并必须保留水印。

## 5. 输出目录

~~~text
outputs/
  checkpoints/          正式和可恢复 checkpoint
    archive/            历史权重，不自动删除
    stages/             v12–v15 阶段权重
  comparisons/          定量比较及逐样本记录
  figures/              论文/PPT 图
  fits/v16/             用户点云部署
  logs/                 训练与无人值守流水线日志、状态 manifest
  paper/                论文案例
  archive/              可恢复历史产物
  tmp/                  临时 smoke；验证后可删除
~~~

文件命名：

- checkpoint：method_version_role.pt；
- 对比：version_methods_Kmin-Kmax_nN_seedS；
- 临时产物：outputs/tmp/_smoke_name。

正式结果目录不要使用 smoke、test 或 latest。

当前 1e-4 串行入口按 checkpoint SHA-256 自动区分 `formal_<hash>` 与
`diagnostic_<hash>`。运行总状态和每阶段命令、返回码、耗时及最终目录记录在
`outputs/logs/candidate_selection_v16_mse1e-4_k56/pipeline_manifest.json`。旧
`mse5e-5_k64` overnight manifest 只属于历史消融。

## 6. 测试

v16 主测试：

| 文件 | 覆盖 |
|---|---|
| tests/test_v16_network.py | mask、selected-only KV、参数和节点有序性、梯度 |
| tests/test_v16_subset_loss.py | 集合代价、策略梯度、在线目标和可微 refit |
| tests/test_train_v16.py | 两阶段 gate、保存、恢复、资格和配置校验 |
| tests/test_v16_checkpoint_integration.py | checkpoint 构建、未达标拒绝及诊断放行 |
| tests/test_fit_v16_point_cloud.py | 用户点云部署、JSON/PNG 和诊断水印 |
| tests/test_visualize_v16_real_deployments.py | 真实六方法 3×2 图与资格守卫 |
| tests/test_benchmark_v15_datasets.py | v15/v16 共享 benchmark、六方法主集合、附加控制和报告 |
| tests/test_plot_v15_dataset_benchmark.py | 指标绘图及诊断水印 |
| tests/test_plot_v16_method_comparison.py | v16 正式方法图、方法集合、输入审计及诊断水印 |

运行：

~~~powershell
$env:PYTHONPATH = 'src'
python -B -m pytest tests -q
~~~

使用 -B 防止重新生成 pyc；pytest 和 Ruff 缓存已被 .gitignore 排除。

## 7. v8–v15 历史

历史训练入口 `scripts/train_candidate_pruning.py` 仍按旧 objective 工作，不会自动升级为 v16。v7–v15 的旧框架、命令、时间口径和实测报告已集中到[历史归档](archive/README.md)。

历史 checkpoint、教师缓存和正式图保留在 outputs/archive、outputs/checkpoints/archive 及已登记的 current 目录，不因 v16 清理而删除。

## 8. 发布前检查

1. 运行统一 v16 qualification，拒绝 proposal 或 target_not_met 权重；
2. 保存 checkpoint、manifest、测试配置和 SHA-256；
3. 逐来源报告 MSE、通过率、节点数和失败样本；
4. 区分 network time 与完整方法时间；
5. 用逐样本 JSON 重绘 PNG，不手动修改数值；
6. 确认所有文档路径存在；
7. 清理只针对可再生成缓存和 smoke，不删除 .last/.proposal/history 或真实数据。
