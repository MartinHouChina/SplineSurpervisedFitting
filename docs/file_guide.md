# 当前工作区文件索引

## 核心代码

| 路径 | 作用 |
|---|---|
| `src/spline_fitting/models/v16_network.py` | v16 编码、参数、候选、Selector、mass-TopK 与 selected-only 重定位 |
| `src/spline_fitting/losses/v16_subset_loss.py` | Proposal/Joint 损失；正式分支为 synthetic ground-truth supervision |
| `src/spline_fitting/data/synthetic.py` | 合成三次 B 样条生成与最简性证书 |
| `src/spline_fitting/data/v16_mixed.py` | 合成训练抽样、真实留出验证加载及来源分组 |
| `src/spline_fitting/checkpointing.py` | objective、结构合同与正式 checkpoint 审计 |
| `src/spline_fitting/evaluation/bspline_inference.py` | 统一标准 B 样条 refit |
| `src/spline_fitting/evaluation/published_knot_baselines.py` | 五个论文方法的 adaptation 与附加控制 |

## 主入口

| 路径 | 作用 |
|---|---|
| `scripts/train_v16.py` | certified Synthetic-only 的 40-epoch Proposal + 64-epoch Joint |
| `scripts/inspect_v16_checkpoint.py` | 检查 supervised v16 结构资格 |
| `scripts/fit_v16_point_cloud.py` | 用户有序点云的一次性部署 |
| `scripts/benchmark_v16_datasets.py` | 六方法×Synthetic/UJI/Natural Earth/USGS 对比 |
| `scripts/plot_v16_method_comparison.py` | 从 `comparison.json` 生成 input/reference 两张 2×2 图 |
| `scripts/visualize_v16_real_deployments.py` | 真实曲线六方法 3×2 案例图 |
| `scripts/visualize_v16_ours_cases.py` | Ours 单独的结构案例图 |
| `scripts/run_v16_mse1e-4_3090.ps1` | Windows/3090 fresh 一条龙入口 |
| `scripts/run_v16_mse1e-4_3090.sh` | Linux 对应入口 |

## 数据

```text
data/raw/                         原始下载
data/processed/                   预处理真实曲线
data/splits/uji_pen_v2.jsonl      UJI 留出清单
.../natural_earth/.../manifest    Natural Earth 清单
.../usgs_contours/.../manifest    USGS 清单
```

正式训练只生成 certified Synthetic；真实 manifest 虽传给训练脚本，但仅用于 validation，随后用于 benchmark 和案例图。不要把真实曲线描述为参与了梯度更新。

## 输出

```text
outputs/
  checkpoints/    .pt、.proposal.pt、.last.pt、.history.json
  logs/<run>/      各阶段日志；PowerShell 入口另含 pipeline_manifest.json
  comparisons/     comparison.json、summary.csv、measurements.csv、report.md
  figures/         两张2×2指标图与真实案例图
  fits/            单点云部署结果
```

主 `.pt` 是最佳成熟 Joint；`.proposal.pt` 只保存候选生成阶段；`.last.pt` 用于完全一致的训练恢复。训练或评测进行中不要移动这些文件。

当前正式命名建议：

```text
candidate_selection_v16_mse1e-4_k56_supervised
```

文件名不能证明模型合格，应执行：

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --mse-tolerance 1e-4
```

## 当前文档

| 文档 | 内容 |
|---|---|
| `training_pipeline.md` | 数据、两阶段训练与一条龙 |
| `v16_counterfactual_subset.md` | 当前 supervised-only v16 算法；文件名仅历史兼容 |
| `math_formulation.md` | MSE、匹配、选择、损失和 checkpoint cost |
| `architecture.md` | 模块与张量流 |
| `deployment_pipeline.md` | 点云部署、六方法表和图 |
| `synthetic_data_minimality_report.md` | source-subset 最简性证书及声明边界 |
| `published_knot_methods_reproduction.md` | 五篇论文 adaptation 与公平协议 |
| `v16_verification.md` | 运行前后验证清单 |
| `presentation_demo.md` | PPT 讲解顺序 |
| `pruning_redesign.md` | v3–v16 演进与当前方案定位 |

`docs/archive/` 只保存历史版本说明，不能作为当前运行协议。旧 Teacher cache、K64/K96 和旧阈值输出若保留，必须明确标为历史消融，不能并入当前 supervised K56 主结果。
