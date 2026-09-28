# Ours 部署后的冗余节点检查

无需重新训练、无需更换初始化模型。在原网络部署与现有公共误差修复之后，增加一次实际误差驱动的冗余节点删除检查。保留历史原始结果，写入独立输出目录。

当前测试沿用 `experiment/v16-1070-overnight-3090` 分支的 `outputs/checkpoints/overnight_stable_k32_3090_r1.pt`（选中 epoch 14，内部节点容量32）。这是部署后处理，不是重新训练了更准确的 KeepMask，也不修改其他对照方法。

## 算法

1. 重新计算当前曲线在输入采样点上的 MSE 与最大单点平方误差。
2. 若当前曲线未同时满足 MSE≤5e-5、MaxSE≤5e-4，则保留原状态，不以删节点掩盖失败。
3. 枚举当前每一个内部节点的删除方案；参数 t、剩余节点位置不变，只用 CPU float64 标准最小二乘重新求控制顶点。
4. 仅接受同时满足两项约束的方案，在可行方案中选归一化最坏误差较小者。
5. 重复到没有可行单节点删除、达到最小节点数，或达到删除次数预算。

三次 B 样条默认允许内部节点降至0（仍有4个控制顶点），最多32个内部节点。重复节点逐个出现次数删除。端点约束沿用输入方法设置；不移动节点、不更新参数、不引入任何真实节点标签。全过程保留每次尝试的误差、接受状态、节点位置与重拟合次数。

结果是当前固定几何下的贪心单节点局部精简，不是任意节点位置上的全局最少节点证明。MSE可能在允许范围内增大；最大平方误差约束只针对实际输入采样点，不是连续曲线最大误差或 Hausdorff 保证。

## 六方法重新测量与绘图

```bash
python scripts/benchmark_historical_dual_error.py \
  --selection outputs/diagnostics/historical_confirm_validation_20260922.json \
  --source-benchmark outputs/comparisons/local_models_paper_20260922_m32 \
  --device cuda \
  --post-prune-ours \
  --post-prune-min-knots 0 \
  --post-prune-max-deletions 32 \
  --output-dir outputs/comparisons/historical_best_dual_error_postprune_fresh

python scripts/plot_dual_error_comparison.py \
  --report outputs/comparisons/historical_best_dual_error_postprune_fresh/comparison.json \
  --paper-style \
  --output-dir outputs/figures/historical_best_dual_error_postprune_fresh
```

模型选择和输入样本沿用先前公共验证及61条测试曲线，不重新挑模型或挑好看的测试样本。五个文献对照组仍采用原来的仓库适配实现与公共插入修复；本次新增的删除检查只应用于 Ours。原对照组没有被减弱或修改参数。

`--paper-style` 只简化图片呈现：名称为 Ours/Park/Liang/Dung/Kang/Luo，不增加水印或方法说明脚注；保留曲线、采样点、控制顶点、节点、真实指标与必要图例。完整方法说明仍保留在 Markdown/JSON 中，便于论文实验节交代。

## 结果与计时

- `raw_measurements`：原方法尚未公共误差修复的输出。
- `pre_pruning_measurements`：公共插入修复后、此次删除检查之前的输出。
- `measurements`：最终输出；Ours 包括删除检查，其余方法保持原流程。
- `raw/`、`before_pruning/`、`geometry/`：对应几何与控制顶点。
- `pruning_ms`、`removed_knots_count`、`pruning_diagnostics`：删除耗时、删除个数与完整尝试记录。

重新运行六方法时，Ours 的 `total_ms = raw_ms + repair_ms + pruning_ms`。网络耗时单独保存，不能用 network-only 时间代替最终方法耗时。已达标的曲线不允许因后处理变成失败；原先未达标的曲线仍计入失败分母。图中不重复说明方法细节，不意味着该数值检查是网络一次前向的一部分。

另有 `scripts/postprune_saved_comparison.py` 可只处理此前保存的几何，不重新训练或重新运行所有基线。该快速路径保留历史计时，新增删除计时单独测量，两者之和是分阶段组合时间，不应当宣称为重新测得的完整端到端时间；正式时间比较使用上面的完整重测入口。

只想快速检查已有结果，可以运行：

```bash
python scripts/postprune_saved_comparison.py \
  --source-report outputs/comparisons/historical_best_dual_error_20260922/comparison.json \
  --output-dir outputs/comparisons/ours_postprune_cached_r1 \
  --torch-num-threads 1

python scripts/plot_dual_error_comparison.py \
  --report outputs/comparisons/ours_postprune_cached_r1/comparison.json \
  --paper-style \
  --output-dir outputs/figures/ours_postprune_cached_r1
```

输出目录必须使用新名称，避免覆盖原始实验。完整逐样本记录可用于分析哪些曲线仍被参数偏差或候选位置限制；不能仅凭最终节点数下降，就认定所有曲线都已优于传统方法。
