# V16 论文消融实验

本文档对应 `scripts/run_v16_paper_ablation.py`。实验固定同一个 K32 检查点和同一批 61 条曲线，只改变一个输入特征或推理模块；每个变体仅执行一次标准三次 B 样条重拟合，不使用残差插入和验证式后剪枝。

因此，这是一组**共享权重的推理消融**：它用于判断当前模型依赖哪些信息，不等价于为每个变体重新训练一个最优模型。

## 正式 Linux/RTX 3090 指令

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight

bash scripts/run_v16_paper_ablation_linux.sh \
  --run-name v16_paper_ablation_k32_3090_r1 \
  --checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt \
  --source-report outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json \
  --device cuda \
  --network-repeats 100 \
  --torch-threads 4 \
  --mse-tolerance 5e-5 \
  --max-squared-error-tolerance 5e-4 \
  --paper-figure-dir paper/overnight/figures
```

`--network-repeats 100` 只重复网络前向以稳定计时，不会把同一曲线重复计入误差统计。正式实验不要传 `--max-cases`。

## Windows 本机诊断

```powershell
python scripts/run_v16_paper_ablation.py `
  --checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt `
  --source-report outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json `
  --device cpu `
  --network-repeats 3 `
  --torch-threads 4 `
  --output-dir outputs/ablations/v16_paper_ablation_k32_cpu_r1 `
  --paper-figure-dir paper/overnight/figures
```

快速检查脚本完整性时可以追加 `--max-cases 2`，但这种输出不能用于论文统计。

## 消融组

特征消融：

- `Coordinates only`：保留坐标，置零一阶和二阶差分；
- `Coordinates + first derivative`：保留坐标和一阶差分，置零二阶差分；
- `Coordinates + first + second derivatives`：完整输入。

模块消融：

- `Complete network`：完整原始网络；
- `Fixed 0.5 KeepMask`：以 `p >= 0.5` 替换 probability-mass Top-K；
- `No survivor-conditioned decoder`：筛选后不执行存活节点条件化更新；
- `No candidate selection`：保留全部 32 个候选；
- `Chord parameters after selection`：筛选后把学习参数替换为弦长参数，并在同一参数变换下移动节点。

## 输出文件

输出目录 `outputs/ablations/<run-name>/` 包含：

- `protocol.json`：检查点、输入报告、代码哈希、阈值、硬件和样本清单；
- `measurements.jsonl`：每条曲线、每个消融臂的节点数、MSE、MaxSE、通过状态和时间；
- `summary.json`、`summary.csv`：按总体和数据源汇总的均值与 95% bootstrap 区间；
- `results.md`、`ablation_table.tex`：可直接检查或引用的表格；
- `feature_ablation.png/.pdf`、`module_ablation.png/.pdf`：论文消融图。

指定 `--paper-figure-dir paper/overnight/figures` 后，还会同步生成：

- `paper/overnight/figures/ablation_features.png/.pdf`；
- `paper/overnight/figures/ablation_modules.png/.pdf`。

## 三阶段框架图

```bash
python scripts/plot_overnight_framework.py \
  --checkpoints outputs/checkpoints/overnight_stable_k32_3090_r1.pt \
  --output-dir paper/overnight/figures

python scripts/plot_three_stage_coupling.py \
  --output paper/overnight/figures/three_stage_coupling
```

第一张图给出完整网络及训练/部署边界；第二张图专门给出三个阶段之间的状态接口 $S_1$、$S_2$ 和 $S_3$。

## 解释规则

1. 双阈值通过要求同时满足 `MSE <= 5e-5` 和 `MaxSE <= 5e-4`，两者均为平方欧氏误差，不开方。
2. 失败样本保留在通过率分母中，不能静默丢弃。
3. `No candidate selection` 若提高通过率但使用全部 32 个节点，表示用复杂度换取可行性，不能解释为筛选模块无用。
4. 固定 0.5 阈值在当前 raw-network 样本上略优于自适应筛选，必须如实报告为 KeepMask 校准问题。
5. 本机 CPU 时间只用于诊断；论文时间列应采用同一台 RTX 3090 的正式重跑结果。
