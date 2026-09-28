# 已训练 M16/M32：10%真实测试、六方法统计与案例图

这是独立评估，不启动训练，不改权重、不改原训练输出。

| 模型 | 输入检查点 |
|---|---|
| M16 | `outputs/checkpoints/paper_coupled_clean_3090_r2_m16.pt` |
| M32 | `outputs/checkpoints/paper_coupled_clean_3090_r3_m32.pt` |

## Linux 一条龙

在已激活 PyTorch 的环境中，更新代码后执行：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
bash scripts/run_v16_saved_models_evaluation_linux.sh \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --real-test-fraction 0.1 \
  --selection-seed 20260922 \
  --tag test10pct_20260922
```

默认检查两个检查点及四份已预处理 manifest，再依次评估、汇总绘图、逐案例绘图。
仅跑 M32 加 `--capacities 32`；仅跑 M16 加 `--capacities 16`。
追加 `--dry-run` 只打印指令，不读取数据、不写结果。中断后相同指令追加 `--resume`，
复用兼容的已完成测量；参数/输入/权重指纹不同会拒绝复用，不覆盖旧实验。
本入口要求已有处理后的数据，不自动联网下载，不改变训练/验证划分。

## 采样与公平性

- UJI、NaturalEarth、USGS、IndustrialOffset 各自只取 test split。
- 每来源取 `ceil(N_test * 0.1)` 条曲线，无放回逐样本均匀随机抽样；不是抽取10%的整个数据集、不是取前10%、不是按误差筛样本。
- 例如201条test记录抽21条；两个容量使用同一选择种子及同一manifest，选择相同外部样本。
- 这次改为逐样本随机，不是旧的按group轮转均衡抽样；同一地理/书写组可能贡献多条相关曲线。置信区间建议以group为单位bootstrap。
- 合成数据保持K4..24、每K10条、seed20000。M16的K17..24属于超候选容量压力测试；主表应另列K4..16共同范围，不能将全范围排名解释成纯架构优劣。
- 每个容量下：Ours / Park / Liang / Dung / Kang / Luo，统一该容量上限，当前适配实现、不加额外公共补点修复。尚非经过验证的五种作者原版实现。
- IndustrialOffset是程序生成工业等距线，不是真实测量工业数据。

## 输出

名称分别为 `paper_coupled_clean_3090_r2_m16_test10pct_20260922` 和
`paper_coupled_clean_3090_r3_m32_test10pct_20260922`。

- `outputs/comparisons/<名称>/`：comparison.json/CSV及逐方法、逐曲线几何JSON/NPZ，记录采样索引、manifest/hash、输入点、原始参考点、参数、节点、控制顶点、密集拟合曲线、逐点误差和耗时。
- `outputs/figures/<名称>/metrics/`：六方法按数据来源的统计图（MSE、通过率、平均内部节点数、完整耗时及最大平方误差扩展图）。
- `outputs/figures/<名称>/cases/`：全部已抽样真实/等距线案例及合成案例的六方法曲线图，包含参考曲线、输入点、拟合曲线、控制多边形/顶点、内部节点对应的曲线点。时间/误差直接复用测量结果，不重复拟合。

MSE阈值5e-5；最大误差指标为最大平方欧氏残差，不开方，不是连续Hausdorff误差。
同时保存输入点误差与原始参考点误差。完整方法时间用于跨方法对比，network时间另列。
失败保持记录和分母，不能用虚构几何填图。图不加水印。

绘图数量约为两倍“210 + 各来源抽样数之和”，可能较多；六方法数值求解也可能明显慢于网络。
训练阶段使用过的外部验证集不进入本次抽样；如果这些test结果已参与调参，应称开发测试而非全新盲测。
