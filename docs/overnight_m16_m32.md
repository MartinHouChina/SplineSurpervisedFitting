# M16 / M32：两个通用模型

当前容量实验只训练两个模型：最大16、32个内部候选节点。**不是每个数据集训练一个模型，不增加自动路由，也不先跑16、失败后再跑32。** 最终节点数由KeepMask决定，并非固定16或32；当前档位最少保留4个内部节点。

## 配置与边界

| 设置 | M16 | M32 |
|---|---|---|
| 最大内部候选数 | 16 | 32 |
| 完整三次样条节点向量最大长度 | 24 | 40 |
| 有标签合成源内部K | 4..16 | 4..24 |
| 候选/筛选/调节新增交互层 | 2 / 2 / 2 | 2 / 2 / 2 |
| 每模型默认训练 | 4 Proposal + 20 Joint | 4 Proposal + 20 Joint |
| Joint开始冻结候选几何 | 4轮 | 4轮 |
| MSE通过阈值 | 5e-5 | 5e-5 |

两者从同一个浅层K32 **best .pt** 初始化。M16需显式缩容后重训，不是无损转换；不使用旧来源专用训练的 `.last.pt` 续训。

两者均为混合合成训练：35%低K认证样条、40%完整范围认证样条、25%混合程序形状，默认1500训练样本、500合成验证样本、batch32、每曲线192点。真实数据不加入训练；UJI、Natural Earth、USGS参与验证，工业等距线用于测试。程序形状无伪造真节点标签，使用几何Teacher监督。

默认源K范围不同，因此是**容量匹配训练**，不是唯一变量只有容量的消融。若要纯容量比较，给下方指令追加 `--shared-source-max-knots 16`，两者的有标签合成训练和合成验证都用K4..16。混合形状类别、轮数、损失和架构设置一致。

默认 `--capacity-variant deep_peak` 同时保留增层和峰值损失；可选 `baseline`、`deep`、`peak`，每次容量计划仍恰好两个模型。峰值设置为权重0.05、平方误差目标5e-4、最差5%点；它不是MSE通过阈值，也不构成连续曲线最大误差保证。

## 上传与更新

本机 Windows PowerShell：

```powershell
scp "E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_m16_m32_linux_update.zip" feng@10.76.0.64:/home/feng/HouCode/
```

服务器先检查、保存自己的未提交源码修改，再解压更新：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
git status --short
unzip /home/feng/HouCode/overnight_m16_m32_linux_update.zip
python scripts/run_v16_granularity_experiments.py --help
```

包仅更新源码、测试和文档，不含模型或数据。旧中断实验保留，不删除、不覆盖；使用新的 `run-prefix`。旧 `--route specialists`、`--route all`、`--domains` 已禁用，避免再次启动来源专用模型。

## Linux 一条龙

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
python scripts/run_v16_granularity_experiments.py \
  --route capacity \
  --warm-start-checkpoint outputs/checkpoints/overnight_anchored_k32_p12_j48_3090_r1.pt \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --device cuda \
  --prepare-real-data \
  --run-prefix universal_m16_m32_3090_r1
```

执行前可追加 `--dry-run`：预览应显示 `Planned pipelines: 2`，只包含 `_m16`、`_m32`。不读取权重、不训练、不写输出。实际运行先完成M16训练、对比、绘图，再执行M32；任一步失败会停止，不隐瞒失败。默认24轮是短程比较预算，不保证充分收敛或一晚完成。

## 自动输出

每个模型均测试 Synthetic K4..24、UJI、Natural Earth、USGS、IndustrialOffset。M16的K17..24容量外测试保留并报告，不悄悄缩小测试范围。每个模型单独与 Park、Liang、Dung、Kang、Luo 适配实现进行六方法比较，所有方法在该组统一最大内部节点数；两组不得混为同容量比较。

输出时间、MSE通过率、平均内部节点、MSE、最大单点平方误差 `MaxSqErr`，以及Ours和六方法案例图。每个外部来源默认20条测试、6条案例；合成每K两条。MaxSqErr与MSE均不开方，前者另有均值/P95/最坏值；不把它冒称Hausdorff距离。

```text
outputs/experiment_plans/universal_m16_m32_3090_r1.json
outputs/checkpoints/universal_m16_m32_3090_r1_m16*.pt
outputs/checkpoints/universal_m16_m32_3090_r1_m32*.pt
outputs/logs/universal_m16_m32_3090_r1_m16/
outputs/logs/universal_m16_m32_3090_r1_m32/
outputs/comparisons/universal_m16_m32_3090_r1_m16/
outputs/comparisons/universal_m16_m32_3090_r1_m32/
outputs/figures/universal_m16_m32_3090_r1_m16/
outputs/figures/universal_m16_m32_3090_r1_m32/
```

未达标模型保留诊断标记，不按测试误差逐条挑选M16或M32以提高汇总成绩。新模型尚未完成3090完整实验，没有声称筛选效果已提高。可选四组深度消融仍由独立 `--route depth` 启动；默认容量路线不会追加它们。架构与历史结果见 [Granularity说明](overnight_granularity.md)。
