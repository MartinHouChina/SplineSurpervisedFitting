# Overnight Granularity：更细的节点筛选、分来源容量与峰值误差

本轮基于 `e9a47c8`，不覆盖已有模型、不回滚分支。新开关默认关闭，历史 checkpoint 保持原行为。**新增路线尚未完成 3090 泛化实验，不能称为已改善结果。**

## 1. 本次结果说明了什么

详见 [2026-09-21 配对审计](overnight_results_audit_20260921.md)。三次比较的 122 条输入 SHA256 完全相同：

| 已训练 best 模型 | best epoch | 输入通过率 | 平均内部 K | 平均 MSE |
|---|---:|---:|---:|---:|
| Stable K32 | 14/32 | 60.66% | 24.04 | 8.591e-5 |
| Anchored 4+12 | 5/16 | 69.67% | 26.20 | 7.170e-5 |
| Anchored 12+48 | 13/60 | 69.67% | 26.19 | 7.017e-5 |

两次 Anchored 的 best 都是 Joint 第一轮。长训练末尾验证通过率 59.73%、平均 K23.70，Natural Earth 最差来源仅 18.75%；不能把测试 best13 解释为完成48轮 Joint后的网络。

Teacher 紧凑终点重新解码的通过率已从旧版本末尾58.53%升至长训练末尾92.53%，但后者是**被搜索且存在固定几何可行终点的子集**，不是全验证集。其几何一致性显著改善，当前应重点检验筛选/数量学习的泛化，而不是仅增加epoch。深贪心仅覆盖约12.5%的训练曲线。

## 2. 路线 A：三个位置可分别加深

数据流仍为：有序点 → Encoder/ParameterHead → 候选 → 动态阈值与概率质量 Top-K → 存活节点和参数联合更新 → 一次标准 B 样条 refit。

当前档位仍最少保留4个内部节点，并在数量足够时优先覆盖4个参数区间的锚点；“零安全储备”不等于取消覆盖约束或允许零节点。保留数来自概率质量，而不是每个概率单独与0.5比较。

| 开关 | 插入位置及作用 |
|---|---|
| `--candidate-refinement-layers 2` | 候选之间交互 + 围绕当前位置的局部 Gaussian cross-attention；逐层有界更新候选位置 |
| `--selection-refinement-layers 2` | KeepMask之前的候选交互与局部证据融合；保留动态β/概率质量数量预测 |
| `--decoder-refinement-layers 2` | 存活节点交互，并向点参数特征反馈；被删除候选不充当节点KV |

三个开关默认0。新增残差门初始为0：在相同容量、相同旧配置下，扩层的初始输出与旧权重一致；门学习后才逐步引入新计算。沿用 anchored 的顺序、间隔和参数域耦合约束。不是多次试删，也不增加部署refit次数；注意力层数增加仍会增加网络时间。

严格 full warm-start 只允许显式增加这些层，不允许静默丢弃已经训练过的层。Proposal冻结包含新增候选层，学习率分组与保存/重载同步支持。增层并不保证能解决教师覆盖、分布偏移和参数遗忘，须与同预算浅层控制组比较。

## 3. 路线 B：形态专用模型 + 候选容量消融

默认继续遵循**合成训练、真实验证、真实test独立评测**，不自动把真实点或测试标签加进训练。

| 专用路线 | 合成程序形状偏置 | 两个比较容量 | 两个模型共同训练源K范围 |
|---|---|---|---|
| UJI | handwriting | 16 / 32 | 4..16 |
| IndustrialOffset | industrial | 24 / 32 | 4..24 |
| NaturalEarth | terrain | 24 / 32 | 4..24 |
| USGS | terrain | 24 / 32 | 4..24 |

每个专用模型的合成混合比例为25%低K认证样条、25%完整范围认证样条、50%相应程序形状；程序形状没有伪造真节点标签，仍由拟合Teacher提供几何监督。Synthetic验证范围同该模型的训练源范围，外部验证仅该路线来源。形状偏置只是对领域的一种近似，不等同真实数据适配已经成功。

同路线的两种容量使用相同数据分布、轮数和浅层网络，均开启同样的峰值损失。由同一个浅层 K32 best 权重初始化，小容量通过显式 interval-query 重采样重新训练，**不是无损缩容**。容量只能用验证集选择；不得在test逐条选择误差最小的模型或读取真K路由。

所有模型仍评测全部五个来源，包括固定的 Synthetic source K4..24。新增 `--synthetic-source-max-knots 24` 将评测生成范围与专用模型的训练范围分离，避免K16模型找不到K17..24测试例。超容量测试不会删掉或伪称域内测试。每个六方法实验内部的所有方法统一同一候选上限；不同上限实验分表，不能将K16基线与K32模型混成公平比较。

高级可选 `--training-source UJI --training-real-fraction 0.5` 明确开启该来源 **train split** 几何Teacher训练；其他可用名称为 NaturalEarth、USGS、IndustrialOffset。这不是默认方案，也不在下面矩阵中；它改变数据协议，须单独命名、披露真实训练暴露，验证/测试点仍不得进入训练。

## 4. 最大误差与训练

对归一化的有序对应点，记 `r_i = ||C(t_i)-Q_i||²`：

- `MSE = mean_i r_i`，不除以维数，不开方；通过率仍按 MSE≤`5e-5`。
- `MaxSqErr = max_i r_i`，不等于最大欧氏距离、Hausdorff或连续曲线误差保证。
- 对比表已记录每条曲线MaxSqErr，以及数据集均值/P95/最坏值；五指标图第五项为整个数据集最坏MaxSqErr，输入/原始参考两种口径分开。
- 本轮额外让训练验证history、各来源统计和终端报告精确MaxSqErr；并新增可关闭的峰值损失。

示例参数：`--max-point-error-weight 0.05 --max-point-error-tolerance 5e-4 --max-point-error-tail-fraction 0.05`。

损失把精确最大值与最差5%点的平均平方误差，各经现有稳定log/softplus拟合惩罚后等权融合。Proposal阶段用dense实际refit；Joint阶段用dense和实际hard-mask部署refit的平均损失，不新增求解。尾部点提供比仅一个最坏点更广的梯度。

`5e-4`是**独立的平方距离训练目标**，是待验证的初始设置，并非统计确定的最优值，也不保证所有点达标。不把它悄悄设成MSE阈值。Teacher筛选及checkpoint通过资格继续使用原MSE口径；若最大误差下降而K上升，应报告为折中，不能只选有利指标。

## 5. Linux 顺序执行实验

先同步本次所有Python/Bash文件。增量包为 `outputs/delivery/overnight_granularity_linux_update.zip`，基于 `e9a47c8`，不含权重、数据或实验输出，未自动提交/推送Git。先保存服务器本地改动，再从Windows上传并在目标代码库解压（若提示覆盖，仅核对本次源码文件，不覆盖实验结果）：

```powershell
scp "E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_granularity_linux_update.zip" feng@10.76.0.64:/home/feng/HouCode/
```

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
unzip /home/feng/HouCode/overnight_granularity_linux_update.zip
```

初始化使用这次长训练保存的 **best .pt**，不是 `.last.pt`。默认下面每个模型24轮=4 Proposal+20 Joint，前4轮Joint冻结候选几何，是控制变量的试验预算，不是“保证够训练”的正式推荐。

路线A四个消融：浅层、增层、仅峰值、增层+峰值。

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
python scripts/run_v16_granularity_experiments.py \
  --route depth \
  --warm-start-checkpoint outputs/checkpoints/overnight_anchored_k32_p12_j48_3090_r1.pt \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --device cuda --prepare-real-data \
  --run-prefix granularity_depth_3090_r1
```

路线B，先对简单来源比较小容量与K32：

```bash
python scripts/run_v16_granularity_experiments.py \
  --route specialists --domains UJI IndustrialOffset \
  --warm-start-checkpoint outputs/checkpoints/overnight_anchored_k32_p12_j48_3090_r1.pt \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --device cuda --prepare-real-data \
  --run-prefix granularity_specialists_3090_r1
```

省略 `--domains` 会执行四来源、八模型；`--route all` 是A+B共十二模型，**不承诺一晚完成**。可用 `--depth-variants deep_peak` 只跑一个版本，但它不能替代完整消融。各模型依次训练→检查→五来源六方法benchmark→四/五指标图→Ours/六方法案例图；20条/外部来源、合成每K两条，案例每来源6条。训练时间不包含对照算法和绘图时间。

执行前加 `--dry-run` 检查计划；不读取权重、不训练、不写输出。实际运行会记录计划与完整命令，拒绝覆盖已有实验；失败时停止。重跑请改run-prefix，或按日志中的单个Bash命令显式续跑该模型，不能换配置后resume。

每个计划成员的文件仍位于：

```text
outputs/checkpoints/<run_name>*.pt
outputs/checkpoints/<run_name>.history.json
outputs/logs/<run_name>/
outputs/comparisons/<run_name>/comparison.json, measurements.csv, report.md
outputs/figures/<run_name>/four_metrics/, ours_cases/, six_method_real_cases/
```

保留诊断/未达标标记，不修改旧数据、误差曲线或失败例。新实验的“正式论文结果”仍需多随机种子、更多独立组以及严格来源隔离验证。

## 6. 论文入口与检查边界

新的论文基础架构在 [paper/overnight](../paper/overnight/README.md)，与旧CountHead/BIC论文草稿分开。介绍、方法、数据、指标、公平对照、消融和失败案例已有结构；图暂不插入，新增实验结果留待真实训练填写。

局部CPU验收覆盖空/全/部分KeepMask、有限反向梯度、旧模型逐张量兼容、扩层warm-start、Proposal冻结/解冻、保存续训、峰值聚合和域专用数据边界。它们验证代码通路，不证明节点数或泛化效果改善。没有在本机替代3090完成十二组训练。

本轮完整 `tests/` 回归：**1253 passed**（207.55秒）。实际CPU小训练完成旧模型2轮 → 扩层3轮 → 同配置续训第4轮；这只是保存/加载/反向传播验收，不是泛化或简化实验。Bash语法、实验计划dry-run、K16/K32下游命令、最新best初始化均已检查。
