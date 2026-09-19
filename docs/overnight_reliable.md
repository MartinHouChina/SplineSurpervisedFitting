# Overnight Reliable：约束参数偏移，改进可行子集教师

这是 `overnight_plus_3090_r1` 的短程改进实验，保留 **Proposal → KeepMask → 存活节点/参数联合更新 → 标准 B 样条 refit**。新增内容由 `--reliable-selection` 显式启用，与 `--enhanced-selection` 互斥；旧 checkpoint 不会被自动改写成新网络。

本版本尚未完成 3090 正式训练。下面的 32 epochs 是先验证机制的预算，不是精度、通过率或耗时保证。

代码以服务器结果分支的 `d473159` 为基点，在本地独立分支 `experiment/overnight-reliable` 修改；没有覆盖原主线、旧一夜工作区或旧实验结果。更新包是针对该 overnight 版本的源码增量，不含数据、权重或训练结果，也没有自动推送到服务器。

## 1. 为什么修改，以及修改什么

上一轮高 `teacher_keep_f1` 表示学生与在线教师的候选槽位一致，并不等于真值节点位置准确。教师主要沿学生排序搜索，可能一起停在冗余较多的解；参数预测又可能让稠密拟合容易、少节点拟合困难。上一轮没有传入真实数据 manifest，因而“验证达标”仅指合成验证。

| 修改 | 训练时作用 | 部署时作用 |
|---|---|---|
| 两级参数 trust gate | 学习每条曲线需要多少参数修正；与弦长反事实拟合比较 | Proposal 参数与严格弦长参数有界融合；Subset 再有界更新参数 |
| 几何引导的教师候选 | 在学生排序之外检查有限的删除/交换候选，以实际 refit 判断可行性 | 无额外搜索 |
| Proposal 有序匹配、局部拟合损失 | 避免仅靠最近邻覆盖；关注局部高残差区段 | 无额外求解 |
| 独立真实验证 | 三个真实来源各取 validation split，参与选模 | 不使用真实节点标签，不读取 test split 选模 |

两个 trust gate 都是每条曲线一个 `0..1` 标量，初值 `0.25`。严格递增的参数向量做凸组合仍保持递增与 `[0,1]` 端点；节点随实际参数同步映射后，继续由原 survivor 交互模块重定位。不是“关闭节点微调”，也不是强制所有曲线只用弦长。

训练中的弦长反事实保持节点个数和 KeepMask 不变，并把节点坐标同步映射到参考参数域再 refit；它是训练信号，不是推理时挑选最好结果。最终部署仍为一次网络前向、一次离散选集、一次标准 refit，不增加逐节点试删。

Teacher 仍只是在当前候选与搜索预算内找更简约的可行子集，不能证明全局最少节点。合成 source K 也不是不同参数化、不同候选集合下唯一正确的最终 K；不能为追求真值数量而强行删除到不达标。

## 2. 推荐配置

| 项目 | `--reliable-selection` 默认值 |
|---|---:|
| 总 epochs / Proposal / Joint | 32 / 4 / 28 |
| 合成训练 / 合成验证 | 1500 / 500 |
| 真实训练比例 / 各来源真实验证 | 0 / 32 |
| batch / 每曲线采样点 | 32 / 192 |
| source 内部节点 / 候选内部节点 | 4..24 / 64 |
| 归一化平方欧氏 MSE 阈值 | `5e-5` |
| Proposal / Joint 基础学习率 | `5e-5` / `5e-5` |
| Joint Proposal / Selector / Decoder 比例 | 0.25 / 1 / 0.5 |
| Joint 余弦学习率终值比例 | 0.25 |
| policy samples / counterfactual edits / prefix steps | 2 / 4 / 6 |
| 原教师精炼 / 边界排序权重 | 1 轮 × 3 候选 / 0.5 |
| 新几何教师候选数 | 4 |
| 参数反事实 / 局部拟合 / Proposal 有序匹配权重 | 0.25 / 0.1 / 0.5 |

此档传入 `--allow-infeasible-proposals`，阶段按计划进入 Joint；这只是避免 Proposal 门槛阻断机制验证，**不是声明候选或部署已经达标**。达标率、P95 MSE、节点数及正式汇报资格仍独立记录。

三次样条的完整节点向量长为内部节点数 `+8`，控制顶点数为内部节点数 `+4`。所以 `Kc=64` 全保留时是 72 项完整节点向量；source `K=24` 时是 32 项，两者不要混淆。

## 3. Linux 一条龙

先将本次代码更新同步到服务器的 overnight 工作区。更新包路径为 `outputs/delivery/overnight_reliable_linux_update.zip`；它不包含模型或数据。不要在未备份本地修改时覆盖代码，不要把服务器上不存在的 zip 路径当成已经上传。

推荐从已完成的 Plus 权重启动，**使用新的 run name**：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
test -f outputs/checkpoints/overnight_plus_3090_r1.pt

bash scripts/run_v16_1070_overnight_linux.sh \
  --reliable-selection \
  --warm-start-checkpoint outputs/checkpoints/overnight_plus_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --real-val-size 32 \
  --run-name overnight_reliable_3090_r1 \
  --benchmark-profile full
```

缺少真实数据时可显式加 `--prepare-real-data`。先检查命令可追加 `--dry-run`；它不训练，不代替数据和模型的实际验证。解释器不在当前环境时用 `--python /path/to/python`。

Warm-start 复制原 Encoder、ParameterHead、Proposal、Selector 和 Decoder 的兼容权重，仅初始化新加的两套 trust head（12 个 state 字段）；优化器、训练日程与历史重新开始。输入权重不会被覆盖，这不是恢复旧 epoch。

### 先跑两代链路检查

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --reliable-selection \
  --warm-start-checkpoint outputs/checkpoints/overnight_plus_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --epochs 2 --proposal-epochs 1 \
  --train-size 8 --val-size 8 --real-val-size 2 --batch-size 4 \
  --run-name overnight_reliable_smoke_r1 \
  --benchmark-profile quick
```

这只检查训练、评测和图片是否贯通。`quick` 降低测试样本和数值基线预算；不能用于比较论文性能，也不能证明新增机制有效。

### 中断后续跑

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --reliable-selection --resume-run \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --real-val-size 32 \
  --run-name overnight_reliable_3090_r1 \
  --benchmark-profile full
```

续跑不要再传 warm-start。若此前使用自定义 epochs、数据大小等参数，恢复时需保留相同配置。不要启动两个同名 run；不要删除保护输出的检查来规避覆盖错误。

## 4. 数据、对照组与输出

训练仅用合成曲线。UJI Pen、Natural Earth 海岸线和 USGS 等高线分别使用独立 validation split 参与模型选择，再以 test split 做最终评测；同一 writer/tile 不跨集合。不要为提高报告值把 test 数据加入选模。源权重可能带有此前训练经验，`real_fraction=0` 不等于整个模型从未见过真实训练数据；以 provenance 为准。

一条龙顺序：预检 → 训练 → checkpoint 检查 → 合成＋三类真实数据的六方法评测 → 四指标图 → Ours 拟合案例 → 真实数据六方法案例。

六方法为 Ours、Park、Liang、Dung、Kang、Luo；后五项均为本仓库的论文适配，不冒充作者原版。此次修正 Kang 长活跃簇压缩问题，并为 Dung/Kang/Luo 增加明确标注的 `threshold-safe adaptation` 对比模式；详见[复现与公平协议](published_knot_methods_reproduction.md#21-overnight-reliable-的对照组修正)。

四指标为 **平均最终 MSE、阈值通过率、平均最终内部节点数、完整算法耗时**。Ours 网络前向时间另记，不能拿它与传统方法完整时间当作同口径加速比。基线的额外可行性修复全部计入完整耗时，保留修复前后结果；不能隐去修复代价或只报少节点失败解。

```text
outputs/checkpoints/overnight_reliable_3090_r1.{pt,last.pt,proposal.pt,history.json}
outputs/logs/overnight_reliable_3090_r1/
outputs/comparisons/overnight_reliable_3090_r1/
  comparison.json / measurements.csv / summary.csv / native_baseline_summary.csv / report.md
outputs/figures/overnight_reliable_3090_r1/
  four_metrics/ / ours_cases/ / six_method_real_cases/
```

`full` 默认仍为有限预算：合成每个 source K 两条，三个真实来源各八条，共66条配对曲线；不是充分的论文统计实验。最终资格不满足时，应保留 `DIAGNOSTIC NOT FINAL` 标记，不能通过更换标题声称正式达标。

若要查看不加补点保护的论文适配结果，用独立评测 run，追加 `--native-baselines`；它关闭比较包装层，不会重新启用 Kang 已修复的错误聚类代码：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --checkpoint outputs/checkpoints/overnight_reliable_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --native-baselines \
  --run-name overnight_reliable_native_eval_r1 \
  --benchmark-profile full
```

## 5. 如何判断改进是否有效

先比较相同测试曲线的通过率与 P95 MSE，再在同等可行性下看节点数量。额外检查参数反事实差距、教师自身 K、教师精炼收益，以及合成真值节点 F1。`teacher_keep_f1` 不能替代真值节点 F1；真实曲线没有真实节点标签。训练历史同时记录两级参数 gate 的均值及极值，验证按数据来源汇总；gate 饱和需要结合真实拟合误差判断，不能仅凭反事实损失下降宣布参数化最优。

当前完成的是实现与链路验证，不是新模型已达成更高通过率：模型、损失、训练器、Linux 编排、对照算法和绘图的联合回归共 **442项通过**，另有新门控迁移与续训专项检查；关闭 gate 时已对照历史 `d473159` 确认既有输出逐位一致。

本机已用真实旧 Plus 权重跑通 CPU 两代微型流程：1代 Proposal＋1代 Joint，训练仅2条合成曲线，验证为2条合成＋三个真实来源各1条；随后完成24条测试曲线×6方法＝144条记录，生成9张 PNG。真实数据来自既有 split，未加入训练；所有失败样本保留。该运行名为 `overnight_reliable_pipeline_smoke_20260918_r1`，采用 quick 数值预算且保留 `DIAGNOSTIC NOT FINAL` 标记。它仅验证可执行性，不能用于性能结论或估计3090完整耗时；正式效果须由新训练与独立评测确认。
