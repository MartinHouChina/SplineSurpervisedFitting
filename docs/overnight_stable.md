# Overnight Stable：可达删点轨迹与全部外部数据集评测

基于 `1404b15`（“19 13 训练不错”）审查和改进。新入口为 `--stable-selection`，从 `overnight_compact_3090_r1.pt` 的最佳 epoch 7 权重开始新实验，保留 Compact 的模型结构、KeepMask、参数更新、节点重定位和一次性部署。

## 1. 本次结果究竟改善了什么

三轮测试中的 66 个样本标识与内容 SHA256 完全一致，阈值均为归一化欧氏 MSE `5e-5`；可以配对比较。这里是重采样输入点口径，不是原始参考点口径。

| 已保存模型 | 通过曲线 | 平均 MSE | 平均内部 K |
|---|---:|---:|---:|
| Plus，epoch 93 | 51/66，77.27% | 4.430e-5 | 23.85 |
| Reliable，epoch 5 | 60/66，90.91% | 2.385e-5 | 31.33 |
| Compact，epoch 7 | 51/66，77.27% | 3.983e-5 | 27.71 |

Compact 相对 Reliable 少了 3.62 个内部节点，但 10 条由通过变失败、仅 1 条反向改善。因此是精简与误差的权衡改变，不能宣称整体通过率也提高。Reliable 保存时仍有 `2 + 0.2σ` 数量储备，Compact 是 `0 + 0σ`，不能将差异全归因于网络学习。

Compact 的真实来源各仅 8 条：UJI 通过 100%、K=21.62；Natural Earth 通过 50%、K=25.12；USGS 通过 50%、K=34.50。三者原始参考点合计通过 14/24，MSE=5.198e-5。上述均为小样本诊断，原报告的 `DIAGNOSTIC NOT FINAL` 标记保留。

关键训练证据：epoch 7 固定几何 greedy 教师通过 100%，但相同精简 mask 重新经过网络 decoder 后仅通过 38.27%；到 epoch 24 仍为 37.07%。与此同时 teacher-mask F1 约 0.90。主要瓶颈是精简教师几何无法被重新解码稳定复现，而不只是 KeepMask 排序或训练代数不够。epoch 7 到 24，验证平均 K 从 27.71 降至 25.29，最差来源通过率却从 46.88% 降至 31.25%。因此本轮不增加删点压力，也不单纯加长训练。

证据：[Compact 报告](../outputs/comparisons/overnight_compact_3090_r1/report.md)、[训练日志](../outputs/logs/overnight_compact_3090_r1/train_fresh.log)、[完整 history](../outputs/checkpoints/overnight_compact_3090_r1.history.json)。

## 2. 训练改了什么，部署没改什么

旧教师只检查固定几何删点的最终精简节点集。最终集若无法被当前 decoder 拟合，中间已经可达的精简解也没有被利用。

现在保留 greedy 已接受的删除轨迹：

1. 固定种子几何下试删，每次删除仍需真实 B 样条 refit 验收。
2. 每种 seed 最多选择 4 个轨迹检查点，包括近邻、中段和终点，逐个重新 decode/refit。可行性不假设随 K 单调。
3. 只有实际解码达标且更优的节点集才可替换 mask 教师。
4. 每条被搜索曲线最多选 2 个几何蒸馏目标：较精简且可达的过渡解，以及相邻的更难目标；没有可达解时，从较近的删除开始学。目标几何停止梯度，学生解码与实际 refit 保持可微。

参数：`--teacher-greedy-trajectory-checks 4`、`--teacher-geometry-trajectory-targets 2`。关闭两项时保留原 Compact 行为。仍为每 batch 最多 2 条曲线、每条最多两种 seed、每 seed 最多 16 次接受删除；轨迹检查是每 seed 的额外 decode/refit 预算，并非全部最小二乘系统数。所有额外计算仅用于训练。

保留原 24 epochs（4 Proposal + 20 Joint）、batch 32、训练/验证 1500/500、Kc=64、MSE `5e-5`、原学习率与损失权重。新实验不直接复用旧优化器或继续旧 epoch；模型参数结构不变，可完整 warm-start。

部署仍为：有序点 → Proposal → 一次 KeepMask → 一次参数/存活节点更新 → 一次最终标准 B 样条 refit。没有部署时 greedy 搜索、阈值扫描或数值补救。小样本消融发现直接关闭重定位会恶化一些已达标曲线，因此没有用强制 identity 替换现有 decoder。

## 3. 评测与案例图覆盖范围

| 来源 | 性质 | 评测用途 |
|---|---|---|
| Synthetic | 带源节点标签的合成 B 样条 | 六方法四指标 |
| UJI | 实测手写笔迹 | 六方法四指标、Ours 案例、六方法案例 |
| Natural Earth | 地理海岸线数据 | 同上 |
| USGS | 地理等高线数据 | 同上 |
| IndustrialOffsets | 常见工业母轮廓生成的等距线，CAD 半合成，非实测 | 同上 |

工业等距线包含椭圆孔、圆角板、胶囊槽、翼型、径向凸轮、多叶转子、键槽孔。默认本地生成 453 条有效曲线、84 个母轮廓组，train/val/test 为 293/93/67；拒绝了 51 个拓扑改变的构造。同一母轮廓的不同偏置量不跨 split；这些数量对应本次默认配置和种子。没有伪造最简节点标签。

四个外部来源是默认必需评测来源，缺失会报错而不是静默省略。`--prepare-real-data` 在缺少工业数据时本地生成，在缺少实测数据时调用对应准备入口；已有数据复用。可通过重复 `--manifest NAME=PATH` 扩展评测来源，额外测试 manifest 不会进入训练。USGS 的旧 `large_scale_initial` 是同来源快照，不重复计成另一数据集。

旧训练及续跑仍维持原三个真实 validation manifest，工业等距线只加入独立 test；不修改历史 checkpoint 的验证合同。图表和报告分别标记输入点误差、原始参考点误差，保留失败案例和基线适配/补救说明。六方法为 Ours、Park、Liang、Dung、Kang、Luo；后五者是仓库适配，并非作者原代码。

## 4. Linux 运行

先将本次完整源码更新到服务器，不能只替换 Bash。源码增量包为 `outputs/delivery/overnight_stable_linux_update.zip`，基于 `1404b15`，不包含数据、checkpoint 或历史结果。解压前备份服务器未提交修改并确认分支版本；本地修改未自动推送 Git。

Windows PowerShell 上传：

```powershell
scp "E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_stable_linux_update.zip" feng@10.76.0.64:/home/feng/HouCode/
```

Linux 更新源码：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight &&
test -f /home/feng/HouCode/overnight_stable_linux_update.zip &&
unzip /home/feng/HouCode/overnight_stable_linux_update.zip &&
grep -n -- '--stable-selection' scripts/run_v16_1070_overnight_linux.sh
```

### 不重新训练，补齐当前 Compact 的全部数据集测试和图片

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --checkpoint outputs/checkpoints/overnight_compact_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --prepare-real-data \
  --run-name overnight_compact_all_sources_r1 \
  --benchmark-profile full \
  --real-samples-per-dataset 20 \
  --visual-samples-per-dataset 6
```

数据量充足时，这会评测 42 条合成曲线及每个外部来源 20 条，共 122 条 × 六方法；每个外部来源绘制 6 例 Ours 和六方法对比。抽样优先轮流覆盖不同组，组数不足时再取同组其他曲线；同一母轮廓/地理切片的不同偏置或窗口并非完全独立样本，有效独立样本数可能小于图中 n。

### 改进训练 → 六方法测试 → 四指标图 → 全部外部案例图

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --stable-selection \
  --warm-start-checkpoint outputs/checkpoints/overnight_compact_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --prepare-real-data \
  --run-name overnight_stable_3090_r1 \
  --benchmark-profile full \
  --real-samples-per-dataset 20 \
  --visual-samples-per-dataset 6
```

新输出分别位于 `outputs/checkpoints/`、`outputs/logs/<run-name>/`、`outputs/comparisons/<run-name>/`、`outputs/figures/<run-name>/`。保留旧实验，不覆盖同名 run；若需新跑请换 run name。恢复本次训练使用 `--stable-selection --resume-run` 和原 run name/参数，不再传 warm-start。

## 5. 验收重点

先看保存检查点在相同曲线和阈值下的通过率、尾部误差，再比较 K。`compact teacher` 的 decoded_pass 仍特指最精简终点；新增 `reachable teacher` 的通过率统计被检查的中间节点集，两者分母不同，不能用中间检查点通过率冒充终点或学生自由部署通过率。

确认可达教师接受的减点量、几何蒸馏超限量改善后，再判断学生自由部署是否减少 K 且保持通过率。训练教师的搜索结果不等于新模型部署效果，也不是全局最少节点证明。完整 Stable 训练仍需在服务器运行。

`reachable teacher` 中 `checks_total`、`geometry_targets_total` 是整个 epoch 的实际轨迹 decode/refit 次数和辅助目标数量；decoded_pass 按实际检查次数汇总。`target_K`、`target_pass` 按实际辅助目标数统计，包括可达过渡解及尚未可达的较难目标，不能只解读为可达解统计。接受减点量仍按曲线数加权；旧 `compact teacher` 日志保留其历史 batch 样本加权口径。

## 6. 本地验证记录（2026-09-19）

- 最终相关回归共 527 项通过：训练/模型/loss/续跑 330 项，来源覆盖/生成器/Bash/评测/绘图 197 项。旧配置下 loss 与 86 项既有指标数值保持一致；新 history 增加了轨迹诊断字段，不声称 JSON schema 完全相同。
- 使用 Compact 全部 125 个模型张量完成 2-epoch CPU warm-start 小实验（4 条训练、2 条合成验证），Proposal 和 Joint 反向传播、轨迹验收、指标保存均正常。此小实验不是模型提升的证据。
- 现有 Compact 检查点完成 `overnight_all_sources_smoke_20260919_r2`：25 条曲线 × 六方法 = 150 条记录，覆盖 Synthetic 及四个外部来源，输出 11 张 PNG；目视检查五来源四指标图和工业等距线六方法图。CPU、quick 预算、每外部来源仅 1 条，均保留诊断水印，不能替代服务器正式实验。
- 当前 Compact 模型的小样本教师诊断找到中间可达解：UJI `19→11`（MSE=2.506e-5）、`22→17`（4.338e-5）；海岸线 `26→23`（4.253e-5）；USGS `24→14`（2.964e-5）、`13→7`（2.471e-5）。这些目标均通过实际 decode/refit，而各自更精简的终点未达标；原本不达标的另一条海岸线未被标成可行。**这是训练教师搜索的诊断，不是新网络未经训练就能预测这些节点数。**

图像示例：[五来源四指标](../outputs/figures/overnight_all_sources_smoke_20260919_r2/four_metrics/v16_published_methods_input.png)、[工业等距线六方法](../outputs/figures/overnight_all_sources_smoke_20260919_r2/six_method_real_cases/IndustrialOffset__industrial_offset_keyway_bore_v007_o01_m0p0400.png)。诊断图片、数据、权重及测试临时目录均不包含在源码增量包里。
