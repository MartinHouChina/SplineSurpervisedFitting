# V16 短程消融：保留原模型，不从头训练

目标是用现有 Stable K32 权重检验“筛选本身”和“删后几何更新”分别造成多少损失，不推倒重训。所有实验使用独立输出目录，原检查点和原训练入口不变。默认每组训练 8 轮、Proposal 训练 0 轮；实际时间取决于机器和离线教师生成速度，不承诺几分钟或几小时完成。

## 比较什么

| 组别 | 可训练部分 | 推理时的几何 | 用途 |
| --- | --- | --- | --- |
| 原模型基线 | 不更新权重 | 原部署流程 | 记录 warm start 模型的原始表现，并非随机未训练模型 |
| A：`fixed` | 仅筛选器 | 固定 Proposal 给出的参数 t 和候选节点 U | 判断去掉删后几何更新后，筛选能达到什么水平 |
| B：`legacy` | 仅筛选器 | 原解码器参与运行，但解码器权重冻结 | 与 A 使用相同教师标签、相同筛选损失；隔离原几何更新的影响 |
| C：`geometry` | 筛选器与解码器 | 几何可小幅学习 | 在筛选监督之外加入教师 t/U 监督和实际拟合损失，检验删后几何校准 |

A/B/C 从同一份完整检查点分别初始化，不串接前一组的训练结果。A/B 的筛选初始权重、每轮样本顺序、无加权 BCE 标签损失和数量校准损失都相同；B 的“冻结”指参数不更新，并不代表运行时 t/U 不变化。A/B 有意分别执行相同的筛选训练，以保持独立检查点和可审计历史。C 是组合干预，不能把 C 的收益单独归因于某一种损失。这里不重新跑六种传统基线，也不把公共插入修复带来的提升归功于模型。

离线教师由同一份冻结的 Proposal 构造，一份缓存供三组共享。教师删除检查同时使用 MSE ≤ `5e-5` 和最大逐点平方误差 ≤ `5e-4`，不满足双阈值的教师不会被当成可行精简标签。合成训练源复杂度保持 K4..24，模型仍为最多 32 个候选内部节点；源复杂度不等于最少可行节点数。

## Linux 启动

先激活服务器上已有的 PyTorch 环境，在项目根目录运行：

```bash
bash scripts/run_v16_short_ablation_linux.sh \
  --run-name short_ablation_k32_r1 \
  --checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --device cuda
```

默认参数：每组 8 轮，合成 train/val/test 为 256/64/64 条，192 点，batch size 32；筛选器学习率 `3e-5`，解码器学习率 `1e-5`，seed `230923`，CPU 线程 4。三组共用验证集和测试集，但验证与测试之间分离；按验证集各来源宏平均双阈值通过率、最差来源通过率、数值失败数、平均 K、MSE 依次选择权重，再评估测试集，不能用测试成绩挑轮次或调整超参数。第 0 轮也有资格入选，避免短程微调退化后强行替换起点。

外部评估默认包含 UJI、NaturalEarth、USGS 和程序生成的 IndustrialOffset，每个来源的验证/测试各 8 条。IndustrialOffset 是程序生成的工业形状代理，不能称为实测工业数据。外部样本不进入本轮训练；既有预训练检查点是否曾见过相关来源无法由这轮消融撤销。相对于完整论文评测，这是一轮小样本诊断；单种子结果和同一次抽样上的比较不足以宣称统计显著或模型已超越传统方法。

可以先只检查启动参数；不会加载模型、创建输出或训练：

```bash
bash scripts/run_v16_short_ablation_linux.sh \
  --run-name short_ablation_k32_r1 --device cuda --dry-run
```

极小冒烟测试仅用于检查环境和全流程连通性：

```bash
bash scripts/run_v16_short_ablation_linux.sh \
  --run-name short_ablation_k32_smoke \
  --checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt \
  --device cuda --epochs 1 --train-size 4 --val-size 2 --test-size 2 \
  --batch-size 2 --teacher-max-deletions 2 --skip-real-data
```

冒烟配置缩小了数据和教师搜索预算、跳过外部评估，不能用于论文结论，也不能据此判断三组谁更好。`--modes fixed legacy` 可先跑 A/B，但不会得到 C 的结果；新增 C 时使用新的 run-name，保持实验身份清楚。

## 输出与恢复

正常输出位于 `outputs/ablations/short_ablation_k32_r1/`。控制台日志另存于相邻的 `outputs/ablations/short_ablation_k32_r1.run.log`；日志在目录之外，以免启动器提前创建目录影响防覆盖检查。Python 使用无缓冲输出；任何训练错误都会通过日志管道返回非零退出码。

同名输出或日志已存在时默认拒绝覆盖。中断后复用原命令、相同 run-name 和全部原参数，再加 `--resume`；恢复日志会追加，不清空旧日志。协议会核对参数、检查点、数据清单和关键源码指纹。从每组最后完成的 epoch 恢复优化器；已完成的组跳过训练，但重新评估。教师每个 batch 保存 `.partial.pt`，可以从最后完成的 batch 继续；不承诺任意教师删除步骤级别的恢复。不要把改变训练数据规模、阈值、检查点或实验组数当作同一次恢复。要改变配置就用新名字。

主要产物如下：

- `protocol.json`、`runtime.json`、`panels.pt/json`：运行协议、软硬件信息、固定的数据分组与内容指纹。
- `teacher_train.pt`、`teacher_val.pt`、`teacher_replay_before.json`：共享教师缓存和微调前的教师几何复现诊断。
- `baseline_test.json`：原始完整检查点的测试结果；CSV 中 `untrained` 仅指“本轮未微调”。
- 每组目录中的 `best.pt`、`last.pt`、`history.json`、`trainable_parameters.json`：验证最优与最终权重、逐轮记录和实际可训练参数。
- 每组的 `test_best.json`、`test_last.json`、`teacher_replay_validation.json`：逐例测试结果与验证集教师复现。测试 JSON 保存输入点、t、完整 U、控制顶点、稠密曲线、节点在曲线上的位置和残差，可据此复查几何。
- `results.json`、`summary.csv`、`ablation_comparison.png`、`ablation_last_comparison.png`：全局汇总及验证最优/末轮两套对照图；图中显示对应 epoch，不把第0轮当成微调收益。`paired_selector_check` 检查 A/B 末轮筛选权重是否逐位一致。

直接隔离 A/B 的几何更新影响时，优先比较同一末轮的 `test_last.json`。两组可能各自选出不同的验证最优轮次，因此 `test_best.json` 的对照还包含轮次选择的影响，不能完全解释为同一筛选权重下只切换了解码器。

重点对照相同测试样本上的双阈值通过率、保留内部节点数和教师可行解的复现率。只在双方均达标的样本上比较节点数，才能减少“失败但节点少”造成的误导；同时保留全样本通过率，不能只展示交集。精简有效必须同时看达标率，不能只追求少留节点。

## 上传到服务器

交付包路径：`outputs/delivery/v16_short_ablation_linux_update.zip`。包内包含独立运行所需源码、所选 Stable K32 初始化检查点以及本机小样本诊断结果，不包含数据集和各组微调中间权重。可直接使用包内 `outputs/checkpoints/overnight_stable_k32_3090_r1.pt`；也可显式指向服务器已有的同一文件。

本地 PowerShell 示例（替换服务器地址）：

```powershell
scp .\outputs\delivery\v16_short_ablation_linux_update.zip feng@YOUR_SERVER:~/v16_short_ablation_linux_update.zip
```

建议在服务器解压到一个新的独立目录，保留正在使用的项目和权重：

```bash
mkdir /home/feng/HouCode/v16_short_ablation_k32_r1
unzip ~/v16_short_ablation_linux_update.zip -d /home/feng/HouCode/v16_short_ablation_k32_r1
cd /home/feng/HouCode/v16_short_ablation_k32_r1
bash scripts/run_v16_short_ablation_linux.sh \
  --run-name short_ablation_k32_r1 \
  --checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --device cuda
```

若服务器确实没有所选检查点，先另外传输本地 `outputs/checkpoints/overnight_stable_k32_3090_r1.pt`，再将 `--checkpoint` 改为上传后的真实路径。不要误用 `.last.pt` 或 `.proposal.pt` 代替已选中的完整检查点。

实现与测试通过只代表流程可运行；是否改善筛选细致度必须等待这轮消融结果，不预设 A、B 或 C 一定有效。

已完成的本机4轮小样本试验没有显示微调收益，详见 [本机诊断报告](v16_short_ablation_pilot_20260923.md)。时间有限时，不建议立刻扩大为通宵训练；先看 A/B 同末轮和教师复现的差异，再决定下一步。
