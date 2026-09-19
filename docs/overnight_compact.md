# Overnight Compact：在可行性约束下学习更少节点

更新：本档服务器实验已随 `1404b15`（“19 13 训练不错”）返回。下文验证记录是开发时的历史状态；正式返回结果、改善边界与后续可达轨迹教师见 [Overnight Stable](overnight_stable.md)。评测入口现已扩展至全部四个外部来源，包含明确标为半合成的工业等距线。

这是从 Reliable 已选最佳权重继续改进的独立实验，由 `--compact-selection` 显式启用，与 `--enhanced-selection`、`--reliable-selection` 互斥。保留 Reliable 的参数 trust gate、Proposal → KeepMask → 存活节点/参数联合更新 → 标准 B 样条 refit，以及六方法评测和案例图流程；不加新开关时，原有档位不变。

本档尚未完成正式训练。24 epochs 是实验预算，不保证一夜完成、通过率提高或节点数下降；Teacher 找到的子集也不是全局最少节点证明。**部署仍只有一次网络前向、一次离散选集和一次标准 refit，不增加逐节点试删、阈值扫描或数值修复搜索。**

## 1. 四项改进及其边界

| 改进 | 训练中的作用 | 必须区分的边界 |
|---|---|---|
| 按曲线可行性控制简化 | 已达到误差要求的训练曲线获得节点复杂度压力；进入安全误差区间后减弱继续压低 MSE 的梯度 | 不再让一个较差验证来源关闭所有训练曲线的简化信号，但选模与正式资格仍检查 worst-source 通过率 |
| 有预算的 greedy Teacher 与几何蒸馏 | 在固定几何下试删，产生更紧凑的候选教师，再检查经过当前 decoder 的实际拟合，并提供几何蒸馏信号 | 固定参数/节点几何下可行，不等于同一 KeepMask 经当前 decoder 后仍可行；二者分开检查和记录 |
| 数量监督与部署储备对齐 | 对齐 `mass_topk` 的概率质量、方差储备和常数储备，避免学到教师 K 后部署再额外加储备 | 本档初始及最终 `sigma=0`、`safety_knots=0`；仍保留最小节点数和覆盖约束，不承诺输出恰等于教师 K |
| 更广的合成训练混合 | 35% 低 K 简单曲线、25% 程序形状曲线、40% 既有 certified source，每个 epoch 重采样训练集 | 程序形状没有真实节点或最小 K 证书，只接受拟合与 Teacher 监督，不把生成参数或零占位值当作最少 K 标签 |

简化控制器为 `per_curve`：复杂度 ramp 按训练日程推进，不因验证通过率暂时下降而停住；本档明确把 `complexity_max_scale` 限为 `1.0`，不会继续升到旧档的 `4.0`。复杂度项按当前每条曲线的可行性门控，不给当前未通过安全阈值的曲线施加该项简化压力；这不是全局 MSE 保证，也不保证一次梯度更新后或未见曲线仍然达标。`feasible_fit_margin=0.8`、`feasible_fit_weight=0.02` 用于安全区间内的弱拟合压力，不是把部署阈值改成原来的 80%。

Greedy Teacher 默认每个 batch 最多处理 2 条曲线。每条选中曲线最多从两种 seed 开始：当前 Teacher mask 与当前部署 mask；若两者相同，只搜索一种。**每种 seed 最多接受 16 次删除**，不是每条曲线总共 16 步，更不是只做 16 次拟合。每轮会扫描当前所有可删位置，还要验证候选、经过 decoder 验收及执行蒸馏 refit，因此实际最小二乘 refit 数可远多于 16，并单独记入 metrics。它是训练预算内的局部搜索，不是每条曲线的穷举最优。固定几何的搜索结果与 decoded 子集验收分别处理，geometry distillation 权重为 `0.2`。不能因为固定几何 Teacher 达标，就声称学生的一次性部署已经达标；也不能用高 `teacher_keep_f1` 代替最终 MSE 或真实节点位置准确率。

35% 简单分量来自 `K=4..8` 的 certified source；25% 程序形状为平滑的工业轮廓、地形和手写类合成曲线；40% 保留原 `K=4..24` certified source。比例是抽样概率，不要求每个 epoch 的样本数精确等于该比例。低 K 与既有 source 保留其有效证书标签，程序形状没有这类标签；即使有 source 证书，也不证明不同参数化下的全局最少节点数。

训练数据仍完全为合成曲线。UJI、Natural Earth、USGS 各自独立 validation split 参与验证和选模，test split 只用于最终评测。新混合仅改变训练分布，合成验证分布维持原档，也不把测试数据纳入训练。源 checkpoint 的历史数据经验仍应按 provenance 披露，`real_fraction=0` 不表示整个模型从未见过真实训练数据。

## 2. 默认配置与起点

| 项目 | `--compact-selection` 默认值 |
|---|---:|
| 总 epochs / Proposal / Joint | 24 / 4 / 20 |
| 合成训练 / 合成验证 | 1500 / 500 |
| 真实训练比例 / 每个真实来源验证数 | 0 / 32 |
| batch / 每曲线采样点 | 32 / 192 |
| 网络内部候选 / 原 certified source 内部节点 | 64 / 4..24 |
| 最小保留内部节点 `min-selected-knots` | 4（维持旧档下限） |
| MSE 阈值 | `5e-5` |
| Proposal / Joint 基础学习率 | `2e-5` / `3e-5` |
| Joint Proposal / Selector / Decoder 学习率比例 | 0.1 / 1 / 0.25 |
| Joint 余弦学习率终值比例 | 0.25 |
| 初始 sigma / 常数储备节点 | 0 / 0 |
| 最终 sigma / 常数储备节点 | 0 / 0 |
| 简化控制器 / 复杂度倍率上限 | `per_curve` / 1.0 |
| 每种 seed 接受删除数 / 每曲线 seed 数 / 每 batch 曲线数上限 | 16 / 2 / 2 |
| 简单 / 形状 / 既有 certified source 训练比例 | 0.35 / 0.25 / 0.40 |
| 训练集每 epoch 重采样 | 是 |

MSE 为归一化后的 `mean_i ||C(t_i)-Q_i||²`，不取平方根、不除以坐标维数。内部候选 64 对应全保留时完整节点向量 72 项、控制顶点 68 个，不是最终一定保留 64 个节点。本档仍要求至少保留 4 个内部节点，不是在任意节点数（包括零内部节点）的范围内求全局最优。

新 run **必须显式传入 `--warm-start-checkpoint`**，不会自动选择历史 initializer。本次指定起点是 `outputs/checkpoints/overnight_reliable_3090_r1.pt`，即该轮选出的 **best epoch 5**，不是 `.last.pt`。复制兼容的完整模型权重，但优化器、epoch、训练日程和历史重新开始；使用新 run name，不覆盖 Reliable 权重。Best 表示当轮选模结果，不自动等于正式合格模型。

本档保留 `--allow-infeasible-proposals`，让 Proposal 按计划进入 Joint；它不豁免部署可行性或正式资格检查。正式验收仍要求每个验证来源至少 90% 的曲线通过同一 MSE 阈值；worst-source 门槛、最佳 checkpoint 选择、其余 checkpoint 合同及 `DIAGNOSTIC NOT FINAL` 的原有规则不因 `per_curve` 控制器而放宽。

## 3. 先上传更新包，再在 Linux 启动

更新包为 `outputs/delivery/overnight_compact_linux_update.zip`，是基于当前 overnight 代码 `9d155e5` 的源码增量，不包含数据、模型或训练结果，也不是完整仓库。先检查服务器对应 overnight 工作区的 HEAD 与本地修改，备份有冲突的文件；不要解压到其他主线工作区，不使用强制覆盖跳过冲突检查。

在 Windows PowerShell 中上传，把 `SERVER_HOST` 替换成实际服务器地址：

```powershell
Test-Path E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_compact_linux_update.zip
scp "E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_compact_linux_update.zip" feng@SERVER_HOST:/home/feng/HouCode/overnight_compact_linux_update.zip
```

确认上传成功后，在 Linux 中检查工作区：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
git rev-parse --short HEAD
git status --short
```

确认是正确工作区并处理完已有修改后，再执行下面的整段命令。`test -f` 确认包确实已经上传，`&&` 确保解压失败或新开关不存在时不会继续启动旧脚本：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight &&
test -f /home/feng/HouCode/overnight_compact_linux_update.zip &&
unzip /home/feng/HouCode/overnight_compact_linux_update.zip &&
grep -n -- '--compact-selection' scripts/run_v16_1070_overnight_linux.sh &&
test -f outputs/checkpoints/overnight_reliable_3090_r1.pt &&
bash scripts/run_v16_1070_overnight_linux.sh \
  --compact-selection \
  --warm-start-checkpoint outputs/checkpoints/overnight_reliable_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --real-val-size 32 \
  --run-name overnight_compact_3090_r1 \
  --benchmark-profile full
```

`unzip` 对已有文件会询问是否替换；只确认本次已检查的更新文件。若仍看到 `unknown option: --compact-selection`，说明实际执行的脚本没有更新或路径不对，应停止并检查工作目录、上传文件及解压结果，不要删除开关后改跑旧档。没有 zip 或 checkpoint 时上述链会停止；先上传缺失文件，不要把本机路径当作服务器上已存在的文件。

需要先检查命令时，在最后的 runner 调用追加 `--dry-run`；它不训练，也不替代真实数据或运行环境预检。实际启动前移除该参数。运行环境需 Python 3.11+ 和合适的 PyTorch/CUDA；可用 `--python /path/to/python` 指定解释器。真实数据应包含三个 manifest 及其引用的点文件；仅在确实缺少数据时显式加 `--prepare-real-data`，详见 [Linux 数据准备说明](overnight_1070_linux.md#2-推荐复用服务器已有的真实数据)。

### 中断后恢复同一个 Compact run

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight &&
grep -q -- '--compact-selection' scripts/run_v16_1070_overnight_linux.sh &&
bash scripts/run_v16_1070_overnight_linux.sh \
  --compact-selection --resume-run \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --real-val-size 32 \
  --run-name overnight_compact_3090_r1 \
  --benchmark-profile full
```

Resume **不要再传 warm-start**；保留原 run name、输出根、数据根和任何自定义训练参数，需要本次运行的 `.last.pt` 及配套 best/proposal/history 产物。它不是把旧 Reliable 的 `.last.pt` 当 Compact 续训，也不支持通过改写 checkpoint 合同绕过异地输出路径检查。仅评估已有权重时使用 `--checkpoint PATH`，同样不传 warm-start；评估、warm-start、resume 不能混用。

## 4. 输出及如何判断有效

一条龙顺序仍为预检 → 训练 → checkpoint 检查 → 六方法配对评测 → 四指标图 → Ours 案例 → 六方法真实案例。六方法为 Ours、Park、Liang、Dung、Kang、Luo；后五者是本仓库的论文适配。默认保留明确标记的 threshold-safe adaptation 及修复前后记录，修复代价计入基线完整耗时；这不意味着 Ours 部署新增搜索。

```text
outputs/checkpoints/overnight_compact_3090_r1.{pt,last.pt,proposal.pt,history.json}
outputs/logs/overnight_compact_3090_r1/
outputs/comparisons/overnight_compact_3090_r1/
outputs/figures/overnight_compact_3090_r1/
  four_metrics/ / ours_cases/ / six_method_real_cases/
```

先在相同曲线、相同阈值下比较通过率与 P95 MSE，再在相近可行性下比较最终内部节点数。区分固定几何 Teacher、经过 decoder 的教师子集和学生自由选集的拟合结果，并检查教师改进覆盖率、数量误差及真实来源分别统计；不要只用平均 K 下降宣称改进。

Joint 日志中的 `compact teacher` 一行用于分辨这些环节：

| 日志字段 | 含义与解读 |
|---|---|
| `fixed_K` | 已找到的固定几何可行 Teacher 目标的平均内部节点数，不是学生自由部署的平均 K；没有目标时占位为 0，不能解读为零节点可行。 |
| `fixed_pass` | 选中搜索的曲线中，找到固定几何可行目标的比例。 |
| `decoded_pass` | 在已找到固定几何可行目标的曲线中，同一目标 mask 经过当前 decoder 后仍通过阈值的比例；它与 `fixed_pass` 分母不同，也不是全验证集通过率。 |
| `accepted_delta_K` | 经过 decoded 可行性验收后，Teacher 相对 greedy 搜索前的平均净节点减少量；按整个 batch 统计，未改进曲线计 0。正数表示减少；原 Teacher 不可行时，换成更多节点的可行解也可能产生负值。 |
| `geometry_violation` | 几何蒸馏目标经过当前 decoder/refit 后的平均非负对数超限量，约为 `max(log(MSE / tolerance), 0)`；并非原始 MSE。没有目标时也记 0，需结合目标覆盖率判断。 |
| `extra_refits` | greedy 候选扫描、验证、decoder 验收和几何蒸馏的额外最小二乘系统数；不是 Python 调用数，也不含全部旧 Teacher 开销。epoch 日志为 batch 指标的样本加权平均，不是该 epoch 的 refit 总数。 |
| `complexity_active` | 当前 batch 中通过复杂度项安全门控的曲线比例；只描述该训练信号是否启用，不是模型的全局可行性保证。 |

以上定义以单个 batch 为基础；epoch 日志按 batch 样本数加权平均，比例并非把各 batch 的搜索目标直接合并后重算。它们是训练过程统计；正式判断仍看保存 checkpoint 的独立验证和配对测试结果。

四指标继续是平均最终 MSE、阈值通过率、平均最终内部节点数和完整算法耗时，网络前向时间单独记录。`full` 默认仅 66 条配对曲线，`quick` 进一步缩小样本及数值求解预算且强制诊断标记；二者都不是训练改进或全局最简性的证明。完整评测口径与基线说明见 [Overnight Reliable](overnight_reliable.md#4-数据对照组与输出)。

## 5. 本地验证记录（2026-09-19）

- 相关回归测试共 491 项通过，覆盖训练/续跑、loss、数据标签、模型、Linux runner、部署拟合与绘图报告。新增开关默认关闭时，与旧版在相同模型和随机种子下的 Proposal/Joint loss 及 69 项旧指标逐位一致。
- 最终审查补充验证：真实来源仅用于验证时，其有无不改变增强合成训练记录；低 K 分量直接在指定范围生成，再补齐到统一标签宽度，不依赖随机重试找到低 K 样本。修正后已重新跑通微型训练和续跑测试。
- 实际执行了 Linux Bash 入口的 CPU 小实验：3 epochs（1 Proposal + 2 Joint）、训练 4 条/验证 2 条合成曲线、每个真实来源验证 1 条。随后完成 24 条曲线 × 6 方法的 144 条评测记录，输出 9 张 PNG（2 张四指标图、4 张 Ours 案例/总览、3 张六方法真实案例）。
- 小实验中，最差来源通过率为 0% 时，复杂度倍率仍按计划从 0 增至 0.125、0.25；固定几何可行率与重新解码可行率分别记录，未把解码失败的精简目标标成可行 mask。
- 此实验只验证流程、梯度和输出完整性，**不是完整训练后的效果结论**。小样本结果及图已标记为诊断用途，不能据此宣称通过率提高或节点数降低。

本地小实验目录名为 `overnight_compact_pipeline_smoke_20260919_r1`，分别位于 `outputs/checkpoints/`、`outputs/comparisons/`、`outputs/figures/`、`outputs/logs/`；这些诊断产物不包含在服务器源码更新包内。
