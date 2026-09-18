# Overnight Plus：保留历史架构，增强训练与节点选择

`scripts/run_v16_1070_overnight_linux.sh --enhanced-selection` 是可选的新训练配置：沿用历史 Kc64 网络和在线 teacher，增加训练阶段的可行子集搜索、保留/删除边界监督，以及 Joint 分组学习率。部署仍是一次网络前向、一次离散选集和一次标准 B 样条 refit；不增加部署搜索，不切换到主线的离线 teacher，也不更换网络结构。

这是待验证的增强实验，**不是已证明的精度提升**。下文推荐从已训练的 overnight 模型完整迁移权重到新 run，而不是覆盖旧模型或恢复旧优化器。训练与六方法评测的总耗时取决于设备、数据和数值基线，不保证一夜完成。

## 1. 同步更新包与检查环境

本地交付包为 `outputs/delivery/overnight_plus_linux_update.zip`。它包含本次增强代码及之前的六方法绘图/评测更新，使用仓库相对路径；不包含模型、真实数据或实验产物。将它上传后，在历史 worktree `/home/feng/HouCode/SplineFitting_1070_overnight` 解压，**不要应用到主线 worktree**。

这些修改尚未提交，单纯 `git pull` 不会获得它们。先查看包内文件、备份并检查服务器已有代码改动，再确认替换相应文件；不要覆盖自己的未备份改动。

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
git status --short
unzip -l /path/to/upload/overnight_plus_linux_update.zip
unzip /path/to/upload/overnight_plus_linux_update.zip
python --version
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__); print("cuda:", torch.cuda.is_available())'
test -f outputs/checkpoints/overnight_1070_full_pipeline_r1.pt
```

需要 Bash、Python 3.11+ 和可用的 PyTorch/CUDA 环境；可追加 `--python /path/to/venv/bin/python` 指定解释器。`unzip` 对已有文件会询问是否覆盖。本流程不替你 commit、push，也不修改输入 checkpoint。

## 2. 推荐的一条龙命令

确认输入模型存在后执行：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
bash scripts/run_v16_1070_overnight_linux.sh \
  --enhanced-selection \
  --warm-start-checkpoint outputs/checkpoints/overnight_1070_full_pipeline_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --run-name overnight_plus_3090_r1 \
  --benchmark-profile full
```

流程依次执行环境/数据预检、训练、checkpoint 资格检查、四数据源六方法比较、四指标图、Ours 案例和六方法真实案例。可先在上述命令末尾追加 `--dry-run` 检查实际参数；dry-run 不执行 Python、下载或训练，但仍要求输入 checkpoint 存在且新 run name 没有已有输出。

`--warm-start-checkpoint` 完整复制兼容模型的 encoder、参数头、候选生成器、selector 和 subset decoder 权重；重新初始化 epoch、优化器、历史、复杂度/安全控制器及 proposal 阶段门槛。模型配置必须严格兼容，只有新实验的安全余量课程允许重设。它不是从原 epoch 接着训练，也不保证源模型的效果会原样保持。

三个入口不要混用：

| 入口 | 用途 |
|---|---|
| `--warm-start-checkpoint PATH` | 全模型权重迁移到新实验；本页推荐方式 |
| `--init-checkpoint PATH` | 仅迁移兼容的 encoder、参数头、候选生成权重，selector/decoder 从头开始 |
| `--resume-run` | 恢复同一 Linux run 的完整训练状态；已完成训练时继续评测/绘图 |

full-model warm-start 与显式 `--init-checkpoint`、`--no-init-checkpoint`、`--resume-run`、只评估用的 `--checkpoint` 互斥。`--enhanced-selection` 本身不会自动挑选已训练模型：若不传 `--warm-start-checkpoint`，仍使用原默认 `candidate_selection_v16.proposal.pt` 作 proposal-only 初始化。输入文件不存在时先确认路径，不要静默换成另一模型。输出 run name 必须与源实验不同。

## 3. 增强配置具体改变什么

不加 `--enhanced-selection` 时，原 64-epoch 历史配置保持不变：

| 训练项 | 原历史配置 | `--enhanced-selection` |
|---|---:|---:|
| 总 epochs / proposal / Joint 计划 | 64 / 4 / 60 | 96 / 12 / 84 |
| policy samples | 2 | 4 |
| counterfactual edits | 2 | 6 |
| prefix teacher 搜索步数 | 6 | 8 |
| teacher 局部细化 | 关闭 | 1 轮，每类最多 3 个候选编辑 |
| 额外边界排序损失 | 关闭 | 权重 0.5，每侧最多 4 个难例 |
| proposal 阶段学习率 | `2e-4` | `5e-5` |
| Joint 学习率比例：proposal / selector / decoder | 1 / 1 / 1 | 0.25 / 1 / 0.5 |
| Joint 余弦衰减末端比例 | 1（不衰减） | 0.25 |

实际进入 Joint 仍须通过 proposal 门槛，不因总 epoch 数增加而绕过资格检查。增强配置的 proposal 校准阶段使用较小的 `5e-5` 学习率，降低完整权重迁移后大幅改变既有候选的风险。Joint 基础学习率同为 `5e-5`：开始时 proposal 组为 `1.25e-5`、selector 为 `5e-5`、decoder 为 `2.5e-5`；之后分别按余弦日程衰减至各自初值的 25%。proposal 组包括 encoder、参数头和候选生成器，分组和衰减只作用于 Joint。这些设置的实际保留效果仍需验证。

teacher 细化围绕当前最佳 teacher 子集，有限次尝试删除、添加和等节点数交换，遵守实际选集约束。每个候选通过真实 subset decode/refit 计算误差；只接受满足阈值的改进，优先更少内部节点，同节点数再比较 MSE。它是有限局部搜索，不是全局最小节点证明，也不是新增的部署流程。

边界排序额外关注“teacher 要保留、但学生评分最低”和“teacher 要删除、但学生评分最高”的候选，让训练直接处理容易误删/误留的位置。仅可行 teacher 提供该监督；不可行的全保留 fallback 不当作可靠的保留/删除边界。更多 teacher 搜索和样本会增加训练成本，需结合实际验证结果判断收益。

保持不变的合同：`MSE <= 5e-5`、内部候选 `Kc=64`、source 内部节点 `K=4..24`、固定合成 train/validation 为 1500/500、batch 32。三次样条全保留时完整节点向量为 72、控制顶点为 68；这不是 source K，也不是部署最终保留数。MSE 是平均平方欧氏误差，不取平方根、不除以坐标维数。

当前新实验仍只使用合成训练和合成验证；UJI、Natural Earth、USGS 仅以各自 test split 参与当前评测与案例图。完整权重迁移保留源模型已有的数据经验，`real_fraction=0` 不意味着模型从未见过真实数据。`preflight.json` 记录来源哈希和配置，新 checkpoint 的 `initializer_provenance` 记录迁移及已有祖先来源；旧权重缺失的祖先记录仍属未知，不能据此推断纯合成预训练。

## 4. 中断后续跑，或只评估

恢复上述新实验时，**去掉 `--warm-start-checkpoint`，保留 `--enhanced-selection`**：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
bash scripts/run_v16_1070_overnight_linux.sh \
  --enhanced-selection \
  --resume-run \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --run-name overnight_plus_3090_r1 \
  --benchmark-profile full
```

保持原始输出路径、训练配置、数据和评测设置，并保留 `.last.pt`、最佳 proposal / 主模型等配套产物。省略增强开关会变回旧训练合同，不能用于恢复增强 run。训练已完成请求的 epoch 数时会跳过训练；评测仅在实验指纹一致时复用完成记录。不要同时启动两个相同 run name 的进程。

只评估一个已经训练好的增强模型，使用新的评测 run name，不传初始化或续训参数：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --checkpoint outputs/checkpoints/overnight_plus_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --run-name overnight_plus_3090_eval_r1 \
  --benchmark-profile full
```

这个命令不训练，不需要 `--enhanced-selection`；按 checkpoint 中保存的模型部署。`--checkpoint` 仍需满足 Kc64 和一致的 MSE 阈值。未通过正式资格的模型只生成带 `DIAGNOSTIC NOT FINAL` 标记的诊断结果；训练阶段未生成主模型时，一条龙会停止，不伪造合格 checkpoint。

## 5. 数据、产物与如何判断结果

继续复用 `/home/feng/HouCode/SplineSurpervisedFitting/data` 下的完整数据树，JSONL 的相对点文件路径按 manifest 所在目录解析，不能只复制三个 JSONL。缺失数据时可显式加 `--prepare-real-data`，已有数据不必重新准备；详细路径和下载边界见[原 Linux 流程](overnight_1070_linux.md#2-推荐复用服务器已有的真实数据)。

评测保持 **Ours + Park、Liang、Dung、Kang、Luo 五种论文方法适配，共六方法**；可设容量统一为 64。full 为每个 source K 2 条合成曲线、每个真实数据集 8 条，共 66 条配对曲线；真实图每个数据集选 2 个案例。quick 仍会降低样本和数值迭代/计时预算并强制诊断，但不会缩短增强训练的 96 epochs。full 也只是有限评测预算，不等于充分正式统计。

主要产物：

```text
outputs/checkpoints/overnight_plus_3090_r1.{pt,last.pt,proposal.pt,history.json}
outputs/logs/overnight_plus_3090_r1/
outputs/comparisons/overnight_plus_3090_r1/
  comparison.json
  measurements.csv
  summary.csv
  report.md
outputs/figures/overnight_plus_3090_r1/
  four_metrics/
  ours_cases/
  six_method_real_cases/
```

四指标仍为平均 MSE、阈值通过率、平均最终内部节点数和平均完整算法耗时（含最终 refit）。Ours 网络前向耗时在 CSV 独立记录，不能与传统方法完整耗时当作端到端加速比。续跑日志保留、图进入新的 attempt 子目录；`--output-root PATH` 可改变根目录，但同 run 续跑不能搬动原绝对输出路径。

判断增强是否有效，应在相同测试样本和完整评测预算下与源模型配对比较，先看部署通过率和误差，再看节点数；单纯删得更多不等于改进。训练历史另提供 teacher 可行率、局部细化收益、`teacher_keep_f1`、`teacher_false_remove_rate` 和有效 teacher 比例，用于区分“教师本身不可行”和“学生选错组合”。这些训练诊断不能替代独立测试。旧流程的 127 项测试和 CPU 评测联调记录只说明旧版本验证范围，不是本增强模型已经训练成功或效果更好的证据。

## 6. 本次验证结果与限制（2026-09-18）

- 核心网络、loss、训练、checkpoint 和 Bash/preflight 回归测试 264 项通过，报表及案例绘图测试 53 项通过，合计 317 项；Ruff、Bash 语法检查通过。网络结构和推理参数张量没有增加。
- 用已有历史 Kc64 权重实际完成小规模全链路：2 条训练/2 条验证，Proposal 1 + Joint 1；随后完成 24 条曲线 × 6 方法的配对诊断评测，输出四指标图、真实数据案例等 9 张 PNG。这仅验证流程和梯度可运行，不能视为训练收敛或总体达标。
- 同一集成 run 的 `--resume-run` 已实际验证：检测到训练完成后跳过训练，复用配对评测记录，并在新的 attempt 目录重绘全部案例，退出码为 0。
- 独立 toy 配对实验固定 seed=41，双方同初始化、同数据、同逐步 RNG 和优化器，各训练 40 步。80 次更新梯度均有限；所有早期超阈值记录均保留。最终生产 refit：原 loss 平均 K=5.5、MSE=1.138e-6；增强 loss 平均 K=6.0、MSE=8.154e-7，双方均满足该 toy 的 2.5e-5 阈值。**误差下降但节点数反而增加，因此目前不能宣称整体收益。** 这也是把增强版作为独立新 run、保留旧模型的重要原因。
- toy 的增强训练中有 11 个 Joint 步出现 teacher 子集改善，其中 10 步 teacher K 减少；这只说明训练搜索能产生更紧凑的可行标签，不等于学生已经学会这些标签。真实规模的收益需完成 3090 训练后评测。

完整诊断保存在本机 `outputs/tmp/smoke_overnight_plus_seed41_40steps_verified.json`；集成运行目录名为 `overnight_plus_pipeline_smoke_20260918_r1`。代码更新包不包含这些实验产物。可自行执行固定配置诊断（不代替正式训练）：

```bash
python scripts/smoke_overnight_plus.py \
  --output outputs/tmp/overnight_plus_toy_check.json
```
