# 1070 历史架构：Linux 一键训练、六方法比较与案例图

入口：`scripts/run_v16_1070_overnight_linux.sh`。本流程在独立的历史 worktree 中使用旧架构、在线 prefix teacher 训练，再对 Synthetic、UJI、Natural Earth、USGS 运行六方法配对评测，生成四指标图、Ours 案例和六方法真实曲线图。

这是按历史配置重新开始的 **warm-start 新实验**，不是原来夜间实验 epoch 17 的精确续训，也不是当前主线的离线 teacher/cache 训练。不要把新旧方法的结果合并为同一实验。脚本不修改主线 worktree，不承诺训练一定达标或一定在一夜内完成。

需要在同一历史网络上增强训练时，另见 [Overnight Plus](overnight_plus.md)：显式 `--enhanced-selection` 启用 96-epoch 配置，`--warm-start-checkpoint` 可把已训练模型的全部权重迁移到新 run。本页不加该开关的 64-epoch 配置及末尾 2026-09-17 验证记录保持原义；旧测试数量不代表增强版本的验证结论。

## 1. 先同步新增代码，再准备环境

服务器只有旧提交时，`git worktree add` 不会自动获得这次尚未提交的脚本。先将本次更新的历史 worktree 文件同步到服务器对应的 **1070 历史 worktree**，或上传本地 `outputs/delivery/overnight_1070_linux_update.zip` 并在该 worktree 根目录解压。更新包基于历史提交 **`6413ee0`**，不应应用到主线 **`e194a12`**。它只包含代码、文档和测试的相对路径，不包含真实数据或模型，也不是完整仓库。本流程不替你 commit 或 push。

```bash
cd /path/to/SelfSurpervisedSplineFitting_v16_1070_overnight
git rev-parse --short HEAD
git status --short
unzip -l /path/to/upload/overnight_1070_linux_update.zip
unzip /path/to/upload/overnight_1070_linux_update.zip
```

解压前应确认 HEAD 是 `6413ee0` 对应的历史版本。若有本地改动，先备份并检查与包内文件的差异；`unzip` 遇到已有文件时会询问是否覆盖，仅确认替换属于本次更新的文件。不要在主线根目录解压，也不要用强制覆盖跳过冲突检查。

以下命令均在更新后的历史 worktree 根目录执行。需要 Bash、Python **3.11 或更新版本**，以及适合服务器 GPU 的 PyTorch 环境；评测代码使用 Python 3.11 的 `hashlib.file_digest`。可以复用已有环境，不必为此重装 GPU 驱动。

```bash
cd /path/to/SelfSurpervisedSplineFitting_v16_1070_overnight
python --version
python -c 'import torch; print(torch.__version__); print("cuda:", torch.cuda.is_available())'
test -f scripts/run_v16_1070_overnight_linux.sh
test -f outputs/checkpoints/candidate_selection_v16.proposal.pt
```

缺少 Python 包时，在选定的虚拟环境中安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

更新包中的 `requirements.txt` 已补齐数值基线需要的 `scipy>=1.10`。`--device cuda` 要求 CUDA 可用；`auto` 自动选择，`cpu` 用于联调。可用 `--python /path/to/venv/bin/python` 指定环境。`scripts/overnight_linux_preflight.py` 会检查 Python、PyTorch、SciPy、Matplotlib 和设备条件。CPU 或数值基线可能成为瓶颈，换用 3090 不代表全部流程同比加速。

## 2. 推荐复用服务器已有的真实数据

`--data-root` 指向完整的 `data` 目录，而不是仓库根目录。例如主线已准备好数据时：

```bash
DATA_ROOT=/path/to/main_worktree/data
bash scripts/run_v16_1070_overnight_linux.sh \
  --device cuda \
  --data-root "$DATA_ROOT" \
  --run-name overnight_1070_arch_3090_r1 \
  --dry-run
```

`--dry-run` 只打印命令，不执行 Python、下载、训练或写入产物；它仍检查参数、initializer/checkpoint 是否存在和 run name 是否冲突，不能替代实际运行中的数据及环境预检。

需要这三个 manifest 及它们引用的全部点文件：

```text
data/
  splits/uji_pen_v2.jsonl
  processed/uji_pen_v2/...
  processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl
  processed/natural_earth/v5.1.2_10m_coastline/curves/...
  processed/usgs_contours/large_scale/manifest.jsonl
  processed/usgs_contours/large_scale/curves/...
```

JSONL 中的相对 `points_path` 是相对于 **manifest 所在目录** 解析的，不是相对于当前目录或 `--data-root`。只复制三个 JSONL 不够；搬移数据应保留完整目录关系。Windows 绝对路径不能直接作为 Linux 点文件路径使用。历史 worktree 默认不随 Git 携带这些真实数据，复用服务器已有数据通常最安全。

确实没有数据时，显式添加 `--prepare-real-data`：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --device cuda \
  --data-root /path/to/new_data \
  --prepare-real-data \
  --run-name overnight_1070_arch_3090_r1
```

该选项只准备 manifest 缺失的数据集，可能访问 UCI、Natural Earth、USGS 下载源，不默认覆盖已有文件。若 manifest 已存在但点文件丢失，预检会报错，不会静默重建。USGS 使用 `configs/usgs_contour_regions.example.json` 的限定区域，不是全国等高线下载。底层准备入口分别是 `prepare_uji_pen.py`、`prepare_natural_earth.py`、`prepare_usgs_contours.py`；它们没有统一的 `--data-root`，必须分别设置输出路径。USGS 单独调用时还必须提供 `--bbox-file`、`--bbox` 或 `--input`。详细说明见 [UJI 接入](uji_pen_integration.md)和[地理数据接入](geospatial_real_world_data.md)。

新训练不使用真实数据训练或验证，真实数据仅取 `test` split 进行当前实验的评测和作图。不过默认 initializer 曾使用这些数据集的其他拆分，不能称为“从未见过真实数据”的零样本模型，见下节。

## 3. 一条命令完成全流程

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --device cuda \
  --data-root /path/to/main_worktree/data \
  --run-name overnight_1070_arch_3090_r1 \
  --benchmark-profile full
```

默认训练合同：

| 项目 | 本次历史配置 |
|---|---|
| 默认 run name | `overnight_1070_arch_3090_r1` |
| 合成 source 内部节点数 | `K=4..24`，即控制顶点 `8..28` |
| 网络内部候选容量 | `Kc=64`；全保留时完整节点向量 72、控制顶点 68 |
| 误差阈值 | `MSE <= 5e-5`，即 `0.00005` |
| 训练计划 | 64 epochs，proposal 4；通过阶段门槛后，余下最多 60 epochs 为 joint |
| batch / 固定合成数据规模 | batch 32；train 1500 / validation 500，不逐 epoch 换训练集 |
| 本次真实数据采样 | `real_fraction=0`，没有真实验证来源 |
| warm-start initializer | `outputs/checkpoints/candidate_selection_v16.proposal.pt` |

source K、候选容量、输出实际节点数是不同量。source K 最大 24 对应完整源节点向量最大 32，不是网络全保留时的 72。保留旧网络、损失和在线 prefix teacher；本脚本不会调用当前主线的离线 teacher 生成流程。

warm-start 只迁移兼容的 encoder、参数头和候选生成权重，selector、subset decoder 和优化器从新实验开始；不会恢复 initializer 的 epoch、优化器或随机状态。**实际检查的默认 initializer 是 proposal epoch 59，历史 `real_fraction=0.5`，包含 UJI、Natural Earth、USGS 训练来源，且每个来源曾取 100 条真实验证样本。** 因此只能说“本次后续训练/验证为纯合成”，不能说“整个模型只见过合成数据”。`preflight.json` 记录 initializer 哈希、原训练配置和真实数据来源，便于追溯。替换 initializer 后以该文件里的实际记录为准。

initializer 缺失时默认报错；可用 `--init-checkpoint PATH` 指定实际位置。`--no-init-checkpoint` 是显式从头训练消融，不再是这里描述的历史 warm-start 重跑。`--epochs`、`--proposal-epochs`、`--train-size`、`--val-size`、`--batch-size`、`--num-workers`、`--mse-tolerance` 也可覆盖，但改变这些值即改变实验配置，应另取 run name 并披露。

proposal 门槛失败时，产物可能只有 `.proposal.pt`、`.last.pt`、历史记录；一键训练流程会停止，不自动将 proposal 冒充主模型。已有主 checkpoint 但正式资格不通过时，流程继续生成带 `DIAGNOSTIC NOT FINAL` 标记的诊断结果；资格未通过前不能用于正式结论。

## 4. 只评估已有模型，或恢复本次 Linux 运行

跳过训练，评估已有 checkpoint；用独立 run name，避免和训练输出混淆：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --device cuda \
  --data-root /path/to/main_worktree/data \
  --checkpoint /path/to/existing_v16_model.pt \
  --run-name overnight_1070_existing_eval \
  --benchmark-profile full
```

这里的 `--checkpoint` 是 **只评估**，不是 initializer，也不是续训文件。预检要求旧 v16 checkpoint 的内部候选容量为 **64**，且其中保存的 MSE 阈值与本次 `--mse-tolerance` 一致（默认 `5e-5`）；不合格模型只能用于诊断。`--checkpoint` 和 `--resume-run` 不能同时使用。

本脚本新建的 Linux 运行中断后，用原来相同的 run name、数据根、输出根和评测配置追加 `--resume-run`：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --device cuda \
  --data-root /path/to/main_worktree/data \
  --run-name overnight_1070_arch_3090_r1 \
  --benchmark-profile full \
  --resume-run
```

`--resume-run` 只恢复同一个 Linux 实验，需要原始输出路径和完整的 `.last.pt`、最佳 proposal / 主 checkpoint 等产物；已完成请求 epoch 数的训练会被跳过，继续后续评测。评测记录只有在实验指纹一致时才可复用。不要用它承接旧 Windows epoch 17 运行，也不要为了“成功续跑”修改 checkpoint 内保存的训练合同。不要同时启动两个相同 run name 的写入进程。

## 5. 样本量、指标和结果位置

六种方法是 **Ours + 五种论文算法适配**：Kang、Park、Liang、Dung、Luo；不是“六个基线再加 Ours”。本流程不包含 Yeh 和 Greedy。论文方法是仓库适配版本，不声称等同作者原始实现，详见[复现边界](published_knot_methods_reproduction.md)。

| 评测档位 | 每个 source K 的合成曲线 | 每个真实数据集的曲线 | 每个真实数据集的可视化案例 |
|---|---:|---:|---:|
| `full`（默认） | 2 | 8 | 2 |
| `quick` | 1 | 2 | 1 |

因此 full 共 42 条合成 + 24 条真实 = 66 条配对曲线，每条跑六方法；Ours 和六方法真实图各选择每集 2 条、共 6 个案例。quick 共 27 条配对曲线（数据足够时）。`quick` **不缩短 64-epoch 训练计划**；它还降低 Kang ADMM / Luo DE 迭代数至 100 / 10（full 为 400 / 50），并减少计时重复次数，所有结果强制标为诊断，不能与 full 混合当作相同数值求解预算。

full 默认网络前向计时 100 次、完整算法计时 3 次；quick 为 5 次 / 1 次。可分别通过 `--network-repeats` 和 `--end-to-end-repeats` 覆盖。完整计时重复数对所有方法统一；内部节点容量也统一为 64。

可覆盖三项样本数，例如先检查评测链路：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --device cuda \
  --data-root /path/to/main_worktree/data \
  --checkpoint /path/to/existing_v16_model.pt \
  --run-name overnight_1070_smoke_eval \
  --benchmark-profile quick \
  --synthetic-samples-per-k 1 \
  --real-samples-per-dataset 2 \
  --visual-samples-per-dataset 1
```

四指标是平均 MSE、MSE 阈值通过率、平均最终内部节点数、平均完整算法耗时。`MSE = mean_i ||C(t_i)-Q_i||²`，不取平方根、不除以坐标维数。完整耗时包含参数化/方法计算和最终标准 refit；不包含数据文件读取和绘图。Ours 的网络前向耗时在 CSV 中独立记录，不能拿它与传统方法的完整耗时直接作“端到端加速比”。MSE / 节点数均值使用得到有限拟合结果的记录，包括未达阈值者；通过率分母包含失败样本，完整耗时也包含失败尝试。查看汇总时同时检查成功/失败数和诊断状态。

`--output-root PATH` 可改变产物根目录，默认为 `outputs`；默认 run name 的主要结果如下：

```text
outputs/
  checkpoints/overnight_1070_arch_3090_r1.pt
  checkpoints/overnight_1070_arch_3090_r1.last.pt
  checkpoints/overnight_1070_arch_3090_r1.proposal.pt
  checkpoints/overnight_1070_arch_3090_r1.history.json
  logs/overnight_1070_arch_3090_r1/
    preflight.json
    ...各阶段日志...
  comparisons/overnight_1070_arch_3090_r1/
    comparison.json
    measurements.csv
    summary.csv
    report.md
  figures/overnight_1070_arch_3090_r1/
    four_metrics/v16_published_methods_input.png
    four_metrics/v16_published_methods_reference.png
    ours_cases/
    six_method_real_cases/
```

input 图按统一重采样输入点评测；reference 图按可用的原始参考点评测，二者口径不同。续跑时日志追加保留，预检另写 `preflight_<时间>_<进程号>.json`，图写入 `attempt_<UTC时间>_<进程号>` 子目录；以最终日志打印的位置为准。`comparison.json` 和日志保留配置、配对样本、时间口径和资格状态；仅生成图片不代表训练通过资格验证。`full` 的小规模比较也不等于充分的论文统计实验。

## 6. 本次实际验证范围

2026-09-17 在本机以 CPU、历史 `candidate_selection_v16_mse5e-5_k64.pt` 完成了跳过训练的整条评测/绘图链路（`quick`，21 条合成曲线、每个真实数据集 1 条）：24 条配对曲线 × 6 种方法，共 144 条测量记录，生成 2 张四指标图、3 张 Ours 案例、1 张 Ours 总览及 3 张六方法案例图，合计 9 张 PNG。已查看生成图片，确认曲线、采样点、控制顶点、内部节点与耗时标注均能显示。

本地记录位于 `outputs/comparisons/overnight_linux_smoke_20260917_r1/`，图在对应 `outputs/figures/` 目录。更新包不包含这些联调产物。此次验证没有重新执行完整训练，且评测时有其他本机任务，因此它验证的是流程可运行，不是 3090 效率结论或充分的精度统计。核心训练、模型、损失及数值基线算法未改动；新增测试检查命令参数、输出保护、配对统计及失败案例保留。

针对本次改动的自动测试合计 127 项通过（runner/preflight 74 项，报表与案例图 53 项），Bash 语法、Ruff 和 `git diff --check` 通过。续跑额外验证原始绝对输出路径、配置及 last/proposal/best 配套权重；完成训练后的恢复仅继续评测和绘图，不再次调用已结束的训练器。
