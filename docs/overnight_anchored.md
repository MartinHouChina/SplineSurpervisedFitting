# Overnight Anchored：少节点解的稳定解码与短程校准

2026-09-21：本页的新模型已在3090训练并拉回，结果见[配对审计](overnight_results_audit_20260921.md)。进一步增层/专用容量/峰值误差实验见[Granularity](overnight_granularity.md)。以下保留当时的设计与本地验收记录，不把“尚未训练”的历史状态当作当前结果。

本轮基于 `c5a641d` 的 K32 实验改进，不回滚分支，不覆盖已有训练结果。目标仍是：在归一化输入点 **MSE ≤ 5e-5** 的条件下尽量减少内部节点。不存在最少节点或通过率保证。

## 1. 问题与修改

K32 测试并非全面达标：Synthetic/UJI/Natural Earth/USGS/IndustrialOffset 的输入点通过率为 59.5%/90%/40%/30%/85%。简单曲线有明显冗余，而部分复杂曲线全部保留 32 个仍不达标，二者不能用同一“增大删除惩罚”解决。

最佳 epoch 14 的固定几何教师平均 K=12.39，但其紧凑终点集合经网络重新解码后的通过率仅 32.8%。小集合的参数与节点位置发生变化，是教师难以转化为筛选能力的重要原因；不是 Teacher 或 KeepMask 被取消。

| 修改 | 新行为 | 不变项 |
|---|---|---|
| 锚定式子集解码 | 取消向均匀 rank 位置的强制混合；参数在 Proposal 参数附近做有界修正，节点同步变换坐标后局部残差移动 | 保留存活节点交互注意力、参数更新、节点移动；不是禁止微调 |
| 紧凑教师优先级 | 深搜索预算优先分配给可行且有误差余量/冗余的曲线，同时保留困难曲线探索名额 | 搜索只用于训练；实际 refit 验收不放松 |
| 精简 Mask 专项监督 | 对确实重新解码达标、比当前部署更小的教师集合，加强概率质量、保留排序与 Mask 监督 | 不把仅固定几何可行、但网络无法复现的集合当作安全删除标签 |
| 短程几何校准 | Joint 前段冻结 Encoder、ParameterHead、CandidateHead 及 Proposal trust；先校准 Selector/Decoder，后段低学习率联合优化 | 仍保留 Proposal + Joint 两阶段；校准是 Joint 内部子阶段 |

原有 `(Mask,t,U)` 几何蒸馏继续使用；尚不可达的紧凑目标只能提供几何监督，不直接压低筛选数量。新专项损失按获得有效目标的曲线平均，避免少数深搜索目标被整个 batch 稀释。

解码中的顺序/最小间隔限制是几何约束，**不是误差证书**。尤其在切换旧权重到新解码方式时，需要重新训练和评估；不能把“加载成功”称为“精度保持”。

## 2. 训练与部署数据流

```text
训练：合成有序点 + 合成标签
    → Encoder / ParameterHead → t0
    → CandidateHead → 32 个候选 U0
    → Selector → 重要度、动态阈值 β、概率质量 → 一次 Top-K Mask
    → 锚定式 Decoder → 参数 t、存活节点 U
    → 标准 B 样条 refit → MSE 与训练损失

训练教师支路：有预算的固定几何贪心 → 删除轨迹
    → 子集重新解码/refit验收
    → 可行小集合：Mask + 数量 + 排序监督
    → 可行及尚不可达的几何目标：(t,U) 蒸馏 + 实际拟合约束

部署：有序点 → 一次候选/筛选/解码前向 → 一次最终 refit
```

部署不加入贪心删除、阈值扫描或反复求解补救。动态 β 不是固定 0.5 阈值；新档配置不额外保留安全节点。

默认短程配置：16 epochs = 4 Proposal + 12 Joint；Joint 前 6 轮冻结候选几何、关闭复杂度压力，后 6 轮低学习率联调并逐步开启复杂度。使用已训练的原生 K32 检查点进行 **新实验 warm-start**，不继续旧优化器。训练/合成验证为 1500/500，输入 192 点，batch 32，源内部 K=4..24，候选上限 Kc=32。

具体新参数：残差强度 `0.25`；教师最多 `4` 条/批、每个种子最多 `16` 次接受删除、`4` 个轨迹检查点、每条曲线最多 `2` 个几何目标；几何蒸馏权重 `0.4`，紧凑 Mask 专项权重 `0.5`。Proposal 学习率 `2e-5`，Joint selector/decoder 初始 `3e-5`，解冻后的 Proposal 使用其 `0.1` 倍；Joint 余弦衰减末值为初值的 `0.25` 倍。校准结束后的复杂度课程长度为 `4` 轮。教师覆盖增加会增加每步开销，16 轮是实验预算，不是运行分钟数保证。

合成训练仍由低 K 样条 35%、程序化形状 25%、历史合成分布 40% 构成；程序化形状没有真实节点标签，不能当作有标签样条。验证分布不随训练的低 K 过采样改变。

三次夹持 B 样条在保留全部候选时有 40 项完整节点向量、36 个控制顶点；32 指内部节点，不是完整节点向量长度。

## 3. Linux 一条龙

先同步本轮全部 Python/Bash 文件，不能只上传 Bash。不要用旧实验名，不要用 `--resume-run` 切换解码方式。

增量包：`outputs/delivery/overnight_anchored_linux_update.zip`，基于 `c5a641d`，仅包含本轮源码、测试和文档，不包含数据与权重，也未自动提交或推送 Git。PowerShell 上传：

```powershell
scp "E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_anchored_linux_update.zip" feng@10.76.0.64:/home/feng/HouCode/
```

Linux 上先保存服务器未提交修改，确认位于对应 overnight 工作区后解压（按提示确认覆盖源码，不覆盖训练结果）：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
unzip /home/feng/HouCode/overnight_anchored_linux_update.zip
```

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight

bash scripts/run_v16_1070_overnight_linux.sh \
  --anchored-selection \
  --warm-start-checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --prepare-real-data \
  --run-name overnight_anchored_k32_3090_r1 \
  --benchmark-profile full \
  --real-samples-per-dataset 20 \
  --visual-samples-per-dataset 6
```

追加 `--dry-run` 可先检查生成的命令；dry-run 不检查权重内容、不训练。旧权重不存在时请先同步真实文件，不要用改名的 K64 权重冒充 K32。

流水线按顺序执行训练、检查点检查、六方法测试、四/五指标图、Ours 案例与六方法案例图。方法是 Ours、Park、Liang、Dung、Kang、Luo；数值方法仍为仓库明确标注的论文适配，不是作者原代码。全部方法统一最大内部节点 32。

测试覆盖 Synthetic、UJI、Natural Earth、USGS、工业等距线。训练使用合成数据，前三个外部来源参与独立验证；工业等距线仅作测试，是 CAD 半合成，不是实测。输入点与原始参考点的 MSE、最大单点平方误差分别报告，失败样本不剔除。

```text
outputs/checkpoints/overnight_anchored_k32_3090_r1*.pt
outputs/logs/overnight_anchored_k32_3090_r1/
outputs/comparisons/overnight_anchored_k32_3090_r1/
outputs/figures/overnight_anchored_k32_3090_r1/
  four_metrics/
  ours_cases/
  six_method_real_cases/
```

短程结果仍可能标记 `DIAGNOSTIC NOT FINAL`。保留原资格检查，不能为了图表好看更改误差或删掉复杂曲线。对比时重点检查相同测试样本、相同通过条件下的节点数量变化；平均 MSE 达标不代表每条曲线达标。

## 4. 检查小节点集合能否被网络复现

新增诊断直接读取案例 JSON 内保存的归一化点，不依赖服务器与本机数据目录一致。它比较原部署、固定几何贪心删除、同一 Mask 重新解码；搜索不计入 Ours，也不改变 checkpoint。

```bash
python scripts/audit_v16_compact_decoder.py \
  --checkpoint outputs/checkpoints/overnight_anchored_k32_3090_r1.pt \
  --cases-json outputs/figures/overnight_anchored_k32_3090_r1/six_method_real_cases/deployment_visualizations.json \
  --samples-per-dataset 2 \
  --output outputs/comparisons/overnight_anchored_k32_3090_r1/compact_decoder_audit.json
```

诊断保留失败种子，不仅展示成功删点；已有输出不会覆盖。这里仅检查输入点误差，是保存案例的少量机制诊断，不是随机总体实验，也不验证连续曲线最大误差。最终仍以完整六方法测试中的输入/参考两种口径为准。

旧模型缺少新配置时保持 `legacy` 行为；新模型把解码方式和残差强度写入 checkpoint，重载不会丢失。新档必须通过重新训练验证，不能将旧结果重命名为改良结果。

## 5. 本地验收与限制

- 全量 `tests/` 回归 **1143 项通过**，Bash 语法和一条龙 dry-run 通过；覆盖实际小规模 warm-start 训练、冻结/解冻、保存重载、续训一致性和旧检查点逐张量兼容。另补齐两个旧评测测试夹具的 `evaluate()` 接口，以适配 K32 已有的最大误差计算，未修改对照算法或实验结果。
- 使用同两条解析曲线反复优化的 CPU 功能性实验中，候选 6 个，最终均保留 2 个；生产 refit 的 MSE 分别为 `1.092e-5`、`5.505e-6`，均低于 `5e-5`。完整中间记录保存在 `outputs/tmp/anchored_compact_zero_reserve_smoke_20260920.json`。这是训练曲线过拟合测试，不是泛化或六方法胜出证据。
- 旧 K32 权重只切换新解码方式、尚未训练时，部分已通过案例会变失败。这是函数迁移，不是无损升级，必须先校准再看完整测试。
- 未进行完整 3090 重训，不能保证部署通过率、平均 K 或耗时已经改善。旧 K32 和 K64 报告保留。

功能性测试可复现为：

```bash
python scripts/smoke_v16_subset_learning.py \
  --steps 121 --proposal-steps 10 --learning-rate 0.003 \
  --mse-tolerance 5e-5 --subset-geometry-mode anchored \
  --subset-geometry-residual-scale 0.25 --compact-teacher \
  --output outputs/tmp/anchored_compact_smoke_new.json
```

本机有旧回滚源码副本放在 `outputs/` 下，因此运行回归时显式指定 `python -m pytest tests`，不要递归收集输出目录中的旧测试。不要为解决测试导入冲突删除旧实验文件。
