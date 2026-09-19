# Overnight K32：统一容量与最大误差比较

基于 `a6a1ed9`（“19 16 训练结果可以，改良”）的 Stable 架构。保留 Proposal + Joint、在线教师、KeepMask、参数更新和存活节点重定位；不增加部署搜索。新实验显式使用 **最多 32 个内部候选节点**，六种方法采用同一容量。旧 K64 实验及命令默认值保留，不能将旧模型的结果改名成 K32。

## 1. 容量的含义与风险

- 三次夹持 B 样条：32 个内部节点 + 两端各 4 项 = 最多 **40 项完整节点向量**，最多 **36 个控制顶点**。
- 合成源曲线仍为 K=4..24，不把候选上限当作源标签上限；训练/合成验证为 1500/500，192 个输入点，batch 32。
- Ours、Park、Liang、Dung、Kang、Luo 的初始/最大内部容量均设为 32；数值方法仍是仓库标注的论文适配版。Dung/Kang/Luo 的 threshold-safe 修复也受相同上限约束，不能隐藏增加到 64。
- MSE 阈值仍为 `5e-5`。降低容量不保证降低最终节点数或保持通过率，不删除难例，也不只统计成功样本的通过率。

最新 K64 Stable 测试中，Ours 的容量使用情况如下。这些是旧 K64 模型的诊断，不是 K32 的结果，也不证明超过 32 个节点是数学上必需的：

| 来源 | n | 平均内部 K | 最多内部 K | K > 32 的样本数 |
|---|---:|---:|---:|---:|
| Synthetic | 42 | 27.40 | 46 | 16 |
| UJI | 20 | 19.55 | 29 | 0 |
| Natural Earth | 20 | 26.15 | 52 | 2 |
| USGS | 20 | 30.80 | 44 | 10 |
| 工业等距线 | 20 | 17.55 | 23 | 0 |

因此先观察 K32 的 dense proposal 可行性，再看筛选部署；特别注意 USGS 和高复杂度合成曲线。保留 K64 作为容量消融，不能直接截断 K64 模型的输出就宣称完成 K32 训练。

## 2. 迁移权重并重新训练

`--candidate-knots 32` 贯穿训练、预检、基准测试和两类案例图。`--resize-candidate-warm-start` 是显式缩容开关，仅可与 `--warm-start-checkpoint` 一起启动**新实验**：

1. 只沿 interval query 的相对序号线性插值 `65 → 33` 个 query（K+1）。
2. 固定位置 anchors 按 K32 重新构造；其他兼容的编码器、参数头、selector、decoder 张量精确复制。
3. 保留祖先数据暴露记录，记录源/目标容量、插值张量和重建 buffer；优化器、训练历史和控制器重新初始化。
4. 默认严格检查其他结构配置和张量形状，不允许不明形状静默跳过。不传缩容开关时，原 full warm-start 检查保持严格。

缩容不是无损转换，候选间距、注意力和数量概率分布都会变化，需重新适配。建议 **32 epochs = 12 Proposal + 20 Joint**；比上一轮延长 Proposal 适配，其余 Stable 损失与教师设置不变。这是短程验证配置，不承诺达到某个通过率。

不能用 `--resume-run` 将 K64 改成 K32；续跑时必须维持已保存容量、数据和训练配置。评测也会拒绝 checkpoint 容量与 `--candidate-knots` 不一致，避免只给基线降容量。

## 3. 新增最大误差指标

令单个采样点平方残差为 `e_i² = ||C(t_i) - Q_i||²`。所有主误差在归一化坐标下计算，不除以维数、不取平方根：

| 指标 | 定义 | 用途 |
|---|---|---|
| 曲线 MSE | `mean_i e_i²` | 原有通过标准：MSE ≤ 5e-5 |
| 曲线最大单点平方误差 | `max_i e_i²`，字段 `max_squared_error` | 检查平均误差掩盖的局部偏离 |
| 数据集最大单点平方误差 | `max_curve max_i e_i²` | 新增第五项比较图 |
| 数据集最差曲线 MSE | `max_curve MSE`，字段 `mse_max` | 另列报告，不与单点最大误差混淆 |

汇总同时保存每曲线最大单点平方误差的 mean、P95、max；原始参考点按原有参数映射单独计算对应 `reference_*` 指标。新误差计算在方法计时区间之外。旧的四项图保留，额外生成五项图：时间、MSE 通过率、平均内部 K、平均曲线 MSE、最坏单点平方误差。

这是**离散对应点**的最大残差，不是连续整条曲线的最大误差证明，也不是 Hausdorff 距离。未新增最大误差通过阈值；MSE 达标不意味着每一点的平方误差都小于 `5e-5`。若以后要约束最大残差，应单独定义阈值和训练目标。

旧 JSON 没有保存该指标时显示缺失，不能从 MSE 反推最大误差，也不补零。新评测合同会阻止与旧口径记录混合续写，推荐新 run name 重新评测。失败/非有限拟合保留失败记录并计入通过率分母；最大误差汇总仅在成功拟合样本全部有相应测量时给出，任何缺测则标 N/A，并披露有效数/缺测数。失败不当作零误差，因此最大值仍应与失败数一起解释。

## 4. 上传与 Linux 一条龙

更新包 `outputs/delivery/overnight_k32_linux_update.zip` 是基于 `a6a1ed9` 的源码增量，不包含数据、权重、训练结果。已同步这些修改到服务器 Git 的情况下无需重复解压；不要将 Bash 单独更新而漏掉 Python 文件。

Windows PowerShell：

```powershell
scp "E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_k32_linux_update.zip" feng@10.76.0.64:/home/feng/HouCode/
```

Linux（解压前备份服务器未提交代码，按提示确认覆盖）：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight &&
test -f /home/feng/HouCode/overnight_k32_linux_update.zip &&
unzip /home/feng/HouCode/overnight_k32_linux_update.zip
```

训练 → 六方法五来源测试 → 四/五指标图 → 各外部来源案例图：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --stable-selection \
  --candidate-knots 32 \
  --resize-candidate-warm-start \
  --warm-start-checkpoint outputs/checkpoints/overnight_stable_3090_r1.pt \
  --epochs 32 --proposal-epochs 12 \
  --train-size 1500 --val-size 500 --batch-size 32 \
  --mse-tolerance 5e-5 \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --prepare-real-data \
  --run-name overnight_stable_k32_3090_r1 \
  --benchmark-profile full \
  --real-samples-per-dataset 20 \
  --visual-samples-per-dataset 6
```

预览命令可追加 `--dry-run`；它不运行 Python、不验证权重内容，也不启动训练。

覆盖 Synthetic、UJI、Natural Earth、USGS、IndustrialOffset。工业等距线是 CAD 母轮廓生成的半合成数据，不是实测工业数据。训练仍使用历史合成混合分布，三个实测/地理来源仅作验证，工业等距线仅作独立测试，不修改已有拆分。默认规模为 42 条合成 + 每外部来源 20 条，共 122 条 × 六方法；每外部来源 6 张六方法图和 6 张 Ours 图。组优先抽样不等于同一母轮廓的所有曲线统计独立。

输出目录：

```text
outputs/checkpoints/overnight_stable_k32_3090_r1*.pt
outputs/logs/overnight_stable_k32_3090_r1/
outputs/comparisons/overnight_stable_k32_3090_r1/
outputs/figures/overnight_stable_k32_3090_r1/
  four_metrics/       # 原四指标 + 新 five_metrics_input/reference PNG
  ours_cases/
  six_method_real_cases/
```

中断后同 run 续训：保持上述参数，但删去 `--resize-candidate-warm-start` 与 `--warm-start-checkpoint ...`，加入 `--resume-run`。不要把缩容初始化再做一遍。

只测试已训练好的 K32：

```bash
bash scripts/run_v16_1070_overnight_linux.sh \
  --candidate-knots 32 \
  --checkpoint outputs/checkpoints/overnight_stable_k32_3090_r1.pt \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --prepare-real-data \
  --run-name overnight_stable_k32_eval_r1 \
  --benchmark-profile full \
  --real-samples-per-dataset 20 \
  --visual-samples-per-dataset 6
```

所有输出使用新名称，不覆盖已完成的 K64 实验。尚未在服务器完成新 K32 训练前，不宣称精度或速度改善。

## 5. 本地验收记录

- 相关回归合计 **329 项通过**：训练/权重迁移 106 项，容量入口/预检/评测/绘图/数据覆盖 223 项；Bash 语法检查与 `git diff --check` 通过。
- K64 → K32 权重迁移、有限前向/拟合、保存重载、微型 Proposal/Joint 训练、同容量续跑与拒绝跨容量续跑均有测试；旧配置保留。
- 六方法单独绘图也默认拒绝容量不一致。若显式 `--allow-unequal-capacity` 做消融，会标注容量数值与 `UNEQUAL-CAPACITY ABLATION` 诊断水印；Ours-only 不使用数值基线预算。
- 实际运行 `overnight_stable_k32_pipeline_smoke_20260919`：4 条训练、5 条验证，2 epochs（1 Proposal + 1 Joint），CPU quick 对照预算。64 → 32 迁移时 123 个张量精确复制，1 个 query 张量插值、1 个固定 buffer 重建；未修改源 checkpoint。
- 流程完成 25 条曲线 × 六方法 = 150 条评测，五类来源齐全；每条记录均有有限最大单点平方误差，所有方法 `K ≤ 32`。输出 13 张 PNG：四/五指标各含输入和参考两种口径，共 4 张；Ours 四来源案例及总览 5 张；六方法四来源案例 4 张。已目视核对五指标图与等距线六方法图。
- 此短测 Joint 后仅 5 条验证曲线的部署通过率为 20%，checkpoint 标注 `target_not_met`，所有图保留诊断水印。它证明链路能运行，**不证明 K32 训练效果优于 K64**；缩容后需要正式适配，不应把这个 smoke checkpoint 用于汇报或正式部署。

诊断图：[五指标示例](../outputs/figures/overnight_stable_k32_pipeline_smoke_20260919/four_metrics/v16_published_methods_five_metrics_input.png)、[等距线六方法示例](../outputs/figures/overnight_stable_k32_pipeline_smoke_20260919/six_method_real_cases/IndustrialOffset__industrial_offset_keyway_bore_v007_o01_m0p0400.png)。这些诊断权重、数据和图不包含在源码更新包中。
