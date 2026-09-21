# M16 / M32 实际结果审计（2026-09-21）

## 结论

审计对象是拉回提交 `614e839` 中的实际服务器产物，而非实验计划。M16 明显不够可靠；M32 相对旧 Anchored best 少保留约 0.89 个内部节点，但 122 条测试通过数从 85 降到 77。**这轮结果不支持“加深 + 峰值损失已改进总体性能”，也不支持把下一轮预算直接用于延长相同 Joint。** M32 的主要可见问题是紧凑选择/子集解码后的可行性下降；M16 同时存在严重的 dense 几何/容量瓶颈。

小批同输入、同 mask、同步参数坐标变换诊断表明：chord 对部分手写/地理曲线有益，但对工业程序曲线和 M32 合成样本通常更差。不能直接全局替换为 chord，也不能把本次测试诊断当成训练标签或逐样本选模型的依据。

主要证据：

- [M16 comparison](../outputs/comparisons/universal_m16_m32_3090_r1_m16/comparison.json)、[M32 comparison](../outputs/comparisons/universal_m16_m32_3090_r1_m32/comparison.json)。
- [M16 history](../outputs/checkpoints/universal_m16_m32_3090_r1_m16.history.json)、[M32 history](../outputs/checkpoints/universal_m16_m32_3090_r1_m32.history.json)。
- [M16 训练日志](../outputs/logs/universal_m16_m32_3090_r1_m16/train_fresh.log)、[M32 训练日志](../outputs/logs/universal_m16_m32_3090_r1_m32/train_fresh.log)，以及同目录 `inspect_checkpoint.log`、`preflight.json`。
- [旧 Anchored 12+48 comparison](../outputs/comparisons/overnight_anchored_k32_p12_j48_3090_r1/comparison.json)。
- [保存输入的参数化配对诊断](../outputs/diagnostics/m16_m32_parameterization_20260921_saved_inputs.json)。

## 1. 比较口径与实际执行

M16、M32、旧 Anchored 的 122 个 `ours` 样本 dataset/sample ID 顺序及 `metadata.sample_content_sha256` 顺序一致。完整比较中每个方法使用相同的 192 个归一化输入点、同一 `MSE <= 5e-5` 标准、最终 endpoint-constrained float64 LS；无有限拟合失败被排除。Synthetic 为 K4..24 每 K 两条，共 42 条；四个外部来源各 20 条。

M16 每方法最大内部 K=16（完整三次 knot vector 上限 24）；M32 每方法上限 K=32（完整向量上限 40）。因此 M16 与 M32 **不是相同容量下的算法比较**。M16 训练有标签合成源为 K4..16，M32 为 K4..24；训练分布与 warm-start 缩容方式也不同，不能把二者差异全部归因于网络容量。M16 的 K17..24 测试保留在结果中，不应悄悄删去。

两模型均从旧 Anchored **best epoch13** 初始化；M16 的候选查询由 K32 缩到 K16，不是无损模型转换。实际各训练 4 Proposal + 20 Joint；Joint 前 4 轮冻结 Proposal 几何。candidate/selection/decoder 额外层数均为 2，peak weight=0.05、peak squared-error target=5e-4、tail fraction=0.05。训练仍是合成混合（配置 simple 35%、historical 40%、shape 25%，每轮实际计数有随机波动），真实曲线仅用于验证/测试；IndustrialOffset 是程序 CAD 等距线，不是实测工业数据。

## 2. 122 条原始 benchmark：总体与分来源

以下平均数按曲线加权，不是五个来源等权。时间是服务器产物中的每曲线完整方法运行 median 再取均值；不是本机诊断计时，也不是仅网络前向。运行时间跨轮变化还可能受机器状态影响。

| 模型 | best / last epoch | 通过数 / 122 | 通过率 | 平均内部 K | 平均 MSE | 完整 ms/曲线 | history epoch 时间合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 Anchored K32 | 13 / 60 | 85 | 69.67% | 26.189 | 7.017e-5 | 11.023 | 179.5 min |
| M16 | 24 / 24 | 41 | 33.61% | 15.041 | 8.335e-4 | 14.483 | 60.71 min |
| M32 | 5 / 24 | 77 | 63.11% | 25.303 | 7.633e-5 | 14.853 | 80.24 min |

M32 与旧 Anchored 配对：共同通过 76 条；旧通过而新失败 9 条，新通过而旧失败 1 条。共同通过的 76 条上，平均 K 为旧 25.105、新 24.092。可陈述“这部分曲线减少约 1 个节点”，不能省略总通过率下降的代价。新 M32 测试使用的是首个 Joint epoch5，不是 epoch24。

| 模型 | 来源 | 通过数/n | 平均 K | 平均 MSE | 完整 ms | 原参考点通过率 |
|---|---|---:|---:|---:|---:|---:|
| M16 | Synthetic | 7/42 | 15.333 | 1.647e-3 | 14.491 | N/A |
| M16 | UJI | 14/20 | 14.300 | 5.503e-5 | 14.464 | 65% |
| M16 | NaturalEarth | 2/20 | 15.700 | 6.735e-4 | 14.492 | 10% |
| M16 | USGS | 3/20 | 15.450 | 8.021e-4 | 14.491 | 15% |
| M16 | IndustrialOffset | 15/20 | 14.100 | 9.517e-5 | 14.469 | 75% |
| M32 | Synthetic | 26/42 | 26.738 | 4.252e-5 | 14.990 | N/A |
| M32 | UJI | 18/20 | 21.850 | 1.775e-5 | 14.933 | 80% |
| M32 | NaturalEarth | 11/20 | 28.150 | 1.482e-4 | 14.696 | 60% |
| M32 | USGS | 6/20 | 27.950 | 1.626e-4 | 14.703 | 30% |
| M32 | IndustrialOffset | 16/20 | 20.250 | 4.779e-5 | 14.793 | 80% |

`MaxSqErr = max_i ||C(t_i)-Q_i||²`，不取平方根，也不等于最大曲线 MSE，更不是连续曲线最大距离或 Hausdorff 距离。下表 mean/P95/max 汇总的是每条曲线的离散单点最大平方误差。

| 模型 | 来源 | MaxSqErr mean | P95 | max | 原参考点 MaxSqErr max |
|---|---|---:|---:|---:|---:|
| M16 | Synthetic | 6.065e-3 | 1.338e-2 | 1.943e-2 | N/A |
| M16 | UJI | 6.103e-4 | 1.873e-3 | 2.743e-3 | 3.547e-3 |
| M16 | NaturalEarth | 3.643e-3 | 1.245e-2 | 3.001e-2 | 3.357e-2 |
| M16 | USGS | 4.593e-3 | 1.274e-2 | 1.809e-2 | 1.975e-2 |
| M16 | IndustrialOffset | 6.405e-4 | 2.757e-3 | 3.434e-3 | 3.466e-3 |
| M32 | Synthetic | 4.173e-4 | 1.007e-3 | 1.177e-3 | N/A |
| M32 | UJI | 2.871e-4 | 8.311e-4 | 8.488e-4 | 1.271e-3 |
| M32 | NaturalEarth | 1.045e-3 | 3.272e-3 | 8.757e-3 | 9.089e-3 |
| M32 | USGS | 1.092e-3 | 3.750e-3 | 3.860e-3 | 4.548e-3 |
| M32 | IndustrialOffset | 4.636e-4 | 2.156e-3 | 2.164e-3 | 2.168e-3 |

新 M32 的整体 mean MaxSqErr=6.172e-4，旧 Anchored 为 5.862e-4。UJI 有改善、Synthetic 最坏峰值下降，但总体峰值均值及多数来源均值没有改善；深度、peak、训练长度同时变化，不能据此孤立判定 peak loss 有效或无效。

## 3. M16 不只是“测试超出了 K16”

| 合成 source K | n | M16 通过数 / 平均 K | M16 平均 MSE | M32 通过数 / 平均 K | M32 平均 MSE |
|---|---:|---:|---:|---:|---:|
| 4..10 | 14 | 7 / 14.000 | 5.752e-5 | 6 / 19.143 | 5.238e-5 |
| 11..16 | 12 | 0 / 16.000 | 1.108e-3 | 11 / 29.417 | 1.684e-5 |
| 17..24 | 16 | 0 / 16.000 | 3.442e-3 | 9 / 31.375 | 5.314e-5 |

M16 在**训练有标签源范围内** K11..16 也 0/12；这 12 条及 K17..24 的 16 条均保留满 16。M16 总计 70/122 条全保留，M32 为 7/122。仅加强删节点不可能解释/修复这些满容量失败；但这也不是数学证明“任意 K16 样条都不可能拟合”。候选布局、参数化、缩容迁移、几何学习和输入采样都可能参与。

M32 低 K4..10 平均仍保留 19.14，测试 Synthetic 的数量 MAE/bias 均为 +12.738；M16 的整体 bias 仅 +1.333 是低 K 过保留和高 K 容量截断相互抵消，数量 MAE 仍为 4.762。不能把接近目标的平均 K 当作准确学会简化。

## 4. Best/last 与 dense/selection 分解

验证集 n=596（500 合成 + 三来源各32），与上述122条测试不是同一集合。Dense 使用 Proposal 的参数/全候选；deployment 经过 KeepMask 与子集几何解码，因此二者差距不能全部叫作“删掉了必要节点”。

| 模型/epoch | 阶段 | 验证 dense 通过率 | 验证 deployment 通过率 | K | 最差来源 deployment |
|---|---|---:|---:|---:|---:|
| M16 e4 | Proposal 结束 | 45.81% | 45.81% | 16.00 | 12.50% |
| M16 e5 | 冻结 Proposal | 45.81% | 30.54% | 15.06 | 6.25% |
| M16 e24 best/last | Joint | 46.31% | 37.42% | 14.86 | 9.375% |
| M32 e5 best | 冻结 Proposal | 85.91% | 68.96% | 26.61 | 53.125% |
| M32 e8 | 冻结 Proposal 结束 | 85.91% | 52.68% | 24.85 | 28.125% |
| M32 e24 last | Joint | 86.41% | 51.17% | 23.91 | 18.75% |

M32 e5→e8 的 dense 完全不变，deployment 已下降16.28个百分点，这是独立于 Proposal 漂移的选择/子集解码退化证据。e24 的来源情况：

| 模型 | 来源 | Dense 通过率 | Deployment 通过率 | K |
|---|---|---:|---:|---:|
| M16 | Synthetic | 46.20% | 35.40% | 14.810 |
| M16 | UJI | 90.625% | 87.50% | 14.094 |
| M16 | NaturalEarth | 9.375% | 9.375% | 15.719 |
| M16 | USGS | 40.625% | 46.875% | 15.500 |
| M32 | Synthetic | 86.40% | 50.40% | 24.530 |
| M32 | UJI | 100% | 84.375% | 16.625 |
| M32 | NaturalEarth | 75% | 18.75% | 23.156 |
| M32 | USGS | 84.375% | 62.50% | 22.250 |

M32 NaturalEarth dense 从 e5 的81.25%降到75%，说明其 Proposal 也有一些退化，但不足以解释 deployment 的53.125%→18.75%。M16 USGS deployment 反而高于 dense，进一步说明“dense−deployment”并非严格的子集删除损失分解。

两模型均为 `target_not_met`、`DIAGNOSTIC NOT FINAL`。M16 worst dense/deployment=9.375%；M32 best worst dense=81.25%、deployment=53.125%，且首 Joint 的简化 curriculum 尚未成熟。M32 `.last.pt` 没有对应全量六方法测试，不能用 best 测试数字代替 last 结果。

## 5. 参数化偏置：训练统计不是全域证明

训练启用了 parameter-trust 和 chord counterfactual（weight=0.25），默认真参数监督 weight=0.1 只对具有真参数的合成样本有意义。trust `initial=0.25` 是初始化配置，不代表从旧 checkpoint warm-start 后的实际 trust 始终为0.25。

M32 e24 训练：dense learned MSE=9.278e-6、chord counterfactual=2.037e-5；deployment learned=4.016e-5、chord=6.763e-5。history 中 `*_parameter_counterfactual_win_fraction` 计算的是 **learned MSE <= chord MSE**，不是 chord 胜率；e24 Proposal/deployment 分别为90.8%/97.2%。M16 e24 分别82.27%/86.0%；M16 dense 的总体均值却是 learned 1.995e-4、chord 1.977e-4，说明多数曲线胜率与尾部均值可能给出不同印象。

这些是训练混合数据的 epoch 聚合，不是逐来源 held-out 参数化消融。M32 e24 subset trust 训练均值约0.837，但该值不是校准后的“learned 更优概率”；不能用高 trust 证明真实数据上已可靠选择参数化。

## 6. 独立的小批同输入参数化诊断

新增 [诊断脚本](../scripts/diagnose_v16_parameterization.py) 与 [专用测试](../tests/test_diagnose_v16_parameterization.py)。主要产物为 [saved_inputs JSON](../outputs/diagnostics/m16_m32_parameterization_20260921_saved_inputs.json)，内含 checkpoint/code/source SHA256、运行环境、精确归一化输入、mask、两套参数/节点、逐例 MSE/MaxSqErr 和汇总。

范围严格分开：

- 外部24条：直接读取服务器保存的 [M32 图例 JSON](../outputs/figures/universal_m16_m32_3090_r1_m32/ours_cases/deployment_visualizations.json)，每来源6条。M16/M32共享这同一份点数组；未根据本次诊断误差挑选。这是服务器另行选出的图例批次，不是122条 benchmark 的子集：只有1条NaturalEarth、3条IndustrialOffset与benchmark重叠，其内容hash匹配。
- 合成8条：从benchmark的K有序ID中等距选取（K4、7、10、13、15、18、21、24），本机按保存配置再生，再生内容hash与服务器均不匹配。因此只能作为**本机再生合成补充诊断**，不宣称原服务器8条精确重放。精确再生输入已保存，M16/M32在这8条上仍完全配对。
- 未做训练、checkpoint选择或逐曲线oracle路由；不测连续曲线误差，不把小批结果替代全量benchmark。

每个模型每条曲线只编码并选择一次 mask。Final arm 固定该mask及最终节点，Dense arm固定Proposal全候选；将同一个单调分段线性 `t_learned -> t_chord` 同时用于参数与内部节点，然后使用相同 float64 endpoint LS 重拟合。不是“只换t不换U”，也没有重新选择、增删或另行移动节点。非仿射坐标变换会改变B-spline函数空间，本诊断**不声称曲线形状被精确保留**。

外部图例 final-arm 结果：

| 模型 | 来源(n=6) | learned→chord 通过数 | chord MSE降低条数 | learned→chord 平均MSE |
|---|---|---:|---:|---:|
| M16 | UJI | 5→5（救1、伤1） | 4 | 3.206e-5→1.743e-5 |
| M16 | NaturalEarth | 1→3 | 6 | 1.092e-3→8.336e-4 |
| M16 | USGS | 0→3 | 6 | 1.439e-4→9.513e-5 |
| M16 | IndustrialOffset | 3→3 | 1 | 6.955e-5→9.182e-5 |
| M32 | UJI | 6→6 | 4 | 1.442e-5→8.697e-6 |
| M32 | NaturalEarth | 4→4 | 6 | 1.640e-4→1.359e-4 |
| M32 | USGS | 5→5 | 5 | 3.278e-5→2.581e-5 |
| M32 | IndustrialOffset | 4→4 | 1 | 6.194e-5→7.383e-5 |

Dense arm：M16 NaturalEarth通过0→3、USGS2→3；M32 NaturalEarth4→5，其余外部来源通过数不变。M32 UJI final MSE均值虽降低，MaxSqErr均值反而从1.679e-4升到2.083e-4；“MSE变好”不等于“峰值同时变好”。

本机再生合成8条：M32 chord **0/8** 条MSE更低，final通过5→2、dense7→4；final均值4.297e-5→7.464e-5。M16 chord赢4/8，final/dense通过均2→2。足以否定“现有证据支持chord全局替换”，但样本过少，不足以确定新的训练权重或域路由策略。

复现命令（输出必须是新文件）：

```powershell
python scripts/diagnose_v16_parameterization.py `
  --checkpoint outputs/checkpoints/universal_m16_m32_3090_r1_m16.pt `
  --checkpoint outputs/checkpoints/universal_m16_m32_3090_r1_m32.pt `
  --cases-json outputs/figures/universal_m16_m32_3090_r1_m32/ours_cases/deployment_visualizations.json `
  --comparison outputs/comparisons/universal_m16_m32_3090_r1_m32/comparison.json `
  --allow-content-mismatch --samples-per-dataset 8 --device cpu `
  --output outputs/diagnostics/m16_m32_parameterization_new.json
```

`--allow-content-mismatch`仅显式允许诊断；默认hash不匹配失败关闭。它不让结果成为正式benchmark。4项专用测试通过：节点完整warp/空集/单节点、chord不被预设为更优、两checkpoint同输入且不修改权重/覆盖输出、hash不匹配默认拒绝。

本机数据重建另有 [local补充报告](../outputs/diagnostics/m16_m32_parameterization_20260921_local.json)：按可用benchmark ID共36条（8/8/4/8/8），**36条重建content hash均未匹配服务器**，不能当作原benchmark重跑。尤其本机USGS test只有201条，服务器1420条，原20条中本机仅4条存在；缺失ID已记录。hash差异可能来自数据/预处理/运行库差异，但本审计没有证明根因。正式复跑必须使用服务器完整数据根与产物中的manifest/hash协议。

## 7. 基线必须区分 native adaptation 与 threshold-safe adaptation

六方法结果是仓库适配实现，不能写成论文作者原代码复现。Dung/Kang/Luo 默认启用了有限容量的 common-MSE safeguard：native公共refit未达标时进行 residual augmentation/uniform fallback，并把全部修复工作计入完整时间。`native_baseline_summary`实际保存逐样本native/最终结果，应同时披露。

| 容量组 | 方法 | native通过/122 | safeguard后通过/122 | 调用修复条数 | native→最终平均K |
|---|---|---:|---:|---:|---:|
| M16 | Dung | 4 | 63 | 118 | 9.582→12.590 |
| M16 | Kang | 36 | 38 | 86 | 14.951→14.967 |
| M16 | Luo | 5 | 48 | 117 | 2.041→13.426 |
| M32 | Dung | 4 | 94 | 118 | 11.443→17.943 |
| M32 | Kang | 77 | 83 | 45 | 25.672→26.598 |
| M32 | Luo | 17 | 85 | 105 | 4.098→19.861 |

“native”仍只是修正后的仓库适配在可选公共MSE修复之前的结果，不等于精确论文实现。尤其不能把Dung的94/122或Luo的85/122全部归功于其原生方法，也不能把native公共MSE低通过率用作原论文固有缺陷的证据。

原始M32 benchmark中，UJI的Liang适配为100%/K8.85/8.545ms，Dung threshold-safe为100%/K7.35/107.793ms，Ours为90%/K21.85/14.933ms；工业程序曲线对应Liang100%/K11.30/11.002ms、Dung100%/K10.10/132.845ms、Ours80%/K20.25/14.793ms。Ours没有全面时间/节点/精度优势；基线修复开销和原参考点精度也不能省略。

## 8. 可支持的下一步判断

1. M16先修复dense可行性/缩容与候选几何，不能只加压删节点；K11..16失败必须单列，不可全归为K17..24超范围。
2. M32优先验证紧凑选择与子集解码的稳定性及训练-部署一致性。冻结Proposal时期仍明显下降，单独“永久冻结Proposal”缺乏充分依据。
3. 参数化优化应保留learned/chord坐标一致的配对诊断，并在验证集做预先规定的消融。手写/地理小批可见收益，但不能绕过合成与工业反例或峰值恶化。
4. 深度、peak、候选容量、数据源K范围必须分开做控制变量实验；这次两模型不是足以判定每项有效性的消融。
5. 继续报告失败分母、共同通过配对K、离散峰值及原参考点误差；不把diagnostic checkpoint改标为formal，也不根据测试逐条切换M16/M32/chord来抬高汇总成绩。
