# v16 可行子集教师：训练、部署与 3090 一条龙

最新推荐先运行 [Keep 状态交互恢复档](v16_keep_state_recovery.md)：完整 r2 热启动、Kc=56、Proposal 8 + Joint 24，恢复显式 Keep 状态嵌入并修正 Teacher/decoder 参数域对齐。以下 Kc72 结果和命令保留为历史对照；Teacher 的离线缓存原则仍适用。

当前修正针对已观察到的失效：全保留 `Kc=72` 候选时拟合通过率接近 100%，但 Joint 一次性掩码的 MSE 常超过 `1e-4`，节点数也可能低于真正可行的数量。真源节点 `K*` 与真节点匹配掩码是几何标签，却不是**预测候选位置和预测参数域上的可行删除方案**；不能直接把它们当成该域中的最少可行节点教师。

## 最近训练结果的判读与下一步

已拉取的 `source K=4..56, Kc=72` 检查点，其离线教师在训练缓存上平均 `K=33.03`、通过率 `99%`；后期 Joint 在教师强制掩码下约 `97%` 可行，但自由一次性掩码常仅 `20%..32%` 可行。因此首先要解决 **Count/Keep 校准及节点组合选择**，不是增加最大候选数。最新快速测试仅每个合成 K 抽一条、每个真实来源抽五条，不能当论文估计；其中合成数据原始 Ours 为 `MSE=5.32e-4, pass=35.8%, K=46.0`，数值核验修复后为 `MSE=5.25e-5, pass=98.1%, K=54.19`。后者主要靠追加节点达标，不能证明网络已学会最简选择。同一快速测试中 Park 的 `K=41.47`、Dung 的 `K=38.57`，因此当前方法在节点数上没有优势。

训练日志显示 Proposal 64 epoch 约 125 分钟、Joint 64 epoch 约 168 分钟，另有离线教师与评测时间。新 `Kc=56` 配置只作快速架构验证：训练源 K 上限改为 44，保留 12 个冗余候选槽；仍对测试 K=4..56 单独做越界压力测试。旧 `Kc=56` 检查点的部分简单曲线确曾使用更少节点，但其整体验证通过率约 `82%`，且当时 Dung/Luo 防坍缩适配尚未修正，不能直接把旧图当作改进结论。新试验保留 Proposal+Joint 和离线教师，尝试认证真节点优先的可行教师、Count/Keep 梯度耦合；两者都必须通过相同测试样本的**原始一次性** MSE/pass/K 检验。快速配置与命令另见 [Kc56 快速试验](v16_kc56_fast_pilot.md)。

一个仅用于检查教师构建开销的小试验：同一旧 `Kc=56` Proposal、8 条固定认证合成曲线、阈值 `1e-4`、CPU 上，真节点锚定教师耗时 `2.61 s`、平均 `K=37.12`、通过率 `87.5%`；完整贪心教师耗时 `8.70 s`、平均 `K=36.38`、通过率同为 `87.5%`。因此当前锚定构建在此样本上约快 `3.3×`，但平均节点反而多 `0.75`，不能据此断言 3090 上整轮训练一定变快或一次性部署结果改善。锚定缓存中的旧 `teacher_greedy_count` 等字段仅用于兼容，不是实际贪心搜索轨迹。

## 数据进入网络的顺序

训练只用带真参数、真节点和真 `K*` 的认证合成曲线：source 内部节点 `K=4..56`，每条输入 192 个有序点，最大候选内部节点 `Kc=72`，工程误差单位为 mean squared Euclidean（MSE）。UJI、Natural Earth、USGS、IndustrialOffset 只作留出验证/测试，绝不进入 optimizer step。

```text
有序观测点 Q
  -> GeometryEncoder -> 几何序列
  -> ParameterHead  -> 单调预测参数 t_pred
  -> CandidateKnotHead -> Kc=72 个候选节点 U_prop 与候选特征
  -> Selector/Count 校准 -> 一次性 KeepMask
  -> Survivor decoder -> 存活节点位置与部署参数
  -> 标准三次 B 样条 refit -> 拟合曲线、MSE、最终 K
```

真参数/真节点继续监督 Proposal 与参数头；离线教师只负责**在当前 Proposal 候选框架上**给出可行 KeepMask、数量和删除风险，不代替真几何标签。Joint 将真 `K*` 只作为源复杂度诊断，运行参数为 `--synthetic-count-role reference_only`；预测域的可行节点数可能高于或低于源 `K*`。

## 两阶段训练

1. **Proposal（默认 64 epoch）**：监督真参数、有序一一匹配的候选位置、多尺度节点召回和全候选拟合。先训练出可用于搜索的候选集；`Kc=72` 比最大真 `K*=56` 多 16 个槽位。
2. **冻结 Proposal，离线构建教师缓存**：针对固定 Joint 合成样本与固定 Proposal 输出的 `t_pred/U_prop`，从全候选拟合开始搜索满足 `MSE<=1e-4` 的子集，保存槽位 KeepMask、可行 `K`、拟合 MSE 和删除风险。教师必须记录失败/回退，而不是把不可行样本伪装成可行标签。缓存只对相同数据样本、Proposal 权重和阈值有效。
3. **Joint（默认 64 epoch）**：固定 GeometryEncoder、ParameterHead 与 CandidateKnotHead（其 Joint 学习率为 0），只更新 Selector/Count 校准与存活节点解码器，避免教师缓存因 Proposal 漂移失效。Keep 和数量目标改为缓存的可行子集；真 `K*` 只作诊断。分别看教师强制掩码、网络一次性掩码的 MSE/pass/K，以及真节点召回，才能区分 Teacher、Selector、decoder 哪一环失效。

离线缓存要求 Joint 样本身份不变。一条龙脚本固定训练样本并使用 `--no-resample-train-each-epoch`；不要把旧的在线重采样指令与该缓存混用。缓存搜索发生在训练阶段，**不计入**一次性部署网络耗时。

离线教师采用独立的 `--feasible-teacher-batch-size 8`，Joint 优化器仍使用 Batch=64。此设置只限制每次数值求解的显存，不改变教师标签；Kc=72 的贪心搜索仍可能耗时数小时。当前缓存只在全部样本完成后发布，搜索中断不会留下可续算的部分缓存。

候选节点过密时，B 样条拟合矩阵可能秩亏。教师的批量逐节点删除和单曲线 refit 现在对 CUDA 秩亏报错或非有限解改用 CPU 的 SVD 最小二乘，并保持原 MSE 定义；异常样本会增加离线教师或最终 refit 时间，但不改变一次性网络部署的前向次数。

## 部署与方法命名

原始网络部署仍是一次前向、一次最终标准 B 样条 refit，记作 `ours_one_shot`。它能报告工程阈值通过率，但仅靠预测概率和 `TopK` **不能数学保证**每条曲线都满足阈值。

如启用部署后核验/修复，先计算该 refit 的 MSE；超过阈值才追加候选、重新拟合或回退全候选等。该结果应单独记作 `ours_verified_repair`，报告修复率、失败率、增加的节点、额外 refit 次数和完整 wall time。不要把它画成“一次性预测”，也不要把它的 MSE/K 与 `ours_one_shot` 时间混合。即使全候选不足以通过，仍应明示失败。

论文对比的原始网络、核验修复、Greedy 和论文方法使用同一 MSE 定义、同一测试样本和一致的计时范围；`quick` 采样与缩短基线迭代仅用于跑通流程，不能当作正式论文估计。

## Linux RTX 3090 一条龙

首次跑通（新 `run-name`，包含训练、checkpoint 检查、六方法快速诊断表、四指标图和真实案例图）：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --prepare-real-data \
  --device cuda \
  --benchmark-profile quick \
  --run-name candidate_selection_v16_mse1e-4_sourcek56_kc72_feasible_teacher_linux_r1
```

若 Proposal 已完成、离线教师构建阶段中断，先同步最新代码到服务器，并确认同名 `.last.pt` 和 `.proposal.pt` 存在。保持原 `run-name` 续跑；不要改名重训，也不要重新准备已有数据：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --resume-run \
  --device cuda \
  --benchmark-profile quick \
  --run-name candidate_selection_v16_mse1e-4_sourcek56_kc72_feasible_teacher_linux_r1
```

`--resume-run` 从 `.last.pt` 的下一代继续，保留原 best Proposal 和输出路径，不截断旧日志；原训练配置与真实数据 manifest 指纹必须匹配。`--feasible-teacher-batch-size` 是仅影响缓存构建显存的运行参数，允许在续跑时改变；其它训练参数不能随意变更。

若训练已经达到原定总轮数，Linux 一条龙现在先核对配置和模型工件，再直接进入未完成的评测/绘图阶段，不会报“已完成请求轮数”并退出。完整评测须通过配置、代码、数据及记录指纹检查才能复用；部分评测使用原生 journal 续跑。派生图像通过 `.pipeline-stage.json` 记录产物；中断重试写入 `attempt_*` 子目录，不删除或无校验覆盖旧结果。代码或数据已变化的旧测量仍会被安全拒绝，不能把不同实验混成一次续跑。

批量评测额外传入 `--include-verified-ours`，报告中会出现单独的
`ours_verified` 数值修复行；原 `ours` 行始终是未修复的一次性结果。
四指标图仍只画原来的六方法，修复结果保留在逐样本记录与汇总表中。
即使检查点结构审计通过，若实测通过率未达参考值，一条龙仍把表图
标为 `DIAGNOSTIC NOT FINAL`；`quick` 模式无条件采用诊断标记。

正式样本规模将 `--benchmark-profile quick` 换成 `full` 并使用另一个全新 `run-name`；这会重训模型，不应把两次训练的结果当作同一 checkpoint 的两次评测。已有 checkpoint 的正式复测应调用独立 benchmark 脚本，并指向已生成的 `.pt`。可用 `--dry-run` 先打印全部命令。若服务器已有兼容 Proposal checkpoint，可加 `--init-checkpoint <旧.proposal.pt>`，它只初始化 Proposal，不复用旧 Joint 结果。

每次运行的主要产物是：

```text
outputs/checkpoints/<run>.pt                    最佳 Joint
outputs/checkpoints/<run>.proposal.pt           最佳 Proposal
outputs/checkpoints/<run>.proposal.final.pt     Proposal 阶段末审计
outputs/checkpoints/<run>.last.pt               最近恢复状态
outputs/checkpoints/<run>.history.json          逐 epoch 指标
outputs/teachers/<run>/                        离线可行子集缓存
outputs/logs/<run>/                            训练/评测日志
outputs/comparisons/<run>/<tag>/                原始逐样本与汇总指标
outputs/figures/<run>/<tag>/                    四指标及真实案例 PNG
```

同名产物已存在时脚本拒绝覆盖。`quick` 输出带 `diagnostic_` 标记；只有完整跑完、checkpoint/数据指纹审计通过且实测指标达标时，才可用于正式汇报。代码修改不等于通过率已经改善；需在 3090 上重新训练并与旧 checkpoint 同样本比较。
