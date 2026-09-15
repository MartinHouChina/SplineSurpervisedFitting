# v16 可行子集教师：训练、部署与 3090 一条龙

当前修正针对已观察到的失效：全保留 `Kc=72` 候选时拟合通过率接近 100%，但 Joint 一次性掩码的 MSE 常超过 `1e-4`，节点数也可能低于真正可行的数量。真源节点 `K*` 与真节点匹配掩码是几何标签，却不是**预测候选位置和预测参数域上的可行删除方案**；不能直接把它们当成该域中的最少可行节点教师。

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
