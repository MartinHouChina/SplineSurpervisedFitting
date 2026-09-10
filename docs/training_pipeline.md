# v16 训练流程

本文只描述当前 `scripts/train_v16.py`。旧版 v8–v15 仅用于说明设计来源，不再作为当前训练入口。

## 1. 任务与符号

输入是一条按曲线顺序排列、归一化到统一尺度的点序列：

\[
Q\in\mathbb R^{M\times D},\qquad D\in\{2,3\}.
\]

网络预测参数序列、一个高召回候选内部节点集，以及一次性保留的节点子集。最终误差统一定义为平均平方欧氏距离：

\[
\operatorname{MSE}=\frac1M\sum_{i=1}^{M}\lVert \hat Q_i-Q_i\rVert_2^2.
\]

当前默认阈值为 `2.5e-5`。它是 MSE，不开平方，也不再除以坐标维数。

## 2. 容量一定要分清

`--candidate-knots Kc` 表示 **内部候选节点数**。对三次开区间 B 样条：

\[
K_{\mathrm{full}}=K_c+8,\qquad N_{\mathrm{ctrl}}=K_c+4.
\]

因此：

| 命令 | 内部候选 | 全节点向量（全部保留时） | 控制顶点（全部保留时） |
|---|---:|---:|---:|
| `--full-knot-vector-size 64` | 56 | 64 | 60 |
| `--candidate-knots 64` | 64 | 72 | 68 |
| `--candidate-knots 96` | 96 | 104 | 100 |

二者互斥。复杂真实曲线建议以 `Kc=96` 为主实验，并把 `Kc=64` 作为容量消融；主对照表中，所有允许设置容量的方法必须使用相同的**内部节点上限**。

## 3. 每批数据怎样进入网络

### 3.1 数据来源

训练批次由以下来源混合：

- 在线生成的三次 B 样条合成曲线；
- UJI Pen Characters；
- Natural Earth coastline；
- USGS contours。

`--real-fraction` 控制真实样本占比。真实数据没有真节点标签，训练依赖点重建、阈值可行性和在线子集比较，不伪造节点真值。

正式合成样本默认启用 `--certified-minimal-source`：先固定 K，在干净曲线上用 512 个均匀审计点验证完整源节点满足阈值、任意单节点删除均以 20% RMS margin 失败，随后才加入观测噪声。v16 的公开阈值是 MSE，因此证书接收 `sqrt(mse_tolerance)`；完整证明范围见[合成曲线最简性报告](synthetic_data_minimality_report.md)。认证成功的 Synthetic 样本同时返回：

- `target_params`：真采样参数；
- `target_internal_knots` 与 `target_internal_knot_mask`：补齐后的真内部节点及有效位；
- `target_internal_knot_count`：真内部节点数；
- 对应的 `*_valid` 标志。

这些标签只在 certified Synthetic 上有效。UJI、Natural Earth 和 USGS 的有效标志为 false，不进入真参数、真节点或真计数损失。

### 3.2 一次前向的数据流

1. `GeometryEncoder(Q)` 编码坐标、弦长位置、一阶差分和二阶差分。
2. `ParameterHead` 读取编码特征，输出严格递增的采样参数 `t`。
3. `CandidateKnotHead` 同时读取编码特征与参数位置信息，输出固定长度 `Kc` 的有序高召回候选 `U_prop`。
4. `InteractiveSelector` 为每个候选输出 `raw_importance`。
5. 曲线级阈值头输出一个自适应标量 `beta`；选择概率为

   \[
   p_j=\sigma\!\left((r_j-\bar r)-\beta\right).
   \]

   减去曲线内均值后，`raw_importance` 主要学习节点排序，`beta` 主要学习该曲线应保留多少节点，避免二者漂移造成不可辨识。
6. `mass_topk` 预测保留数量：

   \[
   \hat K=\operatorname{ceil}\!\left(
   \sum_jp_j+\sigma_s\sqrt{\sum_jp_j(1-p_j)}+K_s
   \right).
   \]

   训练开始默认 `sigma_s=0.25`、`K_s=2`、`K_min=4`。worst-source pass 达到 92% 时全速退火到 `sigma_s=0.05`、`K_s=0`，在 90%～92% 时以 0.5 倍速度继续退火，跌破 90% 时以 2 倍速度恢复。随后只做一次全局 Top-K；四个参数区间锚点用于防止节点全部挤在局部。空分区不会生成 anchor，也不会覆盖其他分区已经选出的有效 anchor。
7. `SurvivorRelocation` 只读取最终幸存节点及其局部特征，同时更新保留节点的位置；初始 `--relocation-blend 0`，先保持恒等映射，再由训练学习移动。
8. 用部署节点执行一次标准三次 B 样条最小二乘 refit，得到控制顶点和拟合曲线。

这里没有 `CountHead`，也没有部署时的 BIC、阈值扫描或反复试拟合。最终节点向量长度由 `beta + mass_topk` 一次性确定。

## 4. 两个训练阶段

### 4.1 Proposal 阶段

前 `--proposal-epochs` 轮先确保参数头和候选网络具有足够高的拟合可行性。显式目标包括全候选标准 B 样条 refit 的对数拟合惩罚、批内高误差尾部项，以及 certified Synthetic 的真参数和 proposal-to-true-node 覆盖监督。严格参数顺序、候选全域覆盖和最小间距仍由网络参数化本身保证；真实数据不使用伪造标签。

只有每个验证来源的 dense proposal 通过率达到

\[
R_{proposal}+\texttt{complexity-pass-margin}
\]

后，才进入联合筛选阶段。默认 margin 为 `0.02`。若候选集本身不够可行，脚本会停止，而不是让筛选头为 proposal 缺陷背锅。

### 4.2 Joint 阶段

同一批候选上会执行两类用途不同的评价：

- 原始 IID Bernoulli 采样只用于无偏 score-function 策略梯度，不经过最小节点数投影，也不允许成为结构化教师；
- 结构化教师以当前 Selector 排序为固定顺序，在前缀长度上做 training-only 的粗到细可行性搜索：先对所有曲线检查低 K 加密的二次网格；`Kc=96`、`Kmin=4`、7 个区间时约为 `4, 6, 12, 21, 34, 51, 72, 96`。certified Synthetic 额外检查自己的真 K；随后在首个粗网格可行边界内批量二分，再检查最佳前缀的 `K±2` 邻域以及局部增、删、交换组合。这些 mask 复用部署的最小数量与分区覆盖约束。

这些拟合均由网络当前预测的参数和节点位置在线得到，不使用固定离线教师缓存。若结构化教师池中存在满足阈值的子集，先选节点数最少者，再以 MSE 打破同节点数平局；若全部不可行，则选择 MSE 最低者作为临时教师。由于节点重定位和参数反馈会破坏 MSE 对前缀长度的严格单调性，这只是有限搜索预算内的 **coarse-to-fine approximate minimum feasible ranked-prefix teacher**，不构成全局最优证明。该搜索只参与训练；部署没有前缀搜索。

筛选损失包含：

- 加权 mask BCE：漏删教师要求保留的节点代价更高；
- 容量无关数量损失：在 `log1p` 空间监督 requested count score 与教师计数，取消旧式除以 `Kc` 的缩放，因此 Kc=20 和 Kc=96 的梯度尺度可比；
- certified Synthetic 真计数损失：部署尚不可行时只纠正 requested count 低于真值的情况；部署可行后才在 `log1p` 空间对称拉向 `max(K_true-0.25,0)`，使后续 `ceil` 恰好得到逐样本真计数；
- certified Synthetic 额外过预测损失：只对部署 MSE 已可行的有标签样本单边惩罚预测数高于真计数；
- 成对排序损失：教师保留节点分数应高于删除节点；
- 策略损失：低误差、少节点的子集得到更高回报；
- 节点重定位和参数反馈后的真实 refit 误差；
- certified Synthetic 的 proposal/final 真参数监督，以及统一真参数域中的节点监督：proposal 使用真节点到候选的定向覆盖损失；存活节点使用停梯度的一维单调最大基数一一匹配，再对匹配坐标施加 SmoothL1；
- 仅在安全曲线上启用的复杂度惩罚。

位置损失前，proposal 根据 `proposal_params -> true_params`、部署节点根据 `final_params -> true_params` 做分段线性可微 warp，避免直接比较不同参数化的节点值。一一匹配的组合索引由 detached 坐标求得，梯度只通过匹配后的预测坐标；多出来或缺少的未匹配节点交给真计数损失处理。这取代了容易发生多对一聚集的双向 Chamfer。

## 5. 为什么不会再次塌缩到 0 或全部节点

旧版固定 `p>=0.5` 对 logit 整体漂移非常敏感。当前版本使用四层保护与一个可逆课程：

1. 自适应 `beta` 逐曲线调节保留强度；
2. 概率质量而非逐元素阈值决定数量；
3. `safety sigma + safety knots + coverage bins + Kmin` 保护召回和全域覆盖；
4. certified Synthetic 的真计数、统一参数域和一一匹配节点位置监督约束“少而接近真值”；
5. 复杂度权重只在验证通过率稳定高于目标后逐步增加，跌破目标时自动回退，同时安全储备反向恢复。

`complexity_scale` 的默认升降周期由 `--complexity-ramp-epochs 10` 控制，最大为 `--complexity-max-scale 4`。这个机制应称为 **validation-pass feedback complexity multiplier**；它不是严格 Lagrangian/primal-dual 算法，也不是部署参数。以默认 90% target 和 2% margin 为例：pass≥92% 时复杂度全速增加且安全储备全速下降，90%≤pass<92% 时以 0.5 倍速度继续简化，pass<90% 时以 2 倍速度回滚。选择储备由 `--safety-anneal-epochs 10` 在 `2+0.25σ` 与 `0+0.05σ` 之间变化。

## 6. 验证与模型选择

每轮验证分别报告 Synthetic、UJI、Natural Earth 和 USGS：

- dense proposal 通过率；
- 一次性 deployment 通过率；
- worst-source 通过率；
- mean/P95 MSE；
- 最终内部节点数；
- 概率质量与 `beta`。

对 certified Synthetic 还报告：真计数均值、count MAE/bias、exact/within-one rate、参数 RMSE、knot-match precision/recall/F1 与 matched MAE。验证会先按预测参数与真参数的单调对应关系把最终节点 warp 到真参数域，再做一维一对一匹配；`--knot-match-tolerance 0.01` 是节点匹配容差，不是拟合 MSE 阈值。

checkpoint 选择遵循约束优化：

1. 未达到部署目标时，先提高 worst-source 通过率并降低尾部误差；
2. 达到目标但简化课程尚未成熟时，仍按可靠性排序；
3. 成熟且可行后，先偏好同时达到正式 count/F1/matched-MAE 与 dense-pass 门槛的 checkpoint；
4. 正式质量状态相同时，先偏好越过安全余量，再按 0.25 个节点宽度对 Synthetic count MAE 分档；同档内先最大化 knot-match F1、最小化 matched knot MAE，再比较原始 count MAE；
5. 最后才以总体节点数、P95 MSE 和平均 MSE 打破平局。

这对应目标：

\[
\min K\quad\text{s.t.}\quad
\Pr(\operatorname{MSE}\le\varepsilon)\ge R_{target}.
\]

## 7. 推荐训练命令

### 7.1 复杂真实曲线主实验：96 个内部候选

```powershell
python scripts/train_v16.py `
  --epochs 60 --proposal-epochs 20 `
  --train-size 2400 --val-size 500 --real-val-size 100 `
  --batch-size 16 --num-points 192 `
  --min-control-points 8 --max-control-points 24 `
  --candidate-knots 96 `
  --mse-tolerance 2.5e-5 `
  --knot-match-tolerance 0.01 `
  --certified-minimal-source `
  --minimality-margin 0.2 --minimality-max-attempts 16 `
  --minimality-audit-points 512 --oscillation-amplitude 0.3 `
  --proposal-pass-target 0.90 --deployment-pass-target 0.90 `
  --one-shot-selection-policy mass_topk `
  --one-shot-safety-sigma 0.25 --one-shot-safety-knots 2 `
  --final-safety-sigma 0.05 --final-safety-knots 0 `
  --safety-anneal-epochs 10 `
  --one-shot-coverage-bins 4 --min-selected-knots 4 `
  --relocation-blend 0 `
  --teacher-prefix-search-steps 7 `
  --count-weight 2.0 --supervised-count-weight 1.0 `
  --supervised-over-count-weight 1.0 `
  --true-parameter-weight 0.1 `
  --proposal-knot-coverage-weight 1.0 `
  --selected-knot-position-weight 1.0 --knot-position-beta 0.01 `
  --complexity-weight 0.05 `
  --complexity-ramp-epochs 10 --complexity-max-scale 4.0 `
  --complexity-pass-margin 0.02 `
  --real-fraction 0.5 `
  --real-manifest data/splits/uji_pen_v2.jsonl `
  --real-manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --real-manifest data/processed/usgs_contours/large_scale/manifest.jsonl `
  --init-checkpoint outputs/checkpoints/candidate_selection_v16.proposal.pt `
  --device cuda `
  --output outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt
```

这是工程 warm start：只复制形状兼容的 Encoder、ParameterHead 和候选生成张量；Kc=96 新 query、Selector 与 subset decoder 仍重新初始化。严格 Kc 容量消融应去掉 `--init-checkpoint`，并让两组采用相同随机初始化协议。proposal 阶段按 `--proposal-pass-target 0.90` 判断能否进入 Joint；0.02 安全余量只控制 Joint 中复杂度压力的全速/半速区间，不再把 proposal 门槛暗中提高到 92%。未达到 90% 时不会生成主 `.pt`。

### 7.2 “完整节点向量最多 64 项”消融

```powershell
python scripts/train_v16.py `
  --epochs 60 --proposal-epochs 20 `
  --train-size 2400 --val-size 500 --real-val-size 100 `
  --batch-size 16 --num-points 192 `
  --min-control-points 8 --max-control-points 24 `
  --full-knot-vector-size 64 `
  --mse-tolerance 2.5e-5 `
  --proposal-pass-target 0.90 --deployment-pass-target 0.90 `
  --real-fraction 0.5 `
  --real-manifest data/splits/uji_pen_v2.jsonl `
  --real-manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --real-manifest data/processed/usgs_contours/large_scale/manifest.jsonl `
  --device cuda `
  --output outputs/checkpoints/candidate_selection_v16_full64.pt
```

不要用旧 v16 checkpoint 的 `--resume` 进入新结构；旧模型没有 adaptive-beta 参数。若只想迁移编码器、ParameterHead 和 proposal 权重，使用 `--init-checkpoint`，其余新头重新训练。

## 8. 训练后检查

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt
```

检查输出中的：

- `one_shot_selection_policy = mass_topk`；
- `adaptive threshold = True`；
- `candidate capacity (internal) = 96`；
- worst-source deployment pass rate 是否达到 90%；
- 最终 K 是否明显低于 Kc，而不是固定为 0 或 Kc。
- `simplification_ready = True`，且实际安全储备已经等于最终配置；
- Synthetic count MAE/bias、knot-match F1 和 matched MAE。

对于 certified Synthetic 的 source K=4～20，本轮训练目标是平均预测 K 落在 10～14，并让每条曲线的预测计数和节点位置尽量接近真值。因为该生成范围的真计数均值约为 12，优化中心是真值而不是盲目删到更少。10～14 是重新训练后的验收目标，不是未经训练即可保证的常数。即使平均 K 落入区间，也必须同时满足 MSE/pass、较小 count MAE/bias 和较高 knot-match F1；只看均值可能掩盖一部分曲线过删、另一部分过留。

正式 checkpoint 的硬性节点真值门槛为 `count MAE<=2.0`、`knot-match F1@0.01>=0.60`、`matched-knot MAE<=0.005`；worst-source dense/deployment pass 均须达到 90%，最终固定安全节点为 0 且 safety sigma 不大于 0.05。达不到任一项时，检查脚本返回 2，只能作为诊断权重。

该 10～14 范围不强加给没有节点真值的 UJI、Natural Earth 或 USGS；真实曲线的最终 K 仍由误差约束和学习到的复杂度决定。

正式对比协议、论文适配方法和命令见 [v16 在线反事实子集学习](v16_counterfactual_subset.md) 与 [公开节点方法适配说明](published_knot_methods_reproduction.md)。

## 9. 12 小时无人值守训练、比较与案例图

`scripts/run_v16_overnight_12h.ps1` 是本轮 `MSE=5e-5`、Kc=64 实验的单一串行入口。它不是另一套模型实现；内部依次调用当前的训练、资格检查、统一 benchmark、指标绘图和 Ours 案例绘图脚本。

### 9.1 固定实验合同

| 项目 | 默认值 | 准确含义 |
|---|---:|---|
| 网络容量 | `candidate-knots=64` | 64 个内部候选；全部保留时完整三次节点向量为 72 项、控制顶点为 68 个 |
| 合成 source 复杂度 | `min/max-control-points=8/28` | 真内部节点 `K=4..24`；最大 source 控制顶点 28、完整节点向量 32 项 |
| 合成训练/验证集 | 1500 / 500 | 使用固定样本，`no-resample-train-each-epoch`；不把训练样本改成每轮新抽样 |
| 真实训练占比 | 0 | 本轮网络只在合成集训练；UJI、Natural Earth、USGS 只用于之后的独立泛化测试和案例图 |
| 采样点 | 192 | 每条网络输入的有序归一化点数 |
| 拟合阈值 | `MSE <= 5e-5` | 平均平方欧氏距离，不开方；对应 RMS 约 `7.071e-3` |
| fresh batch | 64 | 仅新实验使用；恢复已有 batch=32 的 `.last.pt` 时仍严格使用 32 |
| 初始训练预算 | 56 epochs | fresh 时 proposal=12；现有实验已进入 joint 且 proposal=4 时保留原阶段边界 |
| 条件追加预算 | 64 epochs | 56 轮后仍不合格且总预算剩余至少 6.5 小时时才追加 |
| 工程通过率 | 90% | checkpoint 仍须同时满足简化成熟度、最终安全储备和 Synthetic 节点真值门槛 |

最容易混淆的是两种“最大值”：`24+2*4=32` 是合成真值曲线在三次开区间表示下的最大**完整节点向量**；`candidate-knots=64` 是网络提供给 Selector 的**内部候选数**，对应全保留完整向量 72。前者规定监督数据的复杂度，后者规定网络搜索容量。

### 9.2 一次启动

在仓库根目录执行一次：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_overnight_12h.ps1
```

如果要关闭当前终端，使用独立隐藏进程。脚本本身会持续写日志，所以隐藏运行不会丢失训练记录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_overnight_12h.ps1 -Detach
```

`-Detach` 只负责派生一个隐藏 runner 并立即返回，不会绕过任何检查。隐藏 runner 仍执行相同目标训练等待、不同训练/GPU 冲突检查、自动恢复及正式/诊断分流；其兜底 stdout/stderr 也单独写入同一日志目录。

查看流水线状态或连续日志：

```powershell
Get-Content outputs/logs/candidate_selection_v16_mse5e-5_k64/overnight_manifest.json
Get-Content outputs/logs/candidate_selection_v16_mse5e-5_k64/overnight.log -Wait
```

不要同时手工启动另一个指向同一 output 的 `train_v16.py`。入口会检查 Windows 进程：若发现相同目标的现有训练，会保留它并每 45 秒报告一次等待状态；若发现不同的 v16 训练或明确的其他 Python CUDA 作业，则安全退出且不终止用户进程。WDDM 桌面和普通图形进程不会因此被误判为冲突。

### 9.3 自动恢复规则

默认 checkpoint stem 为 `candidate_selection_v16_mse5e-5_k64`。

1. 如果匹配的训练进程正在运行，先等它自然结束。
2. 如果 `.last.pt` 存在且尚未到 56 轮，从下一轮自动恢复。恢复命令从 checkpoint 的 `training_config` 重建，batch、proposal 边界、数据、损失和随机种子不会被 fresh 默认值覆盖。
3. 如果已经达到目标轮数且主 `.pt` 存在，跳过训练，直接资格检查和评估。
4. 每个完成的 epoch 都原子写入 `.last.pt`，同时更新 `.history.json`；关闭窗口最多损失当前尚未完成的一轮。
5. 不完整产物存在但 `.last.pt` 缺失时，脚本不删除或覆盖它们，只使用可读取的最佳权重生成诊断结果。

要只重跑 benchmark 和图片，可增加 `-SkipTraining`。benchmark 自身带 `--resume`，相同实验指纹下会复用已经完成的逐样本记录。

### 9.4 串行阶段与时间预算

执行顺序固定为：

```text
进程/GPU预检
  -> 等待同目标旧训练
  -> fresh训练或.last.pt续训至56轮
  -> checkpoint资格检查
  -> 条件追加至64轮并再次检查
  -> 66条曲线的八方法配对benchmark
  -> Ours与公开/数值方法2x2指标图
  -> 6个Ours真实拟合案例及总览
```

配对 benchmark 使用 K=4..24 每档 2 条合成曲线，以及 UJI、Natural Earth、USGS 各 8 条，共 66 条。所有可设置容量的方法统一使用 64 个内部节点上限；保留 Greedy 12 次位置梯度更新、Kang 400 次 ADMM/8 次 lambda 二分/8 次重定位、Dung 10 个扫描区间/10 次优化、Luo population=10/DE=50。完整方法计时重复 1 次；Ours 纯网络计时预热 3 次并重复 20 次。纯网络时间和传统方法完整求解时间必须分别标注，不能解释成完全对称的端到端加速比。

当前工作区的 GTX 1070 在已有 batch=32 实验中，joint epoch 截至第 8 轮约为 4.7 分钟。由此估算训练约 4.4～5.1 小时，完整 benchmark 约 3.5～5.5 小时，绘图和 6 个案例约 10～15 分钟，整体按约 8～11 小时规划。`-BudgetHours 12` 只决定是否还有足够余量追加到 64 轮；12 小时不是强制终止时限，脚本也不会为了赶时间静默降低基线迭代或删减样本。RTX 3090 通常会加快网络阶段，但总耗时仍受 CPU 数值基线、GPU 占用和缓存状态影响。

### 9.5 正式与诊断结果分流

训练后会用 `inspect_v16_checkpoint.py` 检查 MSE 阈值、90% pass、joint/simplification 成熟度、最终安全储备、Synthetic `count MAE<=2`、`knot F1>=0.60` 和 `matched MAE<=0.005`。

- 合格：选择合格 checkpoint，目录标签为 `formal_<SHA256前12位>`；
- 不合格：优先选择主 `.pt` 中按验证排序保存的最佳 joint 权重；主权重缺失时才依次使用 `.last.pt` 或 `.proposal.pt`。随后继续完成 benchmark 和案例图，目录标签为 `diagnostic_<SHA256前12位>`，所有报告和 PNG 强制显示 `DIAGNOSTIC NOT FINAL`。

不合格并不等于流水线失败；它表示训练、比较和绘图链路已经完成，但结果只能用于定位问题。checkpoint 无法加载、训练后没有任何权重、benchmark 或绘图命令失败才会把 `overnight_manifest.json` 的顶层状态写为 `failed`。

### 9.6 产物位置

```text
outputs/
  checkpoints/
    candidate_selection_v16_mse5e-5_k64.pt
    candidate_selection_v16_mse5e-5_k64.last.pt
    candidate_selection_v16_mse5e-5_k64.proposal.pt
    candidate_selection_v16_mse5e-5_k64.history.json
  logs/candidate_selection_v16_mse5e-5_k64/
    overnight.log
    overnight_manifest.json
    detached_<timestamp>.stdout.log
    detached_<timestamp>.stderr.log
    train_resume.log 或 train_fresh.log
    inspect_*.log
    benchmark_all_methods.log
    plot_method_comparison.log
    plot_ours_cases.log
  comparisons/candidate_selection_v16_mse5e-5_k64/
    formal_<hash>/ 或 diagnostic_<hash>/
      comparison.json
      summary.csv
      measurements.csv
      report.md
  figures/candidate_selection_v16_mse5e-5_k64/
    formal_<hash>/ 或 diagnostic_<hash>/
      method_comparison/
        v16_published_methods_input.png
        v16_published_methods_reference.png
      ours_cases/
        ours_cases_overview.png
        deployment_visualizations.json
        ours__<dataset>__<sample>.png
```

`overnight_manifest.json` 是第一检查入口：顶层 `status=completed` 表示所有串行阶段结束；`diagnostic_not_final=false` 才表示采用了正式合格 checkpoint。最后还应核对 `comparison.json` 中的阈值、64 节点统一容量、66 条曲线记录，以及案例图是否同时标出采样点、拟合曲线、控制多边形、控制顶点和内部节点。PNG 只从保存的实测 JSON/checkpoint 生成，不能人工调整数值。
