# v16：1e-4 下的候选选择与联合重定位

> 当前推荐实验合同：`Kc=56` 个内部候选（完整三次开放节点向量 64 项）、合成源内部节点 `K=4..56`（控制顶点 8..60）、归一化
> `MSE <= 1e-4`、最差数据源通过率目标 90%。文中的 checkpoint
> `outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt` 是**待训练并通过资格检查的目标文件**，不是已经取得的实验结果。

v16 输入有序点云，一次性选择节点并联合更新参数化和存活节点位置，最后只做一次标准三次 B 样条最小二乘 refit。部署不运行教师搜索、循环删点、BIC 或阈值扫描。

## 1. 任务与计数约定

误差统一为平均平方欧氏距离，不开方：

\[
\operatorname{MSE}=\frac{1}{M}\sum_{i=1}^{M}\lVert \hat Q_i-Q_i\rVert_2^2.
\]

`K`、`Kc` 和 `--candidate-knots` 都表示**内部节点数**。三次开区间 B 样条有：

\[
K_{\mathrm{full}}=K_{\mathrm{internal}}+8,
\qquad
N_{\mathrm{ctrl}}=K_{\mathrm{internal}}+4.
\]

当前主配置为：

| 项目 | 设置 | 含义 |
|---|---:|---|
| 网络候选容量 | `Kc=56` | 高召回搜索容量，不是最终节点数 |
| 合成源控制顶点 | `8..60` | 对应源内部节点 `4..56` |
| 合成节点最小 span | `0.01` | K=56 产生 57 个 span，必须显式覆盖历史默认 0.02 |
| 输入点数 | 192 | 网络输入的有序归一化点数 |
| 拟合阈值 | `MSE <= 1e-4` | 对应 RMS 为 `1e-2` |
| 工程目标 | 90% | 四个数据源中最差者的 deployment pass rate |

当前 `--candidate-knots 56` 表示 56 个内部候选；三次开放样条全部保留时，完整节点向量恰为 `56+8=64` 项、控制顶点为 `56+4=60` 个。不要把“完整节点向量 64 项”误写成 64 个内部候选。

## 2. 为什么简单曲线曾经保留过多节点

问题主要在于“56 个候选中选择多少、选择哪几个”的训练信号不够细：

1. 旧 Selector 初始化约为 `p=0.95`，Joint 开始时天然接近全保留；Proposal 阶段又不训练 Selector，造成明显的高计数惯性。
2. ranked-prefix 教师主要沿当前 Selector 排名找前缀。如果排名尚未学好，真正可行的低节点组合可能根本不在被检查的前缀中。
3. 合成数据的源节点最简证书只证明“固定源位置的所有子集”不能继续删除；允许存活节点重定位后，源 `K` 不一定仍是全局最少计数。
4. 固定四分区覆盖会为每个区间预留候选。简单曲线的关键节点若集中在局部，这些锚点会浪费有限的低 `K` 名额。

当前修正不降低高召回容量，而是提高低复杂度分辨率：

- `--initial-keep-fraction 0.5357142857142857`：即 `30/56`，Joint 初始概率质量目标约为 30；它不是部署最终 K；
- `--teacher-low-count-sweep 16`：训练教师逐个精确检查 `K=Kmin..16` 的 ranked prefix，再进入粗到细搜索；
- `--synthetic-count-role upper_bound`：允许重定位后的可行解少于 certified source K，不再把 source K 错当成全局精确计数；
- `--synthetic-geometry-oracle-teacher`：只在有真节点的 certified Synthetic 上加入与 Selector 分数无关的几何教师；
- `--oracle-teacher-extra-knots 2`：同时加入 `Ktrue+2` 的安全扩展组合，超过 Kc 时截断到 56；Ktrue=56 时即全候选；
- `--one-shot-coverage-bins 0`：当前 1e-4 主实验关闭强制分区锚点，避免简单曲线被迫占用无关区间；
- 混入 UJI、Natural Earth 和 USGS 的无标签训练曲线，使候选与 Selector 不只适应合成几何。

`Kc=56` 覆盖 source K=4..56，并把完整节点向量容量统一为 64 项。source K
描述生成复杂度，Kc 描述候选槽位；最大值相同不代表部署必须全保留。最终
活动节点数由逐曲线自适应概率质量决定。K=56 层没有冗余候选余量，必须单独
报告 dense/deployment pass，失败不能靠放宽 90% 资格处理。

## 3. 一次部署的数据流

```text
ordered points Q [B,M,D]
  -> GeometryEncoder
       local features [B,M,H], global feature [B,H]
  -> chord reference + ParameterHead
       strictly increasing parameters t0 [B,M]
  -> CandidateKnotHead
       ordered candidates U0 [B,Kc], tokens [B,Kc,H]
  -> InteractiveSelector
       centered importance + curve-adaptive beta
       keep probabilities p [B,Kc]
  -> probability-mass cardinality + one Top-K
       discrete KeepMask [B,Kc]
  -> selected-only decoder
       t0 -> t1; survivors jointly relocated
  -> ordered Udeploy
  -> one endpoint-constrained float64 B-spline refit
       curve + control polygon + MSE
```

曲线级 `beta` 与候选相对重要度共同得到：

\[
p_j=\sigma\left((r_j-\bar r)-\beta\right).
\]

`mass_topk` 依据概率质量估计一次性节点数：

\[
\widehat K=\operatorname{ceil}\left(
\sum_jp_j+\sigma_s\sqrt{\sum_jp_j(1-p_j)}+\Delta K
\right).
\]

随后只执行一次 Top-K。KeepMask 产生后，参数反馈与 SurvivorRelocation 只读取存活节点，删除和位置更新因此在同一网络图中耦合。未选节点不能作为重定位 Key/Value。

## 4. 两阶段训练与教师

### 4.1 Proposal 阶段

前 16 个 epoch 全保留候选，先训练 GeometryEncoder、ParameterHead 和 CandidateKnotHead，使四个数据源的 dense proposal 具备高召回与拟合可行性。`Kc=56` 仅在此处作为上限使用。

### 4.2 Joint 阶段

Joint 阶段同时训练 Selector、参数反馈和存活节点重定位。在线教师池包括：

- 当前一次性 deployment mask；
- `Kmin..16` 的逐计数 ranked-prefix 扫描；
- 更高 K 的粗到细前缀搜索及边界附近的增、删、交换组合；
- 全保留组合；
- certified Synthetic 的显式几何 oracle mask 与 `Ktrue+2` 扩展 mask。

几何 oracle 的构造为：先把 proposal 节点映射到真参数域，再进行一维单调一一匹配，得到恰好 `Ktrue` 个候选的 mask；扩展 mask 再补入两个高分候选。它不依赖当前 Selector 排名，可在训练早期打破“错误排名只监督自身前缀”的闭环。它只使用合成训练标签，真实数据和部署均不使用真节点信息。

若教师池存在满足 `MSE <= 1e-4` 的组合，选择节点数最少者，再用 MSE 打破平局；若全部失败，则以 MSE 最小者作为临时教师。该有限搜索只是训练教师，不是全局最优证明。

真实曲线没有节点真值，只参加点重建、可行性、反事实组合与复杂度学习。`real-fraction=0.35` 时每轮约 35% 样本来自三个真实来源，其余为可认证合成样本。

## 5. RTX 3090 推荐训练命令

在仓库根目录运行：

```powershell
python scripts/train_v16.py `
  --epochs 80 --proposal-epochs 16 `
  --train-size 3000 --val-size 600 --real-val-size 100 `
  --batch-size 64 --num-points 192 `
  --min-control-points 8 --max-control-points 60 `
  --knot-min-span 0.01 `
  --candidate-knots 56 --mse-tolerance 1e-4 `
  --knot-match-tolerance 0.01 `
  --certified-minimal-source `
  --minimality-margin 0.2 --minimality-max-attempts 16 `
  --minimality-audit-points 512 --oscillation-amplitude 0.3 `
  --proposal-pass-target 0.90 --deployment-pass-target 0.90 `
  --one-shot-selection-policy mass_topk `
  --initial-keep-fraction 0.5357142857142857 `
  --one-shot-safety-sigma 0.20 --one-shot-safety-knots 2 `
  --final-safety-sigma 0.03 --final-safety-knots 0 `
  --safety-anneal-epochs 12 `
  --one-shot-coverage-bins 0 --min-selected-knots 4 `
  --relocation-blend 0 `
  --teacher-prefix-search-steps 7 `
  --teacher-low-count-sweep 16 `
  --synthetic-count-role upper_bound `
  --synthetic-geometry-oracle-teacher `
  --oracle-teacher-extra-knots 2 `
  --policy-samples 2 --counterfactual-edits 4 `
  --count-weight 2.0 --supervised-count-weight 1.0 `
  --supervised-over-count-weight 1.0 `
  --true-parameter-weight 0.1 `
  --proposal-knot-coverage-weight 1.0 `
  --selected-knot-position-weight 1.0 --knot-position-beta 0.01 `
  --complexity-weight 0.05 `
  --complexity-ramp-epochs 12 --complexity-max-scale 6.0 `
  --complexity-pass-margin 0.02 `
  --real-fraction 0.35 `
  --real-manifest data/splits/uji_pen_v2.jsonl `
  --real-manifest data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --real-manifest data/processed/usgs_contours/large_scale/manifest.jsonl `
  --resample-train-each-epoch `
  --num-workers 4 --torch-num-threads 4 `
  --device cuda `
  --output outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt
```

Windows 上若多进程 DataLoader 不稳定，把 `--num-workers 4` 改为 `--num-workers 0`；这只影响数据加载，不改变模型或实验合同。旧 K64/K96 checkpoint 不能直接 `--resume` 到该配置。

旧 `candidate_selection_v16_mse5e-5_k64.proposal.pt` 可作为可选 warm start：形状兼容的 Encoder、ParameterHead 和 CandidateHead 张量迁移，65 个旧 interval query 沿参数域插值为 57 个；K56 固定锚点重建，Selector、联合解码器和优化器新训。因此它不是跨容量 resume，也不继承旧实验资格。

## 6. 资格检查

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --required-pass-rate 0.90 `
  --mse-tolerance 1e-4
```

检查脚本返回 0 后才能用于正式汇报；返回 2 表示权重只能用于诊断。至少核对：

- objective 和训练阈值与本协议一致；
- worst-source dense/deployment pass rate 均达到 90%；
- 已进入成熟 Joint 阶段，最终 safety knots 为 0、safety sigma 为 0.03；
- 平均最终 K 明显低于 56，简单曲线不再系统性过留；
- Synthetic 的 count bias、knot-match F1 和 matched MAE；
- `oracle_teacher_feasible_fraction` 与 `oracle_teacher_selected_fraction`，用于判断 oracle 是否真正提供了低 K 可行教师。

这些都是验收条件，当前文档不宣称目标 checkpoint 已经达到它们。

## 7. 六方法、四指标对比

六方法固定为：Ours、Park & Lee、Liang、Dung & Tjahjowidodo、Kang 和 Luo。Kang 与 Luo 均为按公开论文目标重写并接入统一 B 样条 refit 的**适配复现**，不是作者原始代码逐行复刻。

所有允许设置容量的方法统一使用最多 56 个内部节点（完整节点向量最多 64 项）；共同使用相同输入、三次样条、端点约束、float64 无正则最终 refit 和 `MSE <= 1e-4` 判定。正式报告四项指标：

1. 最终 MSE；
2. 阈值通过率；
3. 最终内部节点数 K；
4. 完整方法耗时。

Ours 另外报告纯 network-forward 时间，但不得将它与其他方法的完整求解时间伪装成同一计时边界。

合成 `K=4..56` 每档 5 条、三个真实数据集各 20 条：

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --method-set published `
  --samples-per-knot-count 5 `
  --min-knot-count 4 --max-knot-count 56 `
  --real-samples-per-dataset 20 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 `
  --max-internal-knots 56 `
  --paper-initial-knots 56 `
  --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 `
  --paper-relocation-iterations 12 `
  --liang-dense-knots 56 `
  --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 `
  --torch-num-threads 4 --device cuda `
  --output-dir outputs/comparisons/v16_mse1e-4_k56_six_methods
```

若同一实验指纹中断，可在原命令末尾增加 `--resume`。未通过资格检查时，benchmark 会拒绝正式运行；`--allow-unqualified-diagnostic` 只能生成带诊断标记的排错结果。

## 8. 2×2 指标图与真实曲线可视化

从 benchmark 保存的逐样本结果绘制 2×2 四指标图：

```powershell
python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_mse1e-4_k56_six_methods/comparison.json `
  --method-set published `
  --reference `
  --dpi 300 `
  --output-dir outputs/figures/v16_mse1e-4_k56_six_methods/metrics
```

绘制 UJI、Natural Earth 和 USGS 上六种方法的真实拟合案例：

```powershell
python scripts/visualize_v16_real_deployments.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt `
  --real-samples-per-dataset 2 `
  --selection-seed 20260910 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 `
  --max-internal-knots 56 `
  --paper-initial-knots 56 `
  --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 `
  --paper-relocation-iterations 12 `
  --liang-dense-knots 56 `
  --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 `
  --torch-num-threads 4 --device cuda `
  --dpi 300 `
  --output-dir outputs/figures/v16_mse1e-4_k56_six_methods/real_cases
```

每个案例的六个面板使用同一条留出曲线，绘制输入采样点、原始参考折线、拟合曲线、控制多边形、控制顶点和内部节点，并标注 MSE、K 与完整方法时间；Ours 额外标注 network-only 时间。

也可以让仓库脚本依次完成训练、资格检查、六方法 benchmark、2×2 图和真实案例图：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_mse1e-4_3090.ps1
```

该入口不会覆盖同名产物；若 checkpoint 未通过 90%/1e-4 资格检查，默认停止且不生成正式图。只有排错时才增加 `-Diagnostic`，此时目录与图片都会标记为 `diagnostic_<checkpoint-hash>`。若 Windows 多进程加载失败，可增加 `-NumWorkers 0`。入口会在旧的 5e-5 proposal checkpoint 存在时仅迁移 Encoder、ParameterHead 和 CandidateHead；找不到时会提示并从头训练。

`K=56` 有 57 个参数 span；底层通用 synthetic 生成器的旧 `min_span=0.02` 要求总长度至少 1.14，
因此不可能生成该层。本协议显式使用 `--knot-min-span 0.01`，旧默认 0.02 只为
历史数据兼容。边界层 dense/deployment pass 必须单列并纳入原 90% 正式资格。

## 9. Kang/Luo 结果的解释边界

简单曲线中，传统数值方法可直接搜索很小的低维节点集，因此可能同时得到较低 K 和较低 MSE；这不是异常。复杂曲线中，它们的有限迭代、初始化和非凸搜索预算更容易达到节点上限仍不满足阈值。

复查必须确认：

- Kang 使用 `--paper-initial-knots 56`，并完整计入稀疏求解、重定位和 refit；
- Luo 使用同一 56 内部节点上限与固定随机种子，失败样本仍保留在通过率分母中；
- Dung 原生最大距离阈值按 `sqrt(1e-4)=0.01` 换算，而最终横向指标仍统一用 MSE；
- 不按曲线难度或方法单独放宽阈值、容量或删除失败样本。

因此应同时阅读 MSE、通过率、K 与完整耗时，不能仅凭简单曲线上的节点数判定一种方法整体更优。

## 10. 历史实验说明

旧 source K=4..24、旧 K64 以及 `Kc=96、MSE=2.5e-5` 是历史实验/范围或容量消融，不是当前 1e-4 主协议：

- Kc 从当前 56 增到历史 64/96 会增加候选对点注意力约 \(O(K_cM)\) 和候选间注意力约 \(O(K_c^2)\) 的开销；
- 旧 checkpoint 的阈值、候选形状和教师语义不同，不能直接续训或并入当前主表；
- 若做 Kc56/Kc64/Kc96 消融，必须分别重新训练，并保持阈值、数据划分、随机种子和其余参数一致。

## 11. 代码入口

- [网络结构](../src/spline_fitting/models/v16_network.py)
- [在线教师与损失](../src/spline_fitting/losses/v16_subset_loss.py)
- [训练入口](../scripts/train_v16.py)
- [资格检查](../scripts/inspect_v16_checkpoint.py)
- [六方法 benchmark](../scripts/benchmark_v16_datasets.py)
- [四指标绘图](../scripts/plot_v16_method_comparison.py)
- [六方法真实案例](../scripts/visualize_v16_real_deployments.py)
- [公开方法复现边界](published_knot_methods_reproduction.md)
