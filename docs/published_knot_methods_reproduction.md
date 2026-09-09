# 论文节点方法的简化复现与统一对比协议

## 1. 复现范围

本仓库新增的是**可审计的算法适配**，不是作者原始代码的逐语句复刻。各方法接收同一条归一化有序曲线，只使用观测点，不读取真实节点；最终都交给同一个 CPU `float64` 标准 B 样条最小二乘求解器重新计算控制顶点。

| 文献 | 论文原始目标与核心步骤 | 仓库 method ID | 本仓库的明确适配 |
|---|---|---|---|
| [Park & Lee, 2007](https://doi.org/10.1016/j.cad.2006.12.006) | 在误差约束下选取尽量少的 dominant points；内部节点由相邻 dominant-point 参数平均得到，并按曲率和弧长形状指数自适应细分。 | `park_dominant_point_2007_adaptation` | 弦长参数；LCM 种子阈值为平均曲率的 `1/4`；形状指数权重 `r=0.8`；以参数对应残差定位待细分区段。统一 MSE 代替论文的最大距离/正交距离停止准则；离散 Menger 曲率代替噪声情形下可选的平滑基准曲线。 |
| [Liang et al., 2017](https://doi.org/10.1088/1361-6501/aa6a05) | 先用密集均匀节点拟合参考曲线，由弧长和弯曲特征构造单调特征积分；按等特征积分放置初始节点，再执行 iterative knot insertion（IKI）直至满足误差界。 | `liang_feature_iki_2017_adaptation` | **feature-integral + IKI 适配**：特征为归一化累计弧长与累计绝对转角的加权和，默认曲率权重 `0.5`；在最大采样残差所在节点区间插入区间中点。由于全文中的特征组合常数与全部 IKI 细节无法完整核验，不能标为精确复现。 |
| [Dung & Tjahjowidodo, 2017](https://doi.org/10.1371/journal.pone.0173857) | 以最大误差约束串行/并行二分数据，得到粗节点；再通过局部两段 B 样条非线性最小二乘优化节点位置和连续性阶数，最后求控制顶点。 | `dung_direct_knot_2017_adaptation` | 仅复现串行二分和局部节点位置优化；默认原生最大距离阈值为 `sqrt(MSE tolerance)`。**只处理平滑单节点，不复现重节点/连续性分类，也不复现并行 split–join–shift**；容量超限时确定性均匀抽取粗分界。 |
| [Luo–Kang–Yang, 2022](https://doi.org/10.4208/jcm.2012-m2020-0203) | 两阶段优化：先求解 `||P-AC||_F + λ||DC||_{∞,1}`，从导数跳跃的局部峰值确定候选数目；再用 Differential Evolution（DE）全局更新固定数量的节点位置。 | `luo_linf_de_2022_adaptation` | 指定复现的是 **`l_inf,1 + DE` 论文，而不是同年的 DNN 论文**。用 ADMM 求稀疏阶段，并按公共 MSE 预算搜索 `λ`；峰值阈值默认 `eta=0.5`；DE 优化最大采样距离，最后仍按公共 MSE 报告。论文逐例选择 `λ`，本仓库的自动搜索属于对比适配。 |
| [Kang et al., 2015](https://doi.org/10.1016/j.cad.2014.08.022) | 在密集初始节点上用稀疏优化确定活跃节点，再删除冗余节点并调整位置，同时兼顾拟合质量和节点数。 | `kang_sparse_2015_adaptation` | 二维 group-L1/ADMM、跳跃聚类和局部位置调整；不声称复现作者的 CVX 数值路径。正式比较关闭额外 feasibility repair，避免给该基线加入论文之外的补救优势。 |
| Ours v16 | 在给定 MSE 阈值下，先学习高可行率的密集候选，再用在线随机子集和反事实编辑学习最少可行 KeepMask；存活节点、节点位置和参数在一次解码中联动。 | `ours`；checkpoint objective 为 `candidate_selection_counterfactual_bspline_v16` | 无离线硬剪枝标签。部署只执行一次网络前向、一次离散选集和一次标准 B 样条 refit。候选容量必须与 checkpoint 一致。 |

`yeh_feature_cdf_2020` 是另一项论文方法适配；`uniform_gradient_pruning` 是本仓库的传统数值控制组，不属于论文方法。代码分别暴露 `PUBLISHED_ADAPTATION_METHODS` 与 `NUMERICAL_BASELINE_METHODS`，统一 benchmark 使用二者的并集，避免将仓库基线误标成已发表方法。

## 2. 公平协议

1. **配对数据**：所有方法处理完全相同的归一化有序点。合成测试按源内部节点数 `K=4..20` 分层；真实测试使用 UJI Pen、Natural Earth 海岸线和 USGS 等高线的 test split。同一 writer/tile 不跨 train、validation、test。
2. **参数域**：所有路径均以弦长参数为共同参考。所有传统数值基线固定使用弦长参数；Ours 的 ParameterHead 以弦长参数为 reference，学习有界 residual，并在所选子集上再更新。因此主实验比较的是各方法的完整能力，而不是“最终参数完全相同”的纯节点消融。若要单独研究节点选择，应另报固定弦长参数的 Ours 消融，不能把两种口径混在一列。
3. **统一最终拟合**：三次开区间 B 样条、CPU `float64`、无平滑项、无控制点 ridge，并严格插值两个端点。所有表格中的最终误差均来自这次标准 refit，而不是网络 surrogate、ADMM 内部目标或 DE 适应度。
4. **统一误差**：`MSE = mean_i ||C(t_i)-Q_i||_2^2`，不取平方根，也不除以坐标维数。默认通过条件为 `MSE <= 2.5e-5`。Dung 的原生分段仍由最大欧氏距离控制；该值和公共 MSE 必须分别保存。
5. **节点数**：报告最终 refit 实际使用的内部节点数。合成数据的 source K 只是生成复杂度，不自动等于给定阈值下的全局最小 K；真实数据没有节点真值，不能计算节点 precision/recall。
6. **时间**：公共 `total_ms` 从已归一化的 CPU 点开始，包含参数化、方法本身以及最终 refit，不包含文件读取和绘图。Ours 另外报告同步后的 `network_ms`，但它不能与传统方法的完整 `total_ms` 直接当成端到端加速比。
7. **失败样本**：异常或非有限解保留为失败并计入通过率分母，不能静默删除。`comparison.json` 保存逐曲线记录、配置、checkpoint/代码指纹和硬件信息。

v16 benchmark 默认拒绝与 checkpoint `Kc` 不一致的数值容量。只有明确的容量消融才可使用 `--allow-unequal-capacity`，且不能把该结果放入同容量主表。

主表至少同时报告：平均 MSE、P95 MSE、通过率、平均最终内部节点数和完整耗时；Ours 另列网络前向耗时。只比较平均 MSE 会掩盖少量严重失败曲线。

## 3. 快速联调命令

以下命令只验证八个方法、四类数据和绘图链路能运行；数值基线容量与迭代数被缩小，Ours 的候选容量仍由 checkpoint 固定，**不能用于论文结论**。要求已有完成联合阶段的 v16 checkpoint；`.proposal.pt` 或 `target_met=False` 的权重不合格。

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --output-dir outputs/comparisons/v16_published_quick `
  --samples-per-knot-count 1 `
  --real-samples-per-dataset 2 `
  --max-internal-knots 16 `
  --allow-unequal-capacity `
  --paper-initial-knots 16 `
  --paper-admm-iterations 30 `
  --paper-lambda-bisections 2 `
  --paper-relocation-iterations 2 `
  --liang-dense-knots 16 `
  --liang-feature-samples 129 `
  --dung-scan-intervals 3 `
  --dung-optimization-iterations 2 `
  --luo-de-population 5 `
  --luo-de-iterations 2 `
  --network-warmups 1 `
  --network-repeats 3 `
  --end-to-end-repeats 1 `
  --torch-num-threads 1 `
  --device auto

python scripts/plot_v15_dataset_benchmark.py `
  --input outputs/comparisons/v16_published_quick/comparison.json
```

## 4. 正式对比命令

下面以推荐主模型 `Kc=96` 为例，并给所有可配置方法相同的 96 个内部节点上限。Luo 使用 `population=20, iterations=100`。传统算法运行在 CPU，完整配对实验会明显慢于快速联调；中断后用同一命令追加 `--resume`，只有实验指纹完全一致时才会续跑。

当前仓库没有随附该 adaptive Kc=96 权重；命令中的 checkpoint 必须先由 `train_v16.py` 实际生成并经 `inspect_v16_checkpoint.py` 判定合格。旧 fixed-threshold 或全保留节点 checkpoint 会被正式入口拒绝。

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_simplified_certified_k96.pt `
  --output-dir outputs/comparisons/v16_published_formal `
  --samples-per-knot-count 20 `
  --min-knot-count 4 `
  --max-knot-count 20 `
  --scan-size 20000 `
  --real-samples-per-dataset 100 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 2.5e-5 `
  --max-internal-knots 96 `
  --paper-initial-knots 96 `
  --paper-admm-iterations 400 `
  --paper-lambda-bisections 8 `
  --paper-relocation-iterations 8 `
  --park-shape-weight 0.8 `
  --liang-dense-knots 96 `
  --liang-initial-knots 4 `
  --liang-curvature-weight 0.5 `
  --liang-feature-samples 1025 `
  --dung-max-error 0.005 `
  --dung-scan-intervals 10 `
  --dung-optimization-iterations 10 `
  --luo-eta 0.5 `
  --luo-de-population 20 `
  --luo-de-iterations 100 `
  --luo-seed 2022 `
  --network-warmups 10 `
  --network-repeats 100 `
  --end-to-end-repeats 10 `
  --torch-num-threads 1 `
  --device cuda

python scripts/plot_v15_dataset_benchmark.py `
  --input outputs/comparisons/v16_published_formal/comparison.json `
  --dpi 300
```

如果正式 checkpoint 不是 `Kc=96`，应把 `--max-internal-knots`、`--paper-initial-knots` 和 `--liang-dense-knots` 同时改为 checkpoint 的实际**内部候选容量**，并在论文表格中披露该值。若实验规定“完整节点向量最多 64 项”，三次样条应统一改为 56 个内部节点，而不是 64。不要为了改善某个方法的排名而单独更改阈值、样本、最终 refit 或失败样本处理规则。

## 5. 结果解释边界

- 这些实现足以比较“同一输入和统一最终 refit 下的可运行算法”，但不能宣称逐项复现论文表格；原论文的数据、误差范数、参数化、正则参数和硬件并不完全相同。
- Liang 结果必须始终写成 `feature-integral + IKI adaptation`；Dung 结果必须注明 `serial/simple-knot adaptation`；Luo 必须写明 `l_inf,1 + DE adaptation`。
- 对真实数据，优先同时检查 192 点输入 MSE 和原始参考点 MSE。前者通过而后者失败，通常表示重采样网格拟合良好但原始几何泛化不足。
- 比较节点简洁性时必须以“达到同一误差阈值”为前提；不可行的少节点解不能被解释为更优压缩。
