# v16 模型架构与张量流

v16 接收有序点云和 MSE 阈值，一次性输出参数、候选节点、离散 KeepMask 和重定位后的存活节点。最终控制顶点不由网络直接回归，而是通过一次标准三次 B 样条最小二乘 refit 求得。

正式合成训练输入还带有独立的数据合同：源曲线先在干净固定参数化上通过全部单节点删除证书，再加入观测噪声；通过认证的样本返回真参数、真内部节点和真内部节点数。证书与真值监督只用于训练/验证，不增加部署网络分支。详见[合成曲线最简性报告](synthetic_data_minimality_report.md)。

![v16 pipeline](figures/v16_pipeline.svg)

## 1. 输入与最终输出

| 名称 | 形状 | 含义 |
|---|---|---|
| points | [B,M,D] | 沿曲线方向排序并归一化的点，D=2 或 3 |
| mse_tolerance | 标量或 [B] | 归一化 MSE 阈值 ε |
| params | [B,M] | 最终严格递增参数 t1，首尾为 0、1 |
| proposal_internal_knots | [B,Kc] | 稠密有序候选 U0 |
| keep_probabilities | [B,Kc] | 每个候选的一次性保留概率 |
| adaptive_keep_threshold | [B] | 每条曲线的动态阈值 β |
| one_shot_requested_count_score | [B] | 概率质量、置信储备和安全节点之和 |
| learned_keep_mask | [B,Kc] | 由动态数量和一次 Top-K 得到的布尔集合 |
| internal_knots | [B,Kc] | 重定位后候选张量；未选槽位只是占位 |

训练批次还包含补齐后的 `target_params [B,M]`、
`target_internal_knots [B,K_source_max]`、同形状的
`target_internal_knot_mask` 和 `target_internal_knot_count [B]`，并带有效
标志 `target_geometry_valid`、`target_internal_knot_count_valid`。这里的
`K_source_max` 是合成源标签上限，不是候选容量 `Kc`。真实样本的有效标志
为 false，因此这些占位零值不参与监督。

真正部署的内部节点为：

\[
U_{\mathrm{deploy},b}
=
\operatorname{sort}\left(
\text{internal\_knots}_b[
\text{learned\_keep\_mask}_b]\right).
\]

节点数是 KeepMask.sum()。网络没有独立 CountHead，也不会先预测整数 K 再运行变长节点头。

## 2. 几何编码

GeometryEncoder 输入点坐标，并用归一化弦长构造一阶、二阶有限差分。共享一维卷积产生：

- local_features：[B,M,H]，保留每个采样位置的局部几何；
- global_features：[B,H]，通过逐通道最大池化汇总整条曲线。

二阶差分提供弯折信息，不等于精确曲率。

## 3. 初始参数 t0

代码先由相邻点距离得到弦长参考参数 tchord。ParameterHead 同时读取 local_features、global_features 和 tchord，预测弦长间隔的有界残差：

\[
0=t_{0,0}<t_{0,1}<\cdots<t_{0,M-1}=1.
\]

因此参数头的输入不是 KeepMask 或节点数；它在候选生成之前提供第一版参数化。

## 4. 稠密候选 U0

CandidateKnotHead 读取 global_features、local_features 和 t0。Kc 个锚定 query 对带参数位置的逐点特征执行局部 Gaussian cross-attention，输出：

- proposal_internal_knots U0：[B,Kc]；
- candidate_tokens：[B,Kc,H]。

候选位置采用均匀锚点的有界残差，并投影为严格有序正间隔。Kc 是候选容量，不是预期最终节点数。当前工程协议要求 worst-source deployment pass≥90%，同时保持单曲线 MSE 阈值 ε=2.5e-5；通过率阈值与误差阈值不可混为一项。2026-09-09 的 Kc=64 快照来自原 97% 配置，只能按该原配置解释。

`Kc` 在代码中专指内部候选。三次开区间样条全保留时，完整节点向量长度为 `Kc+8`，控制顶点数为 `Kc+4`；因此“完整节点向量64”对应 `Kc=56`，而 `--candidate-knots 64` 对应完整向量72。

## 5. 上下文筛选器

每个候选 token 加入以下显式特征：

| 特征 | 作用 |
|---|---|
| uj | 候选在参数域的位置 |
| uj − uleft、uright − uj | 左右候选间隔 |
| min distance(uj,t0,i) | 到最近采样参数的距离 |
| log(ε / εbase) | 让同一模型感知容差 |

随后经过两类注意力：

1. 候选 self-attention：学习候选之间的互补、冲突和全局分布；
2. 对逐点 memory 的 cross-attention：读取曲线局部几何和参数位置编码。

线性 KeepHead 输出每个候选的相对重要度 `s`。为了让“候选排序”和“曲线复杂度”可辨识，先对同一曲线的 `s` 做中心化；由候选池、全局几何和容差共同输入一个曲线级阈值头，得到 β：

\[
\ell_j=(s_j-\operatorname{mean}(s))-\beta,
\qquad p_j=\sigma(\ell_j).
\]

部署数量不是逐点 `p≥0.5`，而是：

\[
\hat K=\operatorname{clamp}\left(
\left\lceil\sum_jp_j+\sigma_s\sqrt{\sum_jp_j(1-p_j)}+K_s\right\rceil,
K_{\min},K_c\right).
\]

Joint 初期使用 \((K_s,\sigma_s)=(2,0.25)\) 保护召回；验证通过率达到
`target + margin` 后全速退火到 \((0,0.05)\)，位于 target 与
`target + margin` 之间时以 0.5 倍速度继续退火。若 worst-source
deployment pass 跌破 target，则以 2 倍速度降低复杂度压力、恢复选择储备。
随后仅执行一次全局 Top-K。默认还在4个参数区间各预留一个高分覆盖锚点，
前提是预算足够；空分区不产生 anchor，也不会覆盖先前非空分区的 anchor。
这里没有样条求解、阈值扫描或迭代删除；正式 checkpoint 必须在最终安全
储备下完成验证。

## 6. KeepMask 条件下的联合解码

decode_subset(context, KeepMask) 让同一个离散集合同时决定参数更新和节点重定位。

### 6.1 Selected-only memory

候选 token 叠加四个集合特征：

- 到最近左、右存活节点的距离；
- 在存活集合中的相对 rank；
- 存活数量除以 Kc。

未选候选在 attention 的 Key/Value 中被屏蔽。只有空集合时启用一个零 sentinel，避免整行 attention 被完全屏蔽。

### 6.2 参数 t0 → t1

逐点特征作为 Query，存活节点 memory 作为 Key/Value。Parameter update 预测每个参数间隔的有界乘性修正，再归一化到正间隔单纯形，保证 t1 严格递增且首尾固定。

### 6.3 节点 warp 与重定位

候选先按采样点对应关系从 t0 单调映射到 t1，再结合：

- 映射后位置；
- 存活 rank 的均匀位置；
- survivor attention 的位置残差；
- 可学习 blend。

最后只把存活节点的区间投影到满足最小间距的正间隔集合。新训练的 rank-uniform blend 从0开始，即初始严格保持映射后位置；网络有证据时再学习重定位。节点因此能跨越已删除候选留下的区域，而不是停留在原 proposal 位置。

## 7. 训练专用的简化教师与真值约束

Joint 训练按 Selector 分数把候选排成固定顺序，再只在“前 K 个”这一族
组合上做粗到细可行性搜索。所有曲线先检查低 K 加密的二次网格；默认
`Kc=96`、`Kmin=4` 和 7 个区间时约为
`4, 6, 12, 21, 34, 51, 72, 96`。certified Synthetic 再额外检查自己的
真 K，之后在首个粗网格可行边界中批量二分，并检查当前最佳前缀的
`K±2` 邻域及局部增、删、交换组合。搜索输出一个
**coarse-to-fine approximate minimum feasible ranked-prefix teacher**。

这不是全局最优证明：参数更新和存活节点重定位会使 MSE 随 K 的关系出现
非单调。教师只负责提供比旧对数预算更细的训练目标，部署图中不存在该
搜索支路。

计数与位置监督分为四层：

1. 教师计数使用 `log1p` 损失，不除以 `Kc`，保持不同候选容量下的梯度尺度；
2. certified Synthetic 的 `log1p` 真计数损失以
   `max(K_true-0.25,0)` 为目标，使后续 `ceil` 落到 source K；部署不可行
   时只纠正 under-count，可行后才使用对称拉回，并再加单边超量惩罚；
3. proposal/final 参数同时靠近真参数；节点位置比较前，将 proposal 按
   `proposal_params -> true_params`、deployment 按
   `final_params -> true_params` 做分段线性可微 warp；proposal 使用真节点
   到候选的定向覆盖损失；
4. 存活节点的 detached 坐标用于求一维单调、最大基数、最小 L1 一一匹配，
   匹配后的预测坐标接受 SmoothL1 梯度；未匹配数量由真计数损失处理。

上述一一匹配取代双向 Chamfer，避免多个存活节点聚到同一个真节点附近而
仍取得较低位置损失。

复杂度项乘以一个由验证通过率控制的系数：pass≥`target+margin` 时全速
增加到默认上限 4，target≤pass<`target+margin` 时以 0.5 倍速度继续增加，
pass<target 时以 2 倍速度降低。安全储备同步反向变化。它是
**validation-pass feedback complexity multiplier**，不是严格的
Lagrangian 或 primal-dual 算法。

对于 source K=4～20 的 certified Synthetic，真计数均值约为 12，本轮目标
是使平均预测 K 落在 10～14 且逐样本接近真计数和真节点位置。该范围必须
通过重新训练验证，架构本身不作无训练保证。

## 8. 标准 B 样条 refit

网络前向结束后，外部求解器用 t1 和 Udeploy 构造开放三次 B 样条基，固定插值两个端点，并最小二乘求控制顶点。部署只做这一次 refit。

网络 forward 不包含离散搜索和样条求解；训练损失为了比较节点组合，会额外调用多个可微 refit。论文计时必须明确区分：

- network time：仅一次 forward_deployment；
- full deployment time：网络、节点物化和最终标准 refit；
- 数值基线 time：该方法自身从初始化到最终结果的完整时间。

## 9. 与 v8–v15 的边界

v16 使用独立模型 V16CandidateSelectionNetwork 和 objective candidate_selection_counterfactual_bspline_v16。它不读取 v13–v15 的离线 Hard-RMS teacher cache，也不把旧 checkpoint 改名复用。当前版本继承一次性 `adaptive β + mass_topk` 离散化，并把在线教师改为训练专用的粗到细 ranked-prefix 搜索；这不改变一次性部署语义。历史架构见：

- [v14 参数—结构联合反馈](archive/v14_joint_parameter_structure_feedback.md)
- [v15 部署损失对齐](archive/v15_deployment_aligned_optimization.md)

代码入口：[v16_network.py](../src/spline_fitting/models/v16_network.py)。

## 10. 快速训练的容量与迁移边界

默认快速折中使用 batch=16、train=2400、val=500、real-val=100、epochs=60、proposal-epochs=20。GTX 1070 建议先从 batch 16 开始；RTX 3090 可优先尝试 batch 32。batch 只改变训练吞吐和梯度统计，不改变部署网络的固定深度。

推荐的短实验使用 epochs=50、proposal-epochs=5、train=2000、val=500、real-val=100、batch=16，并通过 `--init-checkpoint` 从现有 proposal 迁移：

- `encoder.*`；
- `parameter_head.*`；
- `candidate_head.*`。

Selector 和 subset decoder 保持随机初始化，从 90% 协议下重新学习筛选与联动重定位。该方式必须使用新的 output；`--resume` 不能改变 batch、数据规模、pass target 或实验指纹。
