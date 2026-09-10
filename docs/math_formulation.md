# v16 数学定义

本文只描述 `objective_version=candidate_selection_counterfactual_bspline_v16` 的当前实现。当前主协议使用 `Kc=56` 个内部候选、合成 source `K=4..56`（控制顶点 8..60）和 `MSE<=1e-4`。三次开放样条全保留时完整节点向量为 64 项、控制顶点为 60 个。v16 不使用 v12–v15 的离线教师，但继承并改良了其中有效的一次性 `adaptive beta + mass-TopK` 思路；存活节点重定位由 v16 的集合条件解码器重新实现。

## 1. 阈值约束目标

给定沿曲线方向排序的点序列：

\[
Q=(q_0,\ldots,q_{M-1}),\qquad q_i\in\mathbb R^D,
\]

模型输出参数 t、内部节点集合 U，并通过标准 B 样条最小二乘求控制顶点 P。归一化 MSE 为：

\[
E(U,t,Q)
=
\frac{1}{M}\sum_{i=0}^{M-1}
\lVert C(t_i;U,P^\star)-q_i\rVert_2^2.
\]

理想目标是：

\[
\min_{t,U}|U|
\quad\text{s.t.}\quad
E(U,t,Q)\le\varepsilon.
\]

当前 ε=1×10^-4，对应 RMS=0.01。v16 是该离散连续问题的学习近似，不证明全局最优，也不对单个输入提供硬可行性保证。

### 1.1 合成源的阈值最简证书

当前 `K=4..56` 生成显式使用 `knot_min_span=0.01`。K=56 有 57 个 span；
底层历史默认 0.02 会使最小总长度达到 1.14，因而不能用于当前边界层。

对合成源节点集合 \(U_s\)，干净固定参数化下额外要求

\[
E(U_s,t^\star,Q^\star)\le\varepsilon,
\qquad
\min_{u_j\in U_s}E(U_s\setminus\{u_j\},t^\star,Q^\star)
>\varepsilon(1+m)^2.
\]

这里 margin \(m=0.2\) 施加在 RMS 域，所以在 MSE 域表现为平方。任意真子集都属于某个单删节点空间的子空间；无正则最小二乘误差在缩小函数空间后不会下降，因此该条件证明 \(U_s\) 在其全部源节点子集中达到阈值下最少基数。它不覆盖连续节点重定位或重新参数化，不能解释成全局连续最优。

## 2. 初始参数化

弦长参考参数为：

\[
\bar t_0=0,\qquad
\bar t_i=
\frac{\sum_{r=1}^{i}\lVert q_r-q_{r-1}\rVert_2}
{\sum_{r=1}^{M-1}\lVert q_r-q_{r-1}\rVert_2},
\qquad
\bar t_{M-1}=1.
\]

ParameterHead 对正间隔预测有界乘性残差，归一化后得到：

\[
0=t_{0,0}<t_{0,1}<\cdots<t_{0,M-1}=1.
\]

它在候选和 KeepMask 之前运行。最终集合确定后，subset decoder 还会产生第二次集合条件参数更新 t0→t1。

## 3. 锚定的局部候选

令候选容量为 Kc。CandidateKnotHead 使用 Kc+1 个 interval query，query 锚点为：

\[
a_j=\frac{j+1/2}{K_c+1},\qquad j=0,\ldots,K_c.
\]

逐点 memory 是几何特征与参数位置编码之和。局部 Gaussian 注意力在标准点积 logit 上加入：

\[
b_{j,i}
=
-\frac{1}{2}
\left(\frac{t_{0,i}-a_j}{h}\right)^2,
\]

实现中下限截断为 -30，默认 h=0.08。

候选位置锚点为：

\[
\hat u_j=\frac{j}{K_c+1},\qquad j=1,\ldots,K_c.
\]

令最小候选间距为 δu，允许残差半径：

\[
r_u=\frac12\left(\frac{1}{K_c+1}-\delta_u\right).
\]

相邻 interval score 的差经过 tanh 后得到有界残差：

\[
u_{0,j}
=
\hat u_j+
r_u\tanh(s_{j-1}-s_j).
\]

因此 U0 严格有序、覆盖全域，且候选保持稳定的从左到右槽位身份。

## 4. 上下文保留概率

对每个候选构造覆盖特征：

\[
c_j=
\left[
u_{0,j},
u_{0,j}-u_{0,j-1},
u_{0,j+1}-u_{0,j},
\min_i|u_{0,j}-t_{0,i}|
\right],
\]

其中边界使用 0 和 1。容差以 log(ε/εbase) 嵌入。候选 token 依次经过候选 self-attention 和对逐点特征的 cross-attention。KeepHead 输出原始重要性 \(r_j\)，曲线级阈值头输出 \(\beta\)。先在曲线内中心化，再得到：

\[
p_j=\sigma\!\left(r_j-\bar r-\beta\right).
\]

中心化使 \(r_j\) 主要表达候选排序，\(\beta\) 主要表达曲线复杂度。默认部署数量为：

\[
\widehat K=
\operatorname{clip}\!\left(
\left\lceil
\sum_jp_j+\sigma_s\sqrt{\sum_jp_j(1-p_j)}+K_s
\right\rceil,
K_{\min},K_c
\right).
\]

随后按 \(p_j\) 执行一次全局 Top-K。当前 `coverage_bins=0`，不强制参数区间锚点；安全储备从 \((\sigma_s,K_s)=(0.20,2)\) 退火到 \((0.03,0)\)，且 \(K_{\min}=4\)。Joint 初始 keep fraction 为 \(30/56\)，对应初始概率质量约 30；它只是训练初始化，不是部署最终 K。这里没有额外 CountHead、逐次删除或部署 refit 搜索；`p>=0.5` 仅作为旧式消融保留。

## 5. 集合条件参数更新

对候选 j，使用最近的已选左右邻居、已选 rank 和归一化节点数构造集合特征。未选择候选不能成为 subset attention 的 Key/Value。

逐点 Query 对存活节点 memory 做 attention 后，输出每个旧参数间隔的有界对数修正 hj。设最小参数间距为 δt，旧自由间隔为：

\[
f_i=\max(t_{0,i+1}-t_{0,i}-\delta_t,0).
\]

归一化权重和新间隔为：

\[
w_i=
\frac{f_i\exp(h_i)}
{\sum_r f_r\exp(h_r)},
\qquad
\Delta t_{1,i}
=
\delta_t+
\left(1-(M-1)\delta_t\right)w_i.
\]

累积后得到严格递增 t1。

## 6. 参数域 warp 与存活节点重定位

每个 u0,j 先在相邻采样参数之间做分段线性单调映射：

\[
\tilde u_j
=
t_{1,l}+
\frac{u_{0,j}-t_{0,l}}
{t_{0,r}-t_{0,l}}
\left(t_{1,r}-t_{1,l}\right).
\]

随后 survivor attention 输出位置残差和 blend 修正。基础位置在 warp 结果与存活 rank 的均匀位置之间插值，再允许在最近存活邻居形成的区间内移动。

对每条曲线只将选中节点和两个端点形成的间隔投影到：

\[
\Delta u_i\ge\delta_u,\qquad
\sum_i\Delta u_i=1.
\]

这使存活节点能跨过被删槽位留下的空间，同时保持位于 (0,1) 且严格有序。

## 7. 标准三次 B 样条 refit

给定次数 p=3、内部节点 U 和端点各重复 p+1 次的开放节点向量，构造基矩阵 B(t,U)。控制顶点通过：

\[
P^\star
=
\arg\min_P\lVert BP-Q\rVert_F^2
\quad\text{s.t.}\quad
C(0)=q_0,\ C(1)=q_{M-1}
\]

求得。训练使用可微 float64 求解；部署使用生产 refit。网络不直接回归 P。

## 8. proposal 阶段

全保留掩码 z=1 时计算 dense MSE。为避免早期 MSE/ε 比值产生巨大尺度，拟合惩罚使用对数形式：

\[
L_{\mathrm{fit}}(E,\varepsilon)
=
\log\left(\frac{E+\varepsilon}{\varepsilon}\right)
+\max\left(0,\log\frac{E}{\varepsilon}\right),
\]

实现中对零值加入极小稳定项。proposal 阶段只优化：

\[
L_{\mathrm{proposal}}
=
\lambda_{\mathrm{fit}}\,
\mathbb E[L_{\mathrm{fit}}(E_{\mathrm{dense}},\varepsilon)].
\]

## 9. 节点组合代价

令 r=E/ε。组合评价代价为：

\[
C(z)=
\begin{cases}
\dfrac{K(z)+0.25\min(r,1)}{K_c+1}, & E\le\varepsilon,\\
2+\max(0,\log r), & E>\varepsilon.
\end{cases}
\]

所有可行组合代价小于 1，所有不可行组合至少为 2。可行时少一个节点优先于 MSE tie-break 的全部变化；不可行时不因少节点得到奖励。

在线集合选择遵循：

\[
z^\star=
\begin{cases}
\arg\min_z (K(z),E(z)), & \exists z:E(z)\le\varepsilon,\\
\arg\min_z E(z), & \text{otherwise}.
\end{cases}
\]

这里的候选 z 来自当前部署、有限加/删反事实、全保留集合、`K=4..16` 的逐计数 ranked prefix，以及更高 K 的粗到细预算。certified Synthetic 还加入 geometry-oracle `Ktrue` 和截断到 Kc 的 `Ktrue+2` 组合。独立随机采样只用于策略估计，不能违反部署约束后再充当结构化教师。

## 10. 二值选择梯度

独立采样：

\[
z^{(s)}_j\sim\operatorname{Bernoulli}(p_j).
\]

使用 leave-one-out baseline bs 的 score-function 项：

\[
L_{\mathrm{policy}}
=
\frac1S\sum_s
\operatorname{stopgrad}(C(z^{(s)})-b_s)
\log P_\theta(z^{(s)}).
\]

该损失的梯度用于降低期望组合代价。采样结果保持原始 IID Bernoulli 形式；不会先投影到最小节点数再冒充原分布。确定性的删除、添加、交换和排序预算组合只参与 z* 选择，不进入策略估计器。

联合目标还包括：

- 当前部署集合与 z* 的可微 refit 拟合项；
- 对教师保留项加权的 BCE(logits,z*) 在线蒸馏；
- 概率质量的数量监督与教师保留/删除候选的成对排序监督；
- dense proposal 保持项；
- 默认关闭的熵项；
- 仅对具有严格误差余量的曲线启用、并由验证通过率渐进控制的复杂度项；
- 均值之外的高误差尾部惩罚。

策略标量可以为负，总 loss 不能解释为 MSE。

## 11. 验证与资格

proposal 排序最大化 worst-source dense pass，再最小化 dense MSE。joint 未达 90% 目标时先提高 worst-source deployment pass 并降低 P95/平均 MSE；处于 90% 到 92% 安全线之间时仍优先可靠性；达到 `target + safety margin` 后才先最小化平均 K，再比较 P95 和平均 MSE。

当前 `V16_FORMAL_PASS_RATE=0.90`：工程协议要求 stage=joint、配置目标和实测 worst-source deployment pass 均不低于 0.90，并且 proposal gate 未被消融绕过。这里 0.90 是跨曲线通过比例；每条曲线仍以 E≤1×10^-4 判定，MSE 约束没有放宽。`source K=56` 与 `Kc=56` 同时位于容量边界、没有冗余候选余量，该层的 dense/deployment pass 必须单列，失败不得通过总体平均或放宽 90% 门槛隐藏。

代码对应：

- [v16_network.py](../src/spline_fitting/models/v16_network.py)
- [v16_subset_loss.py](../src/spline_fitting/losses/v16_subset_loss.py)
- [deployment_bspline_loss.py](../src/spline_fitting/losses/deployment_bspline_loss.py)
