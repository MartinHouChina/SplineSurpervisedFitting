# v16 supervised-only 数学定义

## 1. 问题与误差

输入有序点云为

\[
Q=(q_0,\ldots,q_{M-1})\in\mathbb R^{M\times D}.
\]

网络预测严格递增参数 `t`、内部节点集合 `U`，最终通过标准三次 B 样条最小二乘得到控制顶点 `P`。主误差为

\[
\operatorname{MSE}(C,Q)=\frac1M\sum_{i=0}^{M-1}\lVert C(t_i)-q_i\rVert_2^2.
\]

当前工程阈值为 `epsilon=1e-4`。它不取平方根，也不再除以坐标维数。

## 2. 参数与候选

GeometryEncoder 产生局部记忆 `H` 和全局特征 `g`。ParameterHead 预测正参数间隔并归一化，使

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

CandidateKnotHead 用带 `t` 位置编码的局部 cross-attention 输出有序候选

\[
U_{\mathrm{prop}}=(u_1,\ldots,u_{K_c}),\qquad 0<u_1<\cdots<u_{K_c}<1,
\]

当前 `Kc=56`。

## 3. 有序一一匹配

合成标签为真节点

\[
U^*=(u^*_1,\ldots,u^*_{K^*}),\qquad K^*\le K_c.
\]

通过单调注入 `a:{1,...,K*}->{1,...,Kc}` 求

\[
a^*=\arg\min_{a_1<\cdots<a_{K^*}}
\frac1{K^*}\sum_{r=1}^{K^*}\rho(u_{a_r},u^*_r),
\]

其中 `rho` 为位置回归代价。目标 KeepMask 定义为

\[
y_j=\mathbf 1[j\in\{a^*_1,\ldots,a^*_{K^*}\}].
\]

因此 `Kc>K*` 时只标记 K* 个不同候选，`Kc=K*` 时得到严格逐序位对应。directed coverage 仍独立保留，用于强调真节点召回；assignment 防止多个真节点坍缩到同一候选。

## 4. Selector 与一次性计数

Selector 输出候选分数 `s_j` 和曲线级阈值偏移 `beta`：

\[
p_j=\sigma(s_j-\beta),\qquad
\hat k_{\mathrm{soft}}=\sum_j p_j.
\]

部署将概率质量转换为一个整数计数，再对候选分数执行一次全局 Top-K，得到离散 mask `z`。这里没有逐次删除，也不对多个 K 做部署 refit 搜索。

监督 existence 使用正负类分别归一化的 BCE，并提高误删正候选的权重；排序项为

\[
L_{\mathrm{rank}}=
\underset{y_i=1,y_j=0}{\operatorname{mean}}
\operatorname{softplus}(m-s_i+s_j).
\]

计数项在 `log(1+k)` 域约束 probability mass 接近 `K*`，并另加 over-count 惩罚。

## 5. 条件解码与重定位

给定 mask `z`，selected-only decoder 只让存活候选作为 survivor Key/Value 交互，并用集合的邻距、相对 rank 和数量等条件联合更新：

\[
(\hat t_z,\hat U_z)=D(H,U_{\mathrm{prop}},z).
\]

输出经过有序和边界约束。Joint 同时解码两条路径：实际部署 mask `z_deploy` 和标签 mask `y`；两者都接受参数、节点位置和拟合监督。训练正式路径每 batch 只需 dense、deployment-mask、label-mask 三类拟合，不构造在线 Teacher。

## 6. 损失

Proposal 阶段可写为

\[
L_{\mathrm{proposal}}=
\lambda_dL_{\mathrm{dense}}+
\lambda_tL_t+
\lambda_cL_{\mathrm{coverage}}+
\lambda_aL_{\mathrm{assignment}}.
\]

Joint 阶段增加

\[
L_{\mathrm{joint}}=L_{\mathrm{proposal}}+
\lambda_fL_{\mathrm{selected\ fit}}+
\lambda_eL_{\mathrm{exist}}+
\lambda_kL_{\mathrm{count}}+
\lambda_oL_{\mathrm{overcount}}+
\lambda_rL_{\mathrm{rank}}+
\lambda_uL_{\mathrm{relocation}}.
\]

`L_selected fit` 对 deployment-mask 与 label-mask 的阈值感知拟合项取平均；参数和重定位监督也同时作用于两条路径。重定位项同时包含有序一一匹配误差与真节点到存活节点的定向覆盖误差，因此部署节点不足时，遗漏的真节点仍会产生直接几何惩罚。正式 3090 profile 将 source K 视为 exact label，复杂度权重为零，不使用 online Teacher 或伪标签。

## 7. checkpoint 选择量

对单曲线预测 MSE `e`、节点数 `k` 和容量 `Kc`，soft subset cost 为

\[
J(e,k)=
\begin{cases}
\dfrac{k+0.25\min(e/\epsilon,1)}{K_c+1}, & e\le\epsilon,\\[6pt]
2+\max(\log(e/\epsilon),0), & e>\epsilon.
\end{cases}
\]

可行解的任意一次少节点优先于完整 MSE tie-break；不可行解不因节点少而获奖。成熟 Joint checkpoint 以验证集平均 `J` 选择。aggregate pass 仅作为报告统计。

## 8. 最简性声明

认证合成源满足固定干净参数化下：完整 source 节点集达到阈值，删除任一 source 节点后均超过带 margin 的阈值。由样条空间嵌套性，这证明 source 原节点所有真子集均不可行；它不覆盖节点自由移动或重新参数化后的连续空间。因此论文中应写“source-subset threshold-minimal”，不能写“连续全局最优节点数已证明”。
