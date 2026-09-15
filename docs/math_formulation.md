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

当前 source `K^*\in[4,56]`，候选容量 `K_c=72`，所以最大 source K 仍有 16 个冗余候选。三次开放节点向量全保留时共有 `K_c+8=80` 项。

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

当前正式配置始终有 `K_c>K^*`，因此只标记 `K^*` 个不同候选，剩余候选是明确的冗余槽位。`K_c=K^*` 的逐序位情形只保留为通用实现测试。directed coverage 仍独立保留，用于强调真节点召回；assignment 防止多个真节点坍缩到同一候选。

设第 `r` 个真节点到最近候选的距离为 `d_r`。多尺度 Proposal recall surrogate 使用训练尺度 `S={0.0025,0.005,0.010}`：

\[
L_{\mathrm{ms}}=\operatorname{mean}_r\operatorname{mean}_{s\in S}
\log\!\left(1+(d_r/s)^2\right)
+\tfrac12\operatorname{CVaR}_{20\%}(r).
\]

报告时另外直接计算硬阈值 `R@0.005/0.010/0.020`，不能把 surrogate loss 当成 recall 百分比。

## 4. Selector 与一次性计数

Selector 输出候选相对 importance `s_j` 和曲线级阈值偏移 `beta`。部署视图为

\[
p_j=\sigma(s_j-\beta),\qquad
\hat k_{\mathrm{soft}}=\sum_j p_j.
\]

为避免 Keep 排序和计数校准用同一 logit 相互拉扯，训练构造两个数值相同、梯度不同的视图：

\[
\ell^{\mathrm{structure}}_j=s_j-\operatorname{stopgrad}(\beta),\qquad
\ell^{\mathrm{count}}_j=\operatorname{stopgrad}(s_j)-\beta.
\]

existence、ranking、Dice、CDF、fine-teacher 与 entropy 只使用 structure 视图；count 与 over-count 只使用 count 视图。部署仍使用 `s_j-beta`，所以前向概率和一次性 Top-K 语义不变。

部署将概率质量转换为一个整数计数，再对候选分数执行一次全局 Top-K，得到离散 mask `z`。这里没有逐次删除，也不对多个 K 做部署 refit 搜索。

监督 existence 使用正负类分别归一化的 BCE，并提高误删正候选的权重。对于未匹配但靠近任一真节点的负候选，负类权重为

\[
w_j^-=f+(1-f)\operatorname{clip}(d_j/R,0,1),
\]

当前 `R=0.01, f=0.1`。这只降低身份模糊负例的惩罚，不会把它改成正标签。排序项为

\[
L_{\mathrm{rank}}=
\underset{y_i=1,y_j=0}{\operatorname{mean}}
\operatorname{softplus}(m-s_i+s_j).
\]

计数项在 `log(1+k)` 域约束 probability mass 接近 `K*`，并另加 over-count 惩罚。

此外，用概率集合的 soft Dice 约束整体重叠；再按候选位置排序、分别归一化预测概率和标签质量，对两者累计分布函数做 MSE。前者关心“选了多少正确槽位”，后者关心“Keep 概率是否沿参数域放在正确区域”。

## 5. 认证删除风险

最简性认证为每个真节点保存 single-deletion MSE `D_r^*`。同一有序匹配把它搬到对应正候选槽位。对一条曲线的有效真节点，先根据 `log((D_r^*+delta)/epsilon)` 计算 tie-aware midrank 分位数 `q_r\in[0,1]`；相同删除误差共享相同 midrank。风险定义为

\[
r_j=\left[0.25+0.75q_j^{1/T}\right]y_j,
\qquad T=0.5.
\]

单节点曲线的风险取 1；其余有效正节点风险位于 `0.25..1`。`r_j` 表示同一曲线内部的相对删除重要性，不再把已普遍高于阈值的绝对 margin 经 sigmoid 一起压到 1。训练增加 `r_j softplus(-ell_structure_j)`，并用 `r_j` 加强正候选相对负候选的排序 margin。`D*` 在合成最简性认证时由 CPU float64 标准 refit 得到；它不是当前 Selector 产生的伪标签，loss forward 也不为此再运行 subset search。

## 6. 条件解码与重定位

Joint 的前 8 个 epoch 是 Selector warmup：Proposal 与 Parameter 模块的学习率为零，Selector 和 decoder 分别使用 `2e-4` 与 `5e-5`。随后 Proposal 主干、ParameterHead、Selector、decoder 的学习率分别为 `1e-5、5e-5、2e-4、5e-5`。这只改变优化路径，不改变下面的部署函数。

给定 mask `z`，selected-only decoder 只让存活候选作为 survivor Key/Value 交互，并用集合的邻距、相对 rank 和数量等条件联合更新：

\[
(\hat t_z,\hat U_z)=D(H,U_{\mathrm{prop}},z).
\]

输出经过有序和边界约束。Joint 同时解码两条路径：实际部署 mask `z_deploy` 和标签 mask `y`；两者都接受参数、节点位置和拟合监督。训练正式路径每 batch 只需 dense、deployment-mask、label-mask 三类拟合，不构造在线 Teacher。

survivor relocation 先将 Proposal 节点随新参数化 warp，再与 survivor rank 参考做可学习混合。新训练的全局 blend logit 对应初值 `b_0=0.03`，随后叠加逐候选残差并经 sigmoid；这避免从近零饱和状态起步，使 Joint 初期便能获得位置调整梯度。旧 checkpoint 的已保存 logit 不受该初始化规则影响。

节点从预测参数域 warp 到真参数域时使用保持 forward 数值不变的梯度缩放

\[
\operatorname{sg}_{\alpha}(t)=\operatorname{stopgrad}(t)
+\alpha\bigl(t-\operatorname{stopgrad}(t)\bigr).
\]

当前 Proposal `alpha=0`，Joint `alpha=0.1`。因此 Joint 的节点位置误差能有限地修正 ParameterHead，又不会让离散节点任务完全支配参数化。

参数监督除逐点 MSE 外，还包含

\[
L_{\mathrm{gap}}=\operatorname{SmoothL1}
\left(\log\Delta t-\log\Delta t^*\right),\qquad
L_{\mathrm{bias}}=\operatorname{mean}_b
\left(\frac{\operatorname{mean}_i(t_{bi}-t^*_{bi})}{0.01}\right)^2.
\]

log-gap 给每个相邻采样区间相对尺度信号，bias 项显式约束整条曲线向左或向右的系统漂移。

## 7. 损失

Proposal 阶段可写为

\[
L_{\mathrm{proposal}}=
\lambda_dL_{\mathrm{dense}}+
\lambda_tL_t+
\lambda_cL_{\mathrm{coverage}}+
\lambda_aL_{\mathrm{assignment}}+
\lambda_mL_{\mathrm{ms}}+
\lambda_gL_{\mathrm{gap}}+
\lambda_bL_{\mathrm{bias}}.
\]

Joint 阶段增加

\[
L_{\mathrm{joint}}=L_{\mathrm{proposal}}+
\lambda_fL_{\mathrm{selected\ fit}}+
\lambda_eL_{\mathrm{exist}}+
\lambda_D L_{\mathrm{Dice}}+
\lambda_C L_{\mathrm{CDF}}+
\lambda_q L_{\mathrm{delete\ risk}}+
\lambda_kL_{\mathrm{count}}+
\lambda_oL_{\mathrm{overcount}}+
\lambda_rL_{\mathrm{rank}}+
\lambda_uL_{\mathrm{relocation}}.
\]

`L_selected fit` 对 deployment-mask 与 label-mask 的阈值感知拟合项取平均；参数和重定位监督也同时作用于两条路径。重定位项同时包含有序一一匹配误差与真节点到存活节点的定向覆盖误差，因此部署节点不足时，遗漏的真节点仍会产生直接几何惩罚。正式 3090 profile 将 source K 视为 exact label，复杂度权重为零，不使用 online Teacher 或伪标签。

## 8. checkpoint 选择量

对单曲线预测 MSE `e`、节点数 `k` 和容量 `Kc`，soft subset cost 为

\[
J(e,k)=
\begin{cases}
\dfrac{k+0.25\min(e/\epsilon,1)}{K_c+1}, & e\le\epsilon,\\[6pt]
2+\max(\log(e/\epsilon),0), & e>\epsilon.
\end{cases}
\]

可行解的任意一次少节点优先于完整 MSE tie-break；不可行解不因节点少而获奖。成熟 Joint checkpoint 以验证集平均 `J` 选择。aggregate pass 仅作为报告统计。

Proposal 与最终模型承担不同职责：`.proposal.pt` 保存最适合初始化 Joint 的最佳 Proposal，并不要求等于 Proposal 最后一代；`.pt` 保存成熟 Joint 中按上述验证目标选择的最佳部署模型；`.last.pt` 保存最新训练/优化器/RNG 状态用于恢复，不能自动替代最佳 `.pt`。

## 9. 最简性声明

认证合成源满足固定干净参数化下：完整 source 节点集达到阈值，删除任一 source 节点后均超过带 margin 的阈值。由样条空间嵌套性，这证明 source 原节点所有真子集均不可行；它不覆盖节点自由移动或重新参数化后的连续空间。因此论文中应写“source-subset threshold-minimal”，不能写“连续全局最优节点数已证明”。
