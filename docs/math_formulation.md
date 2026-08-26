# v11 数学定义

## 1. 问题

给定有序点：

\[
Q=(q_0,\ldots,q_{M-1}),\qquad q_i\in\mathbb R^D,
\]

预测参数 `t`、内部节点集合 `U` 和控制顶点 `P`，使：

\[
\min |U|
\quad\text{s.t.}\quad
R(U,Q)\le\varepsilon,
\]

其中标准 B 样条重拟合 RMS 为：

\[
R(U,Q)=
\sqrt{\frac1M\sum_{i=0}^{M-1}
\|C(t_i;U,P^*)-q_i\|_2^2}.
\]

v11 学习这个约束问题的一次性近似，不提供逐样本可行性或全局最优证明。

## 2. 严格递增参数

`ParameterHead` 输出正区间并归一化，使：

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

训练时使用：

\[
L_t=\frac1M\sum_i(t_i-t_i^{\mathrm{true}})^2.
\]

节点匹配依赖共享参数化，所以测试时应同时报告 `true_parameter_rmse`。

## 3. 局部 Gaussian cross-attention

设第 `j` 个 interval query 的固定锚点为：

\[
a_j=\frac{j+1/2}{K_c+1}.
\]

点特征加参数位置编码：

\[
x_i=f_i+\operatorname{PE}(t_i).
\]

当局部带宽 `h>0` 时，attention 为：

\[
A_{j,i}=
\operatorname{softmax}_i\left(
\frac{\langle W_Qz_j,W_Kx_i\rangle}{\sqrt d}
-\frac{(t_i-a_j)^2}{2h^2}
\right).
\]

`h=0` 表示不加 Gaussian 偏置，恢复全局 attention。

## 4. 严格有序候选

`Kc+1` 个 interval logit 产生：

\[
\Delta_j=
\delta+
\left[1-(K_c+1)\delta\right]
\frac{\exp \ell_j}{\sum_r\exp\ell_r}.
\]

候选位置：

\[
c_j=\sum_{r=0}^{j-1}\Delta_r,
\qquad j=1,\ldots,K_c.
\]

由构造可知：

\[
0<c_1<\cdots<c_{K_c}<1,
\]

且每个间隔不小于 `δ`。

## 5. 多尺度候选覆盖

对 canonical 节点 `v`，定义到候选集合 `C` 的最近距离：

\[
d(v,C)=\min_j|v-c_j|.
\]

基本 coverage 使用平均最近距离。v11 再加入多尺度 hinge：

\[
L_{\mathrm{multi}}=
\frac{1}{|\mathcal V||\mathcal T|}
\sum_{v\in\mathcal V}
\sum_{\tau\in\mathcal T}
\left[
\max\left(\frac{d(v,C)}{\tau}-1,0\right)
\right]^2,
\]

\[
\mathcal T=\{0.005,0.01,0.02\}.
\]

相同项同时作用于 CandidateKnotHead 的初始候选和 proposal-only 精修候选。

## 6. 截断幂代理与贡献特征

三次截断幂表示为：

\[
C(t)=a_0+a_1t+a_2t^2+a_3t^3+
\sum_{j=1}^{K_c}b_j(t-u_j)_+^3.
\]

全候选代理 solve 得到：

- `\|b_j\|_2^2`：候选系数能量；
- 删除对应列后的解析目标增量；
- 候选附近的局部点残差；
- 左右间距。

这些量经过稳定变换后与候选 token、位置编码一起输入 `InteractivePruningHead`。它们是训练特征，不等价于最终标准 B 样条删除误差。

## 7. 固定 proposal

proposal-only 位置头产生：

\[
u_j^{\mathrm{prop}}=c_j+\Delta u_j^{\mathrm{prop}}.
\]

位移受相邻候选 slack 和最小间隔限制。`U_prop` 是离线教师绑定的不可变槽位集合。

## 8. p0 → u1 → p1

selector 首次预测：

\[
p_j^{(0)}
=\sigma(r_j^{(0)}-\beta^{(0)}).
\]

由 `p0` 构造 hard straight-through gate：

\[
\widetilde m_j^{(0)}
=m_j^{(0)}+p_j^{(0)}
-\operatorname{stopgrad}(p_j^{(0)}).
\]

前向值为 hard mask，反向导数来自 `p0`。其集合上下文驱动临时位置：

\[
u_j^{(1)}
=u_j^{\mathrm{prop}}+\Delta u_j^{(1)}.
\]

`\Delta u1` 不乘 `p0`，因此初始低概率槽位仍能移动并提供第二次选择证据。

将 `u1` 的位置编码变化和归一化位移反馈给 selector：

\[
p_j^{(1)}
=\sigma(r_j^{(1)}-\beta^{(1)}).
\]

这是固定的两次概率计算，不是迭代至收敛。

## 9. 一次性 KeepMask

`mass_topk` 计算概率质量和 Bernoulli 不确定性：

\[
\mu=\sum_jp_j^{(1)},
\qquad
\sigma_K=
\sqrt{\sum_jp_j^{(1)}(1-p_j^{(1)})}.
\]

令请求分数：

\[
q=\mu+s\sigma_K.
\]

请求节点数为：

\[
\widehat K=
\begin{cases}
0,&q<0.5,\\
\lceil q\rceil,&q\ge0.5,
\end{cases}
\]

并截断到有效候选数。然后一次性保留 raw importance 最高的 `Khat` 个有效槽位。可选的参数域覆盖锚点只修改同一次排序得分；默认 v11 不启用分箱锚点。

policy-count 损失不只拟合 `sum p`。教师计数为零时，其连续目标为 `0.25`；教师计数 `K>0` 时，目标为 `K-0.25`。二者分别位于零节点稳定区间和对应正数 `ceil` 区间内部。

## 10. 最终位置 u*

最终 mask 的 hard-ST 上下文驱动第二个位置残差：

\[
u_j^*=u_j^{(1)}+m_j\Delta u_j^{(2)}.
\]

未保留槽位的第二次残差为零。第一阶段先保证：

\[
|u_j^{(1)}-u_j^{\mathrm{prop}}|\le d_{\max}.
\]

第二阶段根据已经消耗的预算限制剩余残差，使：

\[
|u_j^*-u_j^{\mathrm{prop}}|\le d_{\max}.
\]

因此 `d_max` 是两阶段相对 `U_prop` 的合计上限，而不是两个位置头各自的上限。两个阶段还同时受相邻有效节点 slack、`min_gap` 和小于半个可用区间的比例约束，因此选中子序列保持严格有序。

## 11. canonical 与 teacher 双监督

### 11.1 canonical 选择

由候选到 canonical 节点的一维有序分配产生 existence target `y_j^{can}`：

\[
L_{\mathrm{can-sel}}
=\operatorname{BCEWithLogits}(r_j^{(1)}-\beta^{(1)},y_j^{can}).
\]

### 11.2 teacher mask

离线 Hard-RMS 教师给出 `y_j^{teach}`：

\[
L_{\mathrm{keep}}
=L_{\mathrm{weighted\ BCE}}
+L_{\mathrm{soft\ Dice}}.
\]

另外使用：

- retained/remove 排序损失；
- final-state soft-risk BCE；
- critical retained-slot softplus；
- teacher false-positive softplus；
- teacher count 和 policy count；
- teacher 槽位概率质量的累计分布距离；
- complexity。

false-positive 项只把 teacher-negative 且 canonical-negative 的槽位作为负样本。令：

\[
w_j^-=(1-y_j^{\mathrm{teach}})(1-y_j^{\mathrm{can}}),
\]

则其主要形式为：

\[
L_{\mathrm{fp}}=
\frac{\sum_jw_j^-\operatorname{softplus}(\ell_j^{\mathrm{keep}})}
{\max(\sum_jw_j^-,1)}.
\]

canonical-positive 但未被贪心教师保留的邻近替代槽位由此获得豁免。

### 11.3 联合位置

只取教师保留槽位，再与 canonical 节点做有序最小 `L1` 配对：

\[
L_{\mathrm{joint-pos}}
=\operatorname{SmoothL1}
\left(U^*_{\mathrm{teach\ slots}},
U_{\mathrm{canonical}}\right).
\]

因此 teacher 决定“移动哪些槽位”，canonical 决定“这些最终槽位向哪里校准”。`U_prop` 不参与该更新。

## 12. 总损失

候选预训练的主要目标为：

\[
L_{\mathrm{pre}}=
\lambda_tL_t+
\lambda_cL_{\mathrm{coverage}}+
\lambda_mL_{\mathrm{multi}}+
\lambda_uL_{\mathrm{canonical-pos}}+
\lambda_rL_{\mathrm{repulsion}}+
\lambda_fL_{\mathrm{surrogate-fit}}+
\lambda_\varepsilon L_{\mathrm{surrogate-violation}}.
\]

蒸馏/联合校准为：

\[
\begin{aligned}
L_{\mathrm{one}}={}&
\lambda_kL_{\mathrm{keep}}
+\lambda_{\mathrm{risk}}L_{\mathrm{risk}}
+\lambda_{\mathrm{rank}}L_{\mathrm{rank}}\\
&+\lambda_{\mathrm{crit}}L_{\mathrm{critical}}
+\lambda_{\mathrm{fp}}L_{\mathrm{false-positive}}
+\lambda_{\mathrm{dist}}L_{\mathrm{distribution}}\\
&+\lambda_{\mathrm{count}}L_{\mathrm{teacher-count}}
+\lambda_{\mathrm{policy}}L_{\mathrm{policy-count}}
+\lambda_{\mathrm{can}}L_{\mathrm{can-sel}}\\
&+\lambda_{\mathrm{pos}}L_{\mathrm{joint-pos}}
+\lambda_{\mathrm{cmp}}L_{\mathrm{complexity}}.
\end{aligned}
\]

纯选择蒸馏阶段默认令 surrogate fit 和 violation 权重为零。联合位置校准阶段另外加入：

\[
L_{\mathrm{cal}}=L_{\mathrm{one}}
+0.05L_{\mathrm{surrogate-fit}}
+0.5L_{\mathrm{surrogate-violation}}.
\]

代理设计矩阵的 hard-ST fit gate 对选择概率 detach，因此这两项不会通过门控梯度直接增加节点；它们用于约束已选组合的位置可行性。最终 checkpoint 仍依据真实标准 B 样条验证结果排序。

## 13. 标准 B 样条部署

最终节点为：

\[
U=\{u_j^*\mid m_j=1\}.
\]

控制顶点求解：

\[
P^*=\arg\min_P
\|B(t,U)P-Q\|_F^2+
\lambda_s\|D_2P\|_F^2+
\lambda_r\|P\|_F^2,
\]

并施加端点插值：

\[
C(0)=q_0,\qquad C(1)=q_{M-1}.
\]

网络前向一次，标准 B 样条 refit 一次。`R(U,Q)\le\varepsilon` 在默认 one-shot 模式中是测试分布上的统计满足率，而不是每条曲线的硬保证。

## 14. 分阶段节点指标

对容差 `tau`，分别匹配：

\[
U_{\mathrm{prop}},
\qquad
U_{\mathrm{prop}}[M],
\qquad
U^*[M].
\]

得到 proposal、选择后更新前和最终部署的 Precision/Recall/F1/MAE。位置更新增量定义为：

\[
\Delta R=R_{\mathrm{post}}-R_{\mathrm{pre}},
\qquad
\Delta P=P_{\mathrm{post}}-P_{\mathrm{pre}}.
\]

该分解用于判断误差主要来自 proposal 漏检、selector 误删，还是位置更新。
