# 核心数学形式

## 输入与参数

给定有序采样点：

\[
Q=\{Q_i\}_{i=0}^{M-1},\qquad Q_i\in\mathbb R^d.
\]

ParameterHead 预测正参数间隔：

\[
\Delta t_i=\delta_t+\left(1-(M-1)\delta_t\right)
\operatorname{softmax}(r)_i,
\]

\[
t_0=0,\qquad t_i=\sum_{l<i}\Delta t_l,\qquad t_{M-1}=1.
\]

因此参数严格递增。

## 独立节点 query

编码器局部特征加入参数位置编码：

\[
X_i=F_i+\operatorname{PE}(t_i).
\]

第 \(j\) 个 query 通过 cross-attention 得到 \(h_j\)，并独立回归位置：

\[
\widetilde u_j=\delta_u+(1-2\delta_u)\sigma(w_u^Th_j).
\]

最终候选节点为：

\[
U=\operatorname{sort}(\widetilde U).
\]

排序同步作用于 query 特征，但不重新计算节点位置。

## 有序匹配监督

设真实内部节点为 \(U^*=\{u_r^*\}_{r=1}^{K^*}\)。在保持顺序的一对一匹配中求：

\[
\mathcal M=\arg\min_{\mathcal M}
\sum_{(j,r)\in\mathcal M}|u_j-u_r^*|.
\]

每个真实节点必须匹配一个候选节点。匹配 query 的标签 \(y_j=1\)，其余为 0。

位置损失为：

\[
L_{knot}=\frac{1}{|\mathcal M|}
\sum_{(j,r)\in\mathcal M}\operatorname{SmoothL1}(u_j,u_r^*).
\]

## Hard-Concrete existence

ActivityHead 输出 \(\log\alpha_j\)。Hard-Concrete 非零概率为：

\[
\pi_j=P(z_j>0)=
\sigma\left(\log\alpha_j-\beta\log\frac{-\gamma}{\zeta}\right),
\]

默认 \(\gamma=-0.1\)、\(\zeta=1.1\)。existence 损失直接监督该概率的 logits：

\[
L_{exist}=\operatorname{BCEWithLogits}(\operatorname{logit}\pi,y).
\]

节点数损失为：

\[
L_{count}=\frac{1}{B}\sum_b
\left(\frac{\sum_j\pi_{bj}-K_b^*}{K}\right)^2.
\]

## 可微拟合代理

训练设计矩阵使用三次截断幂基：

\[
\Phi(t,U,z)=
\left[
1,t,t^2,t^3,
\operatorname{stopgrad}(z_1)(t-u_1)_+^3,ldots,
\operatorname{stopgrad}(z_K)(t-u_K)_+^3
\right].
\]

线性系数通过带分组正则的岭回归求解：

\[
D^*=\arg\min_D
\|\Phi D-Q\|_F^2
+\lambda_{poly}\|D_{poly}\|_F^2
+\lambda_{knot}\|D_{knot}\|_F^2.
\]

`stopgrad` 保留门值对拟合的数值影响，但阻止拟合损失更新 ActivityHead。

## 总损失

\[
L=
L_{fit}
+0.01L_t
+0.005L_{exist}
+0.01L_{knot}
+0.002L_{count},
\]

其中：

\[
L_{fit}=\frac1{BM}\sum_{b,i}\|\widehat Q_{bi}-Q_{bi}\|_2^2,
\qquad
L_t=\frac1{BM}\sum_{b,i}(t_{bi}-t_{bi}^*)^2.
\]

## 部署 B 样条

评估时使用固定阈值：

\[
m_j=\mathbf1[\pi_j\ge\tau].
\]

保留节点排序为 \(U_{keep}\)，构造开放三次节点向量：

\[
\Xi=[0,0,0,0,U_{keep},1,1,1,1].
\]

再用标准 B 样条基矩阵 \(B(t;\Xi)\) 重求控制点：

\[
P^*=\arg\min_P
\|BP-Q\|_F^2+\lambda_s\|L_2P\|_F^2.
\]

这里的 \(P^*\) 是最终标准 B 样条控制点，与训练代理系数 \(D^*\) 不同。
