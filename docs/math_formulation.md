# v6 数学形式

## Canonical 监督

从源节点集合 \(U_0\) 开始，在几何 RMS 容差 \(\varepsilon\) 下贪心删除节点：

\[
U^*=\operatorname{GreedyRemove}(U_0),\qquad
\operatorname{RMS}(C_{U^*},Q)\le\varepsilon.
\]

监督数量为 \(K^*=|U^*|\)，默认 \(\varepsilon=0.005\)。

## 点参数

ParameterHead 预测正间隔并累加：

\[
t_0=0,\qquad t_i=\sum_{r<i}\Delta t_r,\qquad t_{M-1}=1.
\]

## 交互式结构数量分布

结构 query 对 \(F_{local}+\operatorname{PE}(t)\) 做 cross-attention，再做 query self-attention。第 \(j\) 个 query 输出条件停止概率：

\[
h_j=\sigma(a_j).
\]

数量概率为：

\[
p_0=h_1,
\]

\[
p_k=\left[\prod_{r=1}^{k}(1-h_r)\right]h_{k+1},
\quad 1\le k<K_{max},
\]

\[
p_{K_{max}}=\prod_{r=1}^{K_{max}}(1-h_r).
\]

survival probability 为：

\[
s_j=P(K\ge j)=\prod_{r=1}^{j}(1-h_r).
\]

训练目标使用前缀监督：

\[
L_{structure}=\frac1{K_{max}}\sum_{j=1}^{K_{max}}
\operatorname{BCE}(s_j,\mathbf 1[K^*\ge j]).
\]

最终数量为：

\[
\widehat K=\arg\max_k p_k.
\]

## 动态节点解码

给定正整数数量 \(K\)，只运行 \(K+1\) 个 interval query；\(K=0\) 时无需位置解码。正区间为：

\[
\Delta_j^{(K)}=\delta+
[1-(K+1)\delta]
\frac{\exp a_j^{(K)}}{\sum_r\exp a_r^{(K)}}.
\]

节点位置为：

\[
u_j^{(K)}=\sum_{r=0}^{j-1}\Delta_r^{(K)},
\qquad j=1,\ldots,K.
\]

节点位置监督：

\[
L_{knot}=\frac1{K^*}\sum_{j=1}^{K^*}
\operatorname{SmoothL1}(u_j^{(K^*)},u_j^*).
\]

## 可微拟合代理

\[
\Phi=[1,t,t^2,t^3,m_j(t-u_j)_+^3],
\]

\[
D^*=\arg\min_D\|\Phi D-Q\|_F^2
+\lambda_{poly}\|D_{poly}\|_F^2
+\lambda_{knot}\|D_{knot}\|_F^2.
\]

总损失为：

\[
L=L_{fit}+0.05L_t+0.005L_{structure}
+0.002L_{over}+0.05L_{knot},
\]

其中：

\[
L_{over}=\max(0,\mathbb E[K]-K^*).
\]

## 标准 B 样条部署

部署只使用 \(\widehat K\) 对应的节点并执行一次控制点重拟合：

\[
P^*=\arg\min_P\|BP-Q\|_F^2
+\lambda_s\|D_2P\|_F^2+\lambda_r\|P\|_F^2.
\]

不存在 BIC、第二次数量选择或节点阈值删除。
