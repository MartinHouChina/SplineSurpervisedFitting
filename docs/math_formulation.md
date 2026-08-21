# v6 数学形式

> 本文前半部分是当前已实现的 v6 数学定义；末尾“规划框架”只定义下一阶段目标，尚未对应现有 checkpoint。

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

结构 query 对 \(F_{local}+\operatorname{PE}(t)\) 做 cross-attention，再做 query self-attention。汇聚特征经分类 MLP 得到数量 logits \(\ell_k\)。对数据集合法范围 \([K_{min},K_{max}]\)：

\[
p_k=\frac{\exp \ell_k}{\sum_{r=K_{min}}^{K_{max}}\exp \ell_r},
\qquad p_k=0\ \text{if}\ k<K_{min}.
\]

训练目标为：

\[
L_{structure}=-\log p_{K^*}.
\]

部署使用后验中位数，它是绝对数量误差下的 Bayes 估计：

\[
\widehat K=\min\left\{k:\sum_{r=0}^{k}p_r\ge 0.5\right\}.
\]

后验众数 \(\arg\max_kp_k\) 仍被记录用于置信度诊断。旧 v6 hazard checkpoint 保留原 survival 参数化以兼容权重，但在决策前同样应用合法范围掩码和后验中位数。

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
+0.05L_{knot}.
\]

## 标准 B 样条部署

部署只使用 \(\widehat K\) 对应的节点并执行一次控制点重拟合：

\[
P^*=\arg\min_P\|BP-Q\|_F^2
+\lambda_s\|D_2P\|_F^2+\lambda_r\|P\|_F^2.
\]

默认附加硬约束：

\[
P_0=Q_0,\qquad P_{n-1}=Q_{M-1},
\]

对开放夹持 B 样条即 \(C(0)=Q_0,C(1)=Q_{M-1}\)。实现只求解内部控制点，并把固定端点对数据项和平滑项的贡献移到右端。

不存在 BIC、第二次数量选择或节点阈值删除。

## 规划框架：候选生成与消冗

### 候选热力图

给定真实最简节点 \(U^*=\{u_j^*\}\) 和参数网格 \(g_i\)：

\[
y_i=\max_j\exp\left[-\frac{(g_i-u_j^*)^2}{2\sigma^2}\right],
\]

\[
L_{heatmap}=\operatorname{FocalBCE}(s_i,y_i).
\]

正样本位置残差和单向覆盖损失为：

\[
L_{offset}=\operatorname{SmoothL1}(\widehat\delta_i,u_j^*-g_i),
\]

\[
L_{cover}=\frac1{K^*}\sum_j\min_i|u_j^*-c_i|.
\]

多余候选是允许的，因此不采用对称 Chamfer Loss。候选坍缩由：

\[
L_{rep}=\sum_{i<j}\max(0,d_{min}-|c_i-c_j|)^2
\]

抑制。

### Boehm 消冗监督

\[
(U^*,P^*)\xrightarrow{\mathrm{Boehm}}(\widetilde U,\widetilde P),
\qquad
C(t;U^*,P^*)=C(t;\widetilde U,\widetilde P).
\]

原始必要节点标签为 1，插入且可删除节点标签为 0。删除误差定义为：

\[
E_{del,j}=L_{fit}(\widetilde U\setminus\widetilde u_j)-L_{fit}(\widetilde U).
\]

消冗头输出保留概率 \(p_j\) 和位置残差 \(\delta_j\)：

\[
\widehat K=\sum_j\mathbf 1[p_j\ge\tau],
\qquad
u_j^{final}=c_j+\delta_j.
\]

数量一致性和规划总目标为：

\[
L_{count}=\left|\sum_jp_j-K^*\right|,
\]

\[
L=L_{proposal}+\lambda_kL_{keep}+\lambda_nL_{count}
+\lambda_dL_{delete}+\lambda_rL_{refine}+\lambda_fL_{fit}.
\]

该设计不使用独立 CountHead；Boehm 只生成消冗监督，部署输入仍为点云。完整训练与部署边界见 [proposal_pruning_framework.md](proposal_pruning_framework.md)。
