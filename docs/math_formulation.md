# 数学形式

## 1. Canonical 监督目标

给定采样点 \(Q=(q_i)\) 和源内部节点集合 \(U_0\)，数据标注阶段反复删除单个节点并重拟合控制点。最终标签满足：

\[
U^*=\operatorname{GreedyRemove}(U_0),\qquad
\operatorname{RMS}(C_{U^*},Q)\le\varepsilon.
\]

默认 \(\varepsilon=0.005\)。监督数量为 \(K^*=|U^*|\)，不再等同于随机生成器的控制点数量。

## 2. 点参数

ParameterHead 预测正间隔并归一化：

\[
t_0=0,\qquad t_i=\sum_{r<i}\Delta t_r,\qquad t_{M-1}=1.
\]

## 3. 序数节点计数

局部注意力产生复杂度证据 \(e\)。递增阈值 \(b_r\) 定义 survival probability：

\[
s_r=P(K\ge r)=\sigma(e-b_r),\qquad r=1,\ldots,K_{max}.
\]

类别概率为：

\[
p_0=1-s_1,\quad p_k=s_k-s_{k+1},\quad p_{K_{max}}=s_{K_{max}}.
\]

序数监督为：

\[
L_{ordinal}=\frac1{K_{max}}\sum_r
\operatorname{BCEWithLogits}(e-b_r,\mathbf 1[K^*\ge r]).
\]

过预测惩罚为：

\[
L_{over}=\max(0,\mathbb E[K]-K^*),\qquad
\mathbb E[K]=\sum_k kp_k.
\]

## 4. 数量条件节点解码

共享 query 加入数量 embedding 后对局部特征做 cross-attention。给定 \(K\)，区间为：

\[
\Delta_j^{(K)}=\delta+
[1-(K+1)\delta]\frac{\exp a_j^{(K)}}{\sum_r\exp a_r^{(K)}}.
\]

节点位置为：

\[
u_j^{(K)}=\sum_{r=0}^{j-1}\Delta_r^{(K)},
\qquad j=1,\ldots,K.
\]

所有区间和为 1 且不小于 \(\delta\)，因此节点天然严格有序。位置监督为：

\[
L_{knot}=\frac1{K^*}\sum_{j=1}^{K^*}
\operatorname{SmoothL1}(u_j^{(K^*)},u_j^*).
\]

## 5. 可微拟合代理

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
L=L_{fit}+0.05L_t+0.005L_{ordinal}
+0.002L_{over}+0.05L_{knot}.
\]

## 6. 部署阶次选择

对每个完整数量分支构造标准开放三次 B 样条并重拟合控制点：

\[
P_K^*=\arg\min_P\|B_KP-Q\|_F^2+\lambda_s\|D_2P\|_F^2.
\]

最终数量最小化：

\[
S_K=N\log(\operatorname{SSE}_K/N)+d_K\log N-2\eta\log p_K.
\]

该过程选择完整模型阶次，不对固定候选集合执行逐节点删除。
