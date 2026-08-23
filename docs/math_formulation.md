# v7 数学定义

## 1. 目标与 canonical 标签

对有序点集 (Q=\{q_i\}_{i=1}^M)，目标是：

\[
\min_U |U|\quad\text{s.t.}\quad
R(U)=\sqrt{\frac1M\sum_i\|C_U(t_i)-q_i\|_2^2}\le\varepsilon.
\]

训练标签从源节点 (U_0) 开始做确定性单节点贪心删除：

\[
U^*=\operatorname{GreedyRemove}(U_0;R,\varepsilon).
\]

## 2. 点参数

ParameterHead 预测正间隔并累加：

\[
t_0=0,\qquad t_i=\sum_{r<i}\Delta t_r,\qquad t_{M-1}=1.
\]

## 3. 固定预算候选

CandidateKnotHead 预测 (K_c+1) 个带最小间隔 (delta) 的正区间：

\[
\Delta_j=\delta+
[1-(K_c+1)\delta]\operatorname{softmax}(a)_j,
\]

\[
c_j=\sum_{r=0}^{j-1}\Delta_r,qquad j=1,\ldots,K_c.
\]

因此 (0<c_1<\cdots<c_{K_c}<1)。真实节点到候选的单向覆盖为：

\[
L_{cover}=\frac1{|U^*|}\sum_{u\in U^*}
\frac{\min_j|u-c_j|}{\tau_{match}}.
\]

## 4. 截断幂代理与节点贡献

\[
\Phi=[1,t,t^2,t^3,(t-c_1)_+^3,\ldots,(t-c_{K_c})_+^3],
\]

\[
D^*=\arg\min_D\|\Phi D-Q\|_F^2+D^T\Lambda D.
\]

令 (A=\Phi^T\Phi+\Lambda)，第 (j) 个节点列系数为 (d_j)。删除该列后的正则
二次目标增量可写为：

\[
\Delta J_j=\frac{\|d_j\|_2^2}{(A^{-1})_{jj}}.
\]

消冗头同时读取 (Delta J_j)、(|d_j|^2)、局部残差和左右间距。

## 5. remove/STOP

消冗头输出 (K_c+1) 个动作概率。对当前状态真实可安全删除集合

\[
\mathcal S=\{j:R(U\setminus u_j)\le\varepsilon\},
\]

多正例动作损失为：

\[
L_{action}=
\begin{cases}
-\log\sum_{j\in\mathcal S}P(a=j),&\mathcal S\ne\varnothing,\\
-\log P(a=\mathrm{STOP}),&\mathcal S=\varnothing.
\end{cases}
\]

删除代价使用阈值归一化对数更容易解释：

\[
r_j=\log\frac{R(U\setminus u_j)+\eta}{\varepsilon+\eta}.
\]

(r_j<0) 表示安全，(r_j>0) 表示会越过阈值。解析截断幂增量是输入特征；标准
B 样条删除 RMS 才是阈值监督和部署判据。

## 6. 归一化拟合损失

记全候选截断幂代理的平均欧氏平方误差为 (L_{fit}^{raw})：

\[
L_{fit}=\frac{L_{fit}^{raw}}{\varepsilon^2},
\qquad
L_{violation}=\max\left(0,
\frac{\sqrt{L_{fit}^{raw}}}{\varepsilon}-1\right)^2.
\]

训练设计矩阵始终打开全部候选，keep 概率不接收拟合梯度。

## 7. 标准 B 样条硬部署

对每次候选删除，重新求解：

\[
P^*=\arg\min_P\|BP-Q\|_F^2
+\lambda_s\|D_2P\|_F^2+\lambda_r\|P\|_F^2,
\]

并施加：

\[
P_0=Q_0,\qquad P_{n-1}=Q_{M-1}.
\]

本轮只接受真实 RMS 最低且不超过 (arepsilon) 的删除。最终节点数是接受删除后的集合大小，
与 keep 阈值或 CountHead 无关。
