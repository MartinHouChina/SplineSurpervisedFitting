# 独立节点 Query + 有监督 Existence + Hard-Concrete

当前节点头为每个候选维护独立 query token (h_j)，并直接回归

\[
\widetilde u_j=\delta_u+(1-2\delta_u)\sigma(w_u^Th_j),
\qquad U=\operatorname{sort}(\widetilde U).
\]

排序只同步重排 query token、attention 权重和 existence，不通过区间 softmax
耦合位置。对排序后的预测节点与真值节点做一维有序最小 L1 代价匹配
\(\mathcal M\)。匹配 query 的 existence 标签为 1，未匹配 query 为 0：

\[
L_{exist}=\operatorname{BCEWithLogits}(\operatorname{logit}\pi,y),\qquad
L_{knot}=\frac1{|\mathcal M|}\sum_{(j,r)\in\mathcal M}
\operatorname{SmoothL1}(u_j,u_r^*).
\]

另用归一化计数损失

\[
L_{count}=\left(\frac{\sum_j\pi_j-K^*}{K}\right)^2.
\]

Hard-Concrete 的 \(\pi_j=P(z_j>0)\) 同时接受 existence/count 监督；训练时
采样门 (z_j)，评估时输出固定阈值的严格 0/1 门。以下 interval 累计公式仅为
v0.4 历史参数化说明。

# A 方案数学形式（v0.4 历史分支）

## 1. 输入、参数与候选节点

给定一条含 \(M\) 个有序采样点的曲线

\[
Q=\{Q_i\}_{i=0}^{M-1},\qquad Q_i\in\mathbb R^d,
\]

网络预测投影参数 \(t\)、\(K\) 个候选内部节点 \(U\) 和每个节点的 Hard-Concrete 位置参数 \(\log\alpha\)：

\[
t=f_t(Q),\qquad
U=f_U(Q),\qquad
\log\alpha=f_a(Q,t,U).
\]

### 1.1 严格参数间距

ParameterHead 对 \(M-1\) 个自由 logit \(r_i\) 使用

\[
\Delta t_i
=
\delta_t+
\left(1-(M-1)\delta_t\right)
\operatorname{softmax}(r)_i,
\]

\[
t_0=0,\qquad
t_i=\sum_{\ell=0}^{i-1}\Delta t_\ell.
\]

因此

\[
0=t_0<t_1<\cdots<t_{M-1}=1,
\qquad
\Delta t_i\ge\delta_t,
\]

合法条件为

\[
(M-1)\delta_t<1.
\]

### 1.2 严格候选节点间距

KnotHead 预测包含左右边界在内的 \(K+1\) 个区间：

\[
\Delta u_j
=
\delta_u+
\left(1-(K+1)\delta_u\right)
\operatorname{softmax}(q)_j.
\]

内部节点为

\[
u_j=\sum_{\ell=0}^{j-1}\Delta u_\ell,\qquad j=1,\ldots,K.
\]

于是

\[
0<u_1<\cdots<u_K<1,\qquad
\Delta u_j\ge\delta_u,
\]

合法条件为

\[
(K+1)\delta_u<1.
\]

这与旧版“softplus 后加最小值、再整体归一化”不同；旧公式在归一化后不能保证最小间距。

## 2. 带位置编码的局部 Cross-Attention KnotHead

设共享编码器输出局部特征 \(F_i\in\mathbb R^h\) 和全局特征 \(G\in\mathbb R^h\)。首先将预测参数的位置编码加入局部序列：

\[
M_i=F_i+\operatorname{PE}(t_i).
\]

对包含左右边界的 \(K+1\) 个区间分别设置可学习 query \(e_r\)，并注入全局特征：

\[
q_r=e_r+W_gG.
\]

每个 query 对带位置局部序列做多头 cross-attention：

\[
\widetilde q_r=
\operatorname{LN}\left(
q_r+\operatorname{MHA}(q_r,M,M)
\right),
\]

再经过前馈残差层和标量投影得到第 1.2 节中的区间 logit。这样区间预测可以读取局部几何事件在参数轴上的位置，而不是仅依赖全局最大池化特征。

## 3. 节点局部上下文与 ActivityHead

得到候选节点 \(u_j\) 后，使用带宽 \(h_b\) 的高斯权重汇聚局部上下文：

\[
w_{ji}
=
\frac{
\exp\left[-\frac12((t_i-u_j)/h_b)^2\right]
}{
\sum_{\ell}\exp\left[-\frac12((t_\ell-u_j)/h_b)^2\right]
},
\qquad
\ell_j=\sum_iw_{ji}F_i.
\]

ActivityHead 直接预测 Hard-Concrete 位置参数：

\[
\log\alpha_j
=
\operatorname{MLP}\left([G+\ell_j,\ u_j]\right).
\]

当前方案不使用 pilot、drop-cost 或人工重要性偏置。默认 \(h_b=0.08\)，活动头末层偏置初始化为 -2.0。

## 4. Hard-Concrete 门

### 4.1 训练态采样

对每个候选节点采样

\[
v_j\sim\mathcal U(\epsilon,1-\epsilon),
\]

\[
s_j
=
\sigma\left(
\frac{
\log\alpha_j+\log v_j-\log(1-v_j)
}{\beta}
\right),
\]

\[
\bar s_j=\gamma+(\zeta-\gamma)s_j,
\qquad
z_j=\operatorname{clip}(\bar s_j,0,1).
\]

默认

\[
\gamma=-0.1,\qquad
\zeta=1.1,\qquad
\epsilon=10^{-6}.
\]

\(\beta>0\) 为温度。训练态的 \(z_j\in[0,1]\) 是随机、可重参数化并可反向传播的，同时在 0 和 1 处具有概率质量。

### 4.2 期望非零概率

每个门非零的闭式概率为

\[
\pi_j=P(z_j>0)
=
\sigma\left(
\log\alpha_j-\beta\log\frac{-\gamma}{\zeta}
\right).
\]

代码语义：

| 字段 | 张量形状 | 数学量 |
|---|---|---|
| `activity_logits` | \([B,K]\) | \(\log\alpha\) |
| `activity` | \([B,K]\) | \(\pi\) |
| `l0_probability` | \([B,K]\) | \(\pi\) |
| `expected_l0` | \([B]\) | \(\sum_j\pi_j\) |
| `activity_gate` | \([B,K]\) | 训练态 \(z\)，评估态 \(m\) |

### 4.3 评估态硬门

评估态不采样，而是使用

\[
m_j=\mathbf 1[\pi_j\ge\tau].
\]

因此 `activity_gate` 严格属于 \(\{0,1\}\)。默认 \(\tau=0.5\)，实际实验中应在验证集上确定一次并冻结。

KnotHead 另外输出 `knot_attention_weights`，用于检查每个区间 query 在参数轴上关注的局部位置。

## 5. 可微截断幂训练代理

设样条次数为 \(p=3\)。多项式基和截断幂增量基分别为

\[
V_{ir}=t_i^r,\qquad r=0,\ldots,p,
\]

\[
H_{ij}=(t_i-u_j)_+^p.
\]

A 方案直接使用 Hard-Concrete 门：

\[
\Phi_{\mathrm{sur}}(t,U,z)
=
\left[
V(t)\mid H(t,U)\operatorname{diag}(z)
\right].
\]

代理曲线为

\[
C_{\mathrm{sur}}(t)
=
\sum_{r=0}^{p}c_rt^r+
\sum_{j=1}^{K}z_jd_j(t-u_j)_+^p.
\]

门为零时，对应设计矩阵列严格为零，不再使用 \(\sqrt{a+\epsilon}\) 留下非零列。

## 6. 训练内层线性求解

统一记截断幂系数

\[
D=[c;d].
\]

每个输入样本实时求解

\[
D^*
=
\arg\min_D
\|\Phi_{\mathrm{sur}}D-Q\|_F^2+
D^\mathsf T\Lambda D.
\]

当前可微训练层使用带正则线性系统求解 \(D^*\)。\(D^*\) 是计算图中的中间张量，不是 `nn.Parameter`，也不是最终 B 样条控制点。

## 7. 损失函数

### 7.1 拟合项

\[
\mathcal L_{\mathrm{fit}}
=
\frac1{BM}
\sum_{b=1}^{B}\sum_{i=1}^{M}
\|C_{\mathrm{sur},b}(t_{bi})-Q_{bi}\|_2^2.
\]

该定义是“平均每点平方欧氏距离”。逐坐标 MSE 为其除以坐标维数 \(d\)。

### 7.2 期望 \(L_0\) 节点数

\[
\mathcal L_{L_0}
=
\frac1B
\sum_{b=1}^{B}\sum_{j=1}^{K}\pi_{bj}.
\]

这里对节点维求和而不是求均值，因此其数值直接对应每条曲线的期望非零内部节点数。

### 7.3 真实参数监督和节点间距

\[
\mathcal L_{\mathrm{true\text{-}param}}
=
\frac1{BM}\sum_{b,i}
(t_{bi}-t^*_{bi})^2.
\]

合成数据提供 \(t^*\)。当前新训练默认将弦长 parameter-prior 权重设为 0；`gap_loss` 作为候选节点的附加安全约束，作用于全部候选节点。

### 7.4 总目标

\[
\mathcal L
=
\lambda_f\mathcal L_{\mathrm{fit}}
+\rho_0(e)\lambda_0\mathcal L_{L_0}
+\lambda_g\mathcal L_{\mathrm{gap}}
+\lambda_t\mathcal L_{\mathrm{true\text{-}param}}.
\]

A 方案默认

\[
\lambda_{\mathrm{activity}}
=
\lambda_{\mathrm{binary}}
=0.
\]

这两个字段仅用于旧 checkpoint 兼容。投影正交项也已从当前目标移除：联合优化 \(t\) 时，fit 对 \(t\) 的梯度已经包含残差与切向量的一阶驻点信息，再单独平方惩罚会重复施加相近约束。历史 checkpoint 中的非零 `orthogonal` 字段仍可按原配置复现，但不属于上述当前目标。

## 8. 硬剪枝与标准 B 样条

评估态保留节点集合

\[
\widehat U=\{u_j:m_j=1\},
\qquad
\widehat K=|\widehat U|.
\]

对一般次数 \(p\)，开放夹持节点向量为

\[
\Xi=
\left[
\underbrace{0,\ldots,0}_{p+1},
\widehat U,
\underbrace{1,\ldots,1}_{p+1}
\right].
\]

节点向量长度与控制点数分别为

\[
|\Xi|=\widehat K+2(p+1),
\qquad
n_{\mathrm{ctrl}}=\widehat K+p+1.
\]

对三次样条：

\[
|\Xi|=\widehat K+8,\qquad
n_{\mathrm{ctrl}}=\widehat K+4.
\]

\(\widehat K=0\) 是合法情形，对应 4 控制点开放三次 Bézier 表示。

## 9. 标准 B 样条控制点重拟合

由 \((t,\Xi)\) 构造 Cox--de Boor 标准 B 样条基矩阵 \(B\)。控制点通过

\[
P^*
=
\arg\min_P
\|BP-Q\|_F^2
+\lambda_s\|D_2P\|_F^2
+\lambda_r\|P\|_F^2
\]

求得，其中 \(D_2\) 是控制多边形二阶差分矩阵。

实现求解增广最小二乘：

\[
\underbrace{
\begin{bmatrix}
B\\
\sqrt{\lambda_s}D_2\\
\sqrt{\lambda_r}I
\end{bmatrix}
}_{A_{\mathrm{aug}}}
P
\approx
\underbrace{
\begin{bmatrix}
Q\\0\\0
\end{bmatrix}
}_{Y_{\mathrm{aug}}}.
\]

`torch.linalg.lstsq` 直接处理 \(A_{\mathrm{aug}}\)，不形成 \(B^\mathsf TB\)。CPU 路径使用 `gelsy` 列主元 QR，以提高近邻节点情况下的稳健性。

## 10. 部署指标

\[
\mathrm{fit\_mse}
=
\frac1M\sum_i\|B(t_i)P^*-Q_i\|_2^2,
\]

\[
\mathrm{fit\_rmse}
=
\sqrt{\mathrm{fit\_mse}}.
\]

`augmented_objective` 报告

\[
\|BP^*-Q\|_F^2+
\lambda_s\|D_2P^*\|_F^2+
\lambda_r\|P^*\|_F^2.
\]

它使用未归一化的数据平方和，不能直接与训练 total loss 比较。

## 11. 归一化坐标恢复

若输入归一化为

\[
Q_{\mathrm{norm}}
=
\frac{Q_{\mathrm{original}}-\mathrm{center}}{\mathrm{scale}},
\]

则控制点恢复为

\[
P_{\mathrm{original}}
=
\mathrm{scale}\,P_{\mathrm{norm}}+\mathrm{center}.
\]

## 12. 旧软门兼容形式

pre-A checkpoint 使用

\[
\Phi_{\mathrm{legacy}}
=
\left[
V\mid
H\operatorname{diag}(\sqrt{a+\epsilon})
\right],
\qquad
a=\sigma(\mathrm{logits}).
\]

该形式中 \(a\) 更接近列尺度和有效岭强度，而不是真正的离散结构变量。兼容模式只用于复现旧结果；A 方案需重新训练。

## 参考

Hard-Concrete 与闭式 \(L_0\) 概率来自 Louizos、Welling、Kingma 的 [Learning Sparse Neural Networks through \(L_0\) Regularization](https://openreview.net/forum?id=H1Y8hhg0b)。
