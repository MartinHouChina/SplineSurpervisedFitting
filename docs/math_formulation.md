# v12/v13 数学定义

v13 沿用 v12 的部署方程，但修正实际 survivor 的集合监督、Keep 概率覆盖项和 checkpoint 选择。下文标为 v12 的网络结构仍适用于 v13；训练差异在第 9–10 节明确给出。

## 1. 约束目标

给定沿曲线方向排序的点序列

\[
Q=(q_0,\ldots,q_{M-1}),\qquad q_i\in\mathbb R^D,
\]

模型预测参数 `t`、内部节点集合 `U`，再用标准 B 样条最小二乘求控制顶点 `P`。目标是

\[
\min_U |U|
\quad\text{s.t.}\quad
R(U,Q)\le \varepsilon,
\]

其中

\[
E(U,Q)=\frac1M\sum_{i=0}^{M-1}
\|C(t_i;U,P^*)-q_i\|_2^2,
\qquad
R(U,Q)=\sqrt{E(U,Q)}.
\]

`E` 是平均平方欧氏误差 MSE，`R` 是 RMS 欧氏距离。若
`epsilon=0.005`，对应的 MSE 阈值为 `2.5e-5`。v12 是该离散约束问题的一次性统计
近似，不提供逐样本可行性或全局最少节点证明。

## 2. 严格递增参数

`ParameterHead` 从有序点特征预测正区间并累积归一化，使

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

有真实参数监督时使用

\[
L_t=\frac1M\sum_i(t_i-t_i^{\mathrm{true}})^2.
\]

节点坐标定义在参数域内，因此节点匹配只有在共享或足够接近的参数化下才有直接几何
意义；评估时必须同时检查 ParameterHead 的参数误差。

## 3. 局部 cross-attention 候选生成

`GeometryEncoder` 把坐标、弦长、一阶差分与二阶差分编码为点 token `f_i`，并加入预测
参数的位置编码：

\[
x_i=f_i+\operatorname{PE}(t_i).
\]

设第 `j` 个 interval query 的参数域锚点为

\[
a_j=\frac{j+1/2}{K_c+1}.
\]

带宽 `h>0` 时，局部 Gaussian cross-attention 为

\[
A_{j,i}=\operatorname{softmax}_i\left(
\frac{\langle W_Qz_j,W_Kx_i\rangle}{\sqrt d}
-\frac{(t_i-a_j)^2}{2h^2}
\right).
\]

`h=0` 时不加局部偏置。`K_c+1` 个 interval 输出正 gap，归一化后得到严格有序的
`K_c` 个冗余候选：

\[
0<u_1^{\mathrm{prop}}<\cdots<u_{K_c}^{\mathrm{prop}}<1,
\qquad
u_{j+1}^{\mathrm{prop}}-u_j^{\mathrm{prop}}\ge\delta.
\]

proposal 阶段追求高召回，不直接承担“最少节点”决策。`U_prop` 同时是离线教师缓存的
固定槽位坐标；改变 proposal 权重或定义后，旧 teacher cache 失效。

## 4. canonical 覆盖与代理贡献

对 canonical 节点 `v`，其 proposal 最近距离为

\[
d(v,U_{\mathrm{prop}})=\min_j|v-u_j^{\mathrm{prop}}|.
\]

候选预训练使用平均最近距离和多尺度 coverage hinge：

\[
L_{\mathrm{multi}}=
\frac{1}{|\mathcal V||\mathcal T|}
\sum_{v\in\mathcal V}\sum_{\tau\in\mathcal T}
\left[\max\left(\frac{d(v,U_{\mathrm{prop}})}{\tau}-1,0\right)\right]^2,
\quad
\mathcal T=\{0.005,0.01,0.02\}.
\]

三次截断幂代理写成

\[
C(t)=a_0+a_1t+a_2t^2+a_3t^3+
\sum_{j=1}^{K_c}b_j(t-u_j)_+^3.
\]

它提供每个候选的系数能量、解析删除增量、局部残差与左右间距等可解释特征。这些量只
用于网络特征和训练代理；最终曲线仍由数值更稳定的标准 B 样条基 refit。

v13 在 detached float64 中计算该 pilot 系统和解析删除增量，再将结构描述符转回模型 dtype。这样不增加反向图，也避免病态正规方程的 float32 误差随 batch kernel 变化而被 selector 放大；旧 v12 checkpoint 默认仍保留原有数值语义。

## 5. 一次性 KeepMask

交互选择头先从 proposal token、位置与贡献特征得到初始概率

\[
p_j^{(0)}=\sigma(r_j^{(0)}-\beta^{(0)}),
\]

再产生 provisional 位置反馈 `U^(1)`。该位置变化重新编码后送回选择头，得到最终概率

\[
p_j^{(1)}=\sigma(r_j^{(1)}-\beta^{(1)}).
\]

这是固定深度的两次概率计算，不是在线逐节点删除循环。默认 `mass_topk` 根据

\[
\mu=\sum_jp_j^{(1)},\qquad
\sigma_K=\sqrt{\sum_jp_j^{(1)}(1-p_j^{(1)})},
\]

构造一次性请求分数 `q=mu+s sigma_K`，由它确定保留数量，再按最终重要性分数一次性选出
最高的若干槽位。记最终硬子集为

\[
m_j\in\{0,1\},\qquad
\mathcal S=\{j\mid m_j=1\}.
\]

模型没有独立 `CountHead`，也不会在 learned 部署时逐节点试删。

训练使用 straight-through gate

\[
\widetilde m_j=m_j+p_j^{(1)}-\operatorname{stopgrad}(p_j^{(1)}),
\]

使前向采用硬子集，而位置分支仍可向 keep 概率传递近似梯度。它建立了选择与位置之间的
训练联系，但不把离散 Top-K 组合求解变成精确可微优化。

## 6. final-mask 条件化 survivor 几何

v11 的位置头主要读取稠密 proposal 邻居；删除若干槽位后，这不再等于最终存活序列的
真实邻接关系。v12 在 `KeepMask` 已确定后，对每个存活节点重新寻找前一个和后一个存活
邻居；缺失的一侧使用参数域端点 `0` 或 `1`。

设 relocation 前的位置为 `\tilde u_j`，前后存活边界为 `l_j,r_j`。网络构造八维相对
描述

\[
\phi_j=\big[
\tilde u_j-l_j,
r_j-\tilde u_j,
r_j-l_j,
\frac{\tilde u_j-l_j}{r_j-l_j},
\rho_j,
\frac{|\mathcal S|}{K_{\mathrm{valid}}},
\rho_j-\tilde u_j,
\widetilde m_j
\big],
\]

其中 `rho_j` 是节点在紧凑 survivor 序列中的归一化 rank。rank 与 count 的前向值来自
最终硬集合，反向值保留 straight-through gate 的梯度。

## 7. selected-only survivor relocation

relocation token 由最终 decision token、当前位置编码和相对 survivor 几何组成：

\[
h_j=\operatorname{LN}\left(
d_j+\operatorname{PE}(\tilde u_j)+W_\phi\phi_j
\right).
\]

multi-head attention 的 Key/Value 只允许来自最终存活集合：

\[
\bar h_j=
\operatorname{MHA}\left(
h_j,\{h_k:k\in\mathcal S\},\{h_k:k\in\mathcal S\}
\right).
\]

被删除槽位不能作为 Key/Value 影响最终分布，位置残差也只施加到存活槽位。空集合样本
使用数值安全的零值 fallback，但最终残差仍被硬 mask 置零。

网络输出有界信号 `s_j in [-1,1]`，根据左右 survivor slack 映射成残差 `Delta u_j`：

\[
u_j^*=\tilde u_j+m_j\Delta u_j.
\]

映射同时强制

\[
0<u_{s_1}^*<\cdots<u_{s_K}^*<1,
\qquad
u_{s_{r+1}}^*-u_{s_r}^*\ge\delta,
\]

以及相对固定 proposal 的总位移预算

\[
|u_j^*-u_j^{\mathrm{prop}}|\le d_{\max}.
\]

`d_max` 是 provisional 更新与 survivor relocation 合计后的上限，不是两个位置头各自都可
使用一次的预算。v12 新 relocation head 采用零初始化，所以载入 v11 权重时初始映射为
identity；必须经过 v12 联合校准后才会学习非零移动。

## 8. delete-then-relax 离线教师

v12 教师不是源节点向量，也不是“保留 proposal 原位置”的标签。对每个固定
`U_prop`，离线执行：

```text
greedy Hard-MSE 单节点删除，直到下一次删除超过 epsilon^2
  -> 对当前 survivors 做有序 coordinate relaxation
  -> 在移动后的位置上继续尝试 greedy 删除
  -> 重复有限轮
  -> 若最后一轮又删除节点，对最终 survivors 再做一次 relaxation
  -> 计算最终 leave-one-out risk 并缓存
```

每次删除判断和位置搜索都调用标准 B 样条 refit。relaxation 保持节点有序、满足
`relocation_min_gap`，并限制每个 survivor 相对其 proposal anchor 的最大位移。最终缓存
包括：

- 映射回原 proposal 槽位的 `teacher_retained_mask`；
- 优化后、按存活顺序 packed 的 `teacher_internal_knots`；
- teacher count、最终 RMS/MSE、删除风险与排序；
- relocation 平均/最大位移及 relocation 后额外删除数。

因此 v12 的位置标签明确回答：“这个删除组合确定后，剩余节点应移动到哪里？”v11 cache
缺少该语义，不能直接复用。

## 9. v12 选择与位置监督

proposal 预训练仍用 canonical 节点监督覆盖和初始位置。固定 proposal 后，selector 与
relocation 主要学习离线教师。

对 teacher 保留槽位，按槽位顺序取 student 最终位置 `U_S^*`，以教师优化后的 packed
位置 `\bar U` 为目标：

\[
L_{\mathrm{anchor}}=
\operatorname{SmoothL1}(U_{\mathcal S}^*,\bar U).
\]

为直接监督边界覆盖与 survivor 间距，定义带端点的 gap：

\[
g(U)=\operatorname{diff}([0,U,1]),
\]

并使用

\[
L_{\mathrm{gap}}=
\operatorname{SmoothL1}
\left(g(U_{\mathcal S}^*),g(\bar U)\right).
\]

v12 的位置项为

\[
L_{\mathrm{knot-pos}}=
L_{\mathrm{anchor}}+\lambda_{\mathrm{gap}}L_{\mathrm{gap}},
\]

默认 `lambda_gap=0.25`。这与旧的“把保留槽位拉向 canonical 最近节点”不同；optimized
teacher survivor 才是 v12 联动校准阶段的直接位置目标。

v13 不再用 teacher retained slots 代替实际部署 survivors。记预测集合

\[
P=\operatorname{sort}\{u_j^*:m_j^{final}=1\},\qquad
T=\operatorname{sort}\{\bar u_l:\bar m_l=1\}.
\]

通过一维动态规划寻找保持顺序且覆盖较小集合的最小 L1 配对 \(\mathcal A(P,T)\)，位置项改为

\[
L_{\mathrm{set-pos}}=
\frac{1}{|\mathcal A|}\sum_{(i,l)\in\mathcal A}
\operatorname{SmoothL1}(P_i,T_l).
\]

只有 \(|P|=|T|>0\) 时才计算完整边界/相邻 gap 损失，避免对缺失子序列制造错误边界。

为让 Keep logits 在 hard mask 选错时仍有梯度，v13 对每个 teacher 节点定义局部覆盖概率

\[
c_l=1-\prod_j\left[1-p_j
\exp\left(-\frac{(u_j^*-\bar u_l)^2}{2\sigma_c^2}\right)\right],
\qquad
L_{\mathrm{coverage}}=-\frac1{|T|}\sum_l\log(c_l+\delta).
\]

该项与 proposal 槽位编号无关，并与 relocated teacher set 的平滑 CDF 距离共同构成 `teacher_distribution`。

选择监督还包括 teacher mask 的加权 BCE/Dice、keep-risk、删除排序、critical recall、
false-positive、teacher count、policy count、概率质量分布与复杂度项。canonical existence
只承担辅助几何语义，不能替代固定 proposal 上的可部署组合教师。

## 10. 总损失与训练阶段

候选预训练可概括为

\[
L_{\mathrm{proposal}}=
\lambda_tL_t+
\lambda_cL_{\mathrm{coverage}}+
\lambda_mL_{\mathrm{multi}}+
\lambda_uL_{\mathrm{canonical-pos}}+
\lambda_rL_{\mathrm{repulsion}}+
\lambda_fL_{\mathrm{surrogate-fit}}+
\lambda_\varepsilon L_{\mathrm{surrogate-violation}}.
\]

selector 蒸馏阶段在冻结 proposal 的前提下学习 teacher mask、risk、ranking、count 和
complexity。联合校准阶段再训练 selector 与 survivor relocation：

\[
\begin{aligned}
L_{\mathrm{v13}}={}&
\lambda_kL_{\mathrm{keep}}+
\lambda_{\mathrm{risk}}L_{\mathrm{risk}}+
\lambda_{\mathrm{rank}}L_{\mathrm{rank}}+
\lambda_{\mathrm{count}}L_{\mathrm{count}}\\
&+\lambda_{\mathrm{pos}}(L_{\mathrm{set-pos}}+
\lambda_{\mathrm{gap}}L_{\mathrm{gap}})+
\lambda_{\mathrm{dist}}(L_{\mathrm{CDF}}+L_{\mathrm{coverage}})+
\lambda_{\mathrm{cmp}}L_{\mathrm{complexity}}+
\lambda_fL_{\mathrm{surrogate-fit}}+
\lambda_\varepsilon L_{\mathrm{surrogate-violation}}
+\text{辅助选择项}.
\end{aligned}
\]

实际代码保留更细的 risk/distribution/critical/false-positive 等分量；公式用于说明职责，
具体权重以 checkpoint 的 loss schema 为准。v13 默认令截断幂 surrogate 的
`fit/threshold_violation` 权重为零，仅保留为显式 opt-in 诊断；checkpoint 选择读取验证集
真实标准 B 样条部署指标，并同时比较 distillation 与 calibration checkpoint。

## 11. 标准 B 样条部署

最终内部节点集合为

\[
U_{\mathrm{deploy}}=\{u_j^*\mid m_j=1\}.
\]

控制顶点通过标准开放 B 样条基求解：

\[
P^*=\arg\min_P
\|B(t,U_{\mathrm{deploy}})P-Q\|_F^2+
\lambda_s\|D_2P\|_F^2+
\lambda_r\|P\|_F^2,
\]

并施加端点插值

\[
C(0)=q_0,\qquad C(1)=q_{M-1}.
\]

learned 部署只执行一次固定深度网络 forward 和一次最终标准 refit；教师的多次删除与
refit 全部发生在离线阶段。

## 12. 分阶段节点指标

在匹配容差 `tau` 下，分别评估

\[
U_{\mathrm{prop}},\qquad
U_{\mathrm{pre}}[M],\qquad
U_{\mathrm{deploy}}[M],
\]

得到 proposal、KeepMask 后 relocation 前、最终 deployment 的 Precision/Recall/F1 与
matched MAE。还应报告

\[
|u_j^{\mathrm{deploy}}-u_j^{\mathrm{pre}}|
\quad\text{和}\quad
|u_j^{\mathrm{deploy}}-u_j^{\mathrm{prop}}|
\]

的均值、最大值与非零移动比例，用来证明 relocation 是否真的工作。拟合方面同时报告
标准 refit 的 MSE、mean/P95/max RMS、阈值通过率和保留节点数分布。

## 13. 保证边界

- proposal recall 高不等于最终节点 precision 高；selector 仍可能误删或多留。
- straight-through 是近似梯度，不能保证找到全局最优节点子集。
- `mass_topk` 一次性部署不会按单样本误差回退，因此 `epsilon` 在 learned 模式只形成测试
  分布上的统计通过率。
- `min_gap` 和最大位移是建模约束。增大 `min_gap` 可抑制聚集，也可能损伤确实需要近邻
  节点的曲线，必须单独做消融。
- `verified` 会精确检查并只修复失败曲线；可选残差插点可能超过固定候选上限，因此它是
  自适应质量守卫，不是纯 one-shot 或最少节点证明。
- 质量优先的 hybrid 会在线执行 beam 删除与 coordinate refinement，但它是独立的慢速
  搜索模式，也不构成全局最优证明。
