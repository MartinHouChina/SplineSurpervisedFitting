# v10 数学定义

## 1. 目标

对有序点集 \(Q=\{q_i\}_{i=1}^M\)，在预测参数 \(t_i\) 下求内部节点集合 \(U\)：

\[
\min_U |U|\qquad
\text{s.t.}\qquad
R(U)=\sqrt{\frac1M\sum_i\|C_U(t_i)-q_i\|_2^2}\le\varepsilon .
\]

离线教师采用确定性的逐节点贪心删除，因此得到的是该删除路径上的近似最简集合，不是所有
节点子集上的组合全局最优证明。

## 2. 参数与高召回候选

ParameterHead 预测正间隔并累加：

\[
t_0=0,\qquad t_i=\sum_{r<i}\Delta t_r,\qquad t_{M-1}=1.
\]

CandidateKnotHead 预测 \(K_c+1\) 个带最小间隔 \(\delta\) 的正区间：

\[
\Delta_j=\delta+[1-(K_c+1)\delta]\operatorname{softmax}(a)_j,
\qquad
c_j=\sum_{r=0}^{j-1}\Delta_r .
\]

所以 \(0<c_1<\cdots<c_{K_c}<1\)。候选阶段优先提高真实必要节点的覆盖率，不直接决定最终
节点数量。

## 3. 可解释贡献特征

全候选截断幂代理为

\[
\Phi=[1,t,t^2,t^3,(t-c_1)_+^3,\ldots,(t-c_{K_c})_+^3].
\]

若 \(D^*=\arg\min_D\|\Phi D-Q\|_F^2+D^T\Lambda D\)，则每个节点拥有独立列系数、
系数能量和解析删列目标增量。InteractivePruningHead 同时读取：

- 候选 token 与位置；
- 截断幂系数能量和解析删列增量；
- 候选附近的局部残差；
- 左右间距。

截断幂基只产生可微代理和贡献证据；最终曲线始终由标准 B 样条基求解。

## 4. 离线 Hard-RMS 教师

从全候选集合开始，每轮枚举当前集合的所有单节点删除，对每个子集完整重拟合标准 B 样条，
选择 RMS 最低的删除，并且仅当其 RMS 不超过 \(\varepsilon\) 时接受。最终集合给出
\(m_j^*\in\{0,1\}\) 和教师数量 \(K^*=\sum_jm_j^*\)。

软风险在最终停止状态计算。对最终保留节点 \(j\)：

\[
s_j^*=\sigma\!\left(
\frac{R(U^*\setminus\{u_j\})/\varepsilon-1}{T}
\right),
\]

并限制 \(s_j^*\ge0.5\)；已删除槽位的风险为 0。这样软风险不会与最终 hard mask 产生相反
监督。完整冗余集合的初始单删 RMS 只作为诊断量保存。

## 5. 一次性双向交互

模型先确定不依赖 KeepMask 的 proposal 位置，再执行一次固定选择，不包含数据相关循环：

```text
基础候选 token
  → proposal-only ordered position
  → 固定位置编码 + 解析贡献
  → 两层 selector self-attention
  → final importance / beta / keep
  → probability-mass Top-K + coverage anchors
```

最终概率和 mask 为

\[
p_j=\sigma(r_j-\beta),\qquad
\widehat K=\left\lceil\sum_jp_j+s\sqrt{\sum_jp_j(1-p_j)}\right\rceil.
\]

最终 mask 在覆盖锚点后选择 raw importance 最高的 \(\widehat K\) 个槽位。固定 proposal 位置参与
selector 的位置编码，但 \(m_j\) 不再进入位置解码器。因此 KeepMask
变化不会使离线 Hard-RMS 教师绑定的槽位发生漂移。位移最多使用相邻可用 slack 的 0.45，故
proposal 精修后仍严格有序。第二次 surrogate solve 仍使用
\(m_j+p_j-\operatorname{stopgrad}(p_j)\)，但默认不参与 selector 选优。

## 6. 训练损失与选优

候选预训练使用拟合、阈值违约和覆盖监督。固定 proposal 后，一次性选择器使用：

\[
L_{select}=L_{mask}^{BCE+Dice}+L_{risk}+L_{rank}+L_{CDF}+L_{critical}
+L_{count}^{hard\text{-}ST}+L_{complexity}.
\]

其中

\[
L_{rank}=\operatorname{mean}_{i\in K^+,j\in K^-}
\operatorname{softplus}(\delta-r_i+r_j)
\]

直接要求Hard-RMS教师保留节点的重要性logit高于被删除节点，降低节点数接近但选错位置的情况。
canonical 位置监督只用于 proposal 预训练；教师生成后的蒸馏不再用第二套位置目标改变槽位。

`CDF` 项比较预测与教师节点质量沿有序候选轴的累计分布，是一维 Wasserstein 距离；
`critical` 项只作用于 Hard-RMS 最终保留节点，并按 final-state risk 加权，直接抑制致命漏检。

其中 hard-ST 数量在前向使用部署二值 mask，反向使用概率梯度。截断幂 `fit` 和 `violation`
仍作为诊断量，但默认权重为零，防止代理拟合通过打开全部基列来支配选择器。不可满足阈值的
教师样本不参与 count/complexity 项。

checkpoint 选优不使用代理 pass-rate，而是在验证集对一次性 mask 执行一次标准 B 样条重拟合。
尚未达到目标满足率时改善真实 pass/mean/P95；达到目标后最小化平均节点数，并以真实 fit 和
最终节点 recall/F1 作 tie-break。

## 7. 标准 B 样条部署

网络一次前向得到最终子集后，只求解一次

\[
P^*=\arg\min_P\|BP-Q\|_F^2
+\lambda_s\|D_2P\|_F^2+\lambda_r\|P\|_F^2,
\]

并严格施加

\[
P_0=Q_0,\qquad P_{n-1}=Q_{M-1}.
\]

默认 v10 部署不再运行逐节点 Hard-RMS 搜索，因此 \(R(U)\le\varepsilon\) 是独立测试分布上的
统计满足率，不是每条新曲线的硬保证。必须逐样本保证时，应显式运行离线 hard diagnostic 或
采用 v7 的硬验证回退。
