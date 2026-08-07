# Sparse Spline Fitting v0.5

> v0.5 当前方案：`K` 个独立节点 query 分别回归位置，排序仅用于样条求值；
> query token 与节点局部特征共同预测 supervised existence，随后由
> Hard-Concrete 完成训练态可微门控和部署态二值化。训练标签通过一维有序
> 最小代价匹配得到，目标包含 `existence + knot_position + count + true_params`。
> 删除某个 query 不再重新分配其余节点位置。旧 v0.4 interval checkpoint 会
> 被显式迁移到历史分支，可继续评估，但新结构需要重新训练。

本项目实现“联合预测参数 \(t\)、候选节点 \(U\) 与节点结构”的三次样条拟合。v0.4 使用 **带位置编码的局部 cross-attention KnotHead + `true_params` 参数监督 + ActivityHead/Hard-Concrete 门控 + 标准 B 样条控制点重拟合**。当前论文方案不使用 pilot drop-cost。

核心目标是：在拟合误差可接受的前提下，让网络直接学习每个候选内部节点的保留概率，并在推演阶段输出真正压缩后的开放夹持 B 样条，而不是依靠事后寻找一个“看起来节点数合适”的阈值。

## 两条明确分开的计算路径

训练路径使用可微截断幂代理：

~~~text
Q
 -> 共享几何编码器
 -> ParameterHead
 -> t 与 true_params 监督
 -> 带 t 位置编码的局部 cross-attention KnotHead
 -> t / 候选 U
 -> 节点局部几何上下文
 -> ActivityHead 直接预测 log_alpha
 -> Hard-Concrete 随机门 z
 -> 截断幂设计矩阵 Phi(t,U,z)
 -> 可微岭回归求线性系数 D*
 -> 代理曲线
 -> fit + expected-L0 + gap + true-parameter
~~~

部署路径使用标准 B 样条：

~~~text
Q
 -> model.eval()
 -> pi = P(z != 0)
 -> m = 1[pi >= threshold]
 -> 物理删除 m=0 的候选节点
 -> 开放夹持节点向量 Xi
 -> 标准 B 样条基矩阵 B
 -> 增广最小二乘重求控制点 P*
 -> 可部署 B 样条 (Xi, P*)
~~~

训练中的截断幂系数 \(D^*\) 不是 B 样条控制点。部署阶段不会复用或硬转换这些系数，而是用保留节点重新求解标准 B 样条控制点。

## A 方案的关键改动

- 当前目标使用 fit、expected-\(L_0\)、gap 和 `true_params` 参数监督；弦长 parameter-prior 默认关闭，投影正交项只为历史 checkpoint 兼容保留。
- 旧版 \(\sqrt{a+\epsilon}\) 软缩放改为直接乘 Hard-Concrete 门 \(z\)，门为零时该候选节点列严格为零。
- 稀疏目标改为每条曲线的期望非零节点数 \(\sum_jP(z_j\ne0)\)；A 方案不再依赖 activity/binary 两个经验损失。
- 评估态门是严格的 0/1 值，部署模块只接受该二值门。
- KnotHead 使用带预测 \(t\) 位置编码的局部 cross-attention，使候选区间能够读取几何事件在参数轴上的位置，而不是只依赖全局最大池化特征。
- 活动分支汇聚候选节点附近的局部几何上下文，由 ActivityHead 直接预测 `log_alpha`，再交给 Hard-Concrete；当前训练不计算 pilot 拟合或 drop-cost。
- Hard-Concrete logit 初始偏置设为 -2.0；预热依靠强制全开而不是饱和概率，释放门后仍保留可用的稀疏梯度。
- \(t\) 与 \(U\) 的间距使用“预留最小间距预算 + softmax”参数化，真实保证配置的最小间距。
- 硬删除节点后构造标准开放夹持 B 样条，并用增广 `torch.linalg.lstsq` 重拟合控制点；CPU 使用 `gelsy` 列主元 QR，不形成正规方程。

详细设计见：

- [框架结构](docs/architecture.md)
- [完整数学形式](docs/math_formulation.md)
- [训练与部署流水线](docs/training_pipeline.md)
- [文件功能说明](docs/file_guide.md)

## 环境

~~~bash
pip install torch numpy matplotlib
~~~

## 快速开始

训练一个新的 A 方案检查点：

~~~bash
python scripts/train.py --epochs 60 --output outputs/independent_queries.pt
~~~

默认 60 轮可以完整覆盖 5 轮全开预热和 40 轮温度退火。若仅训练 30 轮，温度不会到达默认终值。

评估完整验证集：

~~~bash
python scripts/evaluate_checkpoint.py --checkpoint outputs/independent_queries.pt --activity-threshold 0.5 --smoothness-weight 1e-6 --control-ridge 0 --json-output outputs/evaluation_report.json
~~~

可视化单条曲线：

~~~bash
python scripts/visualize_result.py --checkpoint outputs/independent_queries.pt --output outputs/result.png --activity-threshold 0.5
~~~

运行测试：

~~~bash
python -m unittest discover -s tests -v
~~~

## 局部 Cross-Attention 与 Hard-Concrete 语义

KnotHead 为 \(K+1\) 个候选区间维护可学习 query。局部特征首先加入由预测参数 \(t_i\) 生成的正弦位置编码：

\[
M_i=F_i+\operatorname{PE}(t_i).
\]

每个区间 query 同时接收全局特征并对整条局部序列做 cross-attention：

\[
q_r=e_r+W_gG,\qquad
h_r=\operatorname{LN}\left(q_r+\operatorname{MHA}(q_r,M,M)\right).
\]

区间 logit 经严格间距预算与 softmax 转换为有序候选节点。ActivityHead 随后结合候选位置及其局部上下文直接输出 \(\log\alpha_j\)，不叠加 pilot 或人工重要性分数。

活动分支输出位置参数 \(\log\alpha_j\)。训练时采样

\[
v_j\sim\mathcal U(\epsilon,1-\epsilon),\qquad
s_j=\sigma\left(
\frac{\log\alpha_j+\log v_j-\log(1-v_j)}{\beta}
\right),
\]

\[
z_j=\operatorname{clip}
\left(\gamma+(\zeta-\gamma)s_j,0,1\right).
\]

默认 \(\gamma=-0.1,\ \zeta=1.1\)。每个门非零的闭式概率为

\[
\pi_j=P(z_j>0)=
\sigma\left(
\log\alpha_j-\beta\log\frac{-\gamma}{\zeta}
\right).
\]

代码字段固定为：

| 字段 | 精确含义 |
|---|---|
| `activity_logits` | \(\log\alpha\) |
| `activity` / `l0_probability` | \(\pi=P(z>0)\) |
| `expected_l0` | 每条曲线的 \(\sum_j\pi_j\) |
| `activity_gate` | 训练态随机可微门 \(z\)；评估态严格二值门 \(m\) |
| `knot_attention_weights` | 每个区间 query 对带位置局部序列的注意力权重 |

评估态使用固定结构规则

\[
m_j=\mathbf 1[\pi_j\ge\tau].
\]

`--activity-threshold` 会设置真正的部署阈值，而不只是改变日志。建议只在验证集上确定一次 \(\tau\)，随后冻结到所有测试和实际推演中；不要为每条测试曲线单独寻找能得到期望节点数的阈值。

## 可微训练代理

三次样条训练代理为

\[
C_{\mathrm{sur}}(t)=
\sum_{r=0}^{3}c_rt^r+
\sum_{j=1}^{K}z_jd_j(t-u_j)_+^3,
\]

\[
\Phi_{\mathrm{sur}}(t,U,z)
=
\left[
V(t)\mid H(t,U)\operatorname{diag}(z)
\right].
\]

线性系数在每次前向传播中实时求解，不是跨样本共享的网络参数：

\[
D^*=
\arg\min_D
\|\Phi_{\mathrm{sur}}D-Q\|_F^2+
D^\mathsf T\Lambda D.
\]

完整训练目标为

\[
\mathcal L=
\lambda_f\mathcal L_{\mathrm{fit}}
+\rho_0(e)\lambda_0\mathcal L_{L_0}
+\lambda_g\mathcal L_{\mathrm{gap}}
+\lambda_t\mathcal L_{\mathrm{true\text{-}param}},
\]

\[
\mathcal L_{L_0}
=
\frac1B\sum_{b=1}^{B}\sum_{j=1}^{K}\pi_{bj}.
\]

\[
\mathcal L_{\mathrm{true\text{-}param}}
=
\frac1{BM}\sum_{b,i}(\hat t_{bi}-t^*_{bi})^2.
\]

默认 \(\lambda_0=2\times10^{-5}\)，`true_parameter` 权重为 \(10^{-2}\)，弦长 `parameter_prior` 权重为 0。`activity` 与 `binary` 权重仍为零；二值化由 Hard-Concrete 完成。

## 默认三阶段训练

| 阶段 | 默认轮次 | 实际门 | 稀疏权重 | 温度 |
|---|---:|---|---|---|
| 全开几何预热 | 1–5 | 所有候选门强制为 1 | \(\rho_0=0\) | 2.0 |
| 释放门控与稀疏渐增 | 6–20 | 随机 Hard-Concrete | 15 轮增至 1 | 从第 6 轮开始退火 |
| 结构稳定 | 21–60 | 随机 Hard-Concrete；验证用硬门 | \(\rho_0=1\) | 40 轮内由 2.0 降至 0.5，随后固定 |

最佳 checkpoint 只从完整 \(L_0\) 渐增结束后开始选择，避免把“尚未施加稀疏目标”的预热模型误保存为最终结构模型。

预热期 `activity=pi` 不必等于 1，但真正进入设计矩阵的 `activity_gate` 被强制为 1。训练日志中的 `gate_nonzero` 和 \(E[K]\) 因而可能不同。

## 标准 B 样条部署

高层 API：

~~~python
model.eval()
with torch.no_grad():
    output = model(points)
    fits = refit_model_output_as_bsplines(
        output,
        points,
        degree=3,
        smoothness_weight=1e-6,
        control_ridge=0.0,
    )
~~~

该 API 严格要求 `activity_gate` 为 0/1；传入训练态随机门或 `activity=pi` 会报错。

若硬门保留 \(\widehat K\) 个内部节点 \(\widehat U\)，开放夹持节点向量为

\[
\Xi=
\left[
\underbrace{0,\ldots,0}_{4},
\widehat U,
\underbrace{1,\ldots,1}_{4}
\right].
\]

三次 B 样条有 \(\widehat K+4\) 个控制点。即使 \(\widehat K=0\)，仍是合法的 4 控制点开放三次 Bézier 表示。

控制点通过下式重拟合：

\[
P^*=
\arg\min_P
\|BP-Q\|_F^2+
\lambda_s\|D_2P\|_F^2+
\lambda_r\|P\|_F^2,
\]

对应增广系统

\[
\begin{bmatrix}
B\\
\sqrt{\lambda_s}D_2\\
\sqrt{\lambda_r}I
\end{bmatrix}P
\approx
\begin{bmatrix}
Q\\0\\0
\end{bmatrix}.
\]

\(D_2\) 是控制多边形二阶差分正则，不等同于严格的连续曲率积分。

## 节点数与损失指标

| 指标 | 含义 |
|---|---|
| `expected_active_count` / \(E[K]\) | \(\sum_j\pi_j\) 的批均值，是期望节点数 |
| `active@tau` | 概率超过固定阈值的节点数 |
| `gate_nonzero_count` | 当前 forward 中真正非零的门数；训练时随机，验证时等于部署节点数 |
| `fit_loss` | 截断幂代理曲线的平均每点平方欧氏距离 |
| `standard_bspline_refit_loss` | 删除节点并重拟合标准 B 样条后的平均每点平方欧氏距离 |
| `augmented_objective` | 未归一化数据平方和加两项控制点正则，不能直接和训练 total loss 比大小 |

\(E[K]\) 可以是小数；最终保留节点数必须读取硬门计数或部署结果的 `retained_count`。

评估报告还包括：

- 保留节点数均值、最小值、最大值和直方图；
- 零节点比例与候选节点阈值敏感性表；
- 标准 B 样条平均控制点数、拟合 RMSE 和求解秩诊断；单样本可视化报告还会打印完整节点向量与控制点数；
- 有真实元数据时的 \(t\) RMSE 与节点匹配指标。

节点位置误差只有在预测参数化与真实参数化可比较时才有明确意义。

## 坐标恢复

数据集默认将点归一化。部署模块返回的控制点也位于归一化坐标。恢复原始坐标：

\[
P_{\mathrm{original}}
=
\mathrm{scale}\,P_{\mathrm{normalized}}+\mathrm{center}.
\]

## 旧检查点兼容

缺少 `gate_mode` 的 checkpoint 会被识别为 pre-A：

- 自动使用 `legacy_soft`；
- 关闭节点局部上下文；
- 恢复旧版间距参数化；
- 缺少 `knot_use_local_cross_attention` 时恢复全局 MLP KnotHead；
- 缺少 `l0` 权重时补为零；
- 缺少 Hard-Concrete 温度 buffer 时使用构造温度；
- 评估脚本先按旧 activity 概率构造二值 mask，再进入标准 B 样条部署模块。

“能够加载旧 checkpoint”只用于复现和诊断，不代表旧 sigmoid activity 与新 Hard-Concrete 概率可直接比较。A 方案实验必须重新训练，不能把旧 `best.pt` 当成 A 方案结果。

历史 Hard-Concrete checkpoint 会保留其 `activity_use_pilot_importance` 语义；v0.4 新训练显式保存为 `false`。该兼容分支不属于当前论文方案。

## 当前边界

- 标准 B 样条阶段目前固定网络预测的 \(t,U\)，只重新求控制点；尚未实现剪枝后对固定结构的 \(t,U\) 非线性精调。
- 训练代理仍使用截断幂基，部署使用标准 B 样条基；两者张成相同的简单节点三次样条空间，但系数正则具有基依赖性，因此两种拟合损失可能略有差别。
- 新训练不执行 pilot 求解；每次 forward 只包含主线性求解与 KnotHead cross-attention。
- `gap_loss` 约束全部候选节点，而不是只约束最终保留节点。
- 合成数据的 `noise_std` 在逐样本归一化之前加入；默认值为 0.001，但归一化坐标中的实际噪声幅度会随每条曲线的 scale 变化。
- 当前配置真值来自命令行参数以及 checkpoint 内的 `model_config`、`dataset_config`、`loss_config` 和 `training_config`；`configs/*.yaml` 尚未接入训练入口。
- 模型候选节点上限应不小于数据集可能出现的最大真实内部节点数，即 `max_control_points - 4 <= max_knots`。

## 参考

- Louizos, Welling, Kingma, [Learning Sparse Neural Networks through \(L_0\) Regularization](https://openreview.net/forum?id=H1Y8hhg0b), ICLR 2018.
- [PyTorch `torch.linalg.lstsq` 文档](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html).
