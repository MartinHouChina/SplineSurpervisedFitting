# v0.5 框架结构

> 当前实现以独立节点 query 取代 v0.4 的共享区间 softmax。每个 query 从带
> 位置编码的局部序列读取特征，独立产生 `u_j`；排序后，query token、节点
> 局部上下文和位置共同输入 ActivityHead。真值节点通过有序最小代价匹配监督
> existence 与位置，Hard-Concrete 仍是实际设计矩阵门和部署二值化机制。
> 下文若提到 `K+1` 区间累计，仅描述 checkpoint 兼容分支。

v0.4 将训练代理和最终部署明确分开。网络负责学习 \(t,U\) 以及节点保留分布；标准 B 样条控制点在硬结构确定后重新求解。KnotHead 使用带位置编码的局部 cross-attention，当前训练不使用 pilot drop-cost。

## 训练路径

~~~mermaid
flowchart TD
    Q[输入有序采样点 Q] --> E[共享 GeometryEncoder]
    E --> LF[局部特征 F]
    E --> GF[全局特征 G]
    LF --> T[ParameterHead]
    GF --> T
    T --> TP[严格单调参数 t]
    TP --> TS[true_params 监督]
    LF --> PE[加入 t 位置编码]
    TP --> PE
    GF --> U[区间 queries]
    PE --> U
    U --> CA[局部 Cross-Attention KnotHead]
    CA --> UP[有序候选节点 U]
    LF --> LC[候选节点局部高斯上下文]
    TP --> LC
    UP --> LC
    GF --> LC
    LC --> LA[ActivityHead 直接预测 log_alpha]
    LA --> HC[Hard-Concrete]
    HC --> PI[非零概率 pi]
    HC --> Z[训练随机门 z]
    TP --> PHI[截断幂代理 Phi]
    UP --> PHI
    Z --> PHI
    PHI --> LS[可微岭回归求 D*]
    LS --> CS[代理曲线 C_sur]
    CS --> LOSS[fit + expected-L0 + gap + true-parameter]
    TS --> LOSS
    PI --> LOSS
    LOSS --> BP[反向传播更新网络]
~~~

### 带位置编码的节点预测与活动特征

KnotHead 不再只从全局最大池化特征预测所有区间。局部特征加入预测参数的位置编码

\[
M_i=F_i+\operatorname{PE}(t_i),
\]

每个候选区间 query 对 \(M\) 做 cross-attention，再产生对应区间 logit。候选节点产生后，ActivityHead 对每个 \(u_j\) 汇聚局部上下文：

\[
w_{ji}=
\operatorname{softmax}_i
\left[
-\frac12\left(\frac{t_i-u_j}{h}\right)^2
\right],
\qquad
\ell_j=\sum_iw_{ji}F_i.
\]

ActivityHead 直接使用 \([G+\ell_j,u_j]\) 预测

\[
\log\alpha_j
=
\operatorname{MLP}([G+\ell_j,u_j]),
\]

然后由 Hard-Concrete 完成训练态可微采样和评估态二值化。没有额外的人工重要性分数或 pilot 路径。

## 部署路径

~~~mermaid
flowchart TD
    Q[输入采样点 Q] --> M[model.eval]
    M --> T[预测参数 t]
    M --> U[候选节点 U]
    M --> PI[非零概率 pi]
    PI --> MASK[固定阈值 m = 1 pi>=tau]
    MASK --> PRUNE[物理删除未保留节点]
    U --> PRUNE
    PRUNE --> XI[开放夹持节点向量 Xi]
    T --> B[标准 B 样条基矩阵 B]
    XI --> B
    B --> QR[增广 LSTSQ / pivoted QR]
    Q --> QR
    QR --> P[控制点 P*]
    XI --> OUT[可部署标准 B 样条]
    P --> OUT
~~~

部署模块位于 `evaluation/bspline_inference.py`，在 `torch.no_grad()` 下运行，不属于反向传播图。

## 模块职责

| 模块 | 职责 |
|---|---|
| `GeometryEncoder` | 从有序点序列提取局部与全局几何特征 |
| `ParameterHead` | 预测严格为正且满足最小间距的参数间隔，累计得到 \(t\) |
| `KnotHead` | 区间 query 对带 \(t\) 位置编码的局部序列做 cross-attention，再累计得到 \(U\) |
| `ActivityHead` | 结合全局特征、节点局部上下文和节点位置直接预测 \(\log\alpha\) |
| `HardConcreteGate` | 训练时采样可微门，评估时输出严格二值门 |
| `truncated_power_basis.py` | 构造训练代理设计矩阵 \([V\mid H\operatorname{diag}(z)]\) |
| `differentiable_solver.py` | 求解截断幂线性系数；历史 drop-cost 工具仅供旧实验兼容 |
| `SplineFittingLoss` | 组合拟合、期望 \(L_0\)、真实参数监督与几何约束 |
| `bspline_inference.py` | 删除节点、构造标准 B 样条并稳定重求控制点 |
| `checkpointing.py` | 显式迁移 pre-A 配置，避免静默改变旧权重语义 |

## 状态和系数所有权

- 网络长期参数：编码器、三个预测头的权重。
- 每次前向临时量：\(t,U,\log\alpha,\pi,z,D^*\)。
- 部署结果：压缩节点向量 \(\Xi\) 与标准 B 样条控制点 \(P^*\)。
- \(D^*\) 与 \(P^*\) 属于不同基，不共享同一组系数。

## 兼容分支

`gate_mode="legacy_soft"` 仅用于读取 pre-A checkpoint：

\[
\Phi_{\mathrm{legacy}}
=
\left[
V\mid H\operatorname{diag}(\sqrt{a+\epsilon})
\right].
\]

旧模式同时关闭节点局部上下文并恢复旧间距归一化。新的论文实验必须使用 `gate_mode="hard_concrete"` 重新训练。

## 当前边界

- 训练内层仍求截断幂代理系数；标准 B 样条只在部署阶段重拟合。
- 当前训练不执行 pilot 求解；历史兼容开关不属于 v0.4 论文路径。
- 部署阶段固定预测的 \(t,U\)，尚未加入结构固定后的非线性精调。
- 标准 B 样条二阶差分项是控制多边形平滑正则，不是连续曲率积分。
