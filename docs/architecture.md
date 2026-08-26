# v11 模型与数据流

## 1. 输入、输出与目标

输入是沿曲线方向排列的二维或三维点：

```text
points [B,M,D], D∈{2,3}
```

输出包括：

```text
params                    [B,M]
proposal_internal_knots   [B,Kc]
deployment_internal_knots [B,Kc]
learned_keep_mask         [B,Kc]
```

最终内部节点向量是：

\[
U_{\mathrm{deploy},b}
=\{u^*_{b,j}\mid M_{b,j}=1\}.
\]

节点数 `K_b` 等于掩码中 `True` 的数量。模型没有单独的 `CountHead`。

## 2. GeometryEncoder 与 ParameterHead

`GeometryEncoder` 从坐标、一阶差分和二阶差分构造局部几何特征，再聚合全局特征：

```text
points
  → local_features  [B,M,H]
  → global_features [B,H]
```

`ParameterHead` 同时读取局部和全局特征，预测严格递增参数：

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

`params` 用于：

- CandidateKnotHead 的参数位置编码和 Gaussian 局部 attention；
- 两次截断幂代理求解；
- 标准 B 样条部署 refit；
- 真参数监督。

## 3. CandidateKnotHead：局部 Gaussian cross-attention

### 3.1 输入与 query

输入为：

```text
global_features [B,H]
local_features  [B,M,H]
params          [B,M]
```

模型使用 `Kc+1` 个固定左到右身份的 interval query。第 `j` 个 query 有参数域锚点：

\[
a_j=\frac{j+1/2}{K_c+1}.
\]

Key/Value 是带参数位置编码的局部几何特征：

\[
X_i=F_i+\operatorname{PE}(t_i).
\]

Query 同时叠加自身参数、锚点位置编码和曲线全局特征。

### 3.2 Gaussian 局部偏置

当带宽 `h>0` 时，attention logit 加入：

\[
b_{j,i}=-\frac{(t_i-a_j)^2}{2h^2}.
\]

因此 query 优先读取锚点附近的点，同时 Gaussian 尾部仍允许收集较宽的上下文。默认：

```text
--candidate-local-attention-bandwidth 0.08
```

设为 `0` 时不添加偏置，即恢复历史全局 cross-attention。

### 3.3 严格有序候选

每个 interval token 回归一个 logit。经过 softmax 和最小间隔 `δ` 后：

\[
\Delta_j=\delta+
\left[1-(K_c+1)\delta\right]\operatorname{softmax}(a)_j,
\]

\[
c_j=\sum_{r=0}^{j-1}\Delta_r.
\]

因此：

\[
0<c_1<c_2<\cdots<c_{K_c}<1,
\]

无需排序，也不会打乱槽位与 token 的对应关系。相邻 interval token 融合后形成 `candidate_tokens [B,Kc,H]`。

## 4. proposal 特征与固定几何

所有候选先参与一次截断幂代理拟合。对每个候选提取：

- 截断幂系数能量；
- 解析列删除目标增量；
- 节点附近的点残差；
- 左右间距；
- 几何 token 和位置编码。

`InteractivePruningHead` 先做候选 self-attention，再用只依赖 proposal 的位置头产生一次基础精修：

\[
U_{\mathrm{prop}}=C+\Delta U_{\mathrm{prop}}.
\]

`U_prop` 对应：

```text
proposal_internal_knots [B,Kc]
```

离线 Hard-RMS 教师、teacher mask 和 slot risk 都绑定这套位置。教师生成后，proposal 主干在蒸馏和联合校准阶段保持冻结。

## 5. 固定交互：p0 → u1 → p1 → mask → u*

v11 在固定 proposal 之外增加独立 deployment 位置路径。它不改写教师槽位。

### 5.1 初始保留概率 p0

两层位置感知 selector self-attention 得到 `S0`，预测 raw importance 和曲线自适应阈值：

\[
p^{(0)}_j=\sigma(r^{(0)}_j-\beta^{(0)}).
\]

`p0` 先构造临时 hard straight-through mask 和集合上下文。前向使用二值选择，反向使用概率梯度。

### 5.2 临时位置 u1

初始选择上下文和候选 token 共同产生临时位置残差：

\[
u^{(1)}_j=u^{\mathrm{prop}}_j+\Delta u^{(1)}_j.
\]

临时位移对所有候选槽位计算，并不乘 `p0`。这样一个初始低概率槽位仍有机会移动到更合适的位置，再把新证据反馈给下一次选择。

### 5.3 最终概率 p1

`u1` 的位置编码变化、位置 token、`p0` 和归一化位移共同反馈给 selector：

\[
p^{(1)}_j=\sigma(r^{(1)}_j-\beta^{(1)}).
\]

这是第二次也是最后一次概率预测。计算图次数固定，不运行数据相关循环。

### 5.4 一次性 mask

默认 `mass_topk` 先计算：

\[
\mu=\sum_jp^{(1)}_j,\qquad
\sigma_K=\sqrt{\sum_jp^{(1)}_j(1-p^{(1)}_j)},
\]

令 `q=μ+sσ_K`，实现使用：

\[
\widehat K=
\begin{cases}
0,&q<0.5,\\
\lceil q\rceil,&q\ge 0.5.
\end{cases}
\]

结果还会截断到有效候选数。随后一次性保留得分最高的 `Khat` 个槽位。`s` 由 `--one-shot-safety-sigma` 设置，默认 `0.25`。零节点因此有稳定区间；`K>0` 时仍保持 `ceil` 语义。

`--one-shot-coverage-bins` 大于零时，会在参数域分箱中加入最高概率锚点；v11 默认值为 `0`，即不强制分箱锚点。该选项只改变一次性 mask，不调用样条搜索。

### 5.5 最终位置 u*

最终位置头读取 `p1`、最终 hard-ST KeepMask 上下文和候选交互特征，只对保留槽位预测第二个残差：

\[
u^*_j=u^{(1)}_j+M_j\Delta u^{(2)}_j.
\]

两阶段位置约束为：

- 参数域 `[0,1]`；
- 相邻有效节点留下的 slack；
- 最小节点间隔；
- `u1` 相对 `U_prop` 不超过位移预算；
- 第二次残差会扣除第一阶段已经消耗的预算，使最终 `|u*-U_prop|` 不超过 `--one-shot-max-position-shift`；
- 位移比例小于可用 slack 的一半。

因此该参数是相对固定 proposal 的两阶段合计上限，不是每个位置头各自可使用的上限。

最终输出：

```text
deployment_internal_knots [B,Kc]
internal_knots            [B,Kc]  # 兼容别名
learned_keep_mask         [B,Kc]
```

## 6. proposal 与 deployment 为什么必须分开

离线教师是对固定 `U_prop` 做 Hard-RMS 删除得到的。如果蒸馏时直接移动 `U_prop`，teacher mask 的槽位含义和风险值会失效。

v11 因此采用：

```text
U_prop：固定，负责 teacher 对齐和 Hard-RMS 对照
U*    ：可学习，负责 LearnedKeep 的最终部署
```

两者共享同一槽位索引，所以 mask 可以从 proposal 无歧义地转移到 deployment 位置。

## 7. 训练监督

### canonical 监督

- 真参数 MSE；
- proposal 最近距离和多尺度 coverage；
- proposal 有序匹配位置；
- canonical existence 辅助分类；
- 联合校准时，教师保留槽位与 canonical 节点的有序位置匹配。

### Hard-RMS 教师监督

- teacher hard mask 的加权 BCE 和 soft Dice；
- final-state soft risk；
- 保留槽位高于删除槽位的排序；
- critical retained-slot recall；
- 删除槽位 false-positive 惩罚，但豁免 canonical-positive 的替代槽位；
- teacher count 和实际 `mass_topk` policy count；
- 概率质量分布和复杂度。

canonical 与 teacher 的职责不同：canonical 提供几何目标，teacher 提供固定 proposal 上的可行组合目标。

## 8. 两次代理 solve 与一次部署 refit

网络 `forward` 内：

1. 所有 proposal 参与截断幂代理 solve，产生贡献特征。
2. 最终 hard-ST mask 和 `u*` 再参与一次截断幂代理 solve，产生训练诊断。

部署时：

1. 网络前向一次；
2. 取 `U*[mask]`；
3. 构造标准开放 B 样条基；
4. 端点约束下重新求解控制顶点一次。

代理 solve 不是标准 B 样条 refit。默认纯选择蒸馏阶段的 surrogate fit/violation 权重为零。最后的联合校准阶段默认使用小权重：

```text
--lambda-joint-fit 0.05
--lambda-joint-threshold-violation 0.5
```

此时截断幂 fit 使用的 hard-ST gate 会从选择概率上 detach，避免 fit 梯度沿门控路径把模型推回 all-keep。该信号用于改善已选节点位置的代理可行性，不替代 teacher 来监督 KeepMask。checkpoint 选优仍使用真实标准 B 样条部署 RMS。

## 9. 兼容语义

- v10 checkpoint 继续按 `one_shot_joint_position_refinement=False` 恢复：固定 proposal、结构化 mask、一次 refit。
- v11 checkpoint 启用局部 attention 和联合 deployment 位置路径。
- 旧输出若没有 `proposal_internal_knots`，评估脚本回退到 `internal_knots`。
- v11 新模块采用兼容初始化，但读取 v10 checkpoint 本身不会改变其 objective version 或部署语义。
- `candidate-pretrain-epochs=0` 时保留 checkpoint 的 proposal attention 带宽，并校验已记录 `model_config` 中影响 proposal 的非权重语义；新写出的 proposal checkpoint 会携带该配置。缺少配置的历史文件只能警告后按兼容语义加载。

v11 是一次性学习近似。它不宣称每个样本严格满足 `ε`，也不宣称获得全局最少节点。
