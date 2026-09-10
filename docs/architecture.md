# v16 模型架构与张量流

v16 接收归一化有序点云与误差阈值，一次性输出参数、候选节点、离散
KeepMask 和重定位后的存活节点。最终控制顶点不由网络回归，而是由一次标准
三次 B 样条最小二乘 refit 求得。

当前主协议固定为：

| 项目 | 当前设置 |
|---|---:|
| 输入点数 | 192 |
| 合成源控制顶点 | 8～60 |
| 合成源内部节点 `K_source` | 4～56 |
| 合成节点最小 span | 0.01（显式设置；底层生成器旧 0.02 默认仅兼容历史） |
| 网络内部候选容量 `Kc` | 56 |
| 单曲线阈值 | `MSE <= 1e-4` |
| 正式资格 | worst-source deployment pass `>= 90%` |
| Joint 初始保留比例 | `30/56≈0.535714`（目标初始 keep mass 约 30） |
| 低计数教师穷举 | `K=4..16` |
| 合成计数语义 | `upper_bound` |
| 合成教师 | geometry-oracle + `Ktrue+2` 储备 |
| 强制覆盖分区 | `bins=0` |

`Kc=56` 是高召回容量，不是最终节点数。三次开放样条全保留时，完整节点
向量有 64 项，控制顶点有 60 个；部署节点数由每条曲线的一次性 KeepMask
决定。

合成源曲线先在干净固定参数化上通过全部单节点删除证书，再加入观测噪声。
证书与真值监督只用于训练和验证，不进入部署。详见
[合成曲线最简性报告](synthetic_data_minimality_report.md)。

![v16 pipeline](figures/v16_pipeline.svg)

## 1. 输入与输出

| 名称 | 形状 | 含义 |
|---|---|---|
| `points` | `[B,M,D]` | 沿曲线排序的归一化点，`D=2` 或 `3` |
| `mse_tolerance` | 标量或 `[B]` | 归一化 MSE 阈值，本协议为 `1e-4` |
| `params` | `[B,M]` | 最终严格递增参数 `t1`，首尾为 0、1 |
| `proposal_internal_knots` | `[B,56]` | 稠密有序候选 `U0` |
| `keep_probabilities` | `[B,56]` | 候选的一次性保留概率 |
| `adaptive_keep_threshold` | `[B]` | 曲线级动态阈值 `beta` |
| `learned_keep_mask` | `[B,56]` | 概率质量与一次 Top-K 得到的离散集合 |
| `internal_knots` | `[B,56]` | 条件重定位后的候选张量 |

真正部署的内部节点为

\[
U_{\mathrm{deploy},b}=\operatorname{sort}\left(
\text{internal\_knots}_b[\text{learned\_keep\_mask}_b]\right).
\]

最终节点数是 `KeepMask.sum()`。网络没有 CountHead，也不会先预测整数 K 再
启动变长节点头。

训练批次还包含真参数、真内部节点、有效 mask 与源节点数。真实样本没有
真节点标签，其有效标志为 false，因此占位值不参与真值监督。

## 2. 一次前向的数据流

```text
ordered points Q [B,192,D]
  -> GeometryEncoder
       local_features [B,192,H]
       global_features [B,H]
  -> ParameterHead(local, global, chord params)
       t0 [B,192]
  -> CandidateKnotHead(local, global, t0)
       U0 [B,56], candidate_tokens [B,56,H]
  -> InteractiveSelector(candidate tokens, point memory, U0, tolerance)
       importance logits + curve-level beta
  -> adaptive probability-mass Top-K once
       KeepMask [B,56]
  -> decode_subset(context, KeepMask)
       t0 -> t1; selected knots warp + relocate
  -> Udeploy
  -> endpoint-constrained cubic B-spline refit x 1
       control vertices + fitted curve
```

## 3. 几何与参数编码

GeometryEncoder 读取点坐标、归一化弦长以及一阶、二阶有限差分。共享一维
卷积输出逐点局部特征和全局池化特征。二阶差分表达局部弯折，不等于精确
曲率。

ParameterHead 同时读取局部特征、全局特征和弦长参考参数，预测相邻弦长
间隔的有界残差，再投影到正间隔单纯形：

\[
0=t_{0,0}<t_{0,1}<\cdots<t_{0,M-1}=1.
\]

因此参数头在候选生成之前提供第一版参数化；它的输出同时进入候选头和后续
KeepMask 条件解码。

## 4. 56 个高召回候选

CandidateKnotHead 读取 `local_features`、`global_features` 和 `t0`。锚定 query
对带参数位置的逐点特征执行局部 Gaussian cross-attention，输出 56 个严格
有序候选 `U0` 及其 token。

56 个内部候选覆盖当前合成 source 的 `K=4..56` 范围，同时比旧 Kc=64 少
12.5% 的候选计算。source K 表示生成复杂度，Kc 表示候选槽位；两者最大值
相同不表示部署必须全保留。`source K=56` 位于容量边界、没有冗余候选余量，
必须单独报告该层 dense/deployment pass。

## 5. 自适应一次性筛选

每个候选 token 加入候选位置、左右间距、最近采样参数距离和容差编码，再经
候选 self-attention 与对点特征的 cross-attention。KeepHead 输出相对重要度
`s_j`，曲线级阈值头输出 `beta`：

\[
\ell_j=(s_j-\operatorname{mean}(s))-\beta,
\qquad p_j=\sigma(\ell_j).
\]

部署请求数量为

\[
\hat K=\operatorname{clamp}\left(
\left\lceil\sum_jp_j+
\sigma_s\sqrt{\sum_jp_j(1-p_j)}+K_s\right\rceil,
K_{\min},56\right),
\]

然后只执行一次全局 Top-K。当前训练从 `(sigma_s,K_s)=(0.20,2)` 退火到
`(0.03,0)`；Joint 初始保留概率质量设为 `(30/56)*Kc≈30`。它只定义训练
起点，不是部署最终 K；部署数量仍由每条曲线的 beta 与概率质量决定。当 worst-source pass 低于
90% 时，训练反馈会减弱复杂度压力并恢复安全储备。

当前设置 `--one-shot-coverage-bins 0`，即不强制在固定参数区间预留节点。
固定分区会让简单曲线即使关键几何集中在局部，也被迫消耗低 K 预算。复杂
曲线的安全性改由候选召回、教师可行性和拟合损失保护。

部署中没有 `p>=0.5` 固定阈值扫描、CountHead、BIC、Hard-Concrete、逐节点
删除或误差回补。

## 6. KeepMask 条件下的参数—节点联动

`decode_subset(context, KeepMask)` 让同一个离散集合同时决定参数更新和节点
重定位：

1. 候选 token 加入最近左右存活节点距离、存活 rank 和存活数量；
2. 未选候选在 attention 的 Key/Value 中被屏蔽；
3. 点特征以存活节点为 memory，将 `t0` 更新为严格递增的 `t1`；
4. 候选从 `t0` 单调 warp 到 `t1`；
5. 存活节点依据邻居、rank、count 和 survivor attention 联合重定位；
6. 仅对存活节点做正间隔投影，保证有界、有序。

因此筛选后的节点不是原候选的静态子集，它们可以移动到被删除候选留下的
区域。

## 7. 当前训练教师

Joint 阶段建立训练专用的候选组合池：

1. 当前一次性 deployment mask；
2. `K=4..16` 的每一个 ranked prefix，逐计数检查可行性；
3. 高于 16 的粗到细前缀搜索和边界邻域编辑；
4. 全保留组合；
5. certified Synthetic 的 geometry-oracle `Ktrue` mask；
6. 同一 oracle 的 `Ktrue+2` 安全扩展 mask；超过 Kc 时截断到 56，故
   `Ktrue=56` 时就是全候选 mask。

geometry oracle 先将候选映射到真参数域，再做一维单调一对一匹配。它在训练
早期提供与 Selector 当前错误排序无关的可行组合，只对合成训练样本启用。

教师在所有可行组合中先取节点最少者，再以 MSE 打破平局。由于源最简性
证书只覆盖固定源节点位置的所有子集，而解码器允许参数和节点重定位，当前
使用 `--synthetic-count-role upper_bound`：`K_source` 是有证书的上界，不是
必须精确回归的全局最小 K。网络可以在仍满足 `MSE<=1e-4` 时学到比源 K 更小
的组合。

这解决两类相反问题：复杂曲线可使用至多 56 个候选维持召回；简单曲线则由
逐计数低 K 扫描、oracle 教师、约 30 的初始质量和取消固定分区得到更细的简化信号。
容量边界 `source K=56` 没有冗余余量，其失败不能靠降低 90% 正式资格解决。

## 8. 两阶段训练

### Proposal：前 16 个 epoch

全保留 56 个候选，训练几何编码、参数化和候选覆盖，先确保四个验证来源的
dense proposal 可行。Selector 在该阶段冻结。

### Joint：后 64 个 epoch

同时训练候选排序、一次性数量选择、参数反馈和存活节点重定位。计数、位置、
反事实组合、拟合可行性和复杂度共同优化。复杂度权重最多放大到 6 倍，但只在
worst-source deployment pass 保持 90% 及安全余量时增强。

训练为了比较候选组合会调用多次可微 refit；部署不存在这些教师搜索和额外
refit。

## 9. 最终标准 refit 与计时

网络前向结束后，外部求解器用 `t1` 和 `Udeploy` 构造三次开放 B 样条基，
固定插值两个端点，并用 CPU `float64` 最小二乘求控制顶点。部署只做一次
refit。

- `network time`：仅一次 `forward_deployment`；
- `full deployment time`：网络、节点物化和一次最终 refit；
- 数值方法时间：从其初始化到最终统一 refit 的完整方法时间。

## 10. 当前版本与历史消融

当前结构入口是
[v16_network.py](../src/spline_fitting/models/v16_network.py)，训练与完整评估由
[run_v16_mse1e-4_3090.ps1](../scripts/run_v16_mse1e-4_3090.ps1) 串行执行。

若使用旧 K64 proposal warm start，Encoder、ParameterHead 和 CandidateHead 中形状
兼容的张量直接迁移；65 个旧 interval query 沿参数域插值成 57 个新 query，固定
锚点由 K56 模型重新生成。Selector、selected-only 联合解码器和优化器从头训练，
因此它不是跨容量 `resume`。

旧 `Kc=96 / MSE=2.5e-5 / 97%`、旧 K64（包括 1e-4 或 2.5e-5）和固定 0.5
KeepMask 结果仅是历史容量或阈值消融，不是当前主协议，也不能通过改名或
`--resume` 解释成当前模型。
