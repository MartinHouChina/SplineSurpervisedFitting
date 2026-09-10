# v16 网络架构

## 1. 总览

```text
ordered points
  -> GeometryEncoder
  -> ParameterHead
  -> CandidateKnotHead
  -> interactive Selector + adaptive beta
  -> probability-mass Top-K
  -> survivor-conditioned parameter/relocation decoder
  -> one standard cubic B-spline refit
```

当前容量为 `Kc=56` 个内部候选。网络一次产生全部候选和一次离散组合，不是自回归生成，也不是在线逐节点剪枝。

## 2. 张量与模块

| 模块 | 主要输入 | 主要输出 | 作用 |
|---|---|---|---|
| GeometryEncoder | `Q[B,M,D]` | `H[B,M,d]`, `g[B,d]` | 编码坐标、局部差分及全局几何 |
| ParameterHead | `H,g` 与弦长参考 | `t[B,M]` | 预测严格递增参数 |
| CandidateKnotHead | `g,H,t` | `U_prop[B,Kc]`, candidate tokens | 带参数位置编码的局部 cross-attention 候选生成 |
| Selector | candidate tokens、几何 memory、阈值 embedding | logits、`beta`、probabilities | 候选交互和曲线级数量调节 |
| mass-TopK | probabilities | `KeepMask[B,Kc]` | 一次性离散选集 |
| subset decoder | context、KeepMask | 更新后的 `t`、`U[B,Kc]` | 仅基于 survivor 交互，联动参数与节点位置 |
| standard refit | `Q,t,U[KeepMask]` | 控制顶点与曲线 | 最终 CPU float64 三次 B 样条解 |

CandidateKnotHead 的候选按参数域有序；局部 cross-attention 的 memory 为 `H + PosEnc(t)`。Selector 再通过候选 self-attention 和对几何 memory 的 cross-attention 计算 keep 分数。

## 3. 训练标签如何进入网络

正式训练 batch 只有 certified Synthetic，因此每行都有 `t*、U*、K*`。标签不作为网络输入；它们只用于计算损失：

```text
U_prop --ordered assignment--> target KeepMask
K* --------------------------> mass/count target
t* --------------------------> ParameterHead 与 subset parameter target
U* --------------------------> proposal/relocation position target
```

真实曲线不出现在训练 batch。真实 manifest 仅供 epoch 后的 held-out validation 和训练后的 benchmark/作图使用。

## 4. Proposal 与 Joint

Proposal 先把参数域和 56 个候选位置学稳定；高 K 分层保证复杂样本在该阶段被充分看到。有序 assignment 与 directed coverage 同时使用：前者保证一一对应，后者保持召回方向。

Joint 解冻完整选择和 subset decoder。目标 mask 由当前 proposal 与真节点的有序匹配直接构造；Selector 学 existence、ranking 和 K，decoder 在实际部署 mask及标签 mask条件下学习参数反馈与 survivor relocation。

正式 Joint 没有在线 Teacher 搜索、Hard-RMS 循环或 Teacher cache。历史 `online_teacher` 分支只用于显式消融，不能与正式结果合并。

## 5. 部署不变式

- 输入只有归一化有序点云和请求 MSE 阈值；
- 不读取合成标签或真实参考折线；
- 只执行一次网络 forward、一次 Top-K 和一次最终 refit；
- 节点严格位于 `(0,1)` 且有序；
- 三次开放完整节点向量为 `[0,0,0,0] + U + [1,1,1,1]`；
- `Kc` 是容量，最终 K 是 KeepMask 的元素数。

## 6. 当前主线与历史版本

v8–v15 的核心是 Hard-RMS/离线 Teacher 蒸馏；早期 v16 的 counterfactual 版本仍需在线组合搜索。当前 supervised-only v16 用认证合成标签直接监督选择与移动，保留一次性部署结构，同时移除正式训练中的 Teacher 依赖。旧 checkpoint 不能靠改名升级为当前合同，只能在明确允许且形状兼容时用作 proposal 初始化。
