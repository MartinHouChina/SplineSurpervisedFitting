# 结构预测方案演化

## v3：逐节点剪枝

v3 使用固定候选节点、ActivityHead 和 Hard-Concrete。实验中容易出现两种极端：保留全部候选或删除全部候选；同一曲线内的 activity 区分度也不足。

根因不是单纯的阈值偏移，而是独立 Bernoulli 门同时承担“节点数量”和“节点身份”判断，并与拟合损失相互牵制。

## v4：直接预测节点数量

v4 改为：

```text
CountHead 预测 K → 选择 K 专属分支 → 输出 K 个有序节点
```

它移除了主路径中的 Hard-Concrete，但每个数量拥有独立分支，训练数据被拆散，参数共享不足。

## v5：canonical 标签与条件数量分支

v5 引入：

- 容差约束的 canonical 最简节点标签；
- 局部位置编码的序数 CountHead；
- 共享参数的 count-conditioned decoder；
- BIC 与学习先验联合选择完整数量分支。

这一版改善了标签歧义和分支共享，但部署仍需计算 (K=0,1,\ldots,K_{max}) 的全部节点分支。interval query 总量为

\[
\sum_{K=0}^{K_{max}}(K+1)=O(K_{max}^2),
\]

而 BIC 造成训练时数量决策与部署时数量决策不一致。

## v6：交互式结构预测与动态节点解码

当前主流程是 v6：

```text
带参数位置编码的局部特征
  → Kmax 个结构 query 做 cross-attention
  → 结构 query 之间做 self-attention
  → continuation/stop 概率构造 P(K)
  → 单次 argmax 得到 K
  → DynamicKnotDecoder 只解码该 K 的节点
```

关键变化：

1. 不再实例化独立 CountHead。数量判断来自与节点局部证据交互后的结构 query。
2. 结构 query 通过 self-attention 联合判断“还需要多少节点”，不再逐节点独立阈值剪枝。
3. 训练时用 canonical 真值数量驱动位置解码，分别稳定监督数量和位置。
4. 验证与部署只使用网络的 (\arg\max P(K))，不使用 BIC 或第二次筛选。
5. 当 (K>0) 时只运行 (K+1) 个 interval query；(K=0) 时跳过位置解码。

因此 v6 的节点位置 query 计算只覆盖所选数量，不再枚举全部数量分支。结构 query 仍包含一次 \(K_{max}\) 规模的 cross-attention 和 \(O(K_{max}^2)\) self-attention。

## 版本对比

| 版本 | 数量机制 | 节点标签 | 位置解码 | 部署决策 |
|---|---|---|---|---|
| v3 | 独立 activity 门 | 源表示 | 固定候选后剪枝 | threshold |
| v4 | categorical CountHead | 源表示 | 数量专属分支 | argmax |
| v5 | ordinal CountHead | canonical | 全数量条件分支 | BIC + prior |
| v6 | 交互式 continuation/stop | canonical | 仅所选数量动态解码 | network argmax |

统一报告节点数量 accuracy/MAE、节点 Precision/Recall/F1、匹配 MAE、标准 B 样条 RMS、控制点数量和推理时间。
