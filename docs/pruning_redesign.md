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

历史 v6 主流程为：

```text
带参数位置编码的局部特征
  → Kmax 个结构 query 做 cross-attention
  → 结构 query 之间做 self-attention
  → 汇聚 token 直接分类 P(K)，屏蔽非法数量
  → 后验中位数得到 K
  → DynamicKnotDecoder 只解码该 K 的节点
```

关键变化：

1. 不再实例化独立 CountHead。数量判断来自与节点局部证据交互后的结构 query。
2. 结构 query 通过 self-attention 联合判断“还需要多少节点”，不再逐节点独立阈值剪枝。
3. 训练时用 canonical 真值数量驱动位置解码，分别稳定监督数量和位置。
4. 验证与部署只使用网络数量后验的中位数，不使用 BIC 或第二次筛选；argmax 仅保留为诊断值。
5. 当 (K>0) 时只运行 (K+1) 个 interval query；(K=0) 时跳过位置解码。

因此 v6 的节点位置 query 计算只覆盖所选数量，不再枚举全部数量分支。结构 query 仍包含一次 \(K_{max}\) 规模的 cross-attention 和 \(O(K_{max}^2)\) self-attention。

## v7：高召回候选生成与阈值硬消冗

v6 仍让结构数量和连续节点位置强耦合：数量错误会改变解码长度，interval 前缀和误差还会向后累计。v7 已改为：

```text
点云
  → CandidateKnotHead 生成固定预算的高召回候选
  → 冗余 B 样条拟合
  → InteractivePruningHead 读取截断幂贡献，预测 remove/STOP 并精修位置
  → 标准 B 样条逐节点重拟合，RMS≤ε 才接受删除
  → 最终节点数量由硬删除轨迹产生
```

Boehm 插入可用于后续离线消冗增强，但不要求候选头复现随机插入位置。当前候选头由 canonical 真值的单向 coverage、位置监督和拟合监督训练。

部署外部输入仍只有点云；候选节点和冗余控制点均在系统内部生成。该方案已作为独立 v7 objective 实现，详见 [proposal_pruning_framework.md](proposal_pruning_framework.md)。

## 版本对比

| 版本 | 数量机制 | 节点标签 | 位置解码 | 部署决策 |
|---|---|---|---|---|
| v3 | 独立 activity 门 | 源表示 | 固定候选后剪枝 | threshold |
| v4 | categorical CountHead | 源表示 | 数量专属分支 | argmax |
| v5 | ordinal CountHead | canonical | 全数量条件分支 | BIC + prior |
| v6 | 交互式 categorical 分布 + 合法范围 | canonical/源表示 | 仅所选数量动态解码 | posterior median |
| v7 | 无 CountHead；remove/STOP 仅作建议 | 阈值 canonical + 真实删除 RMS | 固定高召回候选与局部精修 | 逐节点标准 B 样条硬验证 |

统一报告节点数量 accuracy/MAE、节点 Precision/Recall/F1、匹配 MAE、标准 B 样条 RMS、控制点数量和推理时间。
