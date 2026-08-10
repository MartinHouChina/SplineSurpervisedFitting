# 结构预测重构记录

## v3：候选节点剪枝

v3 使用固定候选节点、ActivityHead 和 Hard-Concrete。验证结果出现两种极端：大量曲线保留全部候选，另一些曲线删除全部候选；单曲线 activity 区分度不足，固定阈值同时承担数量判断和节点身份选择。

结论：问题不是简单的阈值偏移，而是独立 Bernoulli 门与拟合目标之间存在结构冲突。

## v4：直接预测数量

v4 将问题改写为：

```text
CountHead 预测 K → K 专属分支直接生成 K 个有序节点
```

它移除了主路径中的 ActivityHead、Hard-Concrete 和部署阈值。但实验中训练数量准确率约为 63.7%，验证仅约为 21.1%，说明存在明显过拟合。进一步分析发现：

- 数量正确样本的节点 precision 约为 0.60；
- 整体 precision 约为 0.44；
- 使用真实数量后整体 precision 约为 0.55；
- 使用真实参数没有继续改善，ParameterHead 不是主要瓶颈。

根因包括：训练集过小、每个数量分支数据割裂，以及随机生成表示的节点数量并不一定能从曲线几何中唯一恢复。

## v5：canonical ordinal count-conditioned

v5 做了四项修正：

1. 在固定几何容差下贪心删除源节点，把结果作为 canonical 最简标签；
2. CountHead 用局部位置编码 cross-attention 读取整条曲线，并使用序数数量损失；
3. 不同数量共享 interval query 和解码参数，只通过 count embedding 条件化；
4. 部署时比较完整数量分支的 BIC 和 CountHead 先验，不逐节点剪枝。

主路径为：

```text
canonical 标签
  → ordinal CountHead
  → shared count-conditioned decoder
  → 完整 K 节点模型
  → BIC + learned prior 阶次选择
```

论文表述可归纳为“容差约束的 canonical spline supervision、学习式序数阶次估计和条件连续参数回归”。

## 建议消融

| 组别 | 数量机制 | 标签 | 部署选择 |
|---|---|---|---|
| v3 | Bernoulli gates | 源表示 | threshold |
| v4 | categorical | 源表示 | argmax |
| v5-a | ordinal local | canonical | argmax |
| v5-b | ordinal local | canonical | BIC + prior |
| v5 full | ordinal local + shared decoder | canonical | BIC + prior |

统一报告数量 accuracy/MAE、节点 Precision/Recall/F1、匹配 MAE、标准 B 样条 RMS、控制点数和推理时间。
