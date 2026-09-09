# v16 PPT 展示速查

这份提纲只描述当前 v16。新工程协议要求 worst-source deployment pass≥90%，单曲线归一化 MSE 阈值仍为 2.5e-5。性能数字必须来自达到资格的 joint checkpoint 和独立测试；当前旧 Kc=64 训练快照仍是 proposal，不能作为最终结果。

## 第 1 页：问题

标题：阈值约束下的最小复杂度 B 样条拟合。

\[
\min |U|
\quad\text{s.t.}\quad
\operatorname{MSE}(C_U,Q)\le 2.5\times10^{-5}.
\]

输入是有序二维/三维点云，输出是参数、内部节点、控制顶点和拟合曲线。核心难点是删掉一个节点以后，合理参数化和其余节点位置都会变化。

## 第 2 页：核心思路

直接放置 [v16 流程图](figures/v16_pipeline.svg)：

~~~text
高覆盖候选
  → 一次性 KeepMask
  → KeepMask 条件下联合更新参数与存活节点
  → 一次标准 B 样条 refit
~~~

一句话：先保证候选空间能拟合，再学习一个可行且更小的节点组合。

## 第 3 页：候选网络

展示：

- GeometryEncoder：点坐标、弦长、一阶/二阶有限差分；
- ParameterHead：弦长残差参数 t0；
- CandidateKnotHead：Kc+1 个锚定 interval query；
- 局部 Gaussian cross-attention；
- Kc 个有序、全域覆盖候选 U0。

强调 Kc 是容量上限，不是最终 K。

## 第 4 页：一次性筛选

每个候选 token 同时读取：

- 自身位置、左右间距、最近采样参数距离；
- 其他候选的 self-attention 上下文；
- 点云几何的 cross-attention 上下文；
- 当前 MSE 容差。

KeepHead 输出中心化候选重要性，曲线级阈值头输出自适应 `beta`。部署用概率质量、不确定性安全余量和一次全局 Top-K 决定 KeepMask；`p≥0.5` 只保留为旧式消融。节点数就是 KeepMask.sum()，无 CountHead、无 BIC、无逐次删除。

## 第 5 页：删除与重定位联动

~~~text
KeepMask
  → 只让存活 token 成为 Key/Value
  → t0 更新为 t1
  → U0 从 t0 warp 到 t1
  → 按存活邻居、rank、count 联合重定位
  → 正间隔投影
~~~

强调存活节点可以移动到被删候选留下的区域，不是简单原位取子集。

## 第 6 页：两阶段训练

Proposal：

- 全保留 Kc；
- 真实标准 B 样条可微 refit；
- 先提高每个数据源的 dense feasibility。

Joint：

- Bernoulli 随机集合；
- 删除、添加、交换与分散反事实；
- 每个集合均做条件解码和 refit；
- 可行时先最少 K，再最小 MSE；不可行时只降低 MSE。

训练有额外 refit，部署没有。

## 第 7 页：数据

| 来源 | 配置 / 含义 |
|---|---|
| Synthetic | 三次开放样条；控制顶点 8–24，对应源 K=4–20 |
| UJI | 手写轨迹，writer-disjoint |
| Natural Earth | 海岸线，geographic-group-disjoint |
| USGS | 等高线，geographic-group-disjoint |

默认快速折中每轮 2400 个混合样本，real_fraction=0.5；验证为 500 个合成样本加每个真实来源最多 100 个，batch=16，epochs=60、proposal=20。推荐复用 proposal 的 fast90 实验进一步缩短为 train=2000、epochs=50、proposal=5。真实数据无节点标签。

## 第 8 页：评价口径

必须同时报告：

- MSE mean / P95 / max；
- threshold-satisfied fraction；
- 最终内部节点数；
- 每个数据源及 worst-source 通过率；
- network time；
- full deployment time；
- original-reference MSE。

MSE 是平均平方欧氏距离，不开方。纯网络时间不能直接和数值基线完整搜索时间解释为端到端加速比。

## 第 9 页：对比方法

定量表：

- Ours v16；
- Park–Lee；
- Liang；
- Dung–Tjahjowidodo；
- Kang；
- Luo–Kang–Yang；
- Yeh；
- uniform greedy + position refinement。

所有公开方法均标注为基于论文目标的工程适配，不声称是官方代码。四宫格定性图只放 Ours、Kang、Yeh、uniform greedy。

## 第 10 页：当前结果与下一步

以下是 2026-09-09 11:41 的训练中快照，展示前必须用最终 checkpoint 更新：

| 指标 | 数值 |
|---|---:|
| best proposal epoch | 59 |
| overall dense pass | 98.923% |
| Synthetic / UJI | 100% / 100% |
| Natural Earth / USGS | 93% / 93% |
| worst-source | 93% |
| 该旧实验启动目标 | 97% |

该快照来自原 97% 配置；运行中不会自动切换到新 90% 协议。它的结论仍是尚不能进入最终 selector 结论。下一步分成可区分实验：

1. 新建自适应 Kc=64 output，作为轻量容量消融；
2. 在相同数据、训练轮数和随机设置下训练 Kc=96，作为复杂真实曲线主模型；
3. 只有 Kc=96 的 dense proposal 仍明显受限时，才追加 Kc=128；真实比例强化另开实验。

三类实验都必须使用独立 output，不能用原 Kc=64 的 resume 改配置，也不能把混合改动的收益全部归因于容量。

## 展示时不要说

- 不要说旧实验已达到 97%，也不要把 90% 通过率说成 MSE 阈值放宽；overall 不能替代 worst-source。
- 不要把 proposal checkpoint 当完成的 one-shot 模型。
- 不要声称获得全局最少节点。
- 不要把诊断模式或回退搜索写成纯一次性部署。
- 不要人工调整图中的 Ours MSE。
- 不要用训练或验证曲线代替独立测试。
