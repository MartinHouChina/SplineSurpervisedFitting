# 节点结构方案演化

当前主版本是 v16；历史 checkpoint 通过兼容层读取，但保持各自原有部署语义。
v16 使用在线反事实组合学习和 selected-only 联合解码，完整说明见
[v16 主流程](v16_counterfactual_subset.md)。本页其余内容按版本记录演化，不代表当前部署。

## v3–v6：直接结构预测

- v3：独立 knot query、ActivityHead 与 Hard-Concrete；容易出现概率集中在窄区间、阈值后全保留或全删除。
- v4/v5：直接 CountHead 与 canonical count；数量和位置条件解码更清楚，但离散数量错误会直接改变整条节点向量。
- v6：交互结构头与动态区间解码；减少无效分支，但仍把复杂度决策压在一次 count 分类上。

## v7：冗余 proposal + 传统 Hard-RMS

网络只负责生成高召回冗余候选，部署时逐个尝试删除并执行标准 B 样条 refit。优点是误差判定直接、过程可解释；缺点是需要大量串行/批量 refit，不是一次性部署。

## v8–v10：离线教师 + 一次性 LearnedKeep

- v8 把 Hard-RMS 删除放到离线阶段，在线 student 一次性预测 mask；
- v9 固定 proposal 槽位，避免 selector 更新破坏 teacher cache 对齐；
- v10 用候选交互、mass-TopK 和验证集约束选择强化整组节点组合。

这些版本已把在线计算降为一次 forward + 一次 refit，但最终位置主要仍受固定 proposal 限制。

## v11：Keep 与位置反馈

v11 引入局部 Gaussian proposal attention 和固定深度交互：

```text
p0 -> provisional position -> p1 -> final mask -> deployment position
```

其核心问题不在于“代码完全没有位置头”，而在于教师目标：`teacher_internal_knots` 直接复制 retained proposal 位置。因此位置监督把零移动当作正确答案，节点删除后新的邻接关系也没有被显式编码。旧图又只画 proposal 横坐标和 retained 索引，使 hybrid 的真实位置变化不可见。

## v12：删除与存活节点重定位联动

v12 同时修改离线教师和在线 student。

### 离线教师

```text
greedy delete
  -> relax survivor positions
  -> retry delete
  -> 重复有限轮
  -> 缓存最终槽位 mask、优化后位置、gap 和风险
```

位置优化后如果还能安全删除，教师会继续减少节点；最后一轮产生新 survivor 集合时，还会再对最终集合做一次 relaxation。

### 在线网络

```text
final KeepMask
  -> 按最终 survivors 重算邻居、rank、count 和覆盖特征
  -> Key/Value 仅来自 survivors 的 multi-head attention
  -> 只对 survivors 施加位置残差
  -> 有界单调 U_deploy
```

straight-through gate 让位置损失可以影响 keep score，因此删除和移动不再是完全独立的两个目标。最终更新遵守端点、相邻 survivor、`min_gap` 和相对 proposal 的最大位移。

v12 仍是固定深度一次性网络：它没有在线逐节点试删，也没有第二次网络 forward。标准 B 样条只在最终部署节点上 refit 一次。

## v13–v15：修正监督与部署损失对齐

- v13 改为按实际预测 survivor 集合匹配位置，并补充槽位无关的集合覆盖监督；
- v14 让结构输出反馈到参数头，再把候选映射到更新参数域；
- v15 用实际离散 mask 的标准 B 样条部署 MSE 校准，并对齐 mass-TopK 的计数语义。

它们仍依赖固定离线 Hard-RMS 教师，候选是否可行与 selector 是否可行没有形成严格的两阶段 gate。

## v16：候选可行性 + 在线反事实组合

v16 使用独立模型和训练入口：

~~~text
proposal：全候选真实 refit → worst-source 可行性 gate
joint：随机/反事实 KeepMask → selected-only 参数与位置联合解码 → 真实 refit
deployment：adaptive beta + probability-mass Top-K 一次选择 → 一次 refit
~~~

它取消离线 teacher 和旧 checkpoint 的隐式兼容迁移，重新引入曲线级自适应 beta 与 mass-TopK，并明确把“候选空间足够”和“组合足够小”分开验证。

## hybrid：独立的离线质量搜索

hybrid 不是 v12 网络层。它在一次网络预测后执行 beam 删除、coordinate 位置精修和多次标准 refit；传统 greedy 作为 fallback。它会实际移动节点，当前图显示 `proposal u -> refined u*`。

hybrid 适合时间不敏感、需要逐样本继续压缩的场景。有限 beam 与局部网格仍不能证明全局最少节点。

## 版本对比

| 版本 | 结构选择 | 连续位置更新 | 默认部署 |
|---|---|---|---|
| v7 | 在线 greedy Hard-RMS | 无 | 多次 refit |
| v8–v10 | 离线教师蒸馏的一次性 mask | 固定/弱 | 一次 forward + 一次 refit |
| v11 | 一次性 mask 与位置反馈 | 有头，但教师偏向零位移 | 一次 forward + 一次 refit |
| v12 | final-mask 条件化 selector + relocation | delete-then-relax 监督、selected-only attention | 一次 forward + 一次 refit |
| v13–v15 | 离线教师的一次性 mask | 参数、mask、位置逐步联动并对齐部署 MSE | 一次 forward + 一次 refit |
| v16 | 在线 Bernoulli/反事实组合学习 | selected-only 参数更新、warp 与重定位 | 一次 forward + 一次 refit |
| hybrid | beam/greedy 子集搜索 | coordinate refinement | 一次 forward + 多次 refit |

## 兼容边界

- v16 使用独立 objective 和入口，v8–v15 checkpoint 不能改名后当作 v16。
- v16 不读取旧离线 teacher cache；只允许通过 init-checkpoint 部分迁移形状兼容的 encoder/candidate tensors。
- v16 正式结果要求完成 joint 且 worst-source deployment pass 达到协议目标。

- v11 权重可以严格载入 v12；新增 relocation head 零初始化，初始行为保持中性。
- 载入不等于训练完成。需要 v12 delete-then-relax teacher 和联合校准才能获得非零重定位。
- v11 teacher cache 不应直接复用到 v12；首次训练应创建新目录。
- `--teacher-relaxation-min-gap` 默认等于 `--min-knot-gap`，不是 match tolerance。抑制聚集时优先调大 `--min-knot-gap`，使 proposal、student 与 teacher 使用一致约束；这属于建模假设，需要单独做消融。
- learned v12 不提供逐样本阈值或全局最优保证；hybrid 也只在访问过的状态中择优。
