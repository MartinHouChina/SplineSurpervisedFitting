# v7 数据与训练流程

## 1. 优化目标

训练和部署共享同一个归一化欧氏 RMS 阈值 `ε`：

\[
\min |U|\quad \text{s.t.}\quad
\sqrt{\frac1M\sum_i\|C_U(t_i)-q_i\|_2^2}\le\varepsilon.
\]

源文件里原本有多少节点不是学习目标。Boehm 插入可以在不改变曲线的情况下增加节点，因此
恢复任意源节点数既不可辨识，也不等于最简表示。

## 2. 一条训练样本如何生成

1. 在给定范围内随机选择控制点数；三次样条的源内部节点数为 `控制点数-4`。
2. 生成平滑随机控制多边形和非均匀开放节点向量。
3. 生成严格递增的非均匀参数，并采样二维或三维曲线。
4. 加入坐标噪声。
5. 对点集做中心化和最大半径归一化。
6. 从源内部节点开始，逐轮尝试删除每一个节点，并用端点约束的标准 B 样条最小二乘重新求解
   全部控制点。
7. 选择本轮 RMS 最低的删除；仅当 RMS 不超过 `ε` 时接受。重复直到不可继续删除。

最终返回：

```text
points                       [M,D]
chord_params                 [M]
true_params                  [M]
true_internal_knots          [Ksource_max]
true_internal_knot_mask      [Ksource_max]
true_control_points          padded
source_internal_knot_count   scalar
canonical_fit_rms            scalar
center, scale
```

canonical 是确定性贪心标签，不是全局组合最优证明。

## 3. 默认 4–20 节点实验

| 配置 | 值 |
|---|---:|
| 源控制点 | 8–24 |
| 源内部节点 | 4–20 |
| 候选节点 `Kc` | 28 |
| 点数 | 192 |
| 噪声标准差 | 0.001 |
| RMS 阈值 | 0.005 |
| 训练 / 验证 seed | 42 / 10000 |

canonical 删除后允许得到 0–20 个节点，因为“源范围为 4–20”和“阈值下最简数量”是两个
不同概念。不要再把最小合法预测数量强制夹到 4。

高 K canonical 化需要大量标准 B 样条重拟合。训练集默认固定并在当前进程中缓存；验证集
始终固定。`--resample-train-each-epoch` 会每个 epoch 重新支付标签生成成本，通常不建议。

## 4. 两阶段训练

### 4.1 候选预训练

默认前 20 个 epoch 不训练 keep、remove/STOP 和删除代价输出，只优化：

- 真实参数监督；
- true→candidate 单向覆盖；
- 候选间轻量排斥；
- 匹配节点的位置精修；
- 全候选截断幂拟合；
- 超过 `ε` 的全候选拟合惩罚。

单向覆盖允许额外候选存在，直接对应“先保证召回，再消冗”。

### 4.2 联合训练

联合阶段增加：

- canonical keep 辅助 BCE；
- 多正例 remove/STOP 动作损失；
- 删除代价回归；
- 候选 self-attention 与位置精修。

若多个候选均可视为冗余，动作损失优化这些正确删除动作的总概率：

\[
L_{action}=-\log\sum_{j\in\mathcal S}P(a=j).
\]

没有可删除动作时监督 STOP。`count_consistency` 默认权重为 0；它不是最终数量来源。

截断幂删除增量只作为廉价输入特征。训练实现还使用标准 B 样条单节点删除 RMS 教师来监督
删除是否跨越阈值；部署时仍会重新计算，不信任网络预测。

## 5. 损失尺度

覆盖误差除以候选匹配容差，拟合误差除以 `ε²`，避免数值约为 `1e-4` 的拟合项被分类项
淹没：

\[
L_{fit}=\frac{\operatorname{MSE}_{Euclidean}}{\varepsilon^2},\qquad
L_{violation}=\max(0,\operatorname{RMS}/\varepsilon-1)^2.
\]

默认联合权重：

| 项 | 权重 |
|---|---:|
| normalized fit | 0.25 |
| threshold violation | 5.0 |
| true parameters | 0.05 |
| candidate coverage | 5.0 |
| candidate repulsion | 0.05 |
| keep auxiliary | 0.25 |
| remove/STOP | 1.0 |
| refined knot position | 2.0 |
| deletion cost | 0.05 |
| count consistency | 0.0 |

## 6. Checkpoint 选择

v7 不再以 `count_acc` 选权重。当前字典序为：

```text
candidate recall
→ candidate nearest MAE
→ safe-action top-1 / unsafe-delete / false-STOP
→ learned keep 集合的节点 F1/precision/MAE（诊断）
→ validation loss
```

最佳权重和最后权重分别保存，防止再次出现早期 checkpoint 被单一指标误选而无法复核的问题。
最终模型仍必须用独立 seed 的硬部署指标验收。

## 7. 正式训练命令

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --log-every-batches 20 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --candidate-match-tolerance 0.02 `
  --output outputs/candidate_pruning_v7.pt
```

小规模链路检查：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 2 `
  --candidate-pretrain-epochs 1 `
  --train-size 32 `
  --val-size 16 `
  --num-points 64 `
  --min-control-points 4 `
  --max-control-points 8 `
  --candidate-knots 8 `
  --hidden-dim 32 `
  --encoder-layers 1 `
  --output outputs/candidate_pruning_smoke.pt
```

该命令只验证代码链路，不代表模型精度。

## 8. 建议验收项

- candidate recall@0.02 ≥ 0.98；
- candidate recall@0.01 ≥ 0.90；
- 全候选标准 B 样条阈值满足率 ≥ 0.99；
- 硬剔除后阈值满足率 ≥ 0.99；
- 端点最大误差 < `1e-6`；
- 报告最终 RMS mean/P95/max 与节点数分布；
- 同时报告 match@0.005/0.01/0.02，避免宽容差和冗余节点虚增 precision。
