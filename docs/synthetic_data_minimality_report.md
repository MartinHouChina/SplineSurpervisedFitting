# 合成曲线最简性：证书与边界

## 1. 为什么随机 source K 不一定最简

随机控制顶点和随机节点只给出一种生成表示。同一几何可以通过 Böhm 插结得到等价的冗余表示，也可能在有限误差阈值下由更少节点近似。因此不能把未经检查的随机 source K 直接称为“最少节点真值”。

当前 v16 只训练通过 source-subset minimality 证书的合成曲线。

## 2. 当前生成范围

| 项目 | 值 |
|---|---|
| 曲线 | 三次开放 B 样条 |
| source 内部节点 K | 4–56 |
| source 控制顶点 | 8–60 |
| 网络候选 Kc | 56 |
| 每条网络输入点 | 192 |
| 最简性审计点 | 512 个干净均匀参数点 |
| `knot_min_span` | 0.01 |
| 工程阈值 | `MSE<=1e-4`，即 `RMS<=0.01` |
| 删除 margin | 20% RMS |

`K=56` 会产生 57 个参数 span。若最小 span 为 0.02，则总下界 `57×0.02=1.14>1`，数学上不可行；0.01 的下界为 0.57，因此当前范围仍有非均匀分配空间。

## 3. 证书

设干净审计点为 `Q*`，固定真参数为 `t*`，source 内部节点集为 `U`。使用统一 CPU float64、端点插值、无平滑、无 ridge 的标准三次 B 样条 refit，定义 RMS `E(V)`。

曲线被接受当且仅当

\[
E(U)\le 0.01,
\qquad
\min_{u_j\in U}E(U\setminus\{u_j\})>0.012.
\]

任意严格子集都包含于至少一个单删样条空间；在固定参数、固定候选族和无正则的最小二乘条件下，更小空间不可能比其父空间有更低最优误差。因此检查所有单删即可覆盖 source 原节点的所有严格子集。

## 4. 生成顺序

```text
固定 K in 4..56
 -> 生成满足 min_span=0.01 的 source knots 与控制顶点
 -> 生成干净三次 B 样条
 -> 在 512 点网格做完整拟合和全部单删审计
 -> 不合格：保持 K，重新抽几何
 -> 合格：冻结 clean points、t*、U*、K* 和证书字段
 -> 最后加入观测噪声，形成网络输入 Q
```

先固定 K 再拒绝采样，避免高 K 因更难通过证书而被静默过滤。真参数、真节点和真 K 均来自干净源；噪声只进入观测。

## 5. 当前监督语义

正式训练使用：

```text
--certified-minimal-source
--joint-supervision synthetic_ground_truth
--synthetic-count-role exact
--real-fraction 0
```

这里的 `exact` 是训练合同：网络必须重建这条已认证 source 表示的 K、KeepMask 和节点位置。它不是对允许任意重新参数化和连续节点重定位后全局最小 K 的数学声明。

论文建议表述：

> Each synthetic source is certified tolerance-minimal over all subsets of its original knot set under the fixed clean parameterization. Its source count is used as the exact supervised target in the current training protocol.

不得写成“已证明连续全局最少节点数”。

## 6. 训练用途与真实数据边界

只有 certified Synthetic 进入 Proposal 和 Joint 的梯度更新。其 `t*、U*、K*` 通过有序一一匹配直接监督候选、KeepMask、count 与 relocation；不生成在线 Teacher。

UJI、Natural Earth 和 USGS 没有上述节点证书，只用于 validation/test。不得为它们伪造 K 或节点标签，也不得把 original-reference 折线送进部署网络。

## 7. 必须报告的审计字段

每条合成样本至少保留：

- `source_minimality_certified`；
- `source_full_fit_rms` / `source_full_fit_mse`；
- `source_min_single_deletion_rms` / `source_min_single_deletion_mse`；
- `source_minimality_required_rms` / `source_minimality_required_mse`；
- `source_generation_attempts`；
- clean points、真参数和 source knots。

正式验证还应单列 K=56 边界样本数、dense pass 和 deployment pass。该层 `Kc=K*`，没有冗余候选，因此它检验的是容量边界，不应与低 K 简化能力混为一谈。

## 8. 尚未覆盖的更强命题

当前证书不覆盖：

- 删除后把节点移动到 source 集合之外；
- 重新预测参数化；
- 在连续参数域搜索全新节点；
- 对连续曲线所有未采样位置做解析全局证明。

若要加强证据，可另做多启动的 `K-1` 节点自由重定位对抗审计，并准确称为 empirical relocation-resistant audit，仍不能升级为连续全局最优定理。
