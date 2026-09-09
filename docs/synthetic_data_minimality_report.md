# 合成曲线最简性：问题、证书与 v16 处理

## 1. 为什么随机生成的节点数不等于最简节点数

随机控制顶点和随机内部节点只定义了一个“生成表示”，不能自动成为最简表示：

- 同一条 B 样条可通过 Böhm 插结得到任意多的等价冗余节点；
- 即使没有显式插结，平滑随机曲线在允许误差下也可能被更少节点近似；
- 有限采样、噪声和参数化都会改变“多少节点足够”的判断。

对旧 v16 规格做过一次固定 seed 审计：二维、192 点、源内部节点 `K=4..20`、
`RMS=0.005`。100 条干净曲线中，源节点平均为 11.66，贪心阈值简化后平均为
7.24；96% 的源表示还能继续删点，平均可删 4.42 个、最多 11 个。因此旧式
`source K` 只能称为生成节点数，不能当作最简复杂度真值。

## 2. 本仓库采用的可证明范围

v16 正式合成数据默认启用：

```text
source-subset threshold-minimal certificate
```

设干净曲线采样为 `Q*`，固定参数为 `t*`，源内部节点集合为 `U`，工程 MSE
阈值为 `epsilon`。证书使用

```text
RMS tolerance = sqrt(epsilon)
```

并要求：

\[
E(U)\le \sqrt{\epsilon},\qquad
\min_{u_j\in U}E(U\setminus\{u_j\})>
\sqrt{\epsilon}(1+m),
\]

其中 `E` 是 CPU `float64`、端点约束、无平滑正则、无 ridge 的标准三次 B 样条
最小二乘 RMS，`m` 是安全 margin，默认 0.2。

为什么只检查全部单节点删除就足够？任意真子集 `V⊂U` 至少包含在某个
`U\{u_j}` 对应的样条空间中。无正则最小二乘进入更小的函数空间后误差不可能下降。
所以，只要每一个单删空间都不满足阈值，源节点的任何真子集也不可能满足阈值。

这给出了固定 `t*`、固定源节点候选族内的最少基数证书，不只是一次贪心路径的结果。

## 3. 数据生成顺序

```text
先固定目标 K
  → 生成带高频几何细节的干净三次 B 样条
  → 在均匀 512 点审计网格上检查完整拟合和全部单节点删除
  → 不通过则重采控制多边形和节点，但不重抽 K
  → 通过后冻结 source knots、clean points 和证书
  → 最后才加入观测噪声，作为网络输入
```

固定 K 后再拒绝采样，避免较大 K 因更难通过而被悄悄过滤。标签由干净曲线生成，
噪声只进入网络输入，因此不会改变最简节点数和节点位置标签。

## 4. v16 当前接线

`scripts/train_v16.py` 的正式默认值为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--certified-minimal-source` | 开启 | 使用最简性证书 |
| `--minimality-margin` | 0.2 | 单删 RMS 至少越过阈值 20% |
| `--minimality-max-attempts` | 16 | 固定 K 后最多重采次数 |
| `--minimality-audit-points` | 512 | 干净均匀审计点数 |
| `--oscillation-amplitude` | 0.3 | 提高 K 与几何复杂度的一致性 |

当 `--mse-tolerance 2.5e-5` 时，保存到数据配置里的证书阈值严格为
`sqrt(2.5e-5)=0.005`，不会把 MSE 数值误当 RMS。checkpoint 同时保存：

```text
synthetic_data_contract = source_subset_threshold_minimal_v1
dataset_config.certified_minimal_source = true
```

正式资格检查会拒绝没有这两个字段的旧权重。`--no-certified-minimal-source` 仅用于
消融和诊断，其结果不能作为正式最简数据实验。

样本可审计字段包括：

- `source_minimality_certified`；
- `source_full_fit_rms` / `source_full_fit_mse`；
- `source_min_single_deletion_rms` / `source_min_single_deletion_mse`；
- `source_minimality_required_rms` / `source_minimality_required_mse`；
- `source_generation_attempts` 和 `clean_points`。

独立 benchmark 会重新读取 checkpoint 的数据合同，并逐条断言证书为真且
`source K == canonical K`。

## 5. 可以声称什么，不能声称什么

论文中可以准确表述为：

> Each synthetic source is certified tolerance-minimal over all subsets of its
> original knot set under the fixed clean parameterization.

不能把它写成“连续全局最少节点证明”，因为证书不覆盖：

- 删除后允许所有剩余节点任意连续重定位；
- 任意重新参数化；
- 连续曲线上所有未采样位置；
- 非凸节点优化的数学全局最优。

若需要更强的实证，可以额外对 `K-1` 个节点做多启动 `delete + relocation` 对抗审计；
若始终失败，可称为 relocation-resistant empirical audit，仍不能称全局证明。

对于“零误差精确表示”的理论讨论，还可用截断幂形式

\[
C(t)=P_3(t)+\sum_j b_j(t-u_j)_+^3
\]

并约束每个 `b_j` 非零且有下界，使每个 `u_j` 对应不可消失的三阶导数跳变。但工程
任务允许非零误差，所以正式实验仍应以这里的阈值证书为准。

## 6. 复现实验命令片段

以下参数已是 v16 默认值，正式命令中仍建议显式写出，便于审稿复现：

```powershell
--certified-minimal-source `
--minimality-margin 0.2 `
--minimality-max-attempts 16 `
--minimality-audit-points 512 `
--oscillation-amplitude 0.3
```

旧 checkpoint 不会被静默升级；需要用新输出路径重新训练。旧模型可以作为网络参数的
warm-start，但不能提供新数据合同的实验结论。
