# 合成曲线最简性：证书、边界与当前 v16 处理

当前主协议生成三次开放 B 样条：控制顶点 8～60，对应源内部节点
`K_source=4..56`；每条曲线采样 192 点，阈值为 `MSE<=1e-4`。网络使用
56 个内部候选（全保留时完整节点向量 64 项、控制顶点 60 个），但候选容量
与源节点数、最终部署节点数是三个不同概念。

## 1. 为什么随机源节点数不等于最简节点数

随机控制顶点和随机内部节点只定义一个生成表示，不能自动成为最简表示：

- 同一条 B 样条可通过 Böhm 插结得到任意多的等价冗余节点；
- 即使没有显式插结，平滑随机曲线在给定误差下也可能被更少节点近似；
- 有限采样、观测噪声、参数化和是否允许节点重定位都会改变最小所需节点数。

因此训练不能把随机生成时的 `K_source` 无条件当成“连续全局最少 K”的精确
分类标签。

历史 `K=4..20 / MSE=2.5e-5` 消融曾显示大量随机源表示仍可继续删点；该结果
只用于说明问题，不是当前 `K=4..56 / MSE=1e-4` 主实验的统计结果。旧
`K=4..24` 数据也只作为历史范围消融。

## 2. 当前可证明的最简范围

当前 3090 主训练显式启用：

```text
source-subset threshold-minimal certificate
```

设干净采样为 `Q*`，固定参数为 `t*`，源内部节点集合为 `U`，工程阈值为
`epsilon=1e-4`。证书使用同一 CPU `float64`、端点约束、无平滑、无 ridge
的标准三次 B 样条最小二乘，并要求

\[
E(U)\le \sqrt{\epsilon},\qquad
\min_{u_j\in U}E(U\setminus\{u_j\})>
\sqrt{\epsilon}(1+m),
\]

其中 `E` 是 RMS，`m=0.2`。本协议中：

- 完整源表示要求 `RMS<=0.01`，即 `MSE<=1e-4`；
- 每个单删表示要求 `RMS>0.012`，对应 MSE 大于 `1.44e-4`。

为什么检查全部单节点删除足以覆盖源节点的所有真子集？任意真子集
`V subset U` 都落在至少一个 `U\{u_j}` 的样条子空间中；固定参数、固定节点
候选族且无正则时，更小函数空间的最小二乘误差不可能优于包含它的单删空间。

所以该证书准确表明：在固定干净参数化和原始源节点所有子集构成的离散族中，
没有更小子集满足当前阈值。它比一次贪心删除路径更强，但不是连续全局最优
证明。

## 3. 数据生成顺序

```text
固定目标 K in 4..56
  -> 以 knot_min_span=0.01 生成 8..60 个控制顶点和 K 个内部节点
  -> 生成带相应几何细节的干净三次 B 样条
  -> 在 512 点干净网格检查完整拟合和全部单节点删除
  -> 证书失败则重采几何，但不重抽 K
  -> 证书通过后冻结 clean points、true params、source knots 和证书
  -> 最后加入观测噪声，得到网络输入
```

先固定 K 再拒绝采样，避免高 K 样本因更难通过证书而被静默过滤。真参数、
真节点和计数来自无噪声源曲线；噪声只进入网络观测。

`K=56` 个内部节点会把 `[0,1]` 划成 57 个 span。若每个 span 都要求至少
0.02，总长度下界为 `57*0.02=1.14>1`，因此旧 synthetic 默认
`knot_min_span=0.02` 在该层数学上不可行。当前主协议必须显式传入
`--knot-min-span 0.01`；此时下界为 0.57，仍有空间生成非均匀节点。底层
通用 synthetic 生成器中的 0.02 默认值仅为历史调用兼容，不是当前 K=4..56 数据合同。checkpoint 和实验
manifest 必须记录实际的 0.01。

## 4. 为什么当前把 source K 当上界

证书没有覆盖以下变化：

- 删除后把剩余节点移动到原节点集合之外；
- 重新预测参数化；
- 在连续参数域内寻找任意全新节点；
- 对连续曲线上全部未采样位置作全局验证。

v16 的 `decode_subset` 正好允许参数更新和存活节点重定位。因此某个网络组合
可能用少于 `K_source` 的节点达到 `1e-4`，而不与源子集证书矛盾。

当前训练使用

```text
--synthetic-count-role upper_bound
```

含义是：源 K 提供有证书的复杂度上界和几何监督，但当在线教师找到更小且
实际可行的重定位组合时，不用精确计数损失把预测重新拉回 source K。这比把
源 K 当 exact 标签更符合网络允许移动节点的建模范围，也能减少简单曲线
系统性 over-count。

论文中可准确写为：

> Each synthetic source is certified tolerance-minimal over all subsets of its
> original knot set under the fixed clean parameterization. The source count is
> used as an upper bound when continuous relocation is enabled.

不能写成“源节点数是连续全局最少节点证明”。

## 5. geometry-oracle 教师如何避免错误自举

仅按当前 Selector 分数搜索 ranked prefix，训练初期可能发生闭环：错误排序
生成错误教师，错误教师又强化原排序。当前合成训练额外启用：

```text
--synthetic-geometry-oracle-teacher
--oracle-teacher-extra-knots 2
```

oracle 先将 56 个 proposal 节点映射到真参数域，再与源真节点做一维单调
一对一匹配，构造 `Ktrue` mask；同时加入截断到 Kc 的 `Ktrue+2` 安全扩展
mask，故 Ktrue=56 时就是全候选。它提供
与当前 Keep 分数无关的几何锚点，并参与实际 refit 可行性比较。

该 oracle 只用于有真节点的合成训练样本；真实数据训练和任何部署都不读取
真节点。部署仍然只有一次网络前向、一次 Top-K 和一次最终 refit。

## 6. 简单曲线与复杂曲线的共同处理

当前采用 `Kc=56`，使完整三次开放节点向量上限恰为 64 项。source K 描述
生成复杂度，Kc 描述候选槽位；两者最大值相同不代表部署必须全保留。简单
曲线的细粒度选择由以下训练机制完成：

1. `--initial-keep-fraction 0.5357142857142857`，令 Kc=56 的 Joint 初始质量目标约为 30；它不是部署最终 K；
2. `--teacher-low-count-sweep 16`，逐一检查 `K=4..16`，给简单曲线细粒度教师；
3. `--synthetic-count-role upper_bound`，允许可行的低于 source-K 组合；
4. geometry-oracle `Ktrue` 与 `Ktrue+2`，补足早期候选排序；
5. `--one-shot-coverage-bins 0`，取消会强占简单曲线预算的固定区间锚点。

高于 16 的计数仍由粗到细前缀搜索、边界邻域编辑和全保留候选保护。最终
是否兼顾简单与复杂曲线，必须用 `K=4..56` 分层独立测试验证，不能由架构
设置直接宣称。`K=56` 层没有冗余候选余量，必须单独报告 dense/deployment
pass；失败不能用放宽 90% 资格门槛处理。

## 7. 当前训练参数

一键脚本显式传入：

```powershell
--min-control-points 8 `
--max-control-points 60 `
--knot-min-span 0.01 `
--candidate-knots 56 `
--mse-tolerance 1e-4 `
--certified-minimal-source `
--minimality-margin 0.2 `
--minimality-max-attempts 16 `
--minimality-audit-points 512 `
--oscillation-amplitude 0.3 `
--initial-keep-fraction 0.5357142857142857 `
--teacher-low-count-sweep 16 `
--synthetic-count-role upper_bound `
--synthetic-geometry-oracle-teacher `
--oracle-teacher-extra-knots 2 `
--one-shot-coverage-bins 0
```

完整命令由
[run_v16_mse1e-4_3090.ps1](../scripts/run_v16_mse1e-4_3090.ps1) 固定。checkpoint
应保存数据合同、证书配置和教师配置；旧 checkpoint 不会被静默升级，只能
作为允许字段匹配的网络初始化，不能提供当前数据合同下的实验结论。

## 8. 可审计字段与更强实证

单样本至少保存：

- `source_minimality_certified`；
- `source_full_fit_rms` / `source_full_fit_mse`；
- `source_min_single_deletion_rms` / `source_min_single_deletion_mse`；
- `source_minimality_required_rms` / `source_minimality_required_mse`；
- `source_generation_attempts` 和 `clean_points`。

若需要比源子集证书更强的实证，可对 `K_source-1` 个可移动节点执行多启动
`delete + relocation` 对抗审计。若所有启动都失败，可称
`relocation-resistant empirical audit`，仍不能声称数学上的连续全局最优。

旧 source K=4..24、旧 K64、旧 `Kc=96 / MSE=2.5e-5 / 97%` 数据、证书和
checkpoint 只可明确标作历史消融，不能与当前
`Kc=56 / source K=4..56 / MSE=1e-4 / 90%` 主协议合并。
