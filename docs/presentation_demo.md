# v16 PPT 展示速查

本提纲只描述当前主协议：`Kc=56`，合成源内部节点 `K=4..56`（控制顶点
8～60），单曲线 `MSE<=1e-4`，正式资格为 worst-source deployment pass
`>=90%`。所有性能数字必须来自通过资格检查的 joint checkpoint 和独立 test；
本文不预设尚未跑出的结果。

## 第 1 页：任务

标题：阈值约束下的低复杂度 B 样条拟合。

\[
\min |U|
\quad\text{s.t.}\quad
\operatorname{MSE}(C_U,Q)\le 10^{-4}.
\]

输入是 192 个有序二维/三维点；输出是参数、内部节点、控制顶点和拟合曲线。
难点是节点删除以后，合理参数化和其余节点位置也会改变。

## 第 2 页：总体流程

展示 [v16 流程图](figures/v16_pipeline.svg)：

```text
有序点云
  -> 几何与参数编码
  -> 56 个高召回有序候选
  -> 自适应 beta + probability-mass Top-K，一次生成 KeepMask
  -> KeepMask 条件下联合更新参数和存活节点位置
  -> 一次标准三次 B 样条 refit
```

一句话：56 个内部候选使完整三次开放节点向量恰为 64 项；候选容量保护复杂
曲线，教师和自适应选择器让简单曲线使用更少节点。

## 第 3 页：候选网络

- GeometryEncoder：坐标、弦长、一阶和二阶有限差分；
- ParameterHead：在弦长参数上学习有界残差，得到严格递增 `t0`；
- CandidateKnotHead：局部 Gaussian cross-attention；
- 输出 56 个严格有序候选 `U0` 和 56 个候选 token。

强调：`Kc=56` 是内部候选上限，不是最终节点数。全保留时完整节点向量为
64 项、控制顶点为 60 个。

## 第 4 页：为什么不降低全局 Kc

当前 K56 把完整节点向量统一为 64 项，并覆盖 source K=4..56。source K 和
候选 Kc 的最大值虽然相同，语义并不相同；K=56 是没有冗余候选余量的容量
边界。简单曲线保留过多的根因主要是低 K 监督粗糙、早期排序自举和固定覆盖
锚点，而不是最终必须保留全部候选。

当前针对简单曲线使用四项修正：

1. Selector 的 Joint 初始保留比例设为 `30/56≈0.535714`，目标初始概率质量约为 30；这不是部署最终 K；
2. 教师逐一检查 `K=4..16`，不跳过低计数；
3. 合成 `K_source` 使用 `upper_bound` 语义；
4. 加入 geometry-oracle `Ktrue`/`Ktrue+2` 教师（上限截断到 56），并设 `coverage bins=0`。

## 第 5 页：一次性筛选

候选 token 同时读取自身位置、左右间距、最近采样参数距离、其他候选、点云
几何和当前容差。KeepHead 输出相对重要度，曲线级阈值头输出自适应 `beta`。

```text
importance + beta
  -> keep probabilities
  -> probability mass + uncertainty reserve
  -> one global Top-K
  -> KeepMask
```

安全储备从 `(sigma,K)=(0.20,2)` 退火到 `(0.03,0)`。部署无 CountHead、
固定 0.5 阈值、BIC、Hard-Concrete 和逐次删除。

## 第 6 页：删除与重定位联动

```text
KeepMask
  -> 只有存活 token 构成 Key/Value
  -> t0 更新为 t1
  -> U0 从 t0 warp 到 t1
  -> 根据存活邻居、rank、count 联合重定位
  -> 有序正间隔投影
```

所以最终节点不是冗余候选的原位子集；它们能够移动到被删候选留下的区域。

## 第 7 页：训练教师与数据

| 项目 | 当前设置 |
|---|---:|
| Synthetic | 三次开放样条，控制顶点 8～60，源内部 K=4～56 |
| Synthetic min span | 0.01；旧 0.02 无法生成 K=56 的 57 个 span |
| Real | UJI、Natural Earth、USGS 独立 split |
| train / synthetic val | 3000 / 600 |
| real val | 每个来源最多 100 |
| real fraction | 0.35 |
| batch | 64 |
| epochs | 80，其中 proposal 16 |

合成曲线先通过固定参数、固定源节点族内的最简性证书，再加噪声。由于网络
允许参数和节点连续重定位，source K 只作为计数上界；geometry oracle 只用于
合成训练，不进入真实数据或部署。

## 第 8 页：评价口径

四项主指标：

1. MSE（同时报告 mean/P95）；
2. `MSE<=1e-4` 的通过率；
3. 最终内部节点数；
4. 完整方法时间。

另外报告每个数据源及 worst-source、真实数据 original-reference MSE，以及
Ours 的 network-only time。MSE 是平均平方欧氏距离，不开方。纯网络时间只
表示网络延迟，不能代替端到端时间。

Synthetic 还要把 `source K=56` 容量边界单独列出 dense/deployment pass。该层
没有候选冗余余量，失败必须原样计入，不能为展示而放宽 90% 资格。

## 第 9 页：六方法公平对比

当前主表与 3×2 真实曲线图均固定为：

1. Ours v16；
2. Park–Lee adaptation；
3. Liang feature-integral + IKI adaptation；
4. Dung serial/simple-knot adaptation；
5. Kang sparse adaptation；
6. Luo `l_inf,1 + DE` adaptation。

所有方法处理同一输入、使用 56 个内部节点容量上限并交给同一 CPU `float64`
标准 refit。公开方法均为工程适配，不声称是作者官方代码。

Kang/Luo 在简单曲线上表现好、复杂曲线上失败时，要展示阶段诊断：Kang 可能
在活跃簇压缩/重定位后失去可行性；Luo 可能在局部峰值选集后遗漏关键节点，
而固定 K 的 DE 无法补回。不能用只选简单个例的方式形成结论。

## 第 10 页：结果与演示

正式展示前先运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_mse1e-4_3090.ps1
```

脚本串行完成新训练、90% / `1e-4` 资格检查、六方法合成与真实数据比较、
2×2 四指标图和真实曲线 3×2 六方法图。checkpoint 不合格时默认停止，避免
把诊断结果误用于汇报。

结果页只从新生成的 JSON 填入：

| 数据源 | Ours MSE | pass | final K | full time | network time |
|---|---:|---:|---:|---:|---:|
| Synthetic | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| UJI | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| Natural Earth | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| USGS | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

## 展示时不要说

- 不要把完整节点向量 64 项说成 64 个内部候选；当前是 56 个内部候选；
- 不要把 90% 数据集通过率说成误差阈值，单曲线阈值是 `MSE=1e-4`；
- 不要把 source K 说成允许连续重定位后的全局最少节点证明；
- 不要把 proposal 或 `target_met=False` checkpoint 当正式模型；
- 不要删除失败样本、人工调整图中误差或只展示容易样本；
- 不要把公开方法 adaptation 称为作者官方复现；
- 不要用 network-only time 与传统方法完整时间直接计算端到端加速比。

## 历史消融

旧 `Kc=96 / MSE=2.5e-5 / 97%`、旧 K64、旧 `K=4..20` 及 `K=4..24`、固定 0.5 KeepMask 和八
方法图片只可作为历史消融。它们不是当前主协议，不得与本轮
`Kc=56 / source K=4..56 / MSE=1e-4 / 90%` 结果混写。
