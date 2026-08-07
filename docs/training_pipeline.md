# A 方案训练与部署流水线

## 1. 训练前配置检查

训练入口为 `scripts/train.py`。当前配置来源是命令行和 checkpoint 元数据，`configs/*.yaml` 尚未接入。

必须先满足：

\[
\mathrm{max\_control\_points}-4
\le
\mathrm{max\_knots}.
\]

因为含 \(n\) 个控制点的开放三次 B 样条有 \(n-4\) 个内部节点。若候选上限更小，模型不可能覆盖数据集中最复杂的真实结构。

推荐起始命令：

~~~bash
python scripts/train.py --epochs 60 --max-knots 8 --max-control-points 10 --noise-std 0.001 --lambda-l0 2e-5 --output outputs/a_scheme.pt
~~~

上例不会给出容量警告，因为最多 10 个控制点只需最多 6 个内部节点，8 个候选足够。

## 2. 阶段一：全开几何预热

默认前 \(W=5\) 轮：

- `activity_gate=1`，所有候选截断幂列进入训练代理；
- `l0_scale=0`，不施加节点稀疏压力；
- 温度保持 \(\beta_{\mathrm{start}}=2.0\)；
- 活动 logit 初始偏置为 -2.0，避免 \(P(z>0)\) 从训练开始就饱和到 1；
- 优先学习共享几何特征、参数 \(t\) 和候选节点位置 \(U\)。

注意：此阶段

\[
\pi_j=P(z_j>0)
\]

仍由活动头产生，并不一定等于 1；只有实际进入设计矩阵的门被强制为 1。因此：

- \(E[K]=\sum_j\pi_j\) 是活动头当前预测；
- `gate_nonzero_count=K` 是真实 forward 结构；
- 两者不同是正常现象。

## 3. 阶段二：释放门控并渐增 \(L_0\)

从第 \(W+1\) 轮开始关闭强制全开。训练态采样随机 Hard-Concrete 门。

默认用 \(R=15\) 轮把稀疏比例从 0 增至 1：

\[
\rho_0(e)
=
\min\left(
1,
\frac{e-W+1}{R}
\right),
\qquad e\ge W,
\]

其中代码内部 epoch 从 0 开始。

总目标中的稀疏项为

\[
\rho_0(e)\lambda_0
\frac1B\sum_b\sum_j\pi_{bj}.
\]

默认 \(\lambda_0=2\times10^{-5}\)。它是“每个预期节点”的代价；因为没有除以 \(K\)，改变候选节点上限时通常不需要按 \(K\) 重新缩放其语义。

## 4. 温度退火

从预热结束开始：

\[
\beta(e)
=
\beta_{\mathrm{start}}
+
\min\left(
1,
\frac{e-W+1}{A}
\right)
(\beta_{\mathrm{end}}-\beta_{\mathrm{start}}).
\]

默认：

\[
\beta_{\mathrm{start}}=2.0,\qquad
\beta_{\mathrm{end}}=0.5,\qquad
A=40.
\]

默认 60 轮训练能够完成退火。若运行 30 轮，只执行 25/40 的退火，末温约为 1.0625。

温度必须始终为正。过早使用很低温度可能让门迅速饱和；过高温度长期不降则会让训练态门保持过软。

## 5. 阶段三：结构稳定

第 21 轮起默认已经满足：

- `l0_scale=1`；
- 活动头继续采样随机门参与训练；
- 温度继续退火，达到 0.5 后固定；
- 验证态使用严格二值门

\[
m_j=\mathbf1[\pi_j\ge\tau].
\]

checkpoint 只从完整 \(L_0\) ramp 结束后开始参与最佳模型选择。默认索引 19，即人类计数的第 20 轮开始；选择指标仍是验证集组合目标。

## 6. 每个训练 forward

1. 编码输入采样点，得到局部和全局特征；
2. 预测严格单调的 \(t\)；
3. 用 `true_params` 监督 \(t\)；
4. 将预测 \(t\) 的正弦位置编码加入局部特征；
5. 区间 queries 对局部序列做 cross-attention，预测有序候选节点 \(U\)；
6. 在每个 \(u_j\) 周围汇聚局部几何特征；
7. ActivityHead 直接得到 \(\log\alpha\)，Hard-Concrete 计算 \(\pi\) 并采样 \(z\)；
8. 构造 \([V\mid H\operatorname{diag}(z)]\)；
9. 实时求解截断幂系数 \(D^*\)；
10. 重建代理曲线；
11. 计算 fit、expected-\(L_0\)、gap、true-parameter；
12. 梯度穿过主线性求解器、门控、cross-attention 和三个预测分支。

当前训练 forward 不包含 pilot 或 drop-cost 求解。

## 7. 默认损失配置

| 项 | 默认权重 |
|---|---:|
| fit | \(1\) |
| expected \(L_0\) | \(2\times10^{-5}\) |
| legacy activity | \(0\) |
| legacy binary | \(0\) |
| gap | \(10^{-2}\) |
| chord parameter prior | \(0\) |
| true parameter | \(10^{-2}\) |

当前 objective version 为 `cross_attention_true_params_hard_concrete_v1`。投影正交项不属于训练目标，模型也不构造一阶导数。历史 checkpoint 若记录了非零 `orthogonal` 权重，加载器仍会恢复原目标用于复现；因此历史 `best_val` 不应与新目标下的 `best_val` 直接比较。

建议调参顺序：

1. 先确保全开预热的 fit 能下降；
2. 再检查释放门后 fit 是否稳定；
3. 根据验证集的“标准 B 样条部署损失—节点数”折中调整 \(\lambda_0\)；
4. 最后微调温度和阈值。

不要通过逐样本改变阈值来掩盖活动分支未分化的问题。

## 8. 训练日志语义

典型日志包括：

~~~text
train / val / fit / E[K] / active@0.50 / gate_nonzero / l0_scale / temperature
~~~

| 字段 | 解释 |
|---|---|
| `train`, `val` | 当前加权总目标 |
| `fit` | 当前截断幂代理的平均每点平方欧氏距离 |
| `E[K]` | 批平均 \(\sum_j\pi_j\) |
| `active@0.50` | \(\pi_j\ge0.5\) 的节点数 |
| `gate_nonzero` | 当前实际门非零数；显示验证指标时等于硬部署节点数 |
| `l0_scale` | 当前稀疏调度比例 |
| `temperature` | 当前 Hard-Concrete 温度 |

有验证集时日志显示的节点指标来自 `model.eval()`，因此门是确定性二值值。

## 9. checkpoint 内容

新的 checkpoint 保存：

- `model_state_dict`，含最佳 epoch 的 Hard-Concrete 温度 buffer；
- `epoch`、`best_val`、`metrics` 和 `history`；
- `model_config`；
- `dataset_config`；
- `loss_config`；
- `objective_version`，用于区分当前无投影正交项的目标与历史目标；
- `training_config`；
- `activity_threshold` 和 `gate_temperature`。

这些字段用于精确重建门控语义。请勿只保存裸 state dict。

## 10. 验证与阈值冻结

运行：

~~~bash
python scripts/evaluate_checkpoint.py --checkpoint outputs/a_scheme.pt --activity-threshold 0.5 --json-output outputs/evaluation_report.json
~~~

`--activity-threshold` 会同步设置模型评估态硬门和标准 B 样条部署结构。正确流程是：

1. 在验证集比较少量预先定义的阈值；
2. 选择满足精度—复杂度目标的唯一阈值；
3. 将该阈值冻结；
4. 测试集和实际推演均使用同一阈值。

报告中的 threshold sweep 只用于诊断概率分布；主 `standard_bspline_refit_loss` 始终对应命令指定的实际部署阈值。

评估器还输出 `objective version` 和各项 `weighted objective components`。对含历史正交项的 checkpoint，它会同时给出 `current no-orthogonal objective (same saved weights)`；该数值只是对同一组已保存权重扣除旧项，不能替代按新目标重新训练。

## 11. 标准 B 样条部署

高层接口：

~~~python
from spline_fitting.evaluation import refit_model_output_as_bsplines

model.eval()
with torch.no_grad():
    output = model(points)
    fits = refit_model_output_as_bsplines(
        output,
        points,
        degree=3,
        smoothness_weight=1e-6,
        control_ridge=0.0,
    )
~~~

该函数：

1. 校验 `activity_gate` 是否严格为 0/1；
2. 对每条曲线物理删除未保留节点；
3. 构造开放夹持节点向量；
4. 构造标准 B 样条基；
5. 通过增广最小二乘重求控制点；
6. 返回变长的 `list[HardGatedBSplineFit]`。

使用列表是必要的，因为同一 batch 中每条曲线可保留不同节点数，对应不同长度的节点向量和控制点矩阵。

## 12. 部署输出

每条 `HardGatedBSplineFit` 包含：

- 候选数与保留数；
- 二值 mask；
- 保留内部节点；
- 完整开放夹持节点向量；
- 标准 B 样条控制点；
- 采样点处的重建结果；
- fit MSE、欧氏 RMSE、逐坐标 RMSE；
- 数据平方和、二阶差分正则、控制点 ridge 和增广目标；
- 求解秩。

当保留节点为 0 时，返回 4 控制点开放三次 Bézier，而不是错误或 NaN。

## 13. 可视化

~~~bash
python scripts/visualize_result.py --checkpoint outputs/a_scheme.pt --output outputs/result.png --smoothness-weight 1e-6
~~~

左图同时显示：

- 输入采样点；
- 有元数据时的真实 B 样条；
- 评估态硬门截断幂代理曲线；
- 硬删节点后的标准 B 样条重拟合曲线。

右图显示：

- 每个候选节点的 \(\pi=P(z>0)\)；
- 固定阈值；
- 实际 0/1 部署门；
- 候选节点位置及保留颜色。

## 14. 旧 checkpoint

`build_model_from_checkpoint` 会将缺少 `gate_mode` 的模型迁移为：

~~~text
gate_mode = legacy_soft
activity_use_local_context = false
gap_parameterization = legacy
l0 weight = 0
~~~

标准部署 API 不接受旧版 \(\sqrt{a+\epsilon}\) 连续门；评估与可视化脚本会显式用旧 activity 概率构造二值 mask 后再部署。

旧 checkpoint 可用于旧结果诊断，但 A 方案结果必须重训。

若 checkpoint 缺少 `knot_use_local_cross_attention`，迁移器会恢复历史全局 MLP KnotHead。历史 pilot 开关按原配置保留；新训练固定为 `false`。

## 15. 常见异常信号

### 所有门全开或全关

检查：

- \(E[K]\) 与保留数直方图；
- 每条曲线的 \(\pi\) 范围；
- KnotHead attention 是否覆盖不同参数位置；
- \(\lambda_0\) 是否过大或过小；
- 预热是否足够；
- 温度是否过早降得很低。

### 概率仍几乎完全相同

确认新 checkpoint 的 `activity_use_local_context=true`、`knot_use_local_cross_attention=true`、`activity_use_pilot_importance=false`，并检查候选节点是否仍异常聚集。可视化逐节点表只报告节点、最终概率和实际贡献。

### 代理拟合好、标准 B 样条重拟合差

检查：

- 部署是否使用相同的 \(t\) 和保留节点；
- `smoothness_weight` 是否过大；
- 节点是否极近导致秩下降；
- 截断幂训练正则与 B 样条控制点正则的基依赖差异。

### 节点位置匹配差但几何拟合好

节点坐标依赖参数化。只有预测 \(t\) 与真实 \(t\) 可比时，节点位置 MAE/F1 才能作为直接结构指标。
