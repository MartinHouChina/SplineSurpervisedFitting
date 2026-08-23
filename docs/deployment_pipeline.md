# v7 部署、硬剔除与指标

## 1. 部署输入

用户只提供沿曲线方向有序的二维或三维点云。无需初始节点、控制点、Boehm 插入结果或真实
节点数。

```text
ordered points [M,D]
```

点集按训练规则归一化。网络输入会重采样到 checkpoint 记录的长度；最终硬剔除和控制点
重拟合使用全部原始点。

## 2. 网络阶段

```text
points
  → GeometryEncoder
  → ParameterHead: params
  → CandidateKnotHead: Kc 个候选
  → 截断幂贡献特征
  → InteractivePruningHead: 位置精修、删除建议和 keep 诊断
```

部署从网络读取 `params` 和全部 `refined_candidate_knots`。不按 keep 的 0.5 阈值先删，也不
读取 CountHead。

## 3. 标准 B 样条硬剔除

给定当前节点集合 `U`：

1. 分别构造 `U\{u_j}`；
2. 对每个方案重新建立标准开放三次 B 样条基；
3. 固定 `P0=Q0`、`Pn=Qlast`，重新求解其余全部控制点；
4. 计算平均欧氏 RMS；
5. 选择本轮 RMS 最低的单节点删除；
6. 只有该 RMS `≤ ε` 才接受，否则停止；
7. 重复直到停止或达到 `min_internal_knots`。

同一轮的所有单节点删除用批量 Cox–de Boor 和批量最小二乘计算；这只减少运行开销，不改变
“检查全部剩余节点并选真实 RMS 最低者”的规则。

因此每个已接受步骤都有实际几何误差证据。输出包含：

```text
initial/final knots
initial/final standard B-spline fit
每轮被删除的节点和值
每轮所有候选删除 RMS
RMS trajectory
threshold_satisfied
```

这是贪心单节点删除。它保证最终拟合满足阈值，并沿该路径无法再安全删除一个节点；它不保证
在所有节点子集上找到组合全局最少解。

## 4. 端点覆盖

最终控制点求解严格施加：

\[
P_0=Q_0,\qquad P_{n-1}=Q_{M-1}.
\]

因此开放 B 样条覆盖输入首尾点。平滑项只约束未知内部控制点，并正确把固定端点项移到右端；
ridge 也只作用于未知量。

## 5. Checkpoint 评估

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_v7.pt `
  --num-samples 2000 `
  --seed 20000 `
  --batch-size 16 `
  --fit-tolerance 0.005 `
  --json-output outputs/candidate_pruning_v7_evaluation.json
```

未显式给出 `--fit-tolerance` 时，脚本优先读取
`checkpoint.deployment_config.error_tolerance`，再读取数据集 canonical tolerance。

重点指标分三组：

### 候选能力

- true→candidate recall@0.005/0.01/0.02；
- 最近候选 MAE；
- 全候选标准 B 样条 RMS 与阈值满足率。

候选阶段已经不满足阈值时，问题在参数头、候选漏点或候选位置，不应归因于剔除规则。

### 最终硬部署

- 最终阈值满足率；
- RMS mean/P95/max；
- 最终节点数 mean/min/max 与直方图；
- 删除数量和 RMS 轨迹；
- 端点 RMS/max；
- 控制点数量。

### 真值节点诊断

- match precision/recall/F1；
- matched MAE；
- 参数 RMSE。

应同时报告多个匹配容差。宽容差加上大量近均匀候选会自然提高 precision，不能单看
`match@0.02 precision` 判断节点向量恢复是否准确。

learned keep 概率、remove/STOP 和删除代价只作为网络诊断，与最终硬保留集合分开报告。

## 6. 可视化

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_v7.pt `
  --seed 20000 `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view comparison `
  --dpi 600 `
  --output outputs/candidate_pruning_v7_sample_000.png
```

`--pruning-view` 的四种取值如下：

- `all`：保留全部候选节点并重新拟合标准 B 样条；
- `learned`：按 `keep_probability >= activity_threshold` 选节点后重新拟合；
- `hard`：从全部候选开始执行 RMS 硬剔除，是默认部署视图；
- `comparison`：一张图并列显示上述三种标准 B 样条结果及节点决策。

`learned` 是网络剔除能力的诊断，不具备 RMS 硬保证；`hard` 不会先套用 learned mask。
`--dpi` 控制 PNG 分辨率，论文图片建议使用 600。

## 7. 用户点云

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_v7.pt `
  --point-cloud data/my_curve.csv `
  --fit-tolerance 0.005 `
  --json-output outputs/my_curve_fit.json `
  --figure-output outputs/my_curve_fit.png
```

支持 `.csv`、`.txt`、`.xyz`、`.json`、`.npy`、`.pt` 和 `.pth`。CSV 每行一个点：二维为
`x,y`，三维为 `x,y,z`。点序相反时使用 `--reverse-points`。

处理顺序：

```text
加载并检查维度
  → 可选反转点序
  → 归一化
  → 仅为网络重采样
  → 推理 params 和候选
  → 按弦长把 params 插值回所有原始点
  → 在所有原始点上逐节点硬剔除
  → 在所有原始点上最终重拟合
  → 控制点恢复到原坐标系
```

稀疏点经线性重采样不会增加信息。若原始点显著少于训练点数，或最终控制点数接近/超过原始
观测数，低训练输入 RMS 不能证明真实曲线泛化正确，脚本会明确警告。
