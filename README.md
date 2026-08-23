# Minimum-Complexity B-Spline Fitting

本项目从有序二维或三维点云拟合开放三次 B 样条。当前主版本是
`candidate_pruning_minimal_rms_v7`：目标不再是猜测生成器原本用了多少节点，而是

\[
\min |U|\quad\text{s.t.}\quad
\operatorname{RMS}(C_U,Q)\le\varepsilon .
\]

其中 RMS 是归一化坐标中的平均欧氏距离。v6 及更早 checkpoint 仍可严格加载。

## 当前工作流

```text
有序点云 Q [B,M,D]
  → GeometryEncoder
      local_features [B,M,H], global_features [B,H]
  → ParameterHead
      0=t0<t1<...<t(M-1)=1
  → CandidateKnotHead
      固定 Kc 个严格有序、高召回候选节点
  → 全候选截断幂代理拟合
      节点系数能量、精确列删除目标增量、局部残差、左右间距
  → InteractivePruningHead
      候选 self-attention、remove/STOP、keep 诊断、位置精修
  → 标准 B 样条逐节点硬剔除
      每次删除后完整重拟合控制点
      仅当真实 RMS≤ε 时接受
  → 最终节点向量、控制点和覆盖首尾端点的拟合曲线
```

最终节点数来自硬剔除轨迹，不使用 `CountHead`、BIC、Hard-Concrete 或一次性的
`keep_probability >= 0.5`。学习到的 keep 概率只用于诊断；即使分类头判断错误，也不能绕过
部署阶段的 RMS 检查。

截断幂基只用于网络内部提取“每个候选节点的独立贡献”并提供可微拟合代理；最终导出的曲线
始终由标准开放 B 样条基重拟合。

## 数据集

默认 v7 配置如下：

| 项目 | 默认值 |
|---|---:|
| 训练 / 验证样本 | 10000 / 2000 |
| 训练 / 验证 / 独立测试 seed | 42 / 10000 / 20000 |
| 每条曲线采样点 | 192 |
| 源控制点 | 8–24 |
| 源内部节点 | 4–20 |
| 候选预算 `Kc` | 28 |
| 坐标噪声标准差 | 0.001 |
| 默认 RMS 阈值 `ε` | 0.005 |

每条曲线先中心化并按最大半径归一化。监督节点不是随机器任意生成的源表示，而是从源节点
开始逐个尝试删除、重新拟合控制点，并在同一 RMS 阈值下得到的 canonical 表示。该过程是
确定性的贪心消冗；它保证返回结果满足阈值并沿当前删除路径不可再删，但不宣称组合意义上的
全局最少。

## 训练

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --candidate-match-tolerance 0.02 `
  --output outputs/candidate_pruning_v7.pt
```

训练分两段：先优化参数、候选覆盖、位置和全候选拟合，再联合训练候选消冗与位置精修。
高节点数 canonical 标签生成较慢，因此训练集默认固定并缓存；只有明确接受重复标签生成成本时
才启用 `--resample-train-each-epoch`。

脚本同时保存：

- `candidate_pruning_v7.pt`：按候选召回、最近节点误差、安全删除动作和节点匹配选出的最佳权重；
- `candidate_pruning_v7_last.pt`：最后一个 epoch，便于排查选优偏差。

旧的 `scripts/train.py` 保留用于复现实验 v6，不是 v7 主入口。

## 评估与可视化

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint outputs/candidate_pruning_v7.pt `
  --num-samples 2000 `
  --seed 20000 `
  --fit-tolerance 0.005 `
  --json-output outputs/candidate_pruning_v7_evaluation.json
```

```powershell
python scripts/visualize_result.py `
  --checkpoint outputs/candidate_pruning_v7.pt `
  --sample-index 0 `
  --fit-tolerance 0.005 `
  --pruning-view comparison `
  --dpi 600 `
  --output outputs/candidate_pruning_v7_sample_000.png
```

`--pruning-view` 支持 `all`（全部候选）、`learned`（keep 概率阈值）、`hard`
（RMS 硬剔除，默认）和 `comparison`（同图对照三者）。三种结果都会重新拟合为标准
B 样条；`network surrogate` 只作为训练代理诊断。

评估时至少同时检查：候选 recall@0.01/0.02、最终阈值满足率、最终节点数、标准 B 样条
RMS/P95、端点误差和最终节点匹配。单独的 precision 或低训练代理损失不足以说明结构正确。

## 用户点云推演

输入必须沿曲线方向有序，支持 CSV、TXT、XYZ、JSON、NPY、PT/PTH：

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/candidate_pruning_v7.pt `
  --point-cloud data/my_curve.csv `
  --fit-tolerance 0.005 `
  --json-output outputs/my_curve_fit.json `
  --figure-output outputs/my_curve_fit.png
```

网络读取重采样后的序列，但最终参数会插值回全部原始点，硬剔除和控制点重拟合也在全部原始
点上执行。重采样不会增加几何信息；输入点明显少于训练密度时，脚本会给出警告。

## 验证

```powershell
python -m pytest -q
```

## 文档

- [模型数据流](docs/architecture.md)
- [数据与训练流程](docs/training_pipeline.md)
- [部署、硬剔除与指标](docs/deployment_pipeline.md)
- [候选生成与消冗设计说明](docs/proposal_pruning_framework.md)
- [数学定义](docs/math_formulation.md)
- [文件索引](docs/file_guide.md)
