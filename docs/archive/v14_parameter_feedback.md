# v14 ParameterFeedbackHead

v14 在 v13 的一次性结构路径后增加一个晚期参数校正头。它修正的是点参数化，不重新搜索节点，也不替代最终标准 B 样条 refit。

## 数据流与参数域

```text
points
  -> ParameterHead: t0
  -> candidate proposal + pilot descriptors + KeepMask + survivor relocation
  -> ParameterFeedbackHead: structure-conditioned network/chord gap blend
  -> strictly increasing t1
  -> transport surviving knots through the piecewise-linear t0 -> t1 map
  -> transported knots + t1 -> surrogate / one final standard B-spline refit
```

`t0` 是 `ParameterHead` 的初始严格递增参数化。候选生成、固定 proposal 槽位、离线教师标签、KeepMask 筛选及其监督都保留在 `t0` 域；因此 v14 不改变 proposal/teacher 的身份语义。结构已确定后，默认的低延迟 `fast_global` 反馈路径汇总以下证据：

- `t0` 的 gap；
- 观测点的弦长 gap；
- all-candidate pilot 重建残差；
- 删除风险（由 pilot 的解析 deletion delta 和被移除候选的位置加权得到）；
- 最终 KeepMask 的保留比例，以及保留/删除候选的解析 deletion-risk 统计。

这些量被压缩成一个很小的逐曲线结构 token，用来修正可学习的弦长混合权重；随后网络 gap 与满足最小 gap 约束的弦长 gap 做显式凸组合得到 `t1`。因此 `t1` 仍严格有序并覆盖 `[0,1]`。该默认路径只训练 114 个参数，不执行区间级多头注意力；代码仍保留 `cross_attention`/`gaussian_pool` 实验模式，但部署默认采用实测更省训练参数、效果相同的 `fast_global`。新头的弦长权重和末层均从零开始，因此挂载旧 checkpoint 时仍是恒等映射 `t1=t0`；记录完旧模型基线后，训练候选默认以实测较好的 `0.6` 弦长权重启动并继续学习。

节点也必须与参数处在同一个域。候选生成、KeepMask 和 survivor relocation 得到的是 `t0` 域节点；v14 用采样对应关系定义的分段线性单调映射将最终存活节点同步运输到 `t1` 域，再进行 surrogate 或标准 B 样条 refit。这样改变参数化不会被误当成节点结构变化，节点的物理位置与 KeepMask 保持一致。

最终截断幂 surrogate（训练/诊断）和部署的标准 B 样条 control-point refit 都使用 `t1`；候选和教师标签仍解释为 `t0` 域的对象。输出中的 `proposal_params` 保留 `t0`，`params` 是最终 `t1`。

## 一次前向与求解边界

ParameterFeedbackHead 位于同一个固定深度的网络 forward 内：它不会触发第二次网络前向，也不会增加 spline solve。pilot 描述符已经由 selector 路径产生并复用；部署仍在网络外对最终保留、已 relocation 的节点做一次标准 B 样条 refit。`verified` 和 `hybrid` 的额外精确 refit/修复仍属于各自模式，不属于 v14 反馈头。

## 完整训练

下面命令从头训练 proposal、教师、selector/relocation，并在其后执行 10 个 v14 参数反馈校准 epoch：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 20 `
  --keep-position-calibration-epochs 20 `
  --parameter-feedback-epochs 10 `
  --parameter-feedback-lr 1e-4 `
  --parameter-feedback-max-logit-shift 0.5 `
  --parameter-feedback-initial-chord-blend 0.6 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --teacher-survivor-relaxation `
  --teacher-relaxation-rounds 2 `
  --teacher-relaxation-sweeps 2 `
  --teacher-relaxation-grid-size 7 `
  --teacher-relaxation-restarts 1 `
  --one-shot-max-position-shift 0.15 `
  --relocation-lr-scale 0.25 `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v14_teacher `
  --no-resample-train-each-epoch `
  --output outputs/candidate_pruning_one_shot_v14.pt
```

训练脚本先以真实标准 B 样条验证指标选择 v13 结构候选；只有参数反馈校准后的 checkpoint 在相同约束排序下更好时才会被选为最终输出。额外产生的 `*_parameter_feedback.pt` 是该阶段的候选 checkpoint。

## 从既有 checkpoint 微调

若已有语义兼容的 v13 proposal 或结构 checkpoint，可用 `--candidate-pretrain-epochs 0` 与 `--proposal-checkpoint` 作为初始化；缺失的 v14 头只允许按其零初始化新增，其他权重/配置不兼容会被拒绝。v12 默认使用旧 pilot 数值语义，不能在零 proposal 适配模式下伪装成 v13；已有 v12 建议直接使用独立反馈微调脚本，它不重建教师且不会改动原网络结构：

```powershell
python scripts/finetune_parameter_feedback.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --output outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --no-joint-parameter-structure-feedback `
  --epochs 10 `
  --train-size 4000 `
  --val-size 1000 `
  --batch-size 16 `
  --initial-chord-blend 0.6 `
  --device auto `
  --overwrite
```

上述 `--no-joint-parameter-structure-feedback` 明确保留本页所述的旧 v14 单向
`fast_global` 语义。独立脚本的新默认值是 v14_joint：使用 full cross-attention，并在参数
修正后再次联合更新 KeepMask 和节点位置，详见
[v14_joint](v14_joint_parameter_structure_feedback.md)。该脚本先用真实标准 B 样条 refit
记录 `chord=0` 的恒等基线，再训练新增反馈头；验证候选若降低通过率或综合尾部误差，则
自动保存恒等反馈版本。若要走完整 v13→v14 训练流程，例如：

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 150 `
  --candidate-pretrain-epochs 0 `
  --keep-position-calibration-epochs 20 `
  --parameter-feedback-epochs 10 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v13.pt `
  --teacher-cache-dir outputs/candidate_pruning_one_shot_v14_teacher `
  --output outputs/candidate_pruning_one_shot_v14.pt
```

其余数据、节点、教师和优化参数必须与完整命令一致。尤其不能在零 proposal 适配时改变候选数、hidden dimension、最小 gap、候选局部 attention 带宽或固定 proposal 语义。教师缓存只在脚本的严格 proposal/data/shape/teacher 指纹校验通过后才可用 `--reuse-teacher-cache`；从 v13 切到 v14 或任一指纹不一致时，应使用新目录重建缓存。

## 性能说明

3090 会显著加速训练和批量网络前向，尤其是 encoder/attention 的大 batch 吞吐。它不保证按同一比例加速每条曲线的小型 float64 最小二乘、pilot 描述符或最终标准 B 样条 refit；这些小矩阵求解常受精度、同步和启动开销限制。报告速度时应区分网络 forward、pilot/solver 与任何 verified/hybrid 修复时间。

当前 GTX 1070 小规模回归（固定 v12、64 条验证曲线、只微调 1 epoch）中，真实标准 B 样条通过率由 `0.188` 提升到 `0.594`，平均 MSE 由 `5.281e-5` 降到 `2.881e-5`，P95 MSE 由 `1.248e-4` 降到 `6.957e-5`；KeepMask 与节点数不变。相同 16 条曲线的共享参数域节点匹配指标与 v12 相同，而平均部署 MSE 约下降 `50.7%`。这只是实现回归，不替代完整独立测试集实验。

同机 200 次稳态计时中，`forward_deployment`（跳过训练 surrogate）为 B=1 P50 `18.97 ms`，B=16 批耗时 `19.40 ms`、摊销 `1.21 ms/curve`；完整训练 forward 分别为 `20.00 ms` 和 `21.51 ms`。因此批量吞吐已低于 10 ms/curve，但 GTX 1070 的单条请求仍未达标。3090 对训练和批量前向会快很多；单条是否低于 10 ms 必须在目标机实测。3090 还支持当前 GTX 1070（compute capability 6.1）无法使用的 Triton/`torch.compile` 路径，但应把首次编译时间与稳态推理时间分开报告。
