# v14_joint：参数、节点筛选与重定位联合反馈

`v14_joint` 保留一次性部署，不做迭代搜索。它修复了旧 v14 的单向数据流：旧版在
KeepMask 和节点重定位完成后才修正参数 `t0 -> t1`，因此修正后的参数不能再影响节点
选择。新版本在同一个网络前向中关闭这个环路。

## 数据流

```text
有序点云 Q
  -> GeometryEncoder: 局部特征 L、全局特征 G
  -> ParameterHead: 初始严格递增参数 t0
  -> CandidateKnotHead: Kc 个有序候选及 candidate tokens
  -> all-candidate pilot: 残差、解析删除代价、系数能量
  -> InteractivePruningHead: 初始 KeepMask + 初始 survivor relocation
  -> ParameterFeedbackHead: 使用结构、残差和局部特征修正为 t1
  -> 将候选与初始重定位结果从 t0 单调 transport 到 t1
  -> JointParameterStructureHead:
       candidate token 对 (L + t1 位置编码) 做局部 cross-attention
       联合输出 Keep-logit、候选位置和 token 的残差更新
  -> 根据更新后的 Keep 概率重新离散选择一次
  -> 仅以最终 KeepMask 的存活节点作为 Key/Value 再重定位一次
  -> 最终 t1、节点向量和 KeepMask
  -> 一次标准三次 B 样条 control-point refit
```

第二次选择之后不会复用第一次的 survivor mask。最终 relocation 的注意力权重在所有
被删除槽位上均为零，因此输出节点位置和最终 KeepMask 不会错位。

## 训练监督

默认离线教师使用 `canonical_boehm` 启动：在合成数据的 `true_params` 参数域中，从
canonical 真节点开始，按“优先切分最左侧最大区间”的确定性 Boehm 插入补到 `Kc` 个
候选，再运行 delete-then-relax Hard-RMS。教师 mask 与网络的有序候选按 rank 对齐，教师
最终节点位置则从真值参数域映射到固定 proposal 的 `t0` 域后缓存。它提供：

CandidateKnotHead 的完整冗余向量标签使用完全相同的域顺序：先在 `true_params` 域完成
Boehm 插入，再把整个 `Kc` 向量映射到 `t0`。不能先映射稀疏真节点再在 `t0` 中插中点，
因为非线性参数映射与中点插入不可交换，会破坏 ordered-rank 槽位的几何对应关系。

- `teacher_retained_mask`：监督最终 Keep logits；
- `teacher_count`：监督最终 Keep 概率质量对应的节点数；
- `teacher_internal_knots`：先随 `t0 -> t1` 映射到修正参数域，再与最终存活节点做有序集合匹配；
- `true_params`：监督 `t1`；
- 标准化拟合误差与阈值违反量：防止只追求节点少而失去可行性。

这三项结构监督与拟合监督同时存在，避免 straight-through 拟合梯度重新落入
“全部保留”的捷径。

旧 `learned_proposal` 启动先要求网络 proposal 在其自身参数域中满足 RMS 阈值，再产生
删除标签；当 proposal 尚未学好时，教师生成反过来无法开始，形成前置可行性的死循环。
此前的 `true/chord fallback` 虽能绕过报错，但 mask 已不再严格对应部署参数域，因此只适合
诊断。`canonical_boehm` 直接从确定、可审计的监督候选开始，不要求 proposal 预先达到
100% 可行率，也不会把 fallback 标签混入正式缓存。

真值只参与离线合成训练标签。实际部署仍只输入有序点云，执行一次网络前向和一次标准
B 样条 refit，不读取 `true_params`、真节点或教师 cache。

## 兼容性

- 旧目标 `candidate_pruning_one_shot_parameter_feedback_v14` 仍恢复
  `fast_global`，并令 `joint_parameter_structure_feedback=false`；其语义不变。
- 新目标名为
  `candidate_pruning_joint_parameter_structure_feedback_v14`。
- 新 head 的 Keep、位置、token 末层及最终 relocation scale 都从零开始。刚挂载到旧
  checkpoint 时，最终 mask 和 transported 节点与旧 v14 完全相同。
- 从旧 proposal 加载时，只允许缺失 `parameter_feedback_head.*` 和
  `joint_parameter_structure_head.*`；其他缺失或多余权重仍报错。

## 推荐训练命令

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 180 `
  --candidate-pretrain-epochs 50 `
  --keep-position-calibration-epochs 30 `
  --parameter-feedback-epochs 30 `
  --train-size 10000 `
  --val-size 2000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 32 `
  --num-points 192 `
  --certified-minimal-source `
  --parameter-gap-reference uniform_residual `
  --parameter-residual-logit-limit 2.5 `
  --candidate-position-parameterization interval_softmax `
  --candidate-interval-logit-limit 1.75 `
  --lambda-redundant-candidate 5 `
  --lambda-fit 0 `
  --lambda-threshold-violation 0 `
  --parameter-feedback-fusion-mode cross_attention `
  --joint-parameter-structure-feedback `
  --enforce-ordered-joint-candidates `
  --joint-feedback-local-bandwidth 0.08 `
  --joint-feedback-max-keep-logit-shift 2.0 `
  --lambda-feedback-joint-keep 1.0 `
  --lambda-feedback-joint-position 2.0 `
  --lambda-feedback-joint-count 2.0 `
  --fit-tolerance 0.00316227766 `
  --deployment-pass-rate-target 0.97 `
  --teacher-start-mode canonical_boehm `
  --max-teacher-fallback-fraction 0 `
  --teacher-cache-dir outputs/teachers/v14_canonical_boehm `
  --no-reuse-teacher-cache `
  --no-resample-train-each-epoch `
  --output outputs/checkpoints/current/candidate_pruning_v14_canonical_boehm.pt
```

这是一次全新训练，不能再用已经发生候选坍缩的
`candidate_pruning_v14_joint.pt` 作为 proposal。proposal 的标准 B 样条可行率仍用于预训练
checkpoint 排序和诊断，但不会阻止 `canonical_boehm` 教师生成。只有 canonical+Boehm
全候选本身无法达到阈值时才会终止；这代表数据标签、噪声和阈值之间不一致，而不是要求
尚未蒸馏的网络先学会完整部署。

这里训练脚本的 `--fit-tolerance` 使用 RMS，所以
`0.00316227766 = sqrt(1e-5)`；真实数据认证脚本的
`--certified-mse-target 1e-5` 则直接使用 MSE。二者不能混用。

若 proposal checkpoint 与当前数据、候选数、最小间距或稳定 pilot 语义不一致，应使用
新的 teacher cache，并重新训练 proposal。`canonical_boehm` 与旧
`learned_proposal` cache 的监督语义不同，必须使用不同目录；不要把原
`outputs/teachers/v14_feasible` 复制或改名后复用。只有模型、数据集、启动模式和教师配置
的指纹完全一致时才添加 `--reuse-teacher-cache`。

当前 teacher cache 必须带有 `train.pt.start_domains.json` 和
`val.pt.start_domains.json`。v2 sidecar 明确记录 `teacher_start_mode`、生成策略、
`ordered_rank` 槽位对齐方式和 `fixed_proposal_t0` 位置存储域。canonical 模式的来源计数
只能是 `canonical_boehm`，fallback 比例恒为零；`--max-teacher-fallback-fraction` 仅对旧
`learned_proposal` 模式生效。第一次迁移必须用 `--no-reuse-teacher-cache` 在上述新目录
重建；缺少或模式不匹配的认证文件会被拒绝。

sidecar 的 `ordered_rank_anchor_diagnostics` 还记录教师目标映射回 `t0` 后，相对同 rank
proposal 槽位的平均/最大位移，以及超过 `one-shot-max-position-shift` 的样本比例。后者较高
不污染 mask 标签，但说明当前 proposal 与监督候选错位过大，位置重定位头可能无法到达
教师目标，应优先继续 proposal 预训练或增大可移动范围，而不是放宽 teacher fallback。

## 从已有 v13 / v14 独立微调

已有完整 checkpoint 时，不必重建离线教师。独立入口默认挂载
`ParameterFeedbackHead(cross_attention)` 和 `JointParameterStructureHead`，联合训练参数修正、
最终 Keep/位置残差，并以较小学习率更新最终 survivor relocation：

```powershell
python scripts/finetune_parameter_feedback.py `
  --checkpoint outputs/candidate_pruning_one_shot_v13.pt `
  --output outputs/candidate_pruning_v14_joint_finetuned.pt `
  --epochs 20 `
  --learning-rate 1e-4 `
  --final-relocation-lr-scale 0.25 `
  --parameter-feedback-fusion-mode cross_attention `
  --joint-parameter-structure-feedback `
  --lambda-feedback-joint-keep 1.0 `
  --lambda-feedback-joint-position 2.0 `
  --lambda-feedback-joint-count 2.0 `
  --overwrite
```

独立入口在合成数据上用数据集自带的 canonical 真节点做有序候选匹配，由此构造明确的
Keep、节点集合与计数监督；它不声称这些标签来自重新运行的 Hard-RMS 教师。

若输入是旧 `fast_global` v14，切换到默认 `cross_attention` 时只重新零初始化两个新增反馈
head；原 proposal、初始 selector 和已有 relocation 权重原样载入。若要继续使用旧 v14 的
单向、固定 KeepMask 语义，添加 `--no-joint-parameter-structure-feedback`；未显式指定融合
模式时该兼容路径仍使用 `fast_global`。

真实 UJI、Natural Earth 或 USGS manifest 没有节点标签。此时脚本不会构造伪
`teacher_retained_mask`，也不会把缺失标签当全零 mask：`joint_keep`、`joint_position`、
`joint_count` 和 `true_parameter` 的有效监督权重均明确为 `0`，只由拟合误差、阈值违反量
及参数正则调整参数与节点位置。例如：

```powershell
python scripts/finetune_parameter_feedback.py `
  --checkpoint outputs/candidate_pruning_one_shot_v13.pt `
  --real-world-manifest data/processed/natural_earth/manifest.jsonl `
  --real-world-manifest data/processed/usgs/manifest.jsonl `
  --output outputs/candidate_pruning_v14_joint_real_world.pt `
  --epochs 10 `
  --overwrite
```

最终 checkpoint 使用目标名
`candidate_pruning_joint_parameter_structure_feedback_v14`，并记录真实数据上的 Keep、位置、
计数监督权重为零，便于审计。

## 速度口径

`v14_joint` 仍是一次网络前向和一次最终标准 B 样条 refit。相较旧 v14，它增加一次
候选到点特征的局部 cross-attention 和一次固定深度 survivor attention，但不增加内部
spline solve。若做速度消融，可把参数反馈改为 `gaussian_pool`、`fast_structural` 或
`fast_global`；正式效果实验默认 `cross_attention`。
