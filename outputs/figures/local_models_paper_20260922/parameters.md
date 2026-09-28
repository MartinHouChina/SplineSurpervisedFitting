# 本地检查点：实验参数与来源审计

所有参数直接读取指定检查点，不使用一条龙脚本的默认值代替实际记录。模型以严格模式恢复成功；没有训练、修改或覆盖权重。

## 核心实验参数

| Parameter | M16 | M32 |
| --- | --- | --- |
| Candidate internal-knot capacity | 16 | 32 |
| Full clamped knot-vector capacity | 24 | 40 |
| Spline degree | 3 | 3 |
| Ordered input samples x dimensions | 192 x 2 | 192 x 2 |
| Historical spline source internal K | 4--16 | 4--24 |
| Hidden width / embedding dimension | 128 | 128 |
| Encoder layers | 3 | 3 |
| Attention heads | 4 | 4 |
| Base selector blocks | 2 | 2 |
| Proposal / selection / survivor extra blocks | 2 / 2 / 2 | 2 / 2 / 2 |
| Coupled proposal / subset updates | 2 / 2 | 2 / 2 |
| Selection policy | mass_topk | mass_topk |
| Adaptive logit threshold | Yes | Yes |
| Minimum selected internal knots | 4 | 4 |
| Subset geometry | anchored | anchored |
| Final explicit chord-blend coefficient | 0 | 0 |
| Synthetic training / validation samples | 3000 / 500 | 3000 / 500 |
| Real validation samples per listed source | 100 | 100 |
| Real training fraction | 0 | 0 |
| Batch size | 32 | 32 |
| Per-epoch training resampling | Yes | Yes |
| Configured Proposal + Joint epochs | 12 + 48 | 12 + 48 |
| Stored weight epoch / phase | 13 / Joint calibration | 13 / Joint calibration |
| Initial Joint proposal-freeze epochs | 8 | 8 |
| Proposal / Joint base learning rate | 2e-05 / 1e-05 | 2e-05 / 1e-05 |
| Joint proposal / decoder LR scales | 0.1 / 0.25 | 0.1 / 0.25 |
| Weight decay | 0.0001 | 0.0001 |
| Synthetic noise standard deviation | 0.001 | 0.001 |
| Normalized MSE threshold | 5e-05 | 5e-05 |
| Peak squared-point-error training threshold | 0.0005 | 0.0005 |
| Learned model parameters | 3,727,790 | 3,729,838 |
| Saved qualification satisfied | No | No |

## 必须与论文结果一起说明的事项

- 两个文件均保存第 13 轮（12 轮 Proposal 后第 1 轮 Joint calibration）的权重。12 + 48 是配置计划，不是这些权重已经历的轮数。服务器可能继续训练，但仅凭最佳检查点不能判断整次训练是否结束。
- 两者均为 warm start；M16 / M32 的前置模型和源样条复杂度范围不同，不是严格只改变候选容量的消融。
- 当前训练数据中真实数据比例为 0；真实验证名单只有 UJI、Natural Earth、USGS。工业等距线是额外的程序生成测试来源，不能称为实测工业数据。
- 表中的 source K 范围只描述历史样条生成分量。训练混合含 35% simple、25% shape、40% historical，不能将整个混合描述为只有该 K 范围的随机 B 样条。
- 两个保存检查点的 formal_reporting_eligible 均为 False。此前 validation 数字与本次重新测试数字必须分开；成功精选图不能代替全样本统计。
- 误差均在归一化坐标下。MSE 为平均平方欧氏距离；峰值平方误差为样本点中最大平方欧氏距离，不是连续曲线全域 Hausdorff 最大误差。
- min_selected_knots=4 是部署配置；完整端点夹持结点向量容量为 Kc+8（三次样条），控制顶点数量上限为 Kc+4。
- parameter_chord_blend=0 不等于未使用弦长信息：模型仍包含弦长参考、可信度混合和反事实训练。该字段只是最后显式弦长混合的系数。
- CUDA 已记录，但具体 GPU 型号、服务器最终训练轮数和完整训练耗时未记录在该检查点中，不能凭文件名推定。

## 损失权重（检查点原值）

| Parameter | M16 | M32 |
| --- | --- | --- |
| fit_weight | 1 | 1 |
| policy_weight | 0.25 | 0.25 |
| distillation_weight | 2 | 2 |
| dense_weight | 0.25 | 0.25 |
| count_weight | 2 | 2 |
| ranking_weight | 1 | 1 |
| false_remove_weight | 5 | 5 |
| ranking_margin | 1 | 1 |
| entropy_weight | 0 | 0 |
| complexity_weight | 0.05 | 0.05 |
| complexity_activation_ratio | 0.8 | 0.8 |
| tail_weight | 0.5 | 0.5 |
| tail_fraction | 0.2 | 0.2 |
| supervised_count_weight | 1 | 1 |
| supervised_over_count_weight | 1 | 1 |
| true_parameter_weight | 0.1 | 0.1 |
| proposal_knot_coverage_weight | 1 | 1 |
| selected_knot_position_weight | 1 | 1 |
| boundary_ranking_weight | 0.5 | 0.5 |
| parameter_counterfactual_weight | 0.25 | 0.25 |
| local_fit_weight | 0.1 | 0.1 |
| proposal_ordered_weight | 0.5 | 0.5 |
| teacher_geometry_distillation_weight | 0.4 | 0.4 |
| teacher_compact_mask_weight | 0.5 | 0.5 |
| max_point_error_weight | 0.05 | 0.05 |

## 检查点身份与已记录验证（非本次测试）

### M16

- 文件：`E:\SelfSurpervisedSplineFitting\outputs\downloads\server_models_20260922_215418\paper_coupled_clean_3090_r2_m16.pt`
- SHA256：`13300d7124e94557062860d6f2da82b617575e66a2988705fb8215d362c5c022`
- 保存轮次 / phase：13 / `joint_geometry_calibration`
- 已记录验证通过率：45.500%；平均 K：14.94875；MSE：2.845324e-04。
- 保存时优化器参数组：`[{"lr": 0.0, "group_name": "proposal", "lr_scale": 0.1, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0001, "amsgrad": false, "maximize": false, "foreach": null, "capturable": false, "differentiable": false, "fused": null, "decoupled_weight_decay": true}, {"lr": 1e-05, "group_name": "selector", "lr_scale": 1.0, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0001, "amsgrad": false, "maximize": false, "foreach": null, "capturable": false, "differentiable": false, "fused": null, "decoupled_weight_decay": true}, {"lr": 2.5e-06, "group_name": "decoder", "lr_scale": 0.25, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0001, "amsgrad": false, "maximize": false, "foreach": null, "capturable": false, "differentiable": false, "fused": null, "decoupled_weight_decay": true}]`

- This file stores epoch 13; 60 is the configured schedule, not the epoch of these weights. A best checkpoint alone cannot establish whether the server later finished training.
- The saved checkpoint did not meet its recorded reporting qualification. Any newly measured test outcomes must be reported as measured, without substituting this validation snapshot or only selecting successful cases.
- Warm-started weights; this is not a from-scratch training run.
- GPU model, complete-run wall time, and the terminal epoch reached on the server are not established by the checkpoint. device=cuda is not proof of an RTX 3090.

### M32

- 文件：`E:\SelfSurpervisedSplineFitting\outputs\downloads\server_models_20260922_215418\paper_coupled_clean_3090_r3_m32.pt`
- SHA256：`10a1c215e589baa54b2c39ebaebb4e4cfd6e8877166c7759dcee10e5d4048bbe`
- 保存轮次 / phase：13 / `joint_geometry_calibration`
- 已记录验证通过率：63.625%；平均 K：25.51；MSE：5.894943e-05。
- 保存时优化器参数组：`[{"lr": 0.0, "group_name": "proposal", "lr_scale": 0.1, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0001, "amsgrad": false, "maximize": false, "foreach": null, "capturable": false, "differentiable": false, "fused": null, "decoupled_weight_decay": true}, {"lr": 1e-05, "group_name": "selector", "lr_scale": 1.0, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0001, "amsgrad": false, "maximize": false, "foreach": null, "capturable": false, "differentiable": false, "fused": null, "decoupled_weight_decay": true}, {"lr": 2.5e-06, "group_name": "decoder", "lr_scale": 0.25, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0001, "amsgrad": false, "maximize": false, "foreach": null, "capturable": false, "differentiable": false, "fused": null, "decoupled_weight_decay": true}]`

- This file stores epoch 13; 60 is the configured schedule, not the epoch of these weights. A best checkpoint alone cannot establish whether the server later finished training.
- The saved checkpoint did not meet its recorded reporting qualification. Any newly measured test outcomes must be reported as measured, without substituting this validation snapshot or only selecting successful cases.
- Warm-started weights; this is not a from-scratch training run.
- GPU model, complete-run wall time, and the terminal epoch reached on the server are not established by the checkpoint. device=cuda is not proof of an RTX 3090.
