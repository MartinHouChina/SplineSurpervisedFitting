# v15：同时降低节点数与部署 MSE

v15 不改变 v14 的一次性网络结构，也不增加部署阶段的网络分支。它修正联合校准目标，使训练、离散选点和最终标准 B 样条重拟合采用同一口径。

## 关键问题

1. `mass_topk` 部署节点数由
   \[
   K=\left\lceil\sum_jp_j+\sigma\sqrt{\sum_jp_j(1-p_j)}\right\rceil
   \]
   决定；旧联合损失却令 `sum(p)` 逼近整数真值。正的 uncertainty reserve 会令部署结果系统性多保留节点。
2. 旧反馈阶段优化截断幂代理拟合，测试阶段统计标准三次 B 样条重拟合。两个基函数和求解器不一致，代理 MSE 不能可靠代表部署 MSE。

## v15 修改

- **部署计数对齐**：直接监督完整 requested-count score。正节点数 `K` 的连续目标取 `K-0.25`，位于 `ceil` 对应区间内部；零节点目标取 `0.25`。
- **精确部署拟合损失**：硬 KeepMask 后，以预测参数和重定位节点构造标准开区间三次 B 样条；固定首末控制顶点并可微求解其余控制顶点，直接优化 mean squared Euclidean error。
- **全节点集合监督**：有序匹配之外增加双向最近集合损失，使多余的已选节点也收到位置梯度；节点数相同时再监督含边界的区间长度。
- **关键节点保护**：离线教师的 leave-one-out deletion risk 加权关键保留节点召回，降低为了减节点而误删高风险节点的概率。

训练期会多做可微 B 样条求解，因此校准较慢；`forward_deployment()` 未调用该求解，网络推理路径和耗时不变。部署仍是一次网络前向加一次最终标准 B 样条 refit。

## 推荐训练

下面从固定 v14 proposal 重新蒸馏。教师重定位搜索更充分，优先得到更少且仍满足阈值的标签；若 proposal 或数据指纹变化，不要复用旧 cache。

```powershell
python scripts/train_candidate_pruning.py `
  --epochs 100 `
  --candidate-pretrain-epochs 0 `
  --proposal-checkpoint outputs/candidate_pruning_one_shot_v14_proposal.pt `
  --keep-position-calibration-epochs 20 `
  --parameter-feedback-epochs 30 `
  --train-size 4000 `
  --val-size 1000 `
  --batch-size 16 `
  --min-control-points 8 `
  --max-control-points 24 `
  --candidate-knots 28 `
  --num-points 192 `
  --fit-tolerance 0.005 `
  --teacher-survivor-relaxation `
  --teacher-relaxation-rounds 3 `
  --teacher-relaxation-sweeps 3 `
  --teacher-relaxation-grid-size 9 `
  --teacher-cache-dir outputs/teachers/v15_deployment_aligned `
  --no-resample-train-each-epoch `
  --output outputs/checkpoints/candidate_pruning_one_shot_v15.pt
```

默认校准权重：exact deployment fit `0.25`、exact threshold violation `2.0`、joint count `2.0`、critical recall `0.5`、bidirectional set position `1.0`、spacing `0.5`。不要同时打开旧 truncated-power feedback fit，除非专门做消融实验。

已有完整 v14 checkpoint 时，可先做不重建离线教师的快速升级实验：

```powershell
python scripts/finetune_parameter_feedback.py `
  --checkpoint outputs/candidate_pruning_one_shot_v14.pt `
  --epochs 30 `
  --train-size 4000 `
  --val-size 1000 `
  --batch-size 16 `
  --output outputs/checkpoints/candidate_pruning_one_shot_v15_finetuned.pt
```

该入口用合成数据的 canonical 节点构造联合监督，不含离线教师 deletion risk；追求最终节点数和通过率时，仍以重新蒸馏的第一条命令为准。

## 验证重点

- `one-shot K mean`：确认平均节点数下降，而不是只看概率和；
- `refit loss`：部署 mean squared Euclidean error；
- `threshold-satisfied fraction` 与 RMS P95：防止均值改善但尾部失效；
- `offline hard teacher K/RMS`：区分 proposal/teacher 上限与学生蒸馏误差；
- 节点 precision/recall：必须在同一参数化域和同一匹配阈值下比较。

合理验收条件是：相同测试集、阈值和候选 proposal 下，平均节点数与部署 MSE 均不劣于 v14，并且通过率不下降。达不到时应保留 v14 checkpoint，不用展示性调参掩盖结果。
