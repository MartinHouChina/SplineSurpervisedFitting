# Minimum-Complexity B-Spline Fitting

给定沿曲线方向排序的二维或三维采样点，本项目在规定误差内一次性预测尽量少的三次
B 样条内部节点，并用标准最小二乘 refit 求控制顶点。

当前 v16 主协议：

| 项目 | 设置 |
|---|---:|
| 误差 | mean squared Euclidean error，`MSE <= 1e-4`，不开方 |
| 合成 source internal K | 4..56，即控制顶点 8..60 |
| 合成节点最小 span | 0.01；K=56 有 57 个 span，旧 0.02 仅兼容历史 |
| 候选容量 | 56 个内部节点（完整三次开放节点向量上限为 64） |
| Proposal 课程 | 32/96 epoch；合成样本 50% 来自 `K>=40`，Joint 恢复 K=4..56 原分布 |
| 简化合同 | `ranked_prefix_ordered_proposal_high_k_adaptive_complexity_v2` |
| 数据 | Synthetic + UJI + Natural Earth + USGS |
| 验收 | 每个验证来源的 dense/deployment pass 均至少 90% |

## 算法流程

```text
有序点 Q + MSE 阈值 epsilon
  -> GeometryEncoder
  -> ParameterHead：弦长参考 + 有界残差，得到严格递增参数 t0
  -> CandidateKnotHead：带位置编码的局部 cross-attention，得到 Kc=56 个有序候选 U0
  -> Contextual Selector：候选自注意力 + 点特征交叉注意力
  -> 曲线级 beta + 候选相对重要度 -> keep probability mass
  -> 一次 mass-TopK，得到可变长度 KeepMask
  -> selected-only Decoder：筛选与存活节点/参数联合重定位
  -> 一次端点约束的标准三次 B 样条 refit
  -> 内部节点、控制顶点、拟合曲线和实际 MSE
```

部署没有 CountHead、Hard-Concrete、BIC、阈值扫描或逐节点试删；只有一次网络前向、
一次离散 Top-K 和一次最终 refit。反事实集合与多次 refit 只用于训练教师。

为减少简单曲线的过度保留，同时保住复杂曲线容量，当前训练采用：

- Selector 的 Joint 初始 keep fraction 设为 `30/56≈0.535714`，目标初始概率质量约为 30；这只是初始化，不是部署最终 K；
- Proposal 同时使用真节点到候选的 directed coverage 与一维单调一一匹配；前者保护召回，后者避免多个真节点共享同一个最近候选；
- 前 32 个 Proposal epoch 将合成抽样的 50% 固定到 `K>=40`，强化复杂曲线与容量边界；后 64 个 Joint epoch 恢复 K=4..56 原分布，避免 Selector 被高 K 先验推向过留；
- 教师逐个检查低节点数 `K=4..16`，再做粗到细 ranked-prefix 搜索；
- 合成 source K 作为可重定位问题的可行上界，不把更小的可行解拉回 source K；
- 合成真节点通过单调匹配加入训练期 geometry-oracle 教师池，并额外测试 `Ktrue+2`；
- 关闭固定分区强制锚点，由曲线级 beta 自适应决定实际节点数；
- KeepMask 与存活节点位置通过 selected-only 解码器联合训练。

## RTX 3090：训练、测试和作图一键运行

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_v16_mse1e-4_3090.ps1
```

Linux 原生 Bash：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_mse1e-4_k56_ordered_highk_linux
```

若当前环境已经激活，可用 `--python "$(command -v python)"` 明确指定解释器；脚本
不负责切换 Conda/venv。首次换机使用 `--prepare-real-data` 下载并生成三个缺失的
manifest；已存在的数据不会重复处理。`--dry-run` 只打印完整命令，`--help` 查看可调
训练与评测参数。

该入口按顺序执行：

1. 新训练 `Kc=56 / source K=4..56 / MSE=1e-4`；
2. 检查 90% 最差来源通过率和结构资格；
3. 在 Synthetic、UJI、Natural Earth、USGS 上运行六方法对比；
4. 绘制 MSE、通过率、最终内部节点数、完整方法时间四项指标图；
5. 绘制真实曲线六方法 3x2 案例图。

脚本不会覆盖同名实验。目标 checkpoint 尚需在 3090 上训练；若资格检查失败，正式模式
会停止且不生成可汇报图。旧的
`outputs/checkpoints/candidate_selection_v16_mse5e-5_k64.proposal.pt` 若存在，可迁移初始化
Encoder、ParameterHead 和 CandidateHead 中形状兼容的 proposal 张量；其中 65 个旧区间
query 沿参数域插值为 57 个，新模型重新生成固定锚点。Selector、联合解码器和优化器均
从头训练，这属于 warm start，不是把旧 K64 实验 resume 成 K56。

旧安全门失败实验生成的
`candidate_selection_v16_mse1e-4_k56_linux.proposal.pt` 也可以通过
`--init-checkpoint` 迁移 Proposal 模块；由于有序匹配和高 K 课程已升级训练合同，
不能使用旧 `.last.pt` 直接 `--resume`。

3090 会明显加速网络前向/反向，但 Joint 的在线教师仍包含多轮逐样本 float64 B 样条
求解；Kang/Luo 评测也主要在 CPU 上运行，因此整个流水线不会按显卡算力同比缩短。
当前 96-epoch 配置不承诺固定 12 小时完成。该入口是 fresh-only：不要用正式
`RunName` 做 `-DryRun`；中断后按
[训练流程](docs/training_pipeline.md)中的完整参数恢复 `.last.pt`，训练已经成功而仅后处理
失败时直接重跑 benchmark/绘图，不要重新训练。

Windows worker 异常时增加 `-NumWorkers 0`；Linux 对应参数为 `--num-workers 0`。若
batch 64 在 24 GB 显存上仍 OOM，请使用新的实验名并将 batch 调成 32，不要覆盖或混接
原实验。

## 资格检查

```powershell
python scripts/inspect_v16_checkpoint.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk.pt `
  --required-pass-rate 0.90 `
  --mse-tolerance 1e-4
```

返回码 0 才可正式汇报。`.proposal.pt`、未达标权重以及显式
`--allow-unqualified-diagnostic` 生成的结果仅用于排错，并带诊断标识。

## 六方法公平对比

主表固定比较：Ours、Park & Lee、Liang、Dung & Tjahjowidodo、Kang、Luo。公开方法是
根据论文目标编写的可审计适配，不冒充作者原始软件。所有方法处理同一曲线，使用相同
56 内部节点容量、三次样条、最终 CPU float64 无正则 refit 和 `MSE <= 1e-4` 判据。

四项指标为：

- 最终 MSE；
- 阈值通过率；
- 最终内部节点数；
- 完整方法时间。

Ours 另报 network-only 时间，但不能用它替代端到端时间。Kang/Luo 的完整复查见
[Kang/Luo 1e-4 审计](docs/kang_luo_mse1e-4_audit.md)。分步 benchmark 与作图命令见
[训练流程](docs/training_pipeline.md)。

## 节点数量

`--candidate-knots Kc` 表示内部候选数。对三次开放 B 样条：

\[
N_{\mathrm{full\ knot\ vector}}=K_{\mathrm{internal}}+8,\qquad
N_{\mathrm{control}}=K_{\mathrm{internal}}+4.
\]

因此 Kc=56 全保留时完整节点向量长度恰为 64、控制顶点数为 60；合成 source K 最大
56 时也到达同一容量边界。source K 描述生成曲线复杂度，Kc 描述网络候选槽位；二者
数值相同不表示网络必须全保留。由于 K=56 层没有候选冗余余量，必须单独报告该层的
dense/deployment pass；若该层失败，不得通过放宽 90% 正式资格掩盖。
有序一一匹配能缓解多对一候选塌缩，但不能创造第 57 个内部候选，因此也不能替代额外
候选容量；K=56 仍是必须单独审计的硬边界。

K=56 会产生 57 个 span，因此旧 `min_span=0.02` 会要求总长度至少 1.14，无法
生成边界样本。当前 `train_v16.py` 及一键脚本显式使用
`--knot-min-span 0.01`；底层通用 synthetic 生成器的 0.02 默认值只保留用于
历史调用兼容。

## 文档入口

- [文档总索引](docs/README.md)
- [v16 算法与教师](docs/v16_counterfactual_subset.md)
- [训练、资格、六方法评测与作图](docs/training_pipeline.md)
- [部署流程](docs/deployment_pipeline.md)
- [数学定义](docs/math_formulation.md)
- [合成曲线最简性证书](docs/synthetic_data_minimality_report.md)
- [真实数据集](docs/real_world_datasets.md)
- [公开方法适配协议](docs/published_knot_methods_reproduction.md)
- [当前验证状态](docs/v16_verification.md)
- [文件索引](docs/file_guide.md)

旧 Kc=96/2.5e-5、5e-5 和 v8--v15 资料保留在 `docs/archive/`，仅用于历史消融与追溯。
正式权重写入 `outputs/checkpoints/`，比较写入 `outputs/comparisons/`，图片写入
`outputs/figures/`。
