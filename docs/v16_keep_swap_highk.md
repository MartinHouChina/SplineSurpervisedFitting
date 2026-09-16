# v16 KeepMask 交换监督与高 K 诊断

这是一项**尚待训练验证**的实验，不是已达标的正式结果。此前 Kc=56 快速试验虽然减少了节点数，但最佳检查点一次性验证通过率仅 10.7%；合成测试中源 K=45..56 的 12 条曲线，即使保留全部 56 个候选也均无法达到 `MSE <= 1e-4`。因此先分别处理两个瓶颈：节点组合选择，以及高 K 候选本身的可行性。

| 项目 | 新的诊断配置 |
| --- | --- |
| 合成训练源 | 内部节点 K=4..56，全部在训练范围内 |
| 候选容量 | Kc=72；高 K 保留 16 个候选余量，不再把 Kc=56 的失败混入 KeepMask 判断 |
| Proposal | 64 epoch；K>=40 的样本占 65% |
| Joint | 32 epoch，其中前 8 epoch 为 warmup；K>=45 的样本占 50% |
| 数据规模 | 合成 train/val=1500/400；每类真实验证 40；优化器 Batch=64 |
| 高 K 验证 | K>=45 的独立合成分层 32 条，另有 K=56 边界分层 32 条 |
| 教师 | 固定 Proposal 后离线构建 `synthetic_anchor_counterfactual` 标签；每条曲线最多探测 16 个替代槽。可行替代对用“至少保留一个”的 OR 监督，不可行替代则强化原槽保留；不证明全局最简。 |
| 训练/部署 | 训练期缓存数值教师，Joint 每步不做贪心搜索；部署仍一次网络前向、一次 Top-K、一次最终 B 样条 refit |
| 对照 | Synthetic K=4..56、UJI、Natural Earth、USGS、工业等距线；六方法快速对照，均标记为诊断 |

选择 Kc=72 是基于已拉取的**配对样本**：Kc=56 对 K=45..56 的 12 条曲线，数值补节点仍 0/12 达标；旧 Kc=72 在同组中可补节点达标 11/12。后者只能证明较大候选池具备更多可行余地，**不能证明一次性 KeepMask 已解决**。新配置也不保证用时或通过率。

## Linux/3090 一条龙

先同步代码，确认 Python 环境和 CUDA。首次运行请使用一个未被占用的 run name：

```bash
bash scripts/run_v16_mse1e-4_3090.sh \
  --keep-swap-highk72 \
  --prepare-real-data \
  --device cuda \
  --run-name candidate_selection_v16_mse1e-4_keep_swap_highk72_linux_r1
```

如四类真实数据的 manifest 已存在，可去掉 `--prepare-real-data`。先查看完整命令而不执行训练时，在同一命令尾部加 `--dry-run`。脚本拒绝覆盖同名 checkpoint；中断后在**配置不变**的前提下加 `--resume-run`，不要删除 `.last.pt`。重新训练则改用全新的 run name（如 `_r2`），旧教师缓存和检查点不得混用。

运行顺序：Proposal 训练 → 固定 Proposal 构建离线教师缓存 → Joint 训练 → 检查点审计 → 六方法合成/真实数据快速对照 → 四指标 PNG → 真实数据六方法案例 PNG。输出位于 `outputs/checkpoints/<run-name>*`、`outputs/teachers/<run-name>/`、`outputs/logs/<run-name>/`、`outputs/comparisons/<run-name>/diagnostic_*/` 与 `outputs/figures/<run-name>/diagnostic_*/`。此档为 Ours 与数值基线统一设置最多 72 个内部节点初始容量；源曲线仍为 K=4..56。基线使用 72 槽可能增加对照时间，因此快速档每个 K 仅抽 1 条、每类真实数据仅 5 条；该图只能定位问题，正式统计须另行扩大独立样本和基线迭代预算。

## 如何判断是否值得继续扩大实验

1. 先看 Proposal 全 72 个候选在 K>=45、特别 K=56 上的 dense MSE/通过率。若仍不足，Selector 不可能单独救回；先修候选位置或参数化。
2. 对比教师掩码、正确节点数量加网络排序、自由网络掩码三者的达标率。若教师可行而“正确 K＋网络排序”仍失败，继续调 Count 阈值或单纯加长 Joint 没有针对性。
3. 同时看 Keep P/R/F1、关键节点误删率、一次性 K/MSE/通过率，分别报告简单 K=4..20 与复杂 K=45..56。只看平均 K 下降会掩盖误删。
4. `ours_verified` 是额外数值补节点的**另一种方法**，须单列节点数、MSE 和完整耗时。不能用它的通过率代表原始一次性 `ours`。快速基准每个 K 仅 1 条、每类真实数据仅 5 条，只能用于定位问题，不用于论文统计。

如果一次性通过率与节点数同时改善，再扩大独立测试样本和基线迭代预算。若高 K dense 仍不可行，应先停在 Proposal 阶段分析，而不是继续投入完整 Joint 训练。
