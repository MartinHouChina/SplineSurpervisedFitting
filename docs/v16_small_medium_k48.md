# 中小 K 曲线实验：48 个候选内部节点

当前主线只把候选容量从 Kc=24 提高到 **Kc=48**；合成 source 内部节点仍为 **K=4..20**，不扩展到 K=48，也不恢复高 K 混合采样。MSE 阈值保持 `5e-5`，训练保持 Proposal 64 + Joint 64 = 128 代。旧 [K24 配置与检查结果](v16_small_medium_k24.md)、Kc56/72 历史诊断和 1070 配置均保留，不改标成 K48 结果。

## 容量与训练设置

| 项目 | K48 设置 |
|---|---|
| 样条次数 | 3 |
| 合成 source 内部节点 / 控制顶点 | 4..20 / 8..24 |
| 网络内部候选上限 Kc | 48 |
| 六方法内部节点预算 | 全部为 48 |
| 完整部署节点向量最大长度 | 48 + 4 + 4 = 56 项 |
| 部署控制顶点最大数量 | 48 + 3 + 1 = 52 个 |
| 误差约束 | 平均平方欧氏距离 MSE ≤ 5e-5 |
| Proposal / Joint | 64 / 64 代，总计 128 代 |
| Joint warmup | 4 代，已包含在 Joint 64 代内 |
| 合成训练 / 验证 | 600 / 160 条 |
| 外部来源验证 | 每来源最多 20 条，仅验证、不训练 |
| 采样点 / Batch | 192 / 32 |
| Proposal / Selector / Decoder 学习率 | 2e-4 / 5e-5 / 1e-5 |
| 初始 Keep 比例 / 概率质量 | 0.5 / 48×0.5=24，仅初始化值 |
| Teacher 策略 | synthetic_anchor_counterfactual，最多 8 个局部交换探测 |
| 固定安全节点 / sigma | 0 / 0.03 |
| 附加复杂度损失权重 | 0；数量仍由 Teacher 监督 |

完整向量是 `[0,0,0,0] + U_internal + [1,1,1,1]`，56 个向量条目不是 56 个控制顶点。MSE 不开方，也不除以坐标维数。初始概率质量 24 不等于固定保留 24 个节点；最终数量由网络概率质量与 mass-TopK 产生，最少保留 4 个内部节点的既有部署限制不变。

训练从零开始，100% 使用认证合成数据；真实数据只作留出验证和测试。普通合成验证集仍含 16 条指定 source K=20 的边界样本。Keep/Drop 状态交互、全局间隔调整、Count–Keep 耦合、边界排序损失及节点/参数更新不变。Joint 冻结 Proposal，在固定训练样本上离线构建可行 Teacher，不逐轮重建。部署仍为一次网络前向、一次选集、一次标准 B 样条 refit，Teacher 不是部署输入。

## 独立实验与缓存

- 当前范围：`--study-scope small_medium_k48`。
- 认证合同：`certified_source_subset_threshold_minimal_k4_20_kc48_span001_v1`。
- 默认运行名：`candidate_selection_v16_mse5e-5_small_medium_sourcek20_kc48_p64_j64_linux_r1`。

K48 必须使用未占用的唯一运行名，重新生成对应认证数据和 Teacher 缓存。不得加载旧 K24 或 Kc56/72 权重，也不得把旧 `.pt`、`.last.pt`、Teacher 缓存或旧阈值缓存改名复用。即使 source K 与 MSE 相同，K24 和 K48 也不是同一训练/缓存合同。`--resume-run` 仅用于继续同名、同配置的 K48 实验。

`--mse-tolerance 5e-5` 一致传入合成认证、训练/验证、离线 Teacher 及六方法评测。共享脚本的其他历史档未覆盖时仍默认 `1e-4`；原 K24 入口仍保留其自身默认设置。

## Linux / RTX 3090

先检查参数与命令，不执行训练：

```bash
bash scripts/run_v16_small_medium_k48_3090.sh \
  --device cuda \
  --mse-tolerance 5e-5 \
  --epochs 128 \
  --proposal-epochs 64 \
  --run-name candidate_selection_v16_mse5e-5_small_medium_sourcek20_kc48_p64_j64_linux_r1 \
  --dry-run
```

启动训练、同容量六方法对比和外部案例绘图：

```bash
bash scripts/run_v16_small_medium_k48_3090.sh \
  --prepare-real-data \
  --device cuda \
  --mse-tolerance 5e-5 \
  --epochs 128 \
  --proposal-epochs 64 \
  --run-name candidate_selection_v16_mse5e-5_small_medium_sourcek20_kc48_p64_j64_linux_r1
```

数据已齐全时可省略 `--prepare-real-data`。新入口调用共享脚本 `scripts/run_v16_mse1e-4_3090.sh --small-medium-k48`；上述名字也是默认名，已被使用时须换成新的唯一名字，不覆盖旧实验。

同名 K48 实验中断后继续：

```bash
bash scripts/run_v16_small_medium_k48_3090.sh \
  --resume-run \
  --device cuda \
  --mse-tolerance 5e-5 \
  --epochs 128 \
  --proposal-epochs 64 \
  --run-name candidate_selection_v16_mse5e-5_small_medium_sourcek20_kc48_p64_j64_linux_r1
```

首次覆盖过的训练或评测参数，续跑时也须保持相同。未完成训练从 `.last.pt` 恢复；已完成训练校验后跳过，继续评测和绘图。完整评测按检查点、配置、代码、数据与环境指纹校验后复用，部分评测补齐；绘图/案例重试保留旧文件并写入新的 `attempt_*` 目录。未加 `--resume-run` 时拒绝覆盖现有运行。历史流程修复见 [K24 检查与修复记录](v16_k24_code_audit.md)，其测试数值不是 K48 的实训证据。

## 本次代码验证

2026-09-17：255 项相关测试通过，包括 K48/K24 容量、训练与初始化、离线 Teacher、续训及一条龙脚本回归；Bash 语法与修改 Python 文件的 Ruff 检查通过。实际执行了 48 候选前向、4/20/48 个幸存节点的重定位与标准 refit，以及 `5e-5` 下的小规模 CPU Proposal→Joint→续训。完整 128 代训练与最终性能尚未验证。

## 对比与结果边界

- Ours、Park、Liang、Dung、Kang、Luo 使用相同曲线、MSE 定义、`5e-5` 阈值和 48 个内部节点预算；合成对比只扫描 source K=4..20。
- 真实曲线没有已知真 K，不按 Ours 是否通过筛选样本；在 48 节点预算内仍失败的样本须照常计入结果，不能隐藏成“范围外”。
- 保留六方法四指标图和真实数据案例图。`ours_verified` 若启用，须独立报告完整耗时、额外 refit 与节点数，不混入原始 Ours。
- 本范围按诊断协议运行，不改变旧 K4..56/Kc72 的正式资格合同。默认 `quick` 不是论文最终统计，`--benchmark-profile full` 只增加评测预算，不自动赋予正式结论。

尚未用该 K48 配置完成完整 128 代实训；文档不预设通过率、节点数、耗时或每例达标保证。扩大候选容量本身不能证明解决了历史 Keep 排序与组合选择问题；必须结合同一误差约束下的通过率、最终节点数、MSE 和时间评价，不能把全保留当成成功简化。历史 K24 的回归测试与短训数值继续只解释其原配置。
