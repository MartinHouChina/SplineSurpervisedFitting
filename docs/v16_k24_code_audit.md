# Kc24 代码检查报告（2026-09-17）

检查基准：`bc92e8a`，source 内部 K=4..20、候选 Kc=24、MSE 阈值 1e-4。

原检查结论：训练核心和容量配置的现有测试通过，但一条龙存在可复现的绘图与续跑缺陷。下文保留修复前的证据；用户随后授权修复，三项问题现已修正，修复范围和复测见文末。

## 已完成验证

- 训练、网络、损失、Teacher、初始化、K24 参数与恢复等 12 个文件：300 passed。
- 评测、公开方法、绘图、点云与真实数据 I/O 等 24 个文件：221 passed。
- 两组共 521 项通过；它们未覆盖下述跨脚本组合，不能以测试通过替代集成验证。
- 实跑 17 条 Synthetic（source K=4..20 每个 K 一条）和 UJI / Natural Earth / USGS 各一条，共 20 条配对曲线，六方法另加单列 `ours_verified`，140 条执行记录全部完成。
- 核对全部记录：内部节点超出 24 的记录为 0；`fit_pass` 与实际 `MSE<=1e-4` 不一致的记录为 0；候选与各数值方法最大完整节点向量均为 32 项。
- 另运行三个真实数据案例的六方法 PNG 绘制，均成功；四指标图在读入真实评测产物时报错。

短测使用此前仅少量更新的 smoke checkpoint，并降低数值方法迭代预算；多个诊断并行运行。其精度和耗时不用于方法优劣或泛化结论，未执行完整 48 代训练。内部梯度检查覆盖 Keep 状态交互、Count 耦合、selected-only decoder 及 Teacher 参数域对齐，没有发现当前 K24 路径的维度或梯度切断错误。

## 修复前发现的问题

### 1. 默认一条龙的四指标绘图必然拒绝修复方法行

`run_v16_mse1e-4_3090.sh:859` 默认启用 `--include-verified-ours`，评测 JSON 因而含有 `ours_verified`。`plot_v16_method_comparison.py` 调用共享 `read_report`，但 `plot_v15_dataset_benchmark.py:18` 的合法方法列表不包含它，84 行抛错：

```text
ValueError: Unsupported method: ours_verified
```

已使用实际 benchmark 产物复现，和模型性能无关。失败发生在 `plot_four_metrics`，因此一条龙其后的真实案例绘制也不会执行。正确修复应允许读取这种单列诊断方法，同时保持主图只画原六方法，不能将数值修复混作原始 Ours。

### 2. 训练结束后，评测或绘图中断无法用一条龙续跑

`run_v16_mse1e-4_3090.sh:752` 在 `--resume-run` 时总是重新进入训练；`train_v16.py:2037` 拒绝已完成所请求 epochs 的 checkpoint。使用已有 K24 epoch=5 的 last checkpoint 请求相同总轮数，实测得到：

```text
checkpoint already completed the requested epochs
```

即使强行增加训练轮数，只要 best 权重没有变化，脚本 821 行仍拒绝已存在的同 hash 评测目录。benchmark 支持 `--resume`，一条龙没有传递。应区分训练完成和后处理完成，校验相同配置后跳过已完成阶段、恢复不完整阶段，保留防覆盖保护；不能通过删除旧结果解决。

### 3. 一个方法执行失败会导致真值节点均值不一致，阻断绘图

`benchmark_v15_datasets.py:718` 只从 `status=ok` 的行统计 `canonical_k_mean`（745 行），而 `plot_v16_method_comparison.py:145` 要求各方法的该均值一致。

内存复现：所有方法共用真值 K=4 和 K=20 两例，仅 Ours 的 K20 例发生数值失败；汇总后 Ours 真值均值变成 4，其他方法为 12，绘图报：

```text
Synthetic canonical_k_mean is inconsistent across paired methods
```

真值统计应基于完整配对样本，与算法是否成功无关；预测节点误差等需要有效输出的统计则必须保留各自有效样本分母。不能丢弃失败例来使图表通过。

## 环境和解释限制

- 当前本地缺少 `data/processed/industrial_offsets/v1/manifest.jsonl`。因此实际外部测试只覆盖三个已有来源；四来源 wrapper 做了 dry-run 参数检查，没有宣称完整执行。标准命令带 `--prepare-real-data` 时会尝试准备缺失来源。
- 旧 inspector 的正式资格仍针对 K4..56/Kc72，本 K24 配置明确标记为 diagnostic；这不是 K24 张量容量错误，也不能据此声称新范围已具备论文资格。
- Teacher 若全候选已不达标，离线标签仍可能全留；当前 Joint 固定 Proposal，不能将这类失败仅归咎于 Keep。必须分别观察 dense、Teacher、自由部署通过率。

## 复现产物

- `outputs/self_validation/k24_code_review_20260917/comparison/comparison.json`
- `outputs/self_validation/k24_code_review_20260917/comparison/report.md`
- `outputs/self_validation/k24_code_review_20260917/real_cases/deployment_visualizations.json`
- 同目录三张 `six_methods__*.png`

四指标绘图复现命令（修复前报错；修复后已成功输出两张 PNG，不会篡改评测数据）：

```bash
python scripts/plot_v16_method_comparison.py \
  --input outputs/self_validation/k24_code_review_20260917/comparison/comparison.json \
  --output-dir outputs/self_validation/k24_code_review_20260917/metric_figures \
  --method-set published --dpi 90 --reference --allow-unqualified-diagnostic
```

## 三项修复与复测

1. 共享图表读取器接受已知辅助方法 `ours_verified`，但不将其加入六方法主图或替换原始 Ours。新增真实 CLI 路径测试，确认两张 PNG 生成、图中仍只有六方法、输入 JSON 字节不变，未知方法及非法指标仍拒绝。
2. 统计 `canonical_k_mean` 时使用全部有标签的配对样本；新增 `canonical_label_n` 和 `canonical_count_valid_n`，区分真值统计与预测误差的有效分母。选择性数值失败、全失败、超阈值但有有效输出均有回归测试；失败仍计入通过率分母。
3. 新增 `scripts/v16_pipeline_resume.py` 并接入 Linux runner。已完成训练验证配置、阶段、模型合同和数据来源后跳过；未完成训练继续原生恢复。完整评测验证代码/配置/数据/硬件/逐例记录后复用，部分评测通过 journal 续算。派生图像记录 `.pipeline-stage.json` 和产物哈希，重试写新 `attempt_*` 目录，不删除旧文件。历史 paired 检查使用独立 `paired_native_stage` 子目录。

实际复测：

- 最终综合回归 `394 passed, 5 skipped`（59.60 秒）；5 项是本地 Windows 缺少原生 `/bin/bash` 的既有平台跳过，Git Bash 实际 wrapper 用例通过。新增脚本 Ruff、Python 语法、Bash 语法及 diff 空白检查通过。
- 现有 K24 last checkpoint 在请求原总轮数 5 时判为 `complete`；显式扩展到 6 时判为 `resume`，不再要求为继续绘图而增加训练轮数。
- 用修复后的代码重新完成 20 条配对曲线、7 个单列方法（140 条记录）的轻量评测；两张四指标图均生成。
- 完整 benchmark 的恢复检查输出 `Reusing complete, fingerprint-verified benchmark`，没有重新运行拟合算法。
- 实际图像阶段首次生成两张指标图、三个真实案例图；再次运行恢复分别输出 `Reusing verified plot outputs` / `Reusing verified visualize outputs`，没有重新计算案例。中断保留旧文件和新 attempt 重试另有回归测试。
- 产物位于 `outputs/self_validation/k24_three_fixes_20260917/`，未替换原检查目录。

这些修复不改网络、Teacher、训练损失、学习率或已有 checkpoint。无需因流程修复重新训练模型，但代码/数据/硬件指纹改变的旧 benchmark 不能与新测量混合续跑；会保留旧结果并提示。旧报告若统计本身正确，可单独绘图到新目录。

仍未执行完整 48 代训练，工业等距线本地 manifest 缺失的限制也未改变；本记录是工程修复验证，不是新模型效果结论。
