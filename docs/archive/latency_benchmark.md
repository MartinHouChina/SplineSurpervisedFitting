# 10 ms 延迟目标：口径、实测与部署建议

> 本页保留 v12–v15 的历史计时。v16 必须重新报告纯 network time 与包含一次最终 refit 的
> full deployment time；当前未达标 proposal 不能作为正式延迟—精度结果。

## 先区分三个目标

- **单请求延迟**：`batch=1` 从驻留内存的归一化点到最终 B 样条结果。
- **批量吞吐摊销**：一个 batch 的总时间除以曲线数；它不是单请求延迟。
- **训练时间**：完整 forward、loss、backward 和 optimizer step。不能与推理时间混用。

基准脚本会在 CUDA 计时前后显式同步，执行 warmup，并同时报告 P50/P95。数据集生成、文件 I/O、绘图和网络传输不计入模型部署时间。

```powershell
python scripts/benchmark_deployment_latency.py `
  --checkpoint outputs/candidate_pruning_one_shot_v12.pt `
  --modes network-forward refit-chord learned-chord verified-fast `
  --batch-sizes 1 4 8 16 `
  --warmup 10 `
  --repeats 30 `
  --device cuda `
  --network-path structure-only `
  --refit-device cpu `
  --verified-parameterization chord `
  --torch-num-threads 4 `
  --include-training-lower-bound `
  --json-output outputs/logs/v12/v12_latency_benchmark_rerun.json
```

## 当前 GTX 1070 实测

输入为 192 个二维点，候选节点数为 28。以下表格使用历史完整 network path，P50 包含同步；标准 refit 使用 CPU `gelsy`。

| 路径 | B=1 总时延 | B=4 摊销 | B=8 摊销 | B=16 摊销 |
|---|---:|---:|---:|---:|
| network forward | 18.01 ms | 4.58 ms/curve | 2.28 ms/curve | 1.16 ms/curve |
| chord refit only | 3.35 ms | 3.16 ms/curve | 2.87 ms/curve | 2.60 ms/curve |
| learned-chord total | 23.78 ms | 8.32 ms/curve | 5.49 ms/curve | 4.03 ms/curve |
| verified-fast total | 36.46 ms | 14.52 ms/curve | 10.90 ms/curve | 11.20 ms/curve |

这些数值只适用于本机 GTX 1070 与当前软件栈，不能外推为 RTX 3090 的保证。正式论文需在目标硬件空闲时重复运行，并报告 batch size、P50/P95 和 CPU 线程数。

`--network-path structure-only` 会跳过 KeepMask/重定位之后仅供训练诊断使用的最终截断幂 surrogate solve，最终参数、proposal、KeepMask 和重定位节点与完整 forward 完全相同。实测它只把 CPU 单曲线 network P50 从 `11.70` 降到 `11.26 ms`，GTX 1070 从 `18.01` 降到 `17.84 ms`；该 solve 不是主要瓶颈，不能单独解决 10 ms 目标。

## 结论边界

- 当前模型的 `batch=1` 网络 forward 本身已超过 10 ms，因此在不改结构的前提下，端到端单曲线 `<10 ms` 不可行。
- 用户点云部署、分层评估和批量四联图现已调用 `forward_deployment()`，跳过训练专用的最终 truncated-power surrogate；结构输出和最终标准 B 样条 refit 不变。
- `learned-chord` 从 `batch>=4` 开始可以达到**摊销** `<10 ms/curve`，但 batch 总时延仍超过 10 ms。
- `verified-fast` 需要多次按样本精确 refit；当前实现即使批量输入也不能达到 `<10 ms/curve`。
- 完整训练 step 必然慢于网络 forward。当前模型在 GTX 1070 上仅使用重建 MSE 的同步训练下界为：B=1 P50 `47.87 ms/step`，B=8 P50 `47.92 ms/step`（`5.99 ms/sample`）。它还没有计算完整 v13 教师损失，因此“训练 batch 总时间小于 10 ms”不成立；只有大 batch 的单样本摊销可能低于 10 ms。

## 推荐双路径

1. **FAST**：一次网络 forward、弦长域映射、一次标准 refit。对实时流量做 micro-batch；延迟报告必须保留 batch 总时间。
2. **SAFE**：FAST 后精确检查 MSE；达标立即返回，失败曲线进入 `verified-fast` 条件修复。SAFE 保留逐曲线质量兜底，但不能承诺 10 ms。

若必须让 `batch=1` 端到端低于 10 ms，需要训练新的轻量学生，而不是改变计时口径：固定弦长参数、缩小编码器、用网络蒸馏的 deletion-risk 特征替代 forward 内的 pilot solve，并用 v13 教师监督 KeepMask 与 relocation。新学生只有在同一测试集重新验证通过率、MSE、节点数和节点匹配后才能替代现模型。
