# v14_joint 三方法随机分层测试

## 口径

- checkpoint：`outputs/checkpoints/current/candidate_pruning_v14_joint.pt`
- 数据：独立合成测试集，source internal K=4..20，每档随机 3 条，共 51 条
- dataset seed：`20000`；selection seed：`20260905`
- 误差：标准三次 B 样条 refit 后的 mean squared Euclidean error，不开方
- 满足条件：MSE <= `1e-5`
- Ours 时间：GTX 1070、batch=1、输入已在 GPU 上的同步 network forward 中位数；不含最终 refit
- Kang/Greedy 时间：CPU 完整数值搜索与 refit；三者时间范围不对称，不能直接宣称端到端加速倍数

## 总结果

| 方法 | mean / median / P95 MSE | 满足率 | mean final K | 相对 canonical K 的 mean bias/MAE | median / P95 time |
|---|---:|---:|---:|---:|---:|
| Ours v14_joint learned | `1.195317e-3 / 9.771901e-4 / 2.592211e-3` | `0/51 = 0.00%` | `32.000` | `+20.000 / 20.000` | `29.07 / 35.16 ms` network-only |
| Kang-adapted + disclosed feasibility repair | `3.125119e-5 / 2.542540e-5 / 7.492681e-5` | `21/51 = 41.18%` | `33.216` | `+21.216 / 21.216` | `1955.83 / 3261.68 ms` full search |
| Uniform-Kmax greedy deletion + gradient relocation | `2.448690e-4 / 7.179771e-5 / 1.388623e-3` | `8/51 = 15.69%` | `18.765` | `+6.765 / 6.765` | `1585.80 / 11580.78 ms` full search |

按 source K 分层，Ours 的满足率始终为 0%。Kang 在 K=4..9 为 100%，K=10 为
66.7%，K=11 为 33.3%，K>=12 为 0%。独立 Greedy 在 K=4..5 为 100%，K=6
为 66.7%，K>=7 为 0%。

## 结论

这次训练已经塌缩，不能作为 v14_joint 的正面论文结果。模型在 51/51 条曲线上都保留
32/32 个候选节点，但平均 MSE 仍约为阈值的 119.5 倍。checkpoint 内部证据也一致：

- proposal checkpoint 的 validation fit MSE=`2.4397e-2`、RMS=`6.4652e-2`；
- offline teacher 在 train/validation 上都保留 32/32；
- teacher validation mean MSE=`6.8545e-4`，阈值满足率为 0%；
- 最终 checkpoint validation RMS=`3.1976e-2`，满足率仅 `0.15%`。

因此根因发生在 proposal/parameter 阶段：全候选拟合在 teacher 生成前就无法达到
`1e-5`，Hard-RMS teacher 没有任何合法删除动作，之后的 Keep 网络只能学成全保留。

## 文件

- `three_method_comparison.png`：MSE、最终节点数、满足率、时间四面板图
- `three_method_comparison.json`：完整配置、汇总与逐样本诊断
- `three_method_comparison.csv`：逐样本长表
