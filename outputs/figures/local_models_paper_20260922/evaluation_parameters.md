# 本地六方法对比：实际评估参数

| Parameter | Value |
| --- | --- |
| Compared checkpoint | r3_m32.pt |
| Stored weight epoch | 13 |
| Input points per curve | 192 |
| Spline degree | 3 |
| Internal-knot capacity (all methods) | 32 |
| Normalized MSE pass threshold | 5e-05 |
| Additional peak metric | max_i \|\|C(t_i) - Q_i\|\|^2 (no root) |
| Source families / paired selected curves | 5 / 61 |
| Synthetic source K / curves per K | 4--24 / 1 |
| External test selection seed | 20260922 |
| External sampling | Seeded group round-robin |
| UJI selected / available test curves | 10 / 6093 |
| NaturalEarth selected / available test curves | 10 / 533 |
| USGS selected / available test curves | 10 / 201 |
| IndustrialOffset selected / available test curves | 10 / 67 |
| Compared methods | Ours + Park + Liang + Dung + Kang + Luo |
| Literature implementation protocol | adaptation |
| Extra common feasibility repair | No |
| ADMM iterations / lambda bisections | 400 / 8 |
| Kang relocation iterations | 8 |
| Liang dense / initial internal knots | 32 / 4 |
| Dung scan intervals / optimization iterations | 10 / 10 |
| Luo DE population setting / generations | 10 / 50 |
| Network-only warmups / timed repeats | 3 / 10 |
| Complete-method timed repeats per curve | 1 |
| GPU used by Ours in this local evaluation | NVIDIA GeForce GTX 1070 |
| CPU numerical backend threads | 4 |
| Python / PyTorch | 3.13.9 / 2.12.0+cu126 |
| Dense plot sampling points | 512 |

- 参数来源为正在运行/已运行 benchmark 的 experiment.json，不是训练默认参数。所列样本数是已选定数目；全部结果是否完成以最终 rows / summary 为准。
- 本次每外部来源选 10 条，采用固定随机种子、按组轮询，不是每个数据集的十分之一。五列来源为合成、UJI、海岸线、等高线、程序生成等距线。
- IndustrialOffset 为程序生成 CAD 风格等距线，不是实测工业数据。
- 五种文献方法均为仓库适配实现，未经原作者代码一致性认证。额外公共可行性修复关闭。
- MSE 与最大平方点误差均为归一化离散对应点指标，后者不是连续曲线 Hausdorff 距离。通过率仅由 MSE 判断；最大误差单列报告。
- 计时由各方法完整运行得到；Ours 额外报告单独预热测得的 network-only 时间。不能用 GPU network-only 时间代替数值方法完整耗时而宣称端到端公平加速。
- 完整方法每曲线只重复 1 次，适合此次出图诊断，不足以估计运行时间方差；每条曲线内部 network-only 重复 10 次。
- 来源：`E:\SelfSurpervisedSplineFitting\outputs\comparisons\local_models_paper_20260922_m32\experiment.json`
- SHA256：`4400bc5072147a300b10ce2e175f7c1ed26afc34562957c60a5cce12585ec49d`
