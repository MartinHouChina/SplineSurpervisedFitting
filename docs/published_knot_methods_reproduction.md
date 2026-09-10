# 六种节点方法的适配复现与统一协议

## 1. 比较对象

当前主表固定六种方法：

| 标识 | 方法思想 | 本仓库实现边界 |
|---|---|---|
| `ours` | 有序候选、一次性 adaptive mass-TopK、参数/存活节点联动重定位 | 当前 supervised-only v16；部署 1 forward + 1 refit |
| `park_dominant_point_2007_adaptation` | dominant point 与局部曲率驱动节点配置 | Park & Lee (2007) 的可审计 adaptation |
| `liang_feature_iki_2017_adaptation` | 特征积分初始化与 iterative knot insertion | Liang et al. (2017) adaptation |
| `dung_direct_knot_2017_adaptation` | 最大误差分段和局部节点优化 | Dung & Tjahjowidodo (2017) 的串行/simple-knot adaptation |
| `kang_sparse_2015_adaptation` | 稠密节点上的 group sparsity、聚类与重定位 | Kang et al. (2015) 的 group-L1/ADMM adaptation |
| `luo_linf_de_2022_adaptation` | `l_inf,1` 稀疏阶段与 Differential Evolution 重定位 | Luo et al. (2022) adaptation |

这些实现基于论文目标与公开算法描述，不是作者原始软件的逐行、bit-exact 复现。报告和图例必须保留 “adaptation”。Yeh 和统一均匀贪心只作为 `--method-set all` 的附加控制，不属于六方法主表。

## 2. 公平协议

1. 六方法处理完全相同的配对曲线与相同归一化输入。
2. Synthetic 按 source `K=4..56` 分层；真实数据来自 UJI、Natural Earth、USGS 的留出 test split。
3. 最大内部节点容量统一为 56；三次开放完整节点向量全容量为 64 项。
4. 最终结果统一用端点约束、无平滑、无 ridge 的 CPU float64 标准 B 样条 refit。
5. 公共误差为 `MSE=mean_i ||C(t_i)-Q_i||²`，阈值固定 `1e-4`。
6. final K 是最终 refit 实际使用的内部节点数。
7. `total_ms` 包含方法计算和最终 refit，不含文件 I/O 与绘图；Ours 的 `network_ms` 仅作附加指标。
8. 异常、非有限解和未达阈值解均保留在通过率分母。
9. 真实数据同时报告 192 点 input MSE 与 original-reference MSE。

不同论文原始目标可能是最大误差、稀疏正则或几何特征，而公共表统一报告最终 MSE。算法内部目标可保留作诊断，不能冒充公共 MSE。

## 3. Ours 的训练隔离

Ours 的 checkpoint 只由 certified Synthetic 训练；UJI、Natural Earth、USGS 只用于 validation/test。Joint 用真参数、真 K 和有序节点匹配直接监督 KeepMask 与 relocation，不运行在线 Teacher。因此比较时不能把真实测试样本或其 reference 曲线回流到训练。

## 4. 正式运行

推荐使用一条龙入口，确保训练、checkpoint 审计、配对 benchmark 和作图共享同一 run manifest：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v16_mse1e-4_3090.ps1 `
  -RunName candidate_selection_v16_mse1e-4_k56_supervised `
  -Device cuda
```

只运行 benchmark 时：

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_supervised.pt `
  --output-dir outputs/comparisons/v16_supervised_six_methods `
  --method-set published `
  --samples-per-knot-count 5 --min-knot-count 4 --max-knot-count 56 `
  --real-samples-per-dataset 20 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 --max-internal-knots 56 `
  --paper-initial-knots 56 --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 --paper-relocation-iterations 12 `
  --liang-dense-knots 56 --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 --torch-num-threads 4 --device cuda
```

输出 `comparison.json`、`summary.csv`、`measurements.csv` 和 `report.md`。随后执行 `plot_v16_method_comparison.py --reference` 会从同一 JSON 生成 input/reference 两张 2×2 图；`visualize_v16_real_deployments.py` 生成真实六方法案例图。

## 5. 结果解释

- 不能根据少数简单或复杂个例概括方法普遍优劣；应同时看分层 Synthetic 和三个真实来源。
- Kang/Luo 的稀疏阶段可行而压缩/重定位后失败时，应如实计为失败，不能用 dense 初值误差替代最终误差。
- input MSE 通过而 reference MSE 失败，表示对重采样点拟合良好但原始几何保真不足。
- 任何方法的超时或失败都不得从分母删除。
- 本文件不声称任何尚未实际运行得到的领先、通过率或速度结果。
