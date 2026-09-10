# 六种节点方法的适配复现与统一协议

当前主实验固定比较六种方法：Ours v16、Park–Lee、Liang、
Dung–Tjahjowidodo、Kang 和 Luo–Kang–Yang。五种公开方法均为基于论文目标
和算法描述实现的**可审计 adaptation**，不是作者官方代码的逐语句复刻。

主协议统一为：192 个归一化有序点，合成源内部节点 `K=4..56`（控制顶点
8～60），最多 56 个内部节点（完整节点向量 64 项），三次开放 B 样条，
`MSE<=1e-4`，同一最终
CPU `float64` 标准 refit。

## 1. 复现范围

| 方法 | 原始思想 | 本仓库实现与边界 |
|---|---|---|
| Park & Lee, 2007 | 从曲率显著点出发，在误差约束下自适应细分 dominant points，并由相邻参数生成节点。 | `park_dominant_point_2007_adaptation`；弦长参数、离散 Menger 曲率和形状指数细分。统一 MSE 替代论文的原生距离停止准则。 |
| Liang et al., 2017 | 用弧长与弯曲特征积分放置初始节点，再迭代插结。 | `liang_feature_iki_2017_adaptation`；实现 feature-integral + IKI，当前公平协议的密集容量为 56。全文常数和全部 IKI 细节无法逐项核验，不能称精确复现。 |
| Dung & Tjahjowidodo, 2017 | 按最大误差分段，再局部优化节点位置和连续性。 | `dung_direct_knot_2017_adaptation`；只实现串行、平滑单节点版本，不复现重节点/连续性分类和并行 split–join–shift。 |
| Kang et al., 2015 | 密集初始节点上的 group sparsity，随后聚类、删冗余和节点调整。 | `kang_sparse_2015_adaptation`；二维 group-L1/ADMM、跳跃聚类及局部重定位，不声称复现 CVX 数值路径。 |
| Luo–Kang–Yang, 2022 | `l_inf,1` 稀疏阶段确定候选数，再用 Differential Evolution 更新固定数量的节点位置。 | `luo_linf_de_2022_adaptation`；ADMM 稀疏阶段、局部峰值选候选、DE 重定位。指的是 `l_inf,1 + DE` 方法，不是 DNN 工作。 |
| Ours v16 | 高召回候选、一次性自适应 KeepMask、选集条件下参数与存活节点联动重定位。 | `ours`；`Kc=56`，部署一次网络前向、一次离散 Top-K 和一次标准 refit，无教师搜索。 |

Yeh 和统一均匀贪心仍保留为附加控制，但不属于当前六方法主表。需要时可用
`--method-set all` 单独输出附录，不能与主表的方法数混写。

## 2. Kang 与 Luo 的统一复查

两种方法在简单曲线上可能用很少节点达到 `1e-4`，而在复杂曲线上明显失效，
这并不自动表示计时或 MSE 口径错误；应检查每个算法阶段保存的诊断量。

### Kang

旧实现曾把每个活跃连续簇无条件压成一个节点，这会让长簇在复杂曲线上丢失
必要自由度。当前实现已按论文 Algorithm 4/5 的思路修正：检查簇边界对，
使用单/双节点误差判据，并用局部最小二乘误差缩窄位置区间；重复节点按
重数保留，不因诊断去重而从最终拟合中消失。

即使修正后，复杂曲线仍可能出现“稠密 ADMM 解可行、聚类重定位后不可行”。
这是稀疏支撑压缩阶段的算法瓶颈，不能靠提高最终 refit 精度掩盖。正式预算
统一为 `ADMM=1000`、`lambda bisections=10`、`relocation=12`。

论文 Remark 3.2.1 建议：明显成簇的样条采样使用 Algorithm 4，一般数据优先
Algorithm 1。当前统一基线固定为 Algorithm 4/5 adaptation，适合本项目的合成
B 样条，但对 UJI/地理曲线不是完整的 Algorithm 1 复现；主表必须披露这一边界，
后续应把 Algorithm 1/4 选择作为 Kang 专项敏感性附录，不能把当前真实复杂曲线
失败概括成论文方法的普遍上限。

### Luo

当前实现只从完整三点窗口中检测导数跳跃的内部局部峰值，边界不构成完整
窗口，因此不作为峰值；这与算法定义一致。应同时记录：dense 初始 MSE、
sparse-stage MSE、局部峰值候选 refit MSE 以及 DE 后 MSE。

复杂曲线常见失效链路是：稠密或稀疏阶段尚可行，但局部峰值压缩后候选集合
遗漏关键位置；后续 DE 只能移动固定数量的节点，不能恢复已丢失的节点数。
此外 DE 的原生适应度是最大采样欧氏误差，而公共表格报告平均平方欧氏 MSE；
两者都要保存，不能把一个数直接换单位冒充另一个。正式预算统一为
`eta=0.5`、`population=20`、`iterations=100`。

因此正式结论必须来自同一批 `K=4..56` 合成曲线和三个真实 test split 的
统计结果，不能只挑简单或复杂个例，也不能为 Kang/Luo 单独提高容量或调阈值。

## 3. 公平协议

1. **配对数据**：六种方法处理完全相同的曲线。合成测试按源 `K=4..56`
   分层，并显式使用 `knot_min_span=0.01`；真实测试使用 UJI Pen、Natural
   Earth 海岸线和 USGS 等高线的 test split。旧 0.02 默认只兼容历史数据。
2. **容量**：Ours 候选容量、Kang 密集初始容量、Liang 密集容量和公共数值
   上限都为 56 个内部节点。对三次开放样条，这对应完整节点向量最多 64 项、
   控制顶点最多 60 个。
   `source K=56` 是生成复杂度边界，`Kc=56` 是方法容量边界；该层没有冗余
   候选余量，必须单独报告 dense/deployment pass。
3. **参数域**：传统方法使用弦长参数；Ours 学习弦长残差并在选集上再次更新。
   主表比较完整方法能力；固定参数消融应另表报告。
4. **统一 refit**：最终都使用端点约束、无平滑项、无 ridge、CPU `float64`
   三次开放 B 样条最小二乘。表格误差不使用网络 surrogate、ADMM 内部目标
   或 DE 适应度。
5. **统一误差**：`MSE=mean_i ||C(t_i)-Q_i||_2^2`，不取平方根，不除以维数；
   通过条件固定为 `MSE<=1e-4`。
6. **节点数**：报告最终 refit 实际使用的内部节点数。只有达到阈值的少节点解
   才能被称为更简洁。
7. **时间**：公共 `total_ms` 从归一化点开始，包含方法本身和最终 refit，
   不含文件 I/O 与绘图。Ours 的 `network_ms` 另列，不替代端到端时间。
8. **失败样本**：异常、非有限解和超阈值解均保留在通过率分母中。
9. **资格**：Ours checkpoint 必须通过 worst-source 90% / `1e-4` 审计；
   proposal 或 diagnostic checkpoint 不得进入正式表格。

四项主指标固定为平均/P95 MSE、阈值通过率、最终内部节点数和完整方法时间。
报告还应按数据来源分组，防止简单合成样本掩盖真实或复杂曲线失败。

## 4. 快速联调

以下命令仅验证六方法链路。容量仍统一为 56，但数值方法迭代预算被显著压缩，
结果只能标作 diagnostic，不能用于论文结论。

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk.pt `
  --output-dir outputs/comparisons/v16_mse1e-4_k56_six_quick `
  --method-set published `
  --samples-per-knot-count 1 --min-knot-count 4 --max-knot-count 56 `
  --real-samples-per-dataset 2 `
  --mse-tolerance 1e-4 --max-internal-knots 56 `
  --paper-initial-knots 56 --paper-admm-iterations 100 `
  --paper-lambda-bisections 3 --paper-relocation-iterations 3 `
  --liang-dense-knots 56 --liang-feature-samples 257 `
  --dung-scan-intervals 3 --dung-optimization-iterations 2 `
  --luo-eta 0.5 --luo-de-population 5 --luo-de-iterations 5 `
  --network-warmups 1 --network-repeats 5 `
  --end-to-end-repeats 1 --torch-num-threads 4 --device cuda
```

若 checkpoint 尚未通过资格审计，只能额外使用
`--allow-unqualified-diagnostic`，并保留工具生成的诊断水印。

## 5. 正式六方法比较

以下命令与 3090 一键脚本使用同一协议和算法预算：

```powershell
python scripts/benchmark_v16_datasets.py `
  --checkpoint outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk.pt `
  --output-dir outputs/comparisons/v16_mse1e-4_k56_ordered_highk_six_methods `
  --method-set published `
  --samples-per-knot-count 5 `
  --min-knot-count 4 --max-knot-count 56 `
  --real-samples-per-dataset 20 `
  --manifest UJI=data/splits/uji_pen_v2.jsonl `
  --manifest NaturalEarth=data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl `
  --manifest USGS=data/processed/usgs_contours/large_scale/manifest.jsonl `
  --mse-tolerance 1e-4 `
  --max-internal-knots 56 --gradient-steps 12 `
  --paper-initial-knots 56 --paper-admm-iterations 1000 `
  --paper-lambda-bisections 10 --paper-relocation-iterations 12 `
  --liang-dense-knots 56 --liang-feature-samples 1025 `
  --dung-scan-intervals 10 --dung-optimization-iterations 10 `
  --luo-eta 0.5 --luo-de-population 20 --luo-de-iterations 100 `
  --network-warmups 10 --network-repeats 100 `
  --end-to-end-repeats 3 `
  --torch-num-threads 4 --device cuda

python scripts/plot_v16_method_comparison.py `
  --input outputs/comparisons/v16_mse1e-4_k56_ordered_highk_six_methods/comparison.json `
  --output-dir outputs/figures/v16_mse1e-4_k56_ordered_highk_six_methods/metrics `
  --method-set published --reference --dpi 300
```

传统方法主要运行在 CPU；RTX 3090 主要加速 Ours 的训练和网络前向。完整
配对实验会明显慢于快速联调，中断后只有实验指纹完全相同时才能 `--resume`。

## 6. 结果解释边界

- 不能宣称逐项复现原论文表格；原数据、参数化、误差范数、超参数和硬件并不
  完全相同。
- Liang 始终标为 `feature-integral + IKI adaptation`；Dung 标为
  `serial/simple-knot adaptation`；Luo 标为 `l_inf,1 + DE adaptation`。
- 真实数据同时检查输入点 MSE 和 original-reference MSE；前者通过而后者
  失败，表示重采样网格拟合好但原始几何保真不足。
- 合成 source K 是认证源表示的复杂度；因为允许连续重定位，它不是全局最少
  节点真值。当前训练明确采用 `upper_bound` 语义。
- 旧 K64 和 `Kc=96 / MSE=2.5e-5 / 97%` 比较只能标作历史消融，不能与本轮
  K56 主表合并。这里的“完整节点向量 64 项”是当前 K56 的容量换算，不是旧
  K64 内部候选实验。
- 旧 source `K=4..24` 结果也只能标作历史范围消融；当前主表必须覆盖到 56，
  并且不得因边界层失败而放宽 90% 正式资格。
