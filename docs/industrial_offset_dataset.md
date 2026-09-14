# 工业模型等距线外部数据集

## 定位

该数据集用于检验模型对工业型线与几何 offset curve 的泛化，不参与 Proposal 或 Joint 的梯度更新。它由参数化工程轮廓驱动，因此应称为 **CAD-driven semi-synthetic（CAD 驱动半合成）数据集**，不能写成实测 CAD 数据。

输出沿用统一接口：

```text
data/processed/industrial_offsets/v1/
├── manifest.jsonl
├── dataset_metadata.json
└── curves/*.npy
```

`RealWorldCurveDataset`、多方法 benchmark 和真实案例绘图脚本可直接读取该 manifest。

## 曲线构成

默认包含七类常见二维工程轮廓：椭圆孔、圆角板、胶囊槽、NACA 翼型、径向凸轮、多叶转子和带键槽孔。每类生成 12 个确定性参数变体，每个基准轮廓尝试六档偏置：

```text
-8%, -4%, -2%, +2%, +4%, +8%
```

百分比以基准轮廓最大尺寸的一半为尺度，负数表示向内偏置，正数表示向外偏置。坐标以毫米保存，网络读取时仍按项目统一规则归一化。

默认配置当前生成 453 条有效曲线、84 个独立基准轮廓组；实际数量由几何有效性检查决定。

## 几何有效性

- 输入轮廓统一为逆时针闭合有序点列；
- 普通拐角采用 miter join，过长尖角按 `miter_limit=4` 改用 bevel；
- 删除连续重复点并拒绝零长度边、非有限坐标和近零面积轮廓；
- 拒绝 offset 自交；
- 拒绝内偏置越过中轴线、轮廓坍缩或方向翻转；
- 每种拒绝原因和数量均写入 `dataset_metadata.json`，不静默修补或任选一个环。

上述策略优先保证评测几何明确。被拒绝样本不能算作模型失败，也不能放回测试集。

## 防止数据泄漏

同一基准轮廓的所有偏置距离共享一个 `group_id`，通过确定性哈希整体进入 train、val 或 test；不会让同一零件轮廓的相邻偏置同时出现在验证与测试中。当前 v16 只读取其 val/test 几何，不使用其中的 train split 训练网络。

该数据集没有 B 样条节点真值，`has_knot_labels=false`。只报告：

1. 稠密参考点 MSE；
2. `MSE <= epsilon` 通过率；
3. 最终内部节点数；
4. 推理或完整方法时间。

不得在该数据集上报告 knot Precision、Recall 或 F1。

## 生成命令

Linux：

```bash
python scripts/prepare_industrial_offsets.py \
  --output-dir data/processed/industrial_offsets/v1 \
  --variants-per-family 12 \
  --source-points 384 \
  --reference-points 768
```

快速链路检查：

```bash
python scripts/prepare_industrial_offsets.py \
  --output-dir data/processed/industrial_offsets/smoke \
  --family rounded_plate \
  --family keyway_bore \
  --variants-per-family 2 \
  --source-points 96 \
  --reference-points 192
```

若目标目录已存在，脚本会拒绝覆盖；只有明确传入 `--overwrite` 才会重写 manifest 和同名曲线。

## 加入对比实验

```bash
python scripts/benchmark_v16_datasets.py \
  --checkpoint outputs/checkpoints/<run>.pt \
  --output-dir outputs/comparisons/<run>/industrial_offset \
  --method-set published \
  --skip-synthetic \
  --manifest IndustrialOffset=data/processed/industrial_offsets/v1/manifest.jsonl \
  --real-samples-per-dataset 50 \
  --mse-tolerance 1e-4 \
  --device cuda
```

生成六方法案例图：

```bash
python scripts/visualize_v16_real_deployments.py \
  --checkpoint outputs/checkpoints/<run>.pt \
  --output-dir outputs/figures/<run>/industrial_offset_cases \
  --manifest IndustrialOffset=data/processed/industrial_offsets/v1/manifest.jsonl \
  --real-samples-per-dataset 8 \
  --mse-tolerance 1e-4 \
  --device cuda
```
