# UJI Pen Characters v2 接入

## 状态

UJI Pen Characters v2 已接入统一真实曲线 manifest。官方完整文件的本地校验结果为：

| 项目 | 数量 |
|---|---:|
| 原始字符样本 | 11,640 |
| writer | 60 |
| 保留的连续 pen-down stroke | 18,219 |
| 去重后不足 4 点而跳过的 stroke | 394 |
| train / val / test writer | 32 / 8 / 20 |
| train / val / test 曲线 | 9,670 / 2,456 / 6,093 |

这些数量来自预处理版本 `uji-stroke-curves-v1`、划分种子 `20260904` 和
`minimum_unique_points=4`。改变这三个配置后必须生成新的 manifest，不能混用结果。

## 一键下载与预处理

```powershell
python scripts/prepare_uji_pen.py --download
```

已有官方文本时无需联网：

```powershell
python scripts/prepare_uji_pen.py `
  --source D:\datasets\UJI\ujipenchars2.txt
```

默认输出：

- 原始文件：`data/raw/uji_pen_v2/`；
- 每个连续 stroke 的毫米坐标：`data/processed/uji_pen_v2/curves/*.npy`；
- 曲线清单：`data/splits/uji_pen_v2.jsonl`；
- writer 划分、哈希和跳过原因：`data/splits/uji_pen_v2_split.json`。

原始数据来自 [UCI dataset 177](https://archive.ics.uci.edu/dataset/177/uji%2Bpen%2Bcharacters%2Bversion%2B2)，
许可为 CC BY 4.0。论文和发布结果应引用 UCI 页面给出的数据集 DOI
`10.24432/C5FG8S`。

## 曲线定义与预处理

1. 一个字符可能包含 1–5 个 stroke。每个 pen-down stroke 单独作为一条开曲线；
   pen-up 期间没有轨迹记录，因此禁止把多个 stroke 首尾相连。
2. 删除采集设备产生的连续重复坐标，但保留稍后再次经过同一位置的点。
3. UJI 坐标除以 `100`、UPV 坐标除以 `152`，统一为毫米；不改变原始
   `x-right/y-down` 方向。
4. 去重后不足 4 个点的点状或退化 stroke 不进入三次 B 样条实验，原因写入
   split summary。
5. 网络读取时再按弦长重采样到 checkpoint 需要的点数，并采用训练一致的中心化和
   尺度归一化；原始密度点始终保留用于独立几何误差计算。

## Writer-disjoint 划分

UCI 官方已将 60 位 writer 分为 40 位 `trn` 和 20 位 `tst`。本项目完整保留官方
20 位 `tst` writer 作为 test，再从 40 位 `trn` writer 中以稳定 SHA-256 排序选出
8 位作为 val，剩余 32 位作为 train。同一 writer 的全部字符、两次重复和所有
stroke 只能出现在同一个 split。

## 在代码中读取

```python
from spline_fitting.data import RealWorldCurveDataset

test_set = RealWorldCurveDataset(
    "data/splits/uji_pen_v2.jsonl",
    split="test",
    num_points=192,
)
sample = test_set[0]
model_points = sample["points"]                 # [192, 2]，已归一化
reference_points = test_set.load_reference_points(0)  # 原始密度、毫米坐标
metadata = test_set.record(0)
```

单条部署也可直接使用 manifest 中的 `.npy`：

```powershell
python scripts/fit_point_cloud.py `
  --checkpoint outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt `
  --point-cloud data/processed/uji_pen_v2/curves/uji2_02134_stroke_00.npy
```

## 标签与可报告指标

UJI 只有真实有序轨迹，没有 B 样条节点真值。所有记录均显式写入
`has_knot_labels=false`；折线顶点不会被当成内部节点标签。因此：

- 不能在 UJI 上报告 knot Precision / Recall / F1；
- 当前有监督 knot/KeepMask 训练不能直接把 UJI 当作带标签训练集；
- 可用于无标签域适配或外部部署测试，报告原始密度参考点上的 MSE、P95/最大距离、
  阈值通过率、最终节点数和 network-only 时间；
- Greedy/beam 得到的节点组合只能称作算法参考或离线教师，不能称作真实节点标签。

## 首次链路 smoke test（不是正式统计）

使用现有 `candidate_pruning_one_shot_v14_feedback.pt` 的 learned 模式实际运行两个
官方 test stroke，文件读取、192 点重采样、网络前向和标准 B 样条 refit 均已跑通。
29 点样例的 normalized MSE 为 `2.896e-5`（RMS `0.00538`，保留 20 个节点），
221 点样例为 `4.553e-5`（RMS `0.00675`，保留全部 28 个节点）。两者均未满足
RMS `0.005` 阈值；这只是两条曲线，不能代替完整 test 统计，但已经表明合成数据到
真实手写轨迹存在明显域差异，后续需要真实数据适配或 verified 质量守卫，不能直接
宣称当前 learned 模型具有良好的真实数据泛化。对 221 点样例启用 verified 后，
通过 chord fallback 和 2 次 residual-guided insertion 将 K 从 28 增至 30，MSE 降至
`1.939e-5`（RMS `0.00440`）并通过阈值，但 CPU 后处理约 `380 ms`；它是质量守卫，
不属于一次性网络的延迟结果。

通用 JSONL 字段和其他真实数据源遵循同一读取接口，见
[`real_world.py`](../src/spline_fitting/data/real_world.py)。
