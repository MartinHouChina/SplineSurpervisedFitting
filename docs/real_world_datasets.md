# 真实曲线数据集与接入协议

本文列出适合当前“有序点云 → 最少内部节点 B 样条”任务的公开真实数据，并规定许可、指标和数据拆分方式。仓库不再分发原始数据；UJI、Natural Earth 与 USGS 已提供下载/离线解析脚本和统一 manifest，具体命令见 [UJI 接入](uji_pen_integration.md) 与 [Natural Earth/USGS 接入](geospatial_real_world_data.md)。

当前状态必须区分清楚：这三类真实数据已经接入数据层，可以用于外部部署评估；它们仍然没有真实 knot 标签，因此不会自动加入合成数据的 knot 监督训练。若用于无标签微调，只能采用几何拟合或一致性目标。

## 1. 先明确什么可以称为真值

公开数据中几乎不存在同时满足以下两项的数据集：

1. 输入来自真实传感或真实业务几何；
2. 标签是**可证明最简**的 B 样条节点向量。

同一条曲线可以有多套等价或近似等价的 B 样条表示；Boehm 插点还会在不改变曲线的情况下增加节点。因此应区分：

- **原生节点**：CAD 文件保存的建模表示，可能含可移除节点；
- **canonical 节点**：用独立标准 B 样条求解器，在固定误差阈值下重新简化得到的参考表示；
- **无节点标签**：GIS 折线、手写轨迹等只有有序坐标，不应把折线顶点当作节点标签。

只有合成 canonical 数据和经过 canonical 化的 CAD 曲线适合报告节点位置 Precision/Recall/F1。其他真实数据只报告几何误差、阈值通过率、节点数和时间。

## 2. 数据来源

| 数据集 | 内容与格式 | 许可 / 获取 | 节点真值 | 主要用途 |
|---|---|---|---|---|
| **ABC CAD Dataset** | 约一百万个人工设计 CAD B-Rep；STEP/Parasolid，并提供 OBJ、YAML、特征与参数文件。STEP 中可读取 B-spline/NURBS 的 degree、控制点、权重和 knots | NYU 下载页将数据标为 MIT；同时注明模型版权归创建者并链接 Onshape 条款 | 有原生参数曲线；原生 knots 不保证最简，须 canonical 化 | 主 CAD 基准；节点数、节点位置、拟合精度和跨模型泛化 |
| **USGS The National Map 1:24,000 Contours** | 真实地形等高线；Shapefile、FileGDB，通常按 `1×1°` 分块 | 美国政府公共领域，免费且无需账号 | 无 | 等高/等值线上的最少节点拟合、覆盖率和地区外泛化 |
| **NOAA GSHHG** | 全球海岸线；full/high/intermediate/low/crude 五级；ESRI Shapefile 或 native binary | GNU LGPL | 无 | 闭合复杂曲线、多尺度简化，以及与 Douglas–Peucker 的对照 |
| **UJI Pen Characters v2** | 60 位书写者、11640 个字符样本（97 类、每人每类 2 次）；UTF-8 文本逐 stroke 保存有序 `x,y` 点 | CC BY 4.0 | 无 | 真实采样噪声、尖锐转向和跨书写者泛化；官方提供 40/20 writer-disjoint 划分 |
| **OpenStreetMap** | 道路、步道、河流和海岸线；OSM XML 或 PBF | ODbL；需要署名，衍生数据库受相同许可约束 | 无 | 跨城市泛化、非均匀众包采样和长曲线压力测试 |
| **Natural Earth** | `1:10m / 1:50m / 1:110m` 多尺度海岸线等；Shapefile、GeoPackage | Public Domain | 无 | 小体量可视化与快速多尺度 smoke test，不宜作为主精度基准 |
| **CC3D-PSE** | `50k+` CAD/虚拟扫描对，带 line、circle、spline 参数化锐边标注 | 需机构签署许可后申请，不是直接开放下载 | 有参数化 spline 标注，但仍需检查并 canonical 化 | 后续 scan-to-CAD 鲁棒性测试；不作为第一阶段依赖 |

补充的 UCI Character Trajectories 有 2858 条 `x,y,pressure` 轨迹并采用 CC BY 4.0，但只来自一位书写者，且已经 Gaussian 平滑和微分；跨主体评估优先使用 UJI Pen Characters v2。

## 3. 推荐接入顺序

### P0：ABC CAD

这是第一优先级，也是最适合监督节点位置的数据。建议从一个 10k chunk 开始，只保留：

- B-spline edge，而不是 line、circle 或 ellipse；
- 开放、非周期曲线；
- 当前网络第一阶段可直接处理的 degree `3`；
- 非有理曲线，或所有权重近似为 `1` 的 NURBS；
- 没有退化参数区间、自交异常或极短弧长的 edge。

使用 Open Cascade 读取 STEP 后，先按严格 CAD 几何容差反复移除可移除节点，再把结果记为 `canonical_internal_knots`；同时保留 `native_internal_knots`，不能把两者混用。采样点应按等弧长取得，不能只按等参数采样。

### P1：USGS Contours 与 GSHHG

这两组作为不带节点标签的真实外部测试集：

- USGS 检查真实等高线和区域分布变化；
- GSHHG 检查复杂闭合轮廓和多尺度简化。

如果“等距线”实际指几何 offset curve，而不是等高线，可以从 ABC 的真实 CAD 轮廓用 Open Cascade 生成精确偏置线；这属于**真实 CAD 驱动的半合成数据**，论文中不能写成实测数据。

### P2：UJI、OSM 与 Natural Earth

- UJI 用于跨书写者和噪声泛化；预处理时需按元数据处理 UJI 与 UPV 两个采集点不同的坐标比例（100 与 152 ink units/mm）；
- OSM 用于跨城市与长序列压力测试；
- Natural Earth 用于快速展示和程序连通性检查。

### P3：CC3D-PSE

待二维/三维有序曲线适配器稳定后再申请。它能测试扫描伪影下的 CAD 锐边恢复，但访问和再分发受单独协议约束。

## 4. 拆分规则

随机打散曲线段会造成同源泄漏，应采用分组拆分：

| 数据集 | 推荐拆分单元 | 禁止做法 |
|---|---|---|
| ABC | 完整 CAD model；先用归一化几何 hash 去重，再按 model 分 train/val/test | 把同一模型的不同 edge 分到不同集合 |
| USGS | 地理 tile、州或地貌区 | 把同一条 contour 裁剪后随机分散 |
| GSHHG | polygon / island / geographic region | 同一海岸线的不同分辨率跨集合出现 |
| UJI | 优先采用官方 40 train / 20 test 的 writer-disjoint split；验证集再从 40 位训练书写者中按 writer 划分 | 随机拆同一书写者的重复字符 |
| OSM | city / region | 随机拆同一道路的相邻 way segment |

正式结果应固定 split manifest，记录 source ID、group ID、下载版本、文件哈希和预处理版本。

## 5. 统一预处理

真实数据适配器统一输出：

```text
points                    # 按曲线方向排序，[M,D]
sample_id
group_id                  # model / tile / writer / region
source_dataset
normalization_transform   # 可逆的中心与尺度
native_internal_knots     # 可选，仅 CAD
canonical_internal_knots  # 可选，仅完成 canonical 化后提供
```

处理步骤：

1. 提取单条有序 edge 或 `LineString`，去除连续重复点和退化段；
2. 经纬度数据先投影到适当的米制 CRS，不能直接在经纬度上计算欧氏误差；
3. 保存高密度参考曲线，并从中等弧长抽取网络输入，避免用同一批点同时拟合和评价；
4. 用与当前合成数据一致的中心化和尺度归一化，同时保存逆变换；
5. 开曲线直接输入；闭合海岸线应固定可复现的起点和方向，或在模型支持周期 B 样条后单独评价；
6. 任何节点真值必须注明参数域；不同参数域的节点位置不能直接匹配。

## 6. 指标协议

### 有 canonical 节点标签：合成数据与 ABC-canonical

- 匹配后的 knot Precision / Recall / F1 和 MAE；
- 预测节点数与 canonical 节点数的差值和 MAE；
- 标准 B 样条 refit MSE、P95 距离和最大距离；
- 阈值通过率、部署时间。

### 无节点标签：USGS、GSHHG、UJI、OSM

- 独立高密度参考点上的 MSE；
- 对称 Chamfer、Hausdorff 或 P95 几何距离；
- 阈值通过率与保留节点数；
- 复杂度—误差 Pareto 曲线；
- 网络、verified 和 Greedy 在**相同计时边界**下的延迟。

这些数据上可以用独立 Greedy/beam 求解器构造“oracle reference”，但应称为算法参考解，不能称为真实节点标签。

## 7. 官方来源

- [ABC 论文与数据说明](https://openaccess.thecvf.com/content_CVPR_2019/html/Koch_ABC_A_Big_CAD_Model_Dataset_for_Geometric_Deep_Learning_CVPR_2019_paper.html)
- [ABC Chunk 0000：下载、格式与权利说明](https://archive.nyu.edu/handle/2451/44309)
- [USGS Contours：产品格式](https://www.usgs.gov/faqs/what-types-elevation-datasets-are-available-what-formats-do-they-come-and-where-can-i-download)
- [USGS The National Map：许可](https://www.usgs.gov/faqs/what-are-terms-uselicensing-map-services-and-data-national-map?page=1)
- [NOAA GSHHG：格式、分辨率与许可](https://www.ngdc.noaa.gov/mgg/shorelines/shorelines.html)
- [UJI Pen Characters v2](https://archive.ics.uci.edu/dataset/177/uji+pen+characters+version+2)
- [OpenStreetMap：许可](https://www.openstreetmap.org/copyright)
- [OpenStreetMap Planet：XML/PBF 格式](https://wiki.openstreetmap.org/wiki/Planet.osm)
- [Natural Earth Physical Vectors](https://www.naturalearthdata.com/downloads/10m-physical-vectors/)
- [Natural Earth：Public Domain 声明](https://www.naturalearthdata.com/about/terms-of-use/)
- [CC3D-PSE：数据内容与申请方式](https://cvi2.uni.lu/cc3d-pse/)
