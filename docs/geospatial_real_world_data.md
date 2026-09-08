# Natural Earth 与 USGS 等高线接入

这两个数据集现在已有可执行的数据准备链路，而不再只是候选数据源。它们都输出统一的 `manifest.jsonl + curves/*.npy`，可由 `RealWorldCurveDataset` 直接读取。原始数据和生成物默认写入 `data/`，不随仓库再分发。

## 1. 输出约定

每行 manifest 至少包含：

```text
sample_id, source_dataset, group_id, split
points_path, num_points, point_dim
has_knot_labels=false, metadata
```

`points_path` 指向一条高密度参考折线。坐标不是经纬度，而是每条曲线自己的局部米制坐标。加载时再等弧长重采样为网络要求的点数并做与合成训练一致的中心化和尺度归一化：

```python
from spline_fitting.data import RealWorldCurveDataset

dataset = RealWorldCurveDataset(
    "data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl",
    split="test",
    num_points=192,
    normalize=True,
)
sample = dataset[0]
points = sample["points"]       # [192, 2]，可直接给网络
reference = dataset.load_reference_points(0)  # 高密度米制参考线
```

这些 GIS 折线没有 B 样条节点真值。不能把原折线顶点当成 knot，也不能在其上报告 knot Precision/Recall/F1；只报告高密参考上的 MSE/P95/Hausdorff、阈值通过率、最终节点数和时间。

## 2. Natural Earth

默认下载固定 tag `v5.1.2` 的 GeoJSON，因此 URL 本身固定；文件 SHA-256 也会写入数据集元数据。已支持 `coastline`、`rivers_lake_centerlines` 和 `geographic_lines`，以及 `10m/50m/110m` 三种分辨率。这里的 `10m` 是 Natural Earth 对 1:10,000,000 比例尺的命名，不是 10 米空间分辨率。

快速准备 10m 海岸线：

```powershell
python scripts/prepare_natural_earth.py `
  --resolution 10m `
  --layer coastline `
  --reference-points 768
```

先做一个小型连通性测试：

```powershell
python scripts/prepare_natural_earth.py `
  --resolution 110m `
  --layer coastline `
  --max-samples 20 `
  --output-dir data/processed/natural_earth/smoke_110m
```

已有 GeoJSON、Shapefile 或 ZIP 时无需联网：

```powershell
python scripts/prepare_natural_earth.py `
  --input D:/datasets/ne_10m_coastline.geojson `
  --output-dir data/processed/natural_earth/local_10m
```

GeoJSON 无额外依赖；Shapefile/ZIP 需要可选包 `pyshp`，GeoPackage/FileGDB 需要可选的 `geopandas` 环境。

## 3. USGS The National Map 等高线

在线路径使用 USGS 官方 `contours/FeatureServer/5`，即 1:24,000 large-scale contour lines。脚本只允许显式提供小区域，不会隐式请求全美国。它先查询相交对象 ID，再按 ID 排序/确定性采样并分页下载；原始响应缓存后重复运行不会再次访问网络。

使用仓库内的三个示例区域：

```powershell
python scripts/prepare_usgs_contours.py `
  --bbox-file configs/usgs_contour_regions.example.json `
  --max-features-per-region 2000 `
  --output-dir data/processed/usgs_contours/large_scale
```

只取一个小区域：

```powershell
python scripts/prepare_usgs_contours.py `
  --bbox -105.45 39.90 -105.25 40.05 `
  --region-id colorado_front_range `
  --max-features-per-region 2000
```

也可以直接读取从 The National Map 下载器取得的本地文件：

```powershell
python scripts/prepare_usgs_contours.py `
  --input D:/datasets/USGS_contours.geojson `
  --output-dir data/processed/usgs_contours/local_snapshot
```

USGS 在线服务会更新，所以严格复现实验应保留：

- `data/raw/usgs_contours/*.json`：原始查询快照；
- `source_snapshots.json`：每个输入快照的路径、URL 和 SHA-256；
- `dataset_metadata.json`：预处理版本和整体设置；
- `manifest.jsonl`：固定样本和固定 split。

## 4. 几何处理与防泄漏

统一流程如下：

1. 提取 `LineString/MultiLineString` 的每个有序 part；
2. 删除非法点和相邻重复点，并固定曲线方向；
3. 对过长 part 做确定性窗口切分，反向重复曲线按几何 hash 去重；
4. 从 WGS84 经纬度投影到逐样本的球面局部等距方位坐标，单位为米；
5. 在投影后折线上按弦长生成高密参考点；
6. 按曲线中心所在的地理 tile 生成 `group_id`；
7. 对整个 tile 做稳定 SHA-256 分组划分，而不是随机打散曲线。

Natural Earth 默认使用 `5°` tile，USGS 默认使用 `0.25°` tile；USGS 脚本使用固定 split seed `2`，使仓库附带的三地区示例同时具有 train/val/test 地理组。相同 `group_id` 的样本必然进入同一 split，因此不会把同一区域内的相邻曲线随机泄漏到训练与测试两侧。若只准备一个很小的 bbox，某些 split 为空是合理现象，此时应增加地理区域，而不是随机拆分同一 tile。

## 5. 当前验证状态

接入代码已经使用真实官方端点运行。Natural Earth `v5.1.2` 10m coastline 得到 3,908 条曲线、666 个 tile，train/val/test 为 2,625/750/533；USGS layer 5 的 Colorado、Tahoe 和 North Carolina 三个区域得到 820 条曲线、12 个 tile，train/val/test 为 427/192/201。另有 110m coastline 与 Colorado 小 bbox 连通性烟测。自动化测试覆盖离线 GeoJSON/ArcGIS JSON、跨日期变更线、反向去重、分组 split 和统一 loader。

官方来源：

- Natural Earth physical vectors: https://www.naturalearthdata.com/downloads/10m-physical-vectors/
- Natural Earth public-domain terms: https://www.naturalearthdata.com/about/terms-of-use/
- USGS contour FeatureServer: https://cartowfs.nationalmap.gov/arcgis/rest/services/contours/FeatureServer
- USGS elevation product formats: https://www.usgs.gov/faqs/what-types-elevation-datasets-are-available-what-formats-do-they-come-and-where-can-i-download
