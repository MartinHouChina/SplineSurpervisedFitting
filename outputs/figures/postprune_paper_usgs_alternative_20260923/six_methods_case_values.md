# Six-method figure: exact measured values

These are the 30 displayed panels, not aggregate statistics. Errors are normalized squared Euclidean distances; no square root.
Total time includes every measured numerical postprocessing stage. Full records and geometry are in case_panels_manifest.json.

Selection: explicit user-selected illustrative cases for overridden sources; remaining sources use one seeded random paired case per source, independent of all measured outcomes

## Synthetic

Sample: `seed20000_sample2535`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 19 | 3.694184759e-05 | 0.0003674355672 | 1191.3961 | yes |
| Park | 32 | 6.451430836e-05 | 0.0004621361585 | 103.9553 | no |
| Liang | 32 | 5.119231357e-05 | 0.0002049504888 | 92.61059999 | no |
| Dung | 25 | 4.802403318e-05 | 0.0003156323102 | 1683.1417 | yes |
| Kang | 26 | 4.881506161e-05 | 0.0002111764001 | 1292.5981 | yes |
| Luo | 30 | 4.870653576e-05 | 0.000278852965 | 1294.8442 | yes |

## UJI

Sample: `uji2_09247_stroke_01`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 3 | 4.766968723e-05 | 0.0001766144142 | 829.1414 | yes |
| Park | 5 | 4.183215554e-05 | 0.0004150263335 | 18.5098 | yes |
| Liang | 5 | 4.721050282e-05 | 0.0002765751796 | 27.3892 | yes |
| Dung | 4 | 3.096979633e-05 | 0.0001510195611 | 189.7595 | yes |
| Kang | 3 | 4.297067756e-05 | 0.0004696431538 | 3450.8931 | yes |
| Luo | 3 | 4.034638966e-05 | 0.000179163927 | 8508.5125 | yes |

## NaturalEarth

Sample: `natural_earth_10m_coastline_5148fa806eda_w000_5148fa80`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 13 | 4.385148857e-05 | 0.0003806974898 | 1710.5851 | yes |
| Park | 12 | 2.608184796e-05 | 0.0002451352312 | 69.5619 | yes |
| Liang | 13 | 4.235890503e-05 | 0.0004358692751 | 99.82449999 | yes |
| Dung | 14 | 2.191947308e-05 | 0.0003076281302 | 1097.4724 | yes |
| Kang | 13 | 4.018396824e-05 | 0.0004036059661 | 2005.545 | yes |
| Luo | 13 | 3.729006844e-05 | 0.0004266139949 | 8258.4484 | yes |

## USGS

Sample: `usgs_tnm_contours_large_scale_08cddbc994fc_w000_08cddbc9`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 13 | 4.196411706e-05 | 0.000218791473 | 710.5883 | yes |
| Park | 18 | 3.595538286e-05 | 0.0003633575779 | 143.2086 | yes |
| Liang | 16 | 4.330379297e-05 | 0.0002895124289 | 41.6171 | yes |
| Dung | 14 | 4.869953839e-05 | 0.0004607045289 | 700.9399 | yes |
| Kang | 14 | 4.075143546e-05 | 0.0004151655419 | 1815.0588 | yes |
| Luo | 14 | 4.97253507e-05 | 0.0004105669871 | 6488.4167 | yes |

## IndustrialOffset

Sample: `industrial_offset_elliptic_bore_v008_o00_m0p0800`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 5 | 2.84260843e-05 | 0.0001582471563 | 540.7712 | yes |
| Park | 6 | 3.801312747e-05 | 0.0002330729647 | 35.5761 | yes |
| Liang | 6 | 3.819904562e-05 | 0.0001686151579 | 14.50160001 | yes |
| Dung | 6 | 2.718786449e-05 | 0.0001093371137 | 325.5595 | yes |
| Kang | 8 | 1.831276156e-05 | 0.0002521688416 | 3132.5458 | yes |
| Luo | 6 | 4.175557324e-05 | 0.0001591477998 | 6822.1563 | yes |
