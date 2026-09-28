# Six-method figure: exact measured values

These are the 30 displayed panels, not aggregate statistics. Errors are normalized squared Euclidean distances; no square root.
Total time includes every measured numerical postprocessing stage. Full records and geometry are in case_panels_manifest.json.

Selection: explicit user-selected illustrative cases for overridden sources; remaining sources use one seeded random paired case per source, independent of all measured outcomes

## Synthetic

Sample: `2`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 5 | 2e-05 | 0.0001 | 7 | yes |
| Park | 5 | 2e-05 | 0.0001 | 7 | yes |
| Liang | 5 | 2e-05 | 0.0001 | 7 | yes |
| Dung | 5 | 2e-05 | 0.0001 | 7 | yes |
| Kang | 5 | 2e-05 | 0.0001 | 7 | yes |
| Luo | 5 | 2e-05 | 0.0001 | 7 | yes |

## UJI

Sample: `4`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 5 | 4e-05 | 0.0002 | 7 | yes |
| Park | 5 | 4e-05 | 0.0002 | 7 | yes |
| Liang | 5 | 4e-05 | 0.0002 | 7 | yes |
| Dung | 5 | 4e-05 | 0.0002 | 7 | yes |
| Kang | 5 | 4e-05 | 0.0002 | 7 | yes |
| Luo | 5 | 4e-05 | 0.0002 | 7 | yes |

## NaturalEarth

Sample: `0`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 5 | 9e-05 | 0.0009 | 7 | no |
| Park | 5 | 9e-05 | 0.0009 | 7 | no |
| Liang | 5 | 9e-05 | 0.0009 | 7 | no |
| Dung | 5 | 9e-05 | 0.0009 | 7 | no |
| Kang | 5 | 9e-05 | 0.0009 | 7 | no |
| Luo | 5 | 9e-05 | 0.0009 | 7 | no |

## USGS

Sample: `1`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 5 | 1e-05 | 5e-05 | 7 | yes |
| Park | 5 | 1e-05 | 5e-05 | 7 | yes |
| Liang | 5 | 1e-05 | 5e-05 | 7 | yes |
| Dung | 5 | 1e-05 | 5e-05 | 7 | yes |
| Kang | 5 | 1e-05 | 5e-05 | 7 | yes |
| Luo | 5 | 1e-05 | 5e-05 | 7 | yes |

## IndustrialOffset

Sample: `1`

| Method | Internal knots | MSE | Maximum squared error | Total ms | Both bounds pass |
|---|---:|---:|---:|---:|---|
| Ours | 5 | 1e-05 | 5e-05 | 7 | yes |
| Park | 5 | 1e-05 | 5e-05 | 7 | yes |
| Liang | 5 | 1e-05 | 5e-05 | 7 | yes |
| Dung | 5 | 1e-05 | 5e-05 | 7 | yes |
| Kang | 5 | 1e-05 | 5e-05 | 7 | yes |
| Luo | 5 | 1e-05 | 5e-05 | 7 | yes |
