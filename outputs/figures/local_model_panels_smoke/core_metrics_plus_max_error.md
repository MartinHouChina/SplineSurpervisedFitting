# Four core metrics plus maximum squared error

Four core metrics: mean MSE, input-MSE pass rate, mean retained internal knots, mean complete method time; plus maximum squared error.

MaxSE (all) is max over all sampled input points of all finite completed curves, NOT maximum per-curve MSE or mean peak error.

Means and maxima include every finite completed fit, including threshold failures. Numerical failures stay in executed pass denominators; counts are shown.

Unavailable methods are not executed. A missing number is --, never a fabricated zero. Non-network methods have no Net measurement.

MSE threshold = 5e-05; errors are normalized squared Euclidean residuals with no square root.

4 measured curves; cpu; single complete run per curve. Total includes full method execution; Net is separately timed network only.

Literature methods are repository adaptations, not verified native reproductions. Synthetic and procedural offsets are not measured real-world data.

Displayed showcase selection has no effect on these aggregate statistics. Full reference-error statistics are retained in JSON.

The source benchmark is marked diagnostic_not_final; this is a local preview, not a final population benchmark.

| Source | Method | Finite / run | Pass % | Mean K | Mean MSE | MaxSE (all) | Total ms | Net ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UJI | Ours | 1/1 | 100.0 | 20.00 | 7.86e-06 | 1.51e-04 | 25.48 | 18.95 |
| UJI | Park & Lee 2007 | 1/1 | 100.0 | 5.00 | 2.70e-05 | 3.10e-04 | 14.93 | -- |
| UJI | Liang et al. 2017 | 1/1 | 100.0 | 4.00 | 3.93e-05 | 8.13e-04 | 9.20 | -- |
| UJI | Dung & Tjahjowidodo 2017 | 1/1 | 0.0 | 3.00 | 1.57e-04 | 2.34e-03 | 190.59 | -- |
| UJI | Kang 2015 | 1/1 | 0.0 | 1.00 | 2.39e-04 | 2.90e-03 | 2548.73 | -- |
| UJI | Luo et al. 2022 | 1/1 | 0.0 | 2.00 | 1.11e-04 | 6.22e-04 | 5989.74 | -- |
| NaturalEarth | Ours | 1/1 | 0.0 | 27.00 | 7.66e-05 | 7.23e-04 | 21.97 | 19.50 |
| NaturalEarth | Park & Lee 2007 | 1/1 | 100.0 | 25.00 | 3.52e-05 | 8.08e-04 | 99.58 | -- |
| NaturalEarth | Liang et al. 2017 | 1/1 | 100.0 | 24.00 | 4.28e-05 | 4.36e-04 | 43.99 | -- |
| NaturalEarth | Dung & Tjahjowidodo 2017 | 1/1 | 0.0 | 9.00 | 5.96e-04 | 5.21e-03 | 525.57 | -- |
| NaturalEarth | Kang 2015 | 1/1 | 0.0 | 1.00 | 2.38e-02 | 1.05e-01 | 1066.73 | -- |
| NaturalEarth | Luo et al. 2022 | 1/1 | 0.0 | 7.00 | 1.97e-03 | 7.23e-03 | 2562.89 | -- |
| USGS | Ours | 1/1 | 100.0 | 28.00 | 2.16e-05 | 2.05e-04 | 21.04 | 20.42 |
| USGS | Park & Lee 2007 | 1/1 | 100.0 | 9.00 | 4.08e-05 | 3.36e-04 | 22.15 | -- |
| USGS | Liang et al. 2017 | 1/1 | 100.0 | 11.00 | 3.39e-05 | 2.40e-04 | 21.12 | -- |
| USGS | Dung & Tjahjowidodo 2017 | 1/1 | 0.0 | 5.00 | 8.05e-04 | 3.06e-03 | 260.62 | -- |
| USGS | Kang 2015 | 1/1 | 0.0 | 2.00 | 3.23e-03 | 8.47e-03 | 2250.81 | -- |
| USGS | Luo et al. 2022 | 1/1 | 0.0 | 5.00 | 3.49e-04 | 1.21e-03 | 5698.74 | -- |
| IndustrialOffset | Ours | 1/1 | 100.0 | 19.00 | 4.22e-06 | 4.90e-05 | 20.06 | 22.71 |
| IndustrialOffset | Park & Lee 2007 | 1/1 | 100.0 | 10.00 | 2.38e-05 | 5.91e-04 | 94.02 | -- |
| IndustrialOffset | Liang et al. 2017 | 1/1 | 100.0 | 7.00 | 2.20e-05 | 4.14e-04 | 17.12 | -- |
| IndustrialOffset | Dung & Tjahjowidodo 2017 | 1/1 | 0.0 | 4.00 | 2.91e-03 | 2.67e-02 | 174.66 | -- |
| IndustrialOffset | Kang 2015 | 1/1 | 0.0 | 1.00 | 6.00e-03 | 4.59e-02 | 2042.36 | -- |
| IndustrialOffset | Luo et al. 2022 | 1/1 | 0.0 | 3.00 | 5.91e-04 | 2.74e-03 | 6400.86 | -- |
