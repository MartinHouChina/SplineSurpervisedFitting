# Four core metrics plus maximum squared error

Four core metrics: mean MSE, input-MSE pass rate, mean retained internal knots, mean complete method time; plus maximum squared error.

MaxSE (all) is max over all sampled input points of all finite completed curves, NOT maximum per-curve MSE or mean peak error.

Means and maxima include every finite completed fit, including threshold failures. Numerical failures stay in executed pass denominators; counts are shown.

Unavailable methods are not executed. A missing number is --, never a fabricated zero. Non-network methods have no Net measurement.

MSE threshold = 5e-05; errors are normalized squared Euclidean residuals with no square root.

61 measured curves; NVIDIA GeForce GTX 1070; single complete run per curve. Total includes full method execution; Net is separately timed network only.

Literature methods are repository adaptations, not verified native reproductions. Synthetic and procedural offsets are not measured real-world data.

Displayed showcase selection has no effect on these aggregate statistics. Full reference-error statistics are retained in JSON.

The source benchmark is marked diagnostic_not_final; this is a local preview, not a final population benchmark.

| Source | Method | Finite / run | Pass % | Mean K | Mean MSE | MaxSE (all) | Total ms | Net ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic | Ours | 21/21 | 66.7 | 26.14 | 5.64e-05 | 5.95e-03 | 88.00 | 65.16 |
| Synthetic | Park & Lee 2007 | 21/21 | 33.3 | 30.38 | 1.46e-04 | 1.04e-02 | 151.88 | -- |
| Synthetic | Liang et al. 2017 | 21/21 | 38.1 | 27.52 | 1.21e-04 | 3.71e-03 | 129.10 | -- |
| Synthetic | Dung & Tjahjowidodo 2017 | 21/21 | 0.0 | 14.81 | 8.82e-04 | 1.30e-02 | 1301.24 | -- |
| Synthetic | Kang 2015 | 21/21 | 0.0 | 1.00 | 4.71e-03 | 8.36e-02 | 1094.22 | -- |
| Synthetic | Luo et al. 2022 | 21/21 | 0.0 | 3.38 | 6.28e-03 | 9.33e-02 | 3460.53 | -- |
| UJI | Ours | 10/10 | 90.0 | 17.80 | 2.06e-05 | 1.55e-03 | 71.96 | 53.61 |
| UJI | Park & Lee 2007 | 10/10 | 100.0 | 5.70 | 2.93e-05 | 1.48e-03 | 26.03 | -- |
| UJI | Liang et al. 2017 | 10/10 | 100.0 | 5.90 | 2.64e-05 | 7.87e-04 | 30.06 | -- |
| UJI | Dung & Tjahjowidodo 2017 | 10/10 | 50.0 | 3.00 | 2.63e-04 | 1.04e-02 | 192.20 | -- |
| UJI | Kang 2015 | 10/10 | 50.0 | 0.80 | 4.67e-03 | 2.12e-01 | 2541.33 | -- |
| UJI | Luo et al. 2022 | 10/10 | 70.0 | 3.10 | 5.77e-03 | 1.77e-01 | 7326.51 | -- |
| NaturalEarth | Ours | 10/10 | 40.0 | 26.90 | 9.27e-04 | 9.23e-02 | 77.98 | 57.02 |
| NaturalEarth | Park & Lee 2007 | 10/10 | 60.0 | 23.00 | 2.33e-04 | 6.93e-03 | 113.06 | -- |
| NaturalEarth | Liang et al. 2017 | 10/10 | 70.0 | 22.00 | 1.67e-04 | 3.05e-03 | 77.25 | -- |
| NaturalEarth | Dung & Tjahjowidodo 2017 | 9/10 | 0.0 | 13.33 | 5.08e-04 | 9.20e-03 | 832.53 | -- |
| NaturalEarth | Kang 2015 | 10/10 | 0.0 | 1.30 | 2.63e-02 | 2.76e-01 | 1091.50 | -- |
| NaturalEarth | Luo et al. 2022 | 10/10 | 0.0 | 3.60 | 9.01e-03 | 1.28e-01 | 4544.20 | -- |
| USGS | Ours | 10/10 | 50.0 | 26.10 | 7.62e-05 | 1.47e-03 | 80.11 | 63.78 |
| USGS | Park & Lee 2007 | 10/10 | 50.0 | 21.00 | 1.00e-04 | 2.59e-03 | 93.34 | -- |
| USGS | Liang et al. 2017 | 10/10 | 60.0 | 20.40 | 6.23e-05 | 6.93e-04 | 65.80 | -- |
| USGS | Dung & Tjahjowidodo 2017 | 10/10 | 0.0 | 11.80 | 4.02e-04 | 4.29e-03 | 965.05 | -- |
| USGS | Kang 2015 | 10/10 | 0.0 | 1.30 | 7.78e-03 | 1.55e-01 | 1663.90 | -- |
| USGS | Luo et al. 2022 | 10/10 | 10.0 | 3.80 | 2.87e-03 | 4.04e-02 | 5465.57 | -- |
| IndustrialOffset | Ours | 10/10 | 80.0 | 19.60 | 6.61e-05 | 2.16e-03 | 70.93 | 52.73 |
| IndustrialOffset | Park & Lee 2007 | 10/10 | 90.0 | 14.20 | 4.53e-05 | 1.67e-03 | 63.94 | -- |
| IndustrialOffset | Liang et al. 2017 | 10/10 | 100.0 | 11.70 | 3.58e-05 | 4.89e-04 | 31.97 | -- |
| IndustrialOffset | Dung & Tjahjowidodo 2017 | 10/10 | 0.0 | 5.70 | 1.61e-03 | 3.10e-02 | 322.44 | -- |
| IndustrialOffset | Kang 2015 | 10/10 | 0.0 | 1.40 | 5.45e-03 | 7.06e-02 | 2563.99 | -- |
| IndustrialOffset | Luo et al. 2022 | 10/10 | 20.0 | 5.50 | 2.31e-03 | 6.45e-02 | 7338.27 | -- |
