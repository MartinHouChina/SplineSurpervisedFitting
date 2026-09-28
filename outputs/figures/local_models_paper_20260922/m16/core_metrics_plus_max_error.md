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

Synthetic includes 8 source-K > M16 capacity stress curves; the all-case mean is not a capacity-matched score.

| Source | Method | Finite / run | Pass % | Mean K | Mean MSE | MaxSE (all) | Total ms | Net ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic | Ours | 21/21 | 38.1 | 15.43 | 1.63e-03 | 1.97e-02 | 68.32 | 51.28 |
| Synthetic | Park & Lee 2007 | 21/21 | 4.8 | 16.00 | 1.30e-03 | 1.99e-02 | 59.30 | -- |
| Synthetic | Liang et al. 2017 | 21/21 | 14.3 | 15.43 | 1.27e-03 | 1.15e-02 | 46.86 | -- |
| Synthetic | Dung & Tjahjowidodo 2017 | 12/21 | 0.0 | 10.33 | 1.16e-03 | 1.30e-02 | 649.08 | -- |
| Synthetic | Kang 2015 | 21/21 | 0.0 | 1.00 | 4.70e-03 | 8.34e-02 | 312.37 | -- |
| Synthetic | Luo et al. 2022 | 21/21 | 0.0 | 1.95 | 5.74e-03 | 7.23e-02 | 1866.61 | -- |
| UJI | Ours | 10/10 | 90.0 | 13.10 | 1.78e-05 | 1.23e-03 | 73.16 | 50.50 |
| UJI | Park & Lee 2007 | 10/10 | 90.0 | 4.50 | 6.11e-05 | 2.38e-03 | 23.66 | -- |
| UJI | Liang et al. 2017 | 10/10 | 90.0 | 5.70 | 2.69e-05 | 5.64e-04 | 18.10 | -- |
| UJI | Dung & Tjahjowidodo 2017 | 10/10 | 50.0 | 3.00 | 2.63e-04 | 1.04e-02 | 208.66 | -- |
| UJI | Kang 2015 | 10/10 | 50.0 | 0.80 | 4.68e-03 | 2.12e-01 | 1498.98 | -- |
| UJI | Luo et al. 2022 | 10/10 | 40.0 | 1.90 | 5.94e-03 | 1.77e-01 | 5829.84 | -- |
| NaturalEarth | Ours | 10/10 | 10.0 | 15.70 | 1.34e-03 | 4.28e-02 | 71.69 | 52.69 |
| NaturalEarth | Park & Lee 2007 | 10/10 | 30.0 | 14.90 | 1.74e-03 | 8.54e-02 | 70.33 | -- |
| NaturalEarth | Liang et al. 2017 | 10/10 | 30.0 | 14.40 | 6.53e-04 | 1.79e-02 | 54.27 | -- |
| NaturalEarth | Dung & Tjahjowidodo 2017 | 7/10 | 0.0 | 9.00 | 4.46e-04 | 9.20e-03 | 609.87 | -- |
| NaturalEarth | Kang 2015 | 10/10 | 0.0 | 1.40 | 2.18e-02 | 2.78e-01 | 341.83 | -- |
| NaturalEarth | Luo et al. 2022 | 10/10 | 0.0 | 2.40 | 2.00e-02 | 2.40e-01 | 2249.71 | -- |
| USGS | Ours | 10/10 | 50.0 | 15.50 | 3.16e-04 | 8.60e-03 | 75.24 | 50.51 |
| USGS | Park & Lee 2007 | 10/10 | 40.0 | 12.90 | 5.26e-04 | 1.55e-02 | 56.28 | -- |
| USGS | Liang et al. 2017 | 10/10 | 50.0 | 12.40 | 2.32e-04 | 3.66e-03 | 38.05 | -- |
| USGS | Dung & Tjahjowidodo 2017 | 7/10 | 0.0 | 7.57 | 4.21e-04 | 3.42e-03 | 512.34 | -- |
| USGS | Kang 2015 | 10/10 | 0.0 | 1.30 | 7.76e-03 | 1.55e-01 | 852.42 | -- |
| USGS | Luo et al. 2022 | 10/10 | 0.0 | 1.50 | 9.02e-03 | 1.33e-01 | 3261.29 | -- |
| IndustrialOffset | Ours | 10/10 | 80.0 | 14.50 | 1.33e-04 | 4.19e-03 | 74.52 | 51.81 |
| IndustrialOffset | Park & Lee 2007 | 10/10 | 70.0 | 11.80 | 1.62e-04 | 1.09e-02 | 61.93 | -- |
| IndustrialOffset | Liang et al. 2017 | 10/10 | 60.0 | 10.40 | 1.05e-04 | 2.41e-03 | 34.11 | -- |
| IndustrialOffset | Dung & Tjahjowidodo 2017 | 10/10 | 0.0 | 5.70 | 1.61e-03 | 3.10e-02 | 367.54 | -- |
| IndustrialOffset | Kang 2015 | 10/10 | 0.0 | 1.50 | 5.31e-03 | 7.05e-02 | 1138.87 | -- |
| IndustrialOffset | Luo et al. 2022 | 10/10 | 0.0 | 2.00 | 4.80e-03 | 7.41e-02 | 4090.81 | -- |
