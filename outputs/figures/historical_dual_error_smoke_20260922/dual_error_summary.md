# Raw methods versus common dual-error insertion

Dual pass requires input MSE ≤ 5e-05 AND maximum point squared Euclidean residual ≤ 0.0005; neither quantity is square-rooted.

MaxSE is the maximum over all points of all finite fits in the group, not maximum per-curve MSE and not average peak error.

Every requested case, including numerical failures and unavailable results, remains in the pass denominator. Error/count means include all finite fits, even threshold misses; finite counts are shown.

Arrows compare the saved raw method against that same method followed by common residual-guided knot insertion. All five literature methods are repository adaptations, not verified native reproductions. Repaired Ours is a hybrid, not one-shot deployment.

Total time includes original method and common repair; repair time is listed separately. Raw timings may be cached and repair timings freshly measured. Network and numerical methods can use different CUDA/CPU scopes; these are exploratory local measurements, not a controlled hardware-speed claim.

All-case aggregate is curve-weighted, not equal-source-weighted. Synthetic and procedural industrial offsets are not measured real-world data. Contact-sheet selections do not affect statistics.

Declared unified internal-knot budget: 32. A budget-exhausted case is still a failure; no threshold is silently relaxed.

| Source | Method | N / finite raw→final | Dual pass % raw→final | Mean K raw→final | Mean MSE raw→final | MaxSE raw→final | Mean total ms | Mean repair ms | Mean added K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | Ours V16 | 5 / 5→5 | 100.0 → 100.0 | 22.00 → 22.00 | 8.86e-06 → 8.86e-06 | 3.68e-04 → 3.68e-04 | 24.71 | 2.26 | 0.00 |
| All | Park (adaptation) | 5 / 5→5 | 80.0 → 80.0 | 13.40 → 13.40 | 4.19e-05 → 4.19e-05 | 4.62e-04 → 4.62e-04 | 84.03 | 1.82 | 0.00 |
| All | Liang (adaptation) | 5 / 5→5 | 80.0 → 80.0 | 13.20 → 13.20 | 4.34e-05 → 4.34e-05 | 4.36e-04 → 4.36e-04 | 71.96 | 1.98 | 0.00 |
| All | Dung (adaptation) | 5 / 5→5 | 0.0 → 100.0 | 6.80 → 11.60 | 4.05e-04 → 3.14e-05 | 6.46e-03 → 3.16e-04 | 781.06 | 290.47 | 4.80 |
| All | Kang (adaptation) | 5 / 5→5 | 0.0 → 100.0 | 1.40 → 11.80 | 2.20e-03 → 3.84e-05 | 4.70e-02 → 4.70e-04 | 2887.37 | 546.81 | 10.40 |
| All | Luo (adaptation) | 5 / 5→5 | 20.0 → 100.0 | 2.40 → 12.00 | 3.22e-03 → 4.33e-05 | 3.91e-02 → 4.27e-04 | 6711.11 | 591.64 | 9.60 |
| Synthetic | Ours V16 | 1 / 1→1 | 100.0 → 100.0 | 29.00 → 29.00 | 2.02e-05 → 2.02e-05 | 1.47e-04 → 1.47e-04 | 23.28 | 2.24 | 0.00 |
| Synthetic | Park (adaptation) | 1 / 1→1 | 0.0 → 0.0 | 32.00 → 32.00 | 6.45e-05 → 6.45e-05 | 4.62e-04 → 4.62e-04 | 201.92 | 2.04 | 0.00 |
| Synthetic | Liang (adaptation) | 1 / 1→1 | 0.0 → 0.0 | 32.00 → 32.00 | 5.12e-05 → 5.12e-05 | 2.05e-04 → 2.05e-04 | 125.20 | 1.61 | 0.00 |
| Synthetic | Dung (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 15.00 → 25.00 | 4.47e-04 → 4.80e-05 | 6.46e-03 → 3.16e-04 | 1644.26 | 560.00 | 10.00 |
| Synthetic | Kang (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 1.00 → 26.00 | 5.82e-03 → 4.88e-05 | 4.70e-02 → 2.11e-04 | 1412.44 | 1353.11 | 25.00 |
| Synthetic | Luo (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 0.00 → 30.00 | 7.90e-03 → 4.87e-05 | 3.91e-02 → 2.79e-04 | 2107.82 | 2099.00 | 30.00 |
| UJI | Ours V16 | 1 / 1→1 | 100.0 → 100.0 | 21.00 → 21.00 | 6.21e-06 → 6.21e-06 | 1.77e-04 → 1.77e-04 | 19.72 | 2.37 | 0.00 |
| UJI | Park (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 5.00 → 5.00 | 4.18e-05 → 4.18e-05 | 4.15e-04 → 4.15e-04 | 22.21 | 1.09 | 0.00 |
| UJI | Liang (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 5.00 → 5.00 | 4.72e-05 → 4.72e-05 | 2.77e-04 → 2.77e-04 | 11.82 | 1.24 | 0.00 |
| UJI | Dung (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 3.00 → 4.00 | 5.36e-05 → 3.10e-05 | 9.31e-04 → 1.51e-04 | 317.27 | 121.06 | 1.00 |
| UJI | Kang (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 1.00 → 3.00 | 1.84e-04 → 4.30e-05 | 1.97e-03 → 4.70e-04 | 4109.56 | 114.96 | 2.00 |
| UJI | Luo (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 3.00 → 3.00 | 4.03e-05 → 4.03e-05 | 1.79e-04 → 1.79e-04 | 8100.58 | 1.61 | 0.00 |
| NaturalEarth | Ours V16 | 1 / 1→1 | 100.0 → 100.0 | 24.00 → 24.00 | 1.44e-05 → 1.44e-05 | 3.68e-04 → 3.68e-04 | 21.88 | 2.13 | 0.00 |
| NaturalEarth | Park (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 12.00 → 12.00 | 2.61e-05 → 2.61e-05 | 2.45e-04 → 2.45e-04 | 84.04 | 1.71 | 0.00 |
| NaturalEarth | Liang (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 13.00 → 13.00 | 4.24e-05 → 4.24e-05 | 4.36e-04 → 4.36e-04 | 56.90 | 3.21 | 0.00 |
| NaturalEarth | Dung (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 6.00 → 14.00 | 2.76e-04 → 2.19e-05 | 3.30e-03 → 3.08e-04 | 834.69 | 381.38 | 8.00 |
| NaturalEarth | Kang (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 1.00 → 13.00 | 4.06e-03 → 4.02e-05 | 2.84e-02 → 4.04e-04 | 2077.96 | 673.71 | 12.00 |
| NaturalEarth | Luo (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 2.00 → 13.00 | 3.99e-03 → 3.73e-05 | 1.97e-02 → 4.27e-04 | 7292.97 | 545.38 | 11.00 |
| USGS | Ours V16 | 1 / 1→1 | 100.0 → 100.0 | 20.00 → 20.00 | 3.11e-06 → 3.11e-06 | 1.75e-05 → 1.75e-05 | 18.28 | 1.65 | 0.00 |
| USGS | Park (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 12.00 → 12.00 | 3.90e-05 → 3.90e-05 | 3.87e-04 → 3.87e-04 | 68.61 | 2.18 | 0.00 |
| USGS | Liang (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 10.00 → 10.00 | 3.78e-05 → 3.78e-05 | 2.13e-04 → 2.13e-04 | 74.47 | 2.04 | 0.00 |
| USGS | Dung (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 6.00 → 9.00 | 4.38e-04 → 2.91e-05 | 3.30e-03 → 2.36e-04 | 629.67 | 264.95 | 3.00 |
| USGS | Kang (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 2.00 → 9.00 | 2.65e-04 → 4.17e-05 | 1.34e-03 → 2.62e-04 | 3030.59 | 287.86 | 7.00 |
| USGS | Luo (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 6.00 → 8.00 | 9.27e-05 → 4.84e-05 | 3.19e-04 → 3.42e-04 | 7114.52 | 88.33 | 2.00 |
| IndustrialOffset | Ours V16 | 1 / 1→1 | 100.0 → 100.0 | 16.00 → 16.00 | 3.99e-07 → 3.99e-07 | 5.02e-06 → 5.02e-06 | 40.37 | 2.92 | 0.00 |
| IndustrialOffset | Park (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 6.00 → 6.00 | 3.80e-05 → 3.80e-05 | 2.33e-04 → 2.33e-04 | 43.35 | 2.06 | 0.00 |
| IndustrialOffset | Liang (adaptation) | 1 / 1→1 | 100.0 → 100.0 | 6.00 → 6.00 | 3.82e-05 → 3.82e-05 | 1.69e-04 → 1.69e-04 | 91.39 | 1.79 | 0.00 |
| IndustrialOffset | Dung (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 4.00 → 6.00 | 8.09e-04 → 2.72e-05 | 2.90e-03 → 1.09e-04 | 479.39 | 124.97 | 2.00 |
| IndustrialOffset | Kang (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 2.00 → 8.00 | 6.93e-04 → 1.83e-05 | 3.12e-03 → 2.52e-04 | 3806.32 | 304.43 | 6.00 |
| IndustrialOffset | Luo (adaptation) | 1 / 1→1 | 0.0 → 100.0 | 1.00 → 6.00 | 4.08e-03 → 4.18e-05 | 1.00e-02 → 1.59e-04 | 8939.64 | 223.88 | 5.00 |
