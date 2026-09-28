# Raw methods versus common dual-error insertion

Dual pass requires input MSE ≤ 5e-05 AND maximum point squared Euclidean residual ≤ 0.0005; neither quantity is square-rooted.

MaxSE is the maximum over all points of all finite fits in the group, not maximum per-curve MSE and not average peak error.

Every requested case, including numerical failures and unavailable results, remains in the pass denominator. Error/count means include all finite fits, even threshold misses; finite counts are shown.

Arrows compare the saved raw method against that same method followed by common residual-guided knot insertion. All five literature methods are repository adaptations, not verified native reproductions. Repaired Ours is a hybrid, not one-shot deployment.

Total time includes the measured original method and common repair; repair time is listed separately. The report's timing_protocol metadata is authoritative about acquisition and device scopes. Network and numerical methods can use different CUDA/CPU scopes; these are exploratory local measurements, not a controlled hardware-speed claim.

All-case aggregate is curve-weighted, not equal-source-weighted. Synthetic and procedural industrial offsets are not measured real-world data. Contact-sheet selections do not affect statistics.

Declared unified internal-knot budget: 32. A budget-exhausted case is still a failure; no threshold is silently relaxed.

| Source | Method | N / finite raw→final | Dual pass % raw→final | Mean K raw→final | Mean MSE raw→final | MaxSE raw→final | Mean total ms | Mean repair ms | Mean added K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | Ours V16 | 61 / 61→61 | 59.0 → 78.7 | 23.36 → 23.89 | 8.61e-05 → 5.84e-05 | 1.09e-02 → 2.98e-03 | 57.95 | 33.07 | 0.52 |
| All | Park (adaptation) | 61 / 61→61 | 44.3 → 59.0 | 20.93 → 21.10 | 1.17e-04 → 1.14e-04 | 1.04e-02 → 1.04e-02 | 109.66 | 10.60 | 0.16 |
| All | Liang (adaptation) | 61 / 61→61 | 62.3 → 67.2 | 19.31 → 19.39 | 8.93e-05 → 8.86e-05 | 3.71e-03 → 3.71e-03 | 71.01 | 5.70 | 0.08 |
| All | Dung (adaptation) | 61 / 60→60 | 8.2 → 80.3 | 10.60 → 16.78 | 7.65e-04 → 5.57e-05 | 3.10e-02 → 3.58e-03 | 1042.13 | 330.12 | 6.08 |
| All | Kang (adaptation) | 61 / 61→61 | 8.2 → 77.0 | 1.13 → 18.31 | 8.87e-03 → 6.43e-05 | 2.76e-01 → 3.58e-03 | 2511.07 | 871.08 | 17.18 |
| All | Luo (adaptation) | 61 / 61→61 | 16.4 → 78.7 | 3.79 → 18.30 | 5.43e-03 → 6.95e-05 | 1.77e-01 → 7.33e-03 | 5762.74 | 764.72 | 14.51 |
| Synthetic | Ours V16 | 21 / 21→21 | 52.4 → 76.2 | 25.33 → 25.57 | 6.14e-05 → 5.09e-05 | 1.11e-03 → 1.11e-03 | 39.05 | 16.87 | 0.24 |
| Synthetic | Park (adaptation) | 21 / 21→21 | 23.8 → 28.6 | 30.38 → 30.43 | 1.46e-04 → 1.45e-04 | 1.04e-02 → 1.04e-02 | 142.21 | 4.75 | 0.05 |
| Synthetic | Liang (adaptation) | 21 / 21→21 | 38.1 → 38.1 | 27.52 → 27.52 | 1.21e-04 → 1.21e-04 | 3.71e-03 → 3.71e-03 | 98.43 | 2.13 | 0.00 |
| Synthetic | Dung (adaptation) | 21 / 21→21 | 0.0 → 66.7 | 14.81 → 23.24 | 8.82e-04 → 5.26e-05 | 1.30e-02 → 6.89e-04 | 1433.48 | 434.27 | 8.43 |
| Synthetic | Kang (adaptation) | 21 / 21→21 | 0.0 → 57.1 | 1.00 → 24.43 | 4.71e-03 → 6.11e-05 | 8.36e-02 → 2.00e-03 | 2252.97 | 1191.13 | 23.43 |
| Synthetic | Luo (adaptation) | 21 / 21→21 | 0.0 → 61.9 | 3.38 → 25.24 | 6.28e-03 → 6.88e-05 | 9.33e-02 → 1.85e-03 | 4395.26 | 1148.76 | 21.86 |
| UJI | Ours V16 | 10 / 10→10 | 90.0 → 100.0 | 18.30 → 18.50 | 1.35e-05 → 9.84e-06 | 8.32e-04 → 2.84e-04 | 37.31 | 12.61 | 0.20 |
| UJI | Park (adaptation) | 10 / 10→10 | 80.0 → 100.0 | 5.70 → 6.00 | 2.93e-05 → 2.37e-05 | 1.48e-03 → 4.54e-04 | 50.98 | 14.63 | 0.30 |
| UJI | Liang (adaptation) | 10 / 10→10 | 90.0 → 100.0 | 5.90 → 6.20 | 2.64e-05 → 2.47e-05 | 7.87e-04 → 4.58e-04 | 31.79 | 15.37 | 0.30 |
| UJI | Dung (adaptation) | 10 / 10→10 | 50.0 → 100.0 | 3.00 → 4.30 | 2.63e-04 → 2.59e-05 | 1.04e-02 → 4.70e-04 | 303.30 | 61.88 | 1.30 |
| UJI | Kang (adaptation) | 10 / 10→10 | 50.0 → 100.0 | 0.80 → 5.80 | 4.67e-03 → 3.09e-05 | 2.12e-01 → 4.89e-04 | 2759.92 | 235.45 | 5.00 |
| UJI | Luo (adaptation) | 10 / 10→10 | 70.0 → 100.0 | 3.10 → 6.00 | 5.77e-03 → 2.02e-05 | 1.77e-01 → 3.98e-04 | 7729.01 | 149.75 | 2.90 |
| NaturalEarth | Ours V16 | 10 / 10→10 | 30.0 → 70.0 | 26.60 → 28.30 | 2.72e-04 → 1.66e-04 | 1.09e-02 → 2.98e-03 | 122.50 | 102.23 | 1.70 |
| NaturalEarth | Park (adaptation) | 10 / 10→10 | 30.0 → 60.0 | 23.00 → 23.30 | 2.33e-04 → 2.29e-04 | 6.93e-03 → 6.93e-03 | 153.76 | 22.00 | 0.30 |
| NaturalEarth | Liang (adaptation) | 10 / 10→10 | 50.0 → 70.0 | 22.00 → 22.20 | 1.67e-04 → 1.64e-04 | 3.05e-03 → 3.05e-03 | 74.03 | 10.91 | 0.20 |
| NaturalEarth | Dung (adaptation) | 10 / 9→9 | 0.0 → 70.0 | 13.33 → 20.11 | 5.08e-04 → 1.24e-04 | 9.20e-03 → 3.58e-03 | 1210.32 | 338.01 | 6.10 |
| NaturalEarth | Kang (adaptation) | 10 / 10→10 | 0.0 → 70.0 | 1.30 → 21.30 | 2.63e-02 → 1.42e-04 | 2.76e-01 → 3.58e-03 | 2089.95 | 1017.18 | 20.00 |
| NaturalEarth | Luo (adaptation) | 10 / 10→10 | 0.0 → 70.0 | 3.60 → 21.70 | 9.01e-03 → 1.67e-04 | 1.28e-01 → 7.33e-03 | 5268.40 | 953.08 | 18.10 |
| USGS | Ours V16 | 10 / 10→10 | 50.0 → 50.0 | 25.90 → 26.30 | 6.59e-05 → 5.83e-05 | 1.32e-03 → 1.32e-03 | 43.71 | 24.99 | 0.40 |
| USGS | Park (adaptation) | 10 / 10→10 | 40.0 → 50.0 | 21.00 → 21.10 | 1.00e-04 → 9.88e-05 | 2.59e-03 → 2.59e-03 | 94.07 | 6.55 | 0.10 |
| USGS | Liang (adaptation) | 10 / 10→10 | 60.0 → 60.0 | 20.40 → 20.40 | 6.23e-05 → 6.23e-05 | 6.93e-04 → 6.93e-04 | 78.53 | 1.95 | 0.00 |
| USGS | Dung (adaptation) | 10 / 10→10 | 0.0 → 80.0 | 11.80 → 19.40 | 4.02e-04 → 5.03e-05 | 4.29e-03 → 1.36e-03 | 1245.81 | 483.74 | 7.60 |
| USGS | Kang (adaptation) | 10 / 10→10 | 0.0 → 80.0 | 1.30 → 19.50 | 7.78e-03 → 5.73e-05 | 1.55e-01 → 8.77e-04 | 2490.00 | 985.35 | 18.20 |
| USGS | Luo (adaptation) | 10 / 10→10 | 10.0 → 80.0 | 3.80 → 19.40 | 2.87e-03 → 5.65e-05 | 4.04e-02 → 9.36e-04 | 5259.59 | 846.92 | 15.60 |
| IndustrialOffset | Ours V16 | 10 / 10→10 | 80.0 → 100.0 | 18.50 → 18.90 | 4.44e-05 → 1.53e-05 | 2.25e-03 → 3.08e-04 | 67.98 | 26.49 | 0.40 |
| IndustrialOffset | Park (adaptation) | 10 / 10→10 | 70.0 → 90.0 | 14.20 → 14.40 | 4.53e-05 → 4.15e-05 | 1.67e-03 → 1.67e-03 | 71.45 | 11.48 | 0.20 |
| IndustrialOffset | Liang (adaptation) | 10 / 10→10 | 100.0 → 100.0 | 11.70 → 11.70 | 3.58e-05 → 3.58e-05 | 4.89e-04 → 4.89e-04 | 42.13 | 2.05 | 0.00 |
| IndustrialOffset | Dung (adaptation) | 10 / 10→10 | 0.0 → 100.0 | 5.70 → 10.10 | 1.61e-03 → 3.52e-05 | 3.10e-02 → 4.40e-04 | 587.26 | 218.16 | 4.40 |
| IndustrialOffset | Kang (adaptation) | 10 / 10→10 | 0.0 → 100.0 | 1.40 → 13.80 | 5.45e-03 → 3.36e-05 | 7.06e-02 → 3.81e-04 | 3246.41 | 574.24 | 12.40 |
| IndustrialOffset | Luo (adaptation) | 10 / 10→10 | 20.0 → 100.0 | 5.50 → 11.50 | 2.31e-03 → 3.62e-05 | 6.45e-02 → 4.82e-04 | 7665.64 | 302.64 | 6.00 |
