# Final six-method comparison

- **four_groups**: ['complexity', 'feasibility', 'fit_error', 'runtime']
- **complexity**: Mean final internal-knot count across finite returned fits; endpoint repetitions excluded.
- **feasibility**: Both MSE <= 5e-05 and maximum point squared Euclidean error <= 0.0005; denominator includes every requested case and algorithm failure.
- **mean_mse**: Arithmetic mean of per-curve mean squared Euclidean error on normalized input points; all finite returned fits, including tolerance misses. No square root or coordinate-dimension division.
- **max_squared_error**: Maximum single-point squared Euclidean error across all input points of all finite returned curves, not mean peak error and not a continuous/Hausdorff error certificate.
- **runtime**: Arithmetic mean of measured total_ms across all requested cases; includes original method, common insertion, and any Ours post-pruning. Network-only time is stored separately, never substituted.
- **valid_all**: Finite returned fits / requested cases; a finite fit may still fail either tolerance.
- **population**: All report measurements, not only selected examples or passing curves. Industrial offsets are procedurally generated industrial-model curves, not field measurements.
- **protocol**: Five literature repository adaptations retain their existing common insertion wrapper; new redundant-knot post-pruning is Ours-only. Detail is recorded off-figure for the manuscript.
- **timing_scope**: fresh single raw run + fresh common insertion + fresh Ours-only post-pruning; total_ms=raw_ms+repair_ms+pruning_ms; all numerical postprocessing included; metric/export/plot time excluded; network-only independently median of repeated synchronized forwards
- **timing_caution**: Local exploratory timings on the recorded hardware; single raw measurements and CPU/GPU differences do not establish a controlled speedup claim.

## All test curves

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 61/61 | 16.64 | 78.7 | 7.528e-05 | 2.984e-03 | 625.87 |
| Park | 61/61 | 21.10 | 59.0 | 1.145e-04 | 1.041e-02 | 106.38 |
| Liang | 61/61 | 19.39 | 67.2 | 8.861e-05 | 3.710e-03 | 78.58 |
| Dung | 60/61 | 16.78 | 80.3 | 5.565e-05 | 3.575e-03 | 1033.40 |
| Kang | 61/61 | 18.31 | 77.0 | 6.434e-05 | 3.578e-03 | 2348.05 |
| Luo | 61/61 | 18.30 | 78.7 | 6.952e-05 | 7.326e-03 | 5452.86 |

## Synthetic

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 21/21 | 20.48 | 76.2 | 6.523e-05 | 1.107e-03 | 590.73 |
| Park | 21/21 | 30.43 | 28.6 | 1.452e-04 | 1.041e-02 | 132.48 |
| Liang | 21/21 | 27.52 | 38.1 | 1.206e-04 | 3.710e-03 | 107.57 |
| Dung | 21/21 | 23.24 | 66.7 | 5.264e-05 | 6.890e-04 | 1453.25 |
| Kang | 21/21 | 24.43 | 57.1 | 6.105e-05 | 2.005e-03 | 2105.10 |
| Luo | 21/21 | 25.24 | 61.9 | 6.883e-05 | 1.850e-03 | 4192.71 |

## UJI handwriting

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 10/10 | 4.40 | 100.0 | 3.697e-05 | 4.647e-04 | 898.58 |
| Park | 10/10 | 6.00 | 100.0 | 2.366e-05 | 4.537e-04 | 53.50 |
| Liang | 10/10 | 6.20 | 100.0 | 2.471e-05 | 4.577e-04 | 36.97 |
| Dung | 10/10 | 4.30 | 100.0 | 2.590e-05 | 4.698e-04 | 380.82 |
| Kang | 10/10 | 5.80 | 100.0 | 3.094e-05 | 4.891e-04 | 2604.92 |
| Luo | 10/10 | 6.00 | 100.0 | 2.019e-05 | 3.980e-04 | 7896.61 |

## Natural Earth coastline

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 10/10 | 24.70 | 70.0 | 1.721e-04 | 2.984e-03 | 625.00 |
| Park | 10/10 | 23.30 | 60.0 | 2.292e-04 | 6.928e-03 | 129.59 |
| Liang | 10/10 | 22.20 | 70.0 | 1.645e-04 | 3.049e-03 | 79.92 |
| Dung | 9/10 | 20.11 | 70.0 | 1.244e-04 | 3.575e-03 | 1140.19 |
| Kang | 10/10 | 21.30 | 70.0 | 1.425e-04 | 3.578e-03 | 2037.87 |
| Luo | 10/10 | 21.70 | 70.0 | 1.666e-04 | 7.326e-03 | 5013.54 |

## USGS contours

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 10/10 | 20.10 | 50.0 | 7.240e-05 | 1.317e-03 | 434.44 |
| Park | 10/10 | 21.10 | 50.0 | 9.884e-05 | 2.592e-03 | 111.54 |
| Liang | 10/10 | 20.40 | 60.0 | 6.235e-05 | 6.931e-04 | 71.79 |
| Dung | 10/10 | 19.40 | 80.0 | 5.031e-05 | 1.356e-03 | 1157.33 |
| Kang | 10/10 | 19.50 | 80.0 | 5.729e-05 | 8.766e-04 | 2397.48 |
| Luo | 10/10 | 19.40 | 80.0 | 5.652e-05 | 9.357e-04 | 5086.26 |

## Industrial offsets

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 10/10 | 9.30 | 100.0 | 4.076e-05 | 4.943e-04 | 619.27 |
| Park | 10/10 | 14.40 | 90.0 | 4.149e-05 | 1.674e-03 | 76.09 |
| Liang | 10/10 | 11.70 | 100.0 | 3.582e-05 | 4.895e-04 | 64.76 |
| Dung | 10/10 | 10.10 | 100.0 | 3.522e-05 | 4.403e-04 | 573.55 |
| Kang | 10/10 | 13.80 | 100.0 | 3.355e-05 | 3.805e-04 | 2862.14 |
| Luo | 10/10 | 11.50 | 100.0 | 3.621e-05 | 4.825e-04 | 6461.36 |
