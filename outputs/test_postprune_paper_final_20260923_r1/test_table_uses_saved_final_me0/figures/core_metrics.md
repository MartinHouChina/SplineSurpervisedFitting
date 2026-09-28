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
- **timing_scope**: fixture
- **timing_caution**: Local exploratory timings on the recorded hardware; single raw measurements and CPU/GPU differences do not establish a controlled speedup claim.

## All test curves

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 3/3 | 7.33 | 66.7 | 3.033e-05 | 6.000e-04 | 11.00 |
| Dung | 2/3 | 9.00 | 33.3 | 4.500e-05 | 6.000e-04 | 11.00 |

## Synthetic

| Method | Valid/all | Mean internal K | Dual pass (%) | Mean MSE | Maximum squared error | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 3/3 | 7.33 | 66.7 | 3.033e-05 | 6.000e-04 | 11.00 |
| Dung | 2/3 | 9.00 | 33.3 | 4.500e-05 | 6.000e-04 | 11.00 |
