# Paired feature and module ablation

All rows use the same checkpoint and fixed curves. Removed feature channels are zeroed without changing the parameter count. Module arms intervene at inference. No residual insertion or verified post-pruning is included.

| Group | Variant | Dual pass | Mean K | Mean MSE | Mean MaxSE | Worst MaxSE | Network ms |
|---|---|---:|---:|---:|---:|---:|---:|
| feature | Coordinates only | 4.9% | 4.00 | 5.326e-03 | 2.662e-02 | 1.995e-01 | 13.51 |
| feature | Coordinates + first derivative | 6.6% | 4.74 | 3.532e-03 | 1.918e-02 | 1.263e-01 | 12.31 |
| feature | Coordinates + first + second derivatives | 59.0% | 23.36 | 8.609e-05 | 7.004e-04 | 1.088e-02 | 12.49 |
| module | Complete network | 59.0% | 23.36 | 8.609e-05 | 7.004e-04 | 1.088e-02 | 12.93 |
| module | Fixed 0.5 KeepMask | 62.3% | 23.52 | 7.390e-05 | 5.888e-04 | 5.201e-03 | 11.74 |
| module | No survivor-conditioned decoder | 36.1% | 23.36 | 1.499e-04 | 1.704e-03 | 1.454e-02 | 8.84 |
| module | No candidate selection | 75.4% | 32.00 | 5.452e-05 | 3.871e-04 | 4.367e-03 | 11.70 |
| module | Chord parameters after selection | 37.7% | 23.36 | 9.644e-05 | 8.659e-04 | 9.321e-03 | 13.83 |
