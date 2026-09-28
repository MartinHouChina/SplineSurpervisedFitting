# Ours multi-tolerance test

Fixed weights; each tolerance reruns the network, standard refit, insertion and deletion. All requested cases remain in the pass denominator. Errors/counts include every finite returned fit, not only passes.

MSE and MaxSE are squared Euclidean errors on normalized input points, never square-rooted. MaxSE bound is the MSE bound multiplied by the recorded peak_ratio. No continuous-curve/Hausdorff guarantee.

This is a fixed-checkpoint deployment sensitivity study, not separately trained models. No model selection uses these test results. Checkpoint identity and capacity are recorded in the JSON metadata.

## All

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 1/2 | 1/2 (50.0%) | 5e-06 | 4e-05 | 4e-05 | 8 | 4 |
| 2.5e-05 | 0.00025 | 1/2 | 1/2 (50.0%) | 1.25e-05 | 0.0001 | 0.0001 | 8 | 4 |
| 5e-05 | 0.0005 | 1/2 | 1/2 (50.0%) | 2.5e-05 | 0.0002 | 0.0002 | 8 | 4 |
| 0.0001 | 0.001 | 1/2 | 1/2 (50.0%) | 5e-05 | 0.0004 | 0.0004 | 8 | 4 |

## Synthetic

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 1/2 | 1/2 (50.0%) | 5e-06 | 4e-05 | 4e-05 | 8 | 4 |
| 2.5e-05 | 0.00025 | 1/2 | 1/2 (50.0%) | 1.25e-05 | 0.0001 | 0.0001 | 8 | 4 |
| 5e-05 | 0.0005 | 1/2 | 1/2 (50.0%) | 2.5e-05 | 0.0002 | 0.0002 | 8 | 4 |
| 0.0001 | 0.001 | 1/2 | 1/2 (50.0%) | 5e-05 | 0.0004 | 0.0004 | 8 | 4 |
