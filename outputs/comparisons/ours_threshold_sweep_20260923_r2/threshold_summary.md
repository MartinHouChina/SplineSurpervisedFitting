# Ours four-tolerance test

Fixed weights; each tolerance reruns the network, standard refit, insertion and deletion. All requested cases remain in the pass denominator. Errors/counts include every finite returned fit, not only passes.

MSE and MaxSE are squared Euclidean errors on normalized input points, never square-rooted. MaxSE bound is the MSE bound multiplied by the recorded peak_ratio. No continuous-curve/Hausdorff guarantee.

This is a fixed-checkpoint deployment sensitivity study, not four separately trained models. The existing checkpoint was selected previously at 5e-5; no model selection uses these test results.

## All

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 61/61 | 38/61 (62.3%) | 5.42411e-05 | 0.00038072 | 0.00433724 | 21.8852 | 477.923 |
| 2.5e-05 | 0.00025 | 61/61 | 43/61 (70.5%) | 6.04479e-05 | 0.000413067 | 0.0036086 | 18.9016 | 487.768 |
| 5e-05 | 0.0005 | 61/61 | 48/61 (78.7%) | 7.52756e-05 | 0.000495117 | 0.00298375 | 16.6393 | 526.729 |
| 0.0001 | 0.001 | 61/61 | 53/61 (86.9%) | 0.000101849 | 0.000695962 | 0.00377664 | 14.2623 | 578.903 |

## Synthetic

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 21/21 | 11/21 (52.4%) | 4.11218e-05 | 0.000275035 | 0.00109697 | 25.7143 | 319.479 |
| 2.5e-05 | 0.00025 | 21/21 | 14/21 (66.7%) | 4.98372e-05 | 0.00032096 | 0.00109932 | 22.6667 | 408.111 |
| 5e-05 | 0.0005 | 21/21 | 16/21 (76.2%) | 6.52349e-05 | 0.000461016 | 0.00110692 | 20.4762 | 487.078 |
| 0.0001 | 0.001 | 21/21 | 18/21 (85.7%) | 8.68626e-05 | 0.000614582 | 0.00104456 | 18.2857 | 591.045 |

## UJI

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 10/10 | 10/10 (100.0%) | 8.99311e-06 | 6.72422e-05 | 9.7981e-05 | 11.8 | 826.724 |
| 2.5e-05 | 0.00025 | 10/10 | 10/10 (100.0%) | 2.09538e-05 | 0.000172119 | 0.000242613 | 7.5 | 692.973 |
| 5e-05 | 0.0005 | 10/10 | 10/10 (100.0%) | 3.69659e-05 | 0.00022465 | 0.00046468 | 4.4 | 682.992 |
| 0.0001 | 0.001 | 10/10 | 10/10 (100.0%) | 7.29172e-05 | 0.000524693 | 0.00083457 | 2.9 | 590.238 |

## NaturalEarth

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 10/10 | 2/10 (20.0%) | 0.000167582 | 0.00120884 | 0.00433724 | 30.1 | 300.823 |
| 2.5e-05 | 0.00025 | 10/10 | 4/10 (40.0%) | 0.000157629 | 0.00108847 | 0.0036086 | 27.6 | 437.518 |
| 5e-05 | 0.0005 | 10/10 | 7/10 (70.0%) | 0.000172062 | 0.00112306 | 0.00298375 | 24.7 | 490.368 |
| 0.0001 | 0.001 | 10/10 | 7/10 (70.0%) | 0.000197577 | 0.00132417 | 0.00377664 | 20.3 | 562.262 |

## USGS

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 10/10 | 5/10 (50.0%) | 5.96764e-05 | 0.000400261 | 0.00130713 | 23.6 | 425.678 |
| 2.5e-05 | 0.00025 | 10/10 | 5/10 (50.0%) | 6.66283e-05 | 0.000437397 | 0.00131336 | 21.6 | 378.492 |
| 5e-05 | 0.0005 | 10/10 | 5/10 (50.0%) | 7.24022e-05 | 0.000449554 | 0.0013166 | 20.1 | 408.057 |
| 0.0001 | 0.001 | 10/10 | 8/10 (80.0%) | 9.80047e-05 | 0.000628153 | 0.00131462 | 16.7 | 636.121 |

## IndustrialOffset

| MSE bound | MaxSE bound | Valid/all | Dual pass | Mean MSE | Mean MaxSE | Worst MaxSE | Mean K | Mean total ms |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.0001 | 10/10 | 10/10 (100.0%) | 8.26311e-06 | 6.84714e-05 | 9.87841e-05 | 14 | 691.202 |
| 2.5e-05 | 0.00025 | 10/10 | 10/10 (100.0%) | 1.88629e-05 | 0.000147709 | 0.000247455 | 11 | 609.369 |
| 5e-05 | 0.0005 | 10/10 | 10/10 (100.0%) | 4.07579e-05 | 0.000254811 | 0.00049431 | 9.3 | 608.768 |
| 0.0001 | 0.001 | 10/10 | 10/10 (100.0%) | 7.03668e-05 | 0.000477731 | 0.000978219 | 8.7 | 501.493 |
