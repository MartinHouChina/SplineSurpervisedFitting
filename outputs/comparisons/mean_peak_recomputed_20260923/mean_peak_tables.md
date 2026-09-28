# Mean per-curve maximum squared error

Recomputed from saved normalized residuals; 61 curves, six methods. Not a new larger-cohort experiment.
Synthetic: 21; UJI, Natural Earth, USGS, industrial offsets: 10 each. Not 10% sampling.
All methods use insertion repair; only Ours additionally uses post-pruning. Maximum internal knots: 32.
Mean includes every finite fit, including threshold violations. Dung has one nonfinite Natural Earth result, excluded rather than replaced by zero.

## All

| Method | Finite/total | Mean peak squared error | Worst peak squared error |
|---|---:|---:|---:|
| Ours | 61/61 | 4.951169e-04 | 2.983746e-03 |
| Park | 61/61 | 1.815259e-03 | 1.041021e-02 |
| Liang | 61/61 | 5.799253e-04 | 3.709873e-03 |
| Dung | 60/61 | 3.803075e-04 | 3.575195e-03 |
| Kang | 61/61 | 5.182014e-04 | 3.577571e-03 |
| Luo | 61/61 | 5.447337e-04 | 7.326210e-03 |

## Synthetic

| Method | Finite/total | Mean peak squared error | Worst peak squared error |
|---|---:|---:|---:|
| Ours | 21/21 | 4.610157e-04 | 1.106924e-03 |
| Park | 21/21 | 3.577688e-03 | 1.041021e-02 |
| Liang | 21/21 | 7.718915e-04 | 3.709873e-03 |
| Dung | 21/21 | 2.600785e-04 | 6.890305e-04 |
| Kang | 21/21 | 4.590362e-04 | 2.004567e-03 |
| Luo | 21/21 | 5.011207e-04 | 1.850456e-03 |

## UJI

| Method | Finite/total | Mean peak squared error | Worst peak squared error |
|---|---:|---:|---:|
| Ours | 10/10 | 2.246504e-04 | 4.646796e-04 |
| Park | 10/10 | 2.833231e-04 | 4.537458e-04 |
| Liang | 10/10 | 2.399813e-04 | 4.576751e-04 |
| Dung | 10/10 | 2.376800e-04 | 4.698043e-04 |
| Kang | 10/10 | 3.481408e-04 | 4.890676e-04 |
| Luo | 10/10 | 1.700391e-04 | 3.979825e-04 |

## NaturalEarth

| Method | Finite/total | Mean peak squared error | Worst peak squared error |
|---|---:|---:|---:|
| Ours | 10/10 | 1.123064e-03 | 2.983746e-03 |
| Park | 10/10 | 1.850301e-03 | 6.928035e-03 |
| Liang | 10/10 | 1.076634e-03 | 3.048649e-03 |
| Dung | 9/10 | 9.141447e-04 | 3.575195e-03 |
| Kang | 10/10 | 1.132176e-03 | 3.577571e-03 |
| Luo | 10/10 | 1.493687e-03 | 7.326210e-03 |

## USGS

| Method | Finite/total | Mean peak squared error | Worst peak squared error |
|---|---:|---:|---:|
| Ours | 10/10 | 4.495542e-04 | 1.316602e-03 |
| Park | 10/10 | 1.041752e-03 | 2.591561e-03 |
| Liang | 10/10 | 3.548736e-04 | 6.930832e-04 |
| Dung | 10/10 | 4.145706e-04 | 1.356247e-03 |
| Kang | 10/10 | 3.973676e-04 | 8.766059e-04 |
| Luo | 10/10 | 3.495570e-04 | 9.357241e-04 |

## IndustrialOffset

| Method | Finite/total | Mean peak squared error | Worst peak squared error |
|---|---:|---:|---:|
| Ours | 10/10 | 2.548115e-04 | 4.943096e-04 |
| Park | 10/10 | 3.845605e-04 | 1.674413e-03 |
| Liang | 10/10 | 2.450825e-04 | 4.894873e-04 |
| Dung | 10/10 | 2.606992e-04 | 4.402842e-04 |
| Kang | 10/10 | 3.193679e-04 | 3.805372e-04 |
| Luo | 10/10 | 2.572393e-04 | 4.824885e-04 |
