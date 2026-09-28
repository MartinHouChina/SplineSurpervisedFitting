# NaturalEarth: ten-tolerance deployment example

Sample: `natural_earth_10m_coastline_b7529bb1babd_w000_b7529bb1`

An intentionally selected successful example, not an estimate of overall performance. All ten tiers use the same input points and plot limits. Curves are the previously measured outputs, not newly fitted curves.

MSE and MaxSE are squared Euclidean errors in normalized coordinates; neither is square-rooted. MaxSE limit = 10 times the MSE limit.

| MSE limit | MaxSE limit | Actual MSE | Actual MaxSE | Internal K | Control vertices | Dual pass | Total ms |
|---|---|---|---|---|---|---|---|
| 1e-04 | 1e-03 | 9.804575e-05 | 0.0004683912 | 23 | 27 | True | 711.818 |
| 9e-05 | 9e-04 | 8.781549e-05 | 0.0004921839 | 24 | 28 | True | 746.870 |
| 8e-05 | 8e-04 | 7.944707e-05 | 0.000773098 | 24 | 28 | True | 844.801 |
| 7e-05 | 7e-04 | 6.731481e-05 | 0.0005206735 | 25 | 29 | True | 928.790 |
| 6e-05 | 6e-04 | 5.898245e-05 | 0.0005275123 | 27 | 31 | True | 704.658 |
| 5e-05 | 5e-04 | 4.825193e-05 | 0.0003123055 | 28 | 32 | True | 859.398 |
| 4e-05 | 4e-04 | 3.930237e-05 | 0.0002946205 | 32 | 36 | True | 782.937 |
| 3e-05 | 3e-04 | 2.991972e-05 | 0.0001641858 | 35 | 39 | True | 929.681 |
| 2e-05 | 2e-04 | 1.960644e-05 | 0.0001360427 | 40 | 44 | True | 1089.723 |
| 1e-05 | 1e-04 | 9.940712e-06 | 6.518477e-05 | 48 | 52 | True | 2146.348 |
