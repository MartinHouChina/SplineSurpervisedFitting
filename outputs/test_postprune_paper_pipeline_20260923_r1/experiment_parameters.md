# Experiment parameters

Parameters are authenticated against the completed fresh comparison and selected checkpoint, not launcher defaults.

| Group | Parameter | Setting |
| --- | --- | --- |
| Model | Selected weights | overnight_stable_k32_3090_r1.pt |
| Model | Saved epoch / configured Proposal + Joint | 14 / 12 + 20 |
| Model | Spline degree / input points × dimensions | 3 / 192 × 2 |
| Model | Internal / full knot-vector capacity | 32 / 40 |
| Model | Hidden width / attention heads | 128 / 4 |
| Model | Encoder / selector layers; parameters | 3 / 2; 945,290 |
| Model | Selection / adaptive threshold | mass_topk; enabled |
| Training | Synthetic training / validation samples | 1,500 / 500 |
| Training | Synthetic mixture: simple / shape / spline | 35% / 25% / 40% |
| Training | Historical spline internal K / noise σ | 4–24 / 0.001 |
| Training | Real-data training / validation | 0% training; 32 per listed source |
| Training | Batch size / resample each epoch | 32 / yes |
| Training | Proposal / Joint learning rate | 2.0e-05 / 3.0e-05 |
| Training | Teacher greedy steps / trajectory checks | 16 / 4 |
| Evaluation | Paired test curves / compared methods | 61 / 6 |
| Evaluation | Synthetic test curves / source K | 21 / 4–24 |
| Evaluation | UJI / coastline / contours / offset curves | 10 / 10 / 10 / 10 |
| Evaluation | MSE / maximum squared-point-error bound | 5.0e-05 / 5.0e-04 |
| Evaluation | Maximum internal knots, all methods | 32 |
| Evaluation | Common correction / Ours postprocessing | Residual insertion / verified knot deletion |
| Evaluation | Ours deletion floor / deletion budget | 0 / 32 |
| Evaluation | Evaluation GPU / CPU numerical threads | NVIDIA GeForce GTX 1070 / 4 |
| Baselines | Park: shape weight | 0.8 |
| Baselines | Liang: initial / dense knots; curvature weight | 4 / 32; 0.5 |
| Baselines | Dung: scan intervals / optimization steps | 10 / 10 |
| Baselines | Kang: ADMM / bisections / relocation steps | 400 / 8 / 8 |
| Baselines | Luo: population / generations; η | 10 / 50; 0.5 |

## Measurement and provenance notes

- The selected epoch is the epoch of these weights. The configured Proposal + Joint schedule is not a claim of completed from-scratch training.
- These weights were warm-started with capacity resizing from a 64-internal-knot checkpoint; the full initializer chain is retained in checkpoint_audit.
- The 4–24 source-knot range describes only the historical spline component, not every synthetic training sample.
- Real data are validation-only in the selected training run. The listed validation manifests are UJI, Natural Earth, and USGS; the procedural industrial-offset dataset is an additional test source.
- The comparison has 21 synthetic and 10 per external source, not 10% of each source. Every selected curve remains in the numerical comparison, including failures.
- MSE is the mean squared Euclidean input-point residual; MaxSE is the largest single input-point squared Euclidean residual. Both use the normalized coordinates and neither takes a square root. They do not certify continuous-curve or Hausdorff error.
- Five literature baselines are repository adaptations, followed by common residual insertion for all six methods. Ours additionally applies verified post-deployment single-knot deletion with fixed t and surviving knot locations; this is not a pure network-only comparison.
- Total measured time includes raw fitting, common insertion, and Ours pruning. Network-only time is recorded separately. A single fresh complete-method timing per curve does not establish timing variance.
- Maximum internal-knot capacity 32 corresponds to a cubic full clamped vector of 40 entries and at most 36 control points, with four endpoint entries at each side.

- Report: `E:\SelfSurpervisedSplineFitting\outputs\comparisons\historical_best_dual_error_postprune_fresh_20260923\comparison.json`
- Report SHA256: `f22a400cbb0155d955304b0f570210ae8696a5b8345725c39c9427956c57c1fa`
- Checkpoint: `E:\SelfSurpervisedSplineFitting\outputs\checkpoints\overnight_stable_k32_3090_r1.pt`
- Checkpoint SHA256: `0e026b3dca66c666f950cbc8a4794f7c84dd160f870e1575f4130a44981b09fd`
