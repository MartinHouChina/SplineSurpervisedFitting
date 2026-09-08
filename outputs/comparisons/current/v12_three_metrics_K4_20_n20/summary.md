# Source-K stratified comparison: K=4-20

Ours deployment: **Ours verified (adaptive repair)**. This is an adaptive exact-refit verification and conditional-repair path; it is not pure one-shot deployment.

Fit bound: MSE <= `2.500e-05` (equivalent RMS <= `0.005000`).
Canonical K is a true-parameter greedy simplification reference, not a global-minimum certificate.

## K-balanced overall results

| Method | mean K | canonical bias / MAE | mean / P95 MSE | pass rate | match F1@0.010 | median time |
|---|---:|---:|---:|---:|---:|---:|
| Ours verified (adaptive repair) | 8.132 | +0.703 / 1.309 | 2.082e-05 / 2.483e-05 | 100.0% | 0.210 | 51.91 ms |
| Greedy hard | 7.324 | -0.106 / 0.859 | 1.760e-05 / 2.442e-05 | 100.0% | 0.310 | 186.66 ms |

## Adaptive verified repair diagnostics

- Learned-feasible: `42.4%`; zero-cleanup fast path: `0.0%`; repair/fallback: `57.6%`.
- Confidence-prefix cleanup: `100.0%`; hard fallback: `0.0%`.
- Residual-guided dynamic insertion: `0.0%`; inserted K mean/max: `0.000` / `0`.
- Exact direct refits mean: `4.268`; all exact fit evaluations mean: `61.729`; residual-stage refits mean/max: `0.000` / `0`.
- Final sources: `{'chord_parameterization_confidence_add_back_compact': 177, 'chord_parameterization_learned_verified_compact': 144, 'chord_parameterization_proposal_same_mask_compact': 19}`.

## Per-source-K results

| source K | canonical K | Ours verified (adaptive repair) K / Hard K | Ours / Hard mean MSE | Ours / Hard pass | Ours / Hard median ms |
|---:|---:|---:|---:|---:|---:|
| 4 | 3.25 | 3.55 / 3.55 | 1.77e-05 / 1.51e-05 | 100% / 100% | 41.8 / 213.8 |
| 5 | 3.65 | 4.45 / 4.15 | 1.46e-05 / 1.45e-05 | 100% / 100% | 46.6 / 213.4 |
| 6 | 4.20 | 4.90 / 4.40 | 1.78e-05 / 1.46e-05 | 100% / 100% | 41.5 / 199.7 |
| 7 | 4.80 | 5.55 / 5.05 | 1.88e-05 / 1.76e-05 | 100% / 100% | 45.1 / 192.0 |
| 8 | 5.70 | 6.30 / 5.70 | 1.90e-05 / 1.73e-05 | 100% / 100% | 45.2 / 193.5 |
| 9 | 6.25 | 7.20 / 6.65 | 2.13e-05 / 1.72e-05 | 100% / 100% | 46.4 / 189.7 |
| 10 | 6.40 | 6.95 / 6.45 | 2.18e-05 / 1.60e-05 | 100% / 100% | 53.6 / 187.8 |
| 11 | 6.90 | 7.80 / 6.90 | 2.10e-05 / 1.78e-05 | 100% / 100% | 45.6 / 181.9 |
| 12 | 7.50 | 8.75 / 7.55 | 2.09e-05 / 1.53e-05 | 100% / 100% | 53.2 / 182.3 |
| 13 | 8.05 | 8.90 / 7.60 | 2.20e-05 / 1.75e-05 | 100% / 100% | 55.0 / 183.4 |
| 14 | 9.00 | 9.45 / 8.60 | 2.16e-05 / 1.83e-05 | 100% / 100% | 58.6 / 177.1 |
| 15 | 8.80 | 9.65 / 8.40 | 2.33e-05 / 1.84e-05 | 100% / 100% | 56.2 / 171.9 |
| 16 | 9.45 | 10.30 / 9.20 | 2.34e-05 / 1.88e-05 | 100% / 100% | 60.4 / 171.7 |
| 17 | 9.70 | 10.20 / 9.35 | 2.20e-05 / 1.92e-05 | 100% / 100% | 63.1 / 179.6 |
| 18 | 10.80 | 11.90 / 10.20 | 2.31e-05 / 2.01e-05 | 100% / 100% | 64.1 / 161.4 |
| 19 | 10.60 | 10.85 / 10.05 | 2.25e-05 / 2.13e-05 | 100% / 100% | 58.1 / 162.8 |
| 20 | 11.25 | 11.55 / 10.70 | 2.30e-05 / 2.04e-05 | 100% / 100% | 63.7 / 163.4 |

## Timing boundary

- Ours: complete model forward (parameters, proposal, KeepMask, relocation) + exact learned-subset verification + conditional confidence add-back/cleanup/residual insertion/hard fallback; adaptive verified deployment, not pure one-shot.
- Hard: greedy pruning stage only from the materialized shared-domain proposal (chord-warped when Ours uses chord, otherwise network predicted); includes every internal standard refit.
- These scopes are intentionally asymmetric; the ratio is not a same-boundary end-to-end speedup.
