# Raw methods versus common dual-error insertion

Dual pass requires input MSE ≤ 5e-05 AND maximum point squared Euclidean residual ≤ 0.0005; neither quantity is square-rooted.

MaxSE is the maximum over all points of all finite fits in the group, not maximum per-curve MSE and not average peak error.

Every requested case, including numerical failures and unavailable results, remains in the pass denominator. Error/count means include all finite fits, even threshold misses; finite counts are shown.

Arrows compare the saved raw method against that same method followed by common residual-guided knot insertion. All five literature methods are repository adaptations, not verified native reproductions. Repaired Ours is a hybrid, not one-shot deployment.

Total time includes original method and common repair; repair time is listed separately. Raw timings may be cached and repair timings freshly measured. Network and numerical methods can use different CUDA/CPU scopes; these are exploratory local measurements, not a controlled hardware-speed claim.

All-case aggregate is curve-weighted, not equal-source-weighted. Synthetic and procedural industrial offsets are not measured real-world data. Contact-sheet selections do not affect statistics.

Declared unified internal-knot budget: 32. A budget-exhausted case is still a failure; no threshold is silently relaxed.

| Source | Method | N / finite raw→final | Dual pass % raw→final | Mean K raw→final | Mean MSE raw→final | MaxSE raw→final | Mean total ms | Mean repair ms | Mean added K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | Ours V16 | 3 / 2→2 | 33.3 → 66.7 | 8.00 → 8.50 | 1.00e-05 → 1.50e-05 | 1.00e-02 → 2.00e-04 | 3.00 | 1.00 | 0.33 |
| UJI | Ours V16 | 3 / 2→2 | 33.3 → 66.7 | 8.00 → 8.50 | 1.00e-05 → 1.50e-05 | 1.00e-02 → 2.00e-04 | 3.00 | 1.00 | 0.33 |
