# Raw methods versus final methods

Dual pass requires input MSE ≤ 5e-05 AND maximum point squared Euclidean residual ≤ 0.0005; neither quantity is square-rooted.

MaxSE is the maximum over all points of all finite fits in the group, not maximum per-curve MSE and not average peak error.

Every requested case, including numerical failures and unavailable results, remains in the pass denominator. Error/count means include all finite fits, even threshold misses; finite counts are shown.

Arrows compare saved raw methods against their saved final measurements. The final Ours measurements additionally include the redundant-knot post-pruning specified in ours_post_pruning metadata. All five literature methods are repository adaptations, not verified native reproductions; the final pipeline is not one-shot network inference.

Total time includes the measured original method, common repair, and any measured post-pruning; repair and prune times are listed separately. Saved timing_protocol and ours_post_pruning metadata record acquisition and device scopes; these are exploratory local measurements, not a controlled hardware-speed claim.

All-case aggregate is curve-weighted, not equal-source-weighted. Synthetic and procedural industrial offsets are not measured real-world data. Contact-sheet selections do not affect statistics.

Declared unified internal-knot budget: 32. A budget-exhausted case is still a failure; no threshold is silently relaxed.

## Post-pruning metadata

```json
{
  "enabled": true,
  "implementation": "measured greedy deletion",
  "timing": "original + repair + pruning"
}
```

| Source | Method | N / finite raw→final | Dual pass % raw→final | Mean K raw→final | Mean MSE raw→final | MaxSE raw→final | Mean total ms | Mean repair ms | Mean added K | Mean prune ms | Mean removed K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | Ours V16 | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 3.00 | 1.00e-05 → 2.00e-05 | 1.00e-04 → 2.00e-04 | 7.50 | 1.00 | 0.00 | 4.50 | 5.00 |
| All | Park (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| All | Liang (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| All | Dung (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| All | Kang (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| All | Luo (adaptation) | 2 / 1→1 | 50.0 → 50.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| UJI | Ours V16 | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 3.00 | 1.00e-05 → 2.00e-05 | 1.00e-04 → 2.00e-04 | 7.50 | 1.00 | 0.00 | 4.50 | 5.00 |
| UJI | Park (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| UJI | Liang (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| UJI | Dung (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| UJI | Kang (adaptation) | 2 / 2→2 | 100.0 → 100.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| UJI | Luo (adaptation) | 2 / 1→1 | 50.0 → 50.0 | 8.00 → 8.00 | 1.00e-05 → 1.00e-05 | 1.00e-04 → 1.00e-04 | 3.00 | 1.00 | 0.00 | 0.00 | 0.00 |
