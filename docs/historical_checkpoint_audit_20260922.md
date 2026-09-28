# Historical checkpoint audit — 2026-09-22

This is a read-only metadata audit, not a new benchmark and not proof of a globally best model. No model weights were modified.

## Scope and current comparison contract

- Audited 105 local model payloads across `outputs/`, `outputs/checkpoints/` (including archived V8–V15 stages), and downloaded server models.
- Excluded teacher-cache tensors, tests/smoke runs, rollback-tree duplicates, and pre-candidate legacy architectures from this inventory.
- 52 payloads explicitly declare capacity at most 32; 40 remain eligible after excluding Proposal-only stages. These include `.last`, calibrated and distilled stages; SHA deduplication and strict architecture restoration must follow before evaluation.
- Unknown-capacity intermediate payloads require their original companion configuration. Never silently infer a new architecture.
- Current requested limits: maximum 32 internal knots, mean squared Euclidean error at most `5e-5`, maximum point squared Euclidean error at most `5e-4`. Neither error is square-rooted.
- Historical mean-MSE validation does not establish the new maximum-error requirement. Re-evaluate on a common validation split; do not select the model on final test cases or repair-augmented test results.

Machine-readable inventory: `outputs/diagnostics/historical_checkpoint_metadata_20260922.json`. Fields include original path, capacity, epoch, stage, objective, historical tolerance, validation summaries, deployment configuration, exclusion reasons, and a preliminary shortlist.

## Comparable historical V16 checkpoints

The first three rows below use the same historical 596-curve validation design: 500 synthetic curves and 32 curves from each of UJI, Natural Earth and USGS. The local paper model uses a different 800-curve validation design (500 synthetic plus 100 per real source), so its percentage must not be pooled or directly ranked against them. Industrial offset data were not included in these saved validation summaries.

| Existing file under `outputs/checkpoints/` unless noted | Capacity | Saved epoch | Validation pass | Worst-source pass | Mean internal K | Mean MSE |
|---|---:|---:|---:|---:|---:|---:|
| `overnight_stable_k32_3090_r1.last.pt` | 32 | 32 | 69.966% | 18.750% | 23.992 | 5.647e-5 |
| `universal_m16_m32_3090_r1_m32.pt` | 32 | 5 | 68.960% | 53.125% | 26.607 | 4.595e-5 |
| `overnight_anchored_k32_p12_j48_3090_r1.pt` | 32 | 13 | 66.946% | 64.600% | 27.275 | 4.978e-5 |
| downloaded `paper_coupled_clean_3090_r3_m32.pt` | 32 | 13 | 63.625% | 36.000% | 25.510 | 5.895e-5 |

All four are one-network-forward / one-final-refit configurations, without external numerical error repair. The first row has a better sample-weighted average but very poor coastline validation. The third row has better worst-source robustness. No single row dominates every criterion.

## Older V8–V15 candidates

These summaries use synthetic validation and different RMS-derived tolerances; they are candidates for common validation, not cross-dataset winners.

| File | Capacity | Epoch | Historical MSE tolerance | Saved deployment pass | Mean internal K |
|---|---:|---:|---:|---:|---:|
| `outputs/candidate_pruning_one_shot_v14.pt` | 28 | 92 | 2.5e-5 | 62.4% | 11.990 |
| `outputs/checkpoints/current/candidate_pruning_one_shot_v14_feedback.pt` | 28 | 154 | 2.5e-5 | 49.9% | 9.827 |
| `outputs/checkpoints/candidate_pruning_one_shot_v15.pt` | 28 | 121 | 2.5e-5 | 42.1% | 10.241 |
| `outputs/checkpoints/archive/v10/candidate_pruning_v10.pt` | 28 | 24 | 2.5e-5 | 41.5% | 13.820 |

The V14 epoch-92 file retains a `fixed_proposal_for_offline_teacher` deployment-role string despite its feedback-finetune stage; verify actual architecture restoration and execution rather than trusting this stale/ambiguous role. The V14 feedback file also retains older pre-feedback metric entries; its selected-feedback summary must not be confused with those entries.

Other audited V8/V9/V11/V12/V13 selected weights had substantially lower historical deployment pass rates. Their calibrated/distilled/last payloads remain in the inventory where architecture metadata permits screening. V14 joint and canonical-Boehm capacity-32 weights had historical pass rates 0.15% and 0% at MSE approximately `1e-5`; this stricter tolerance still prevents direct numerical comparison to `5e-5`.

## Why older high pass rates are not current winners

- `overnight_reliable_3090_r1.pt`: 94.966% overall, 65.625% worst source, mean K 30.941 at `5e-5`, but candidate capacity is **64**, outside the requested limit.
- `candidate_selection_v16_adaptive_k96.pt`: 99.375% overall at stricter `2.5e-5`, but capacity is **96** and real-source retained counts reach roughly 61–67. It is not a capacity-32 result.
- `overnight_1070_full_pipeline_r1.pt` and `overnight_plus_3090_r1.pt`: saved passes 94% and 92.2%, respectively, at `5e-5`, but those saved validation summaries contain **synthetic data only** and capacity is 64.
- A high-pass epoch mentioned in a complete history is not necessarily available as weights. The inventory lists actual stored model payloads; never reconstruct nonexistent epoch weights from history records.

## Joint-training interpretation

For the newly downloaded M16 and M32 paper checkpoints, embedded histories stop at saved epoch 13. Without their complete server histories or `.last` files, this alone does not prove that all later Joint epochs deteriorated or even that all planned epochs ran.

Complete nearby histories do show a real deterioration pattern. Universal M32 epoch 5 to 24 changes deployment pass from 68.960% to 51.174%, worst-source pass from 53.125% to 18.750%, and K from 26.607 to 23.909, while dense pass remains roughly 86%. Anchored K32 epoch 13 to 60 changes deployment pass from 66.946% to 59.732%, worst-source pass from 64.600% to 18.750%, and K from 27.275 to 23.700. This implicates selection/subset geometry and distribution-specific simplification, not merely a lack of dense candidate capacity.

`scripts/train_v16.py:804` ranks infeasible checkpoints by worst-source pass before tail and mean error. Therefore a first-Joint best checkpoint is not by itself evidence of a broken save function or proof that Joint optimization is universally invalid. An aggregate gain can be rejected when one source degrades. Further common-split raw/repair ablations are required.

## Recommended selection and reporting procedure

1. Deduplicate eligible weights by content hash and strictly restore the saved architecture.
2. Screen all compatible candidates on the same independent validation curves, reporting raw mean-MSE pass, raw maximum-error pass, their intersection, source-macro pass, worst-source pass, K and timing.
3. Select before final test evaluation. Keep numerical repair out of checkpoint selection to avoid concealing weak learned predictions.
4. Evaluate both raw Ours and explicitly named Ours + verified repair on a disjoint test set; charge all repair operations to end-to-end time.
5. Keep the five baseline implementations and their budgets fixed. Report cap exhaustion and failures honestly; never change plotted residuals to resemble another method.
