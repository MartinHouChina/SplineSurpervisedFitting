# Paired paper evidence

All requested samples remain in pass-rate denominators. Error/count means include all finite outputs, including misses. Literature labels refer to repository adaptations, not author code. Common repair and pruning are separate experimental wrappers.

Primary comparison: all methods receive the same repair/deletion rules and internal-knot cap. Historical comparison: only Ours receives deletion; other methods receive insertion only. Raw comparison: no added wrappers. Endpoint conventions inherited from each adapter are explicitly recorded in protocol.json.

Ablations are inference interventions, not separately retrained networks. D/E keep the exact same learned parameters as C and are charged its network forward; E overrides only the final mask. F uses chord parameters without a network. Counts of solver calls below refer ONLY to common postprocessing.

## All

| Method | Stage | MSE limit | Finite/all | Dual pass | Mean K | Mean MSE | Mean MaxSE | Worst MaxSE | All mean / pass mean / median / P95 ms |
|---|---|---|---|---|---|---|---|---|---|
| all_candidates_shared_params | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| all_candidates_shared_params | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| all_candidates_shared_params | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| dung_direct_knot_2017_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| dung_direct_knot_2017_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| dung_direct_knot_2017_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| kang_sparse_2015_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| kang_sparse_2015_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| kang_sparse_2015_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| liang_feature_iki_2017_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| liang_feature_iki_2017_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| liang_feature_iki_2017_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| luo_linf_de_2022_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| luo_linf_de_2022_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| luo_linf_de_2022_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| ours | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| ours | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| ours | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| park_dominant_point_2007_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| park_dominant_point_2007_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| park_dominant_point_2007_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_chord | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_chord | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_chord | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_shared_params | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_shared_params | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_shared_params | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |

## Synthetic

| Method | Stage | MSE limit | Finite/all | Dual pass | Mean K | Mean MSE | Mean MaxSE | Worst MaxSE | All mean / pass mean / median / P95 ms |
|---|---|---|---|---|---|---|---|---|---|
| all_candidates_shared_params | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| all_candidates_shared_params | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| all_candidates_shared_params | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| dung_direct_knot_2017_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| dung_direct_knot_2017_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| dung_direct_knot_2017_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| kang_sparse_2015_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| kang_sparse_2015_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| kang_sparse_2015_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| liang_feature_iki_2017_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| liang_feature_iki_2017_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| liang_feature_iki_2017_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| luo_linf_de_2022_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| luo_linf_de_2022_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| luo_linf_de_2022_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| ours | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| ours | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| ours | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| park_dominant_point_2007_adaptation | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| park_dominant_point_2007_adaptation | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| park_dominant_point_2007_adaptation | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_chord | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_chord | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_chord | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_shared_params | pruned | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_shared_params | raw | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
| uniform_shared_params | repaired | 5e-05 | 1/2 | 1/2 (50.0%) | 8 | 1e-05 | 2e-05 | 2e-05 | 7 / 7 / 7 / 7 |
