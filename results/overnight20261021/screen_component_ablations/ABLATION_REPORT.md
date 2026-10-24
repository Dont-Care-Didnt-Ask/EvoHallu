# EvoHallu frozen-screen component ablations

Status: **complete**; snapshot 2026-10-21T19:52:23.974466+00:00.

All four source-validation screen winners and all eleven predetermined component drops are retained. Coefficients are fixed; no test-based variant selection or target fitting was performed. This is one writer seed per screen policy.

Source uses Llama3.1-8b-instruct features on nine datasets. Target uses Llama3.2-1B-Instruct features and the original shared tokenizer on exactly six datasets. Cross-model comparisons use source and target on those same six datasets. Labels are `is_hallu`.

Positive full-minus-drop means the retained component helps the frozen formula. Directed AUROC and paper AUROC are both shown; paper AUROC folds each dataset with `max(AUC,100-AUC)` before averaging. Confidence intervals use 500 shared claim-level bootstrap replicates and condition on the frozen program. No incomplete dataset subset is silently averaged.

| Cohort | Complete datasets with all 15 programs + NLL | Protocol |
| --- | ---: | --- |
| source_nine | 9/9 | passed |
| target_six | 6/6 | passed |

| Policy | Exact source controls | Exact target controls |
| --- | ---: | ---: |
| original | 9/9 | 6/6 |
| neutral | 9/9 | 6/6 |
| diverse | 9/9 | 6/6 |
| compositional | 9/9 | 6/6 |

Every full control is checked against existing primary predictions, source hashes, NLL predictions, labels, duplicate-disjoint masks, and claim-group IDs. A missing reference remains pending; any mismatch invalidates the report.

| Policy / dropped term | Source9 directed Δ [95% CI] | Source9 paper Δ [95% CI] | Target6 directed Δ [95% CI] | Target6 paper Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| original / normalized_slope | 33.042 [31.655, 34.307] | 5.664 [4.307, 6.724] | 30.888 [28.781, 32.626] | 5.811 [4.051, 7.222] |
| original / inverse_tail_range | 0.226 [-0.325, 0.772] | 0.760 [0.205, 1.303] | 0.140 [-0.970, 1.107] | 1.119 [0.091, 2.166] |
| neutral / total_nll | -0.445 [-0.685, -0.229] | 0.134 [-0.104, 0.356] | -0.774 [-1.043, -0.487] | 0.044 [-0.287, 0.337] |
| neutral / weighted_gap | 1.079 [0.745, 1.382] | 0.335 [-0.171, 0.613] | 1.095 [0.700, 1.450] | 0.162 [-0.238, 0.533] |
| neutral / tail_gap_variance | 1.368 [1.230, 1.518] | 1.362 [1.242, 1.517] | 1.136 [0.957, 1.344] | 1.057 [0.880, 1.260] |
| diverse / total_nll | 3.305 [2.468, 4.057] | 5.367 [4.552, 6.213] | 2.516 [1.539, 3.587] | 5.756 [4.621, 6.912] |
| diverse / relative_gap_drift | 0.003 [-0.004, 0.010] | -0.004 [-0.011, 0.004] | 0.002 [-0.008, 0.012] | -0.006 [-0.016, 0.004] |
| diverse / tail_gap_variance | 4.341 [4.009, 4.716] | 4.150 [3.645, 4.538] | 3.916 [3.421, 4.469] | 3.466 [2.939, 4.036] |
| compositional / tail_surprisal_variance | -0.142 [-0.338, 0.038] | 0.011 [-0.244, 0.246] | -0.022 [-0.189, 0.117] | -0.022 [-0.189, 0.117] |
| compositional / tail_surprisal_mean | -1.086 [-1.453, -0.733] | -1.106 [-1.455, -0.746] | -2.389 [-2.988, -1.836] | -2.325 [-2.904, -1.807] |
| compositional / global_correlation | 10.917 [10.245, 11.547] | 11.279 [10.656, 11.865] | 13.671 [12.811, 14.476] | 12.217 [11.239, 13.040] |

| Program | Source same-six paper AUROC | Target six paper AUROC | Target − source [95% CI] |
| --- | ---: | ---: | ---: |
| new/screen_original__full | 70.366 | 69.304 | -1.062 [-1.716, -0.429] |
| new/screen_original__drop_normalized_slope | 65.190 | 63.493 | -1.697 [-2.397, -0.907] |
| new/screen_original__drop_inverse_tail_range | 69.563 | 68.185 | -1.378 [-2.176, -0.615] |
| new/screen_neutral__full | 74.267 | 71.156 | -3.111 [-3.771, -2.509] |
| new/screen_neutral__drop_total_nll | 73.542 | 71.112 | -2.430 [-3.099, -1.712] |
| new/screen_neutral__drop_weighted_gap | 74.089 | 70.994 | -3.096 [-3.596, -2.613] |
| new/screen_neutral__drop_tail_gap_variance | 73.515 | 70.099 | -3.417 [-4.092, -2.814] |
| new/screen_diverse__full | 74.502 | 71.349 | -3.153 [-3.713, -2.585] |
| new/screen_diverse__drop_total_nll | 67.004 | 65.594 | -1.410 [-2.165, -0.458] |
| new/screen_diverse__drop_relative_gap_drift | 74.512 | 71.355 | -3.157 [-3.723, -2.595] |
| new/screen_diverse__drop_tail_gap_variance | 71.130 | 67.883 | -3.246 [-3.731, -2.792] |
| new/screen_compositional__full | 70.494 | 66.939 | -3.555 [-4.206, -2.825] |
| new/screen_compositional__drop_tail_surprisal_variance | 70.503 | 66.961 | -3.542 [-4.212, -2.784] |
| new/screen_compositional__drop_tail_surprisal_mean | 71.797 | 69.265 | -2.532 [-3.315, -1.772] |
| new/screen_compositional__drop_global_correlation | 59.589 | 54.722 | -4.867 [-5.776, -3.788] |
| original/SequenceLogProb | 71.006 | 67.791 | -3.214 [-3.700, -2.760] |

Complete method/per-dataset scores, directed transfer differences, bootstrap usable counts, prediction hashes, and individual control checks are in JSON/CSV. All terms remain in fixed original/neutral/diverse/compositional source order, not sorted by test performance.

Limitations: final terms share token-probability inputs. Diverse's original short-sequence NLL fallback remains; preflight requires n≥5 for all caches. Original Pearson NaN behavior and all numerical guards remain unchanged. These component effects are exploratory and are not independent causal effects or a multi-seed policy comparison.
