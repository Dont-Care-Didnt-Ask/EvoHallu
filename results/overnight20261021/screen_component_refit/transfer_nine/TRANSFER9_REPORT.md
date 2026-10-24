# Frozen component head: optional full-nine transfer

Status: **complete_exploratory_nine**. The original six-dataset report is preserved.

Same previously frozen 12-column head, coefficients, scaler, wrapper and helper; no target refit or selection. Six target results are byte-identical copies; only the three prespecified additions were scored. Source and target comparisons both use all nine datasets. Labels remain `is_hallu`.

| Scope | Verified datasets | Directed head / NLL | Paper head / NLL | Paper gain [95% CI] | Strict-disjoint paper head / NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_nine | 9/9 | 71.359 / 65.644 | 73.633 / 68.711 | 4.922 [4.193, 5.419] | 73.570 / 68.685 |
| target_nine | 9/9 | 66.552 / 61.459 | 68.800 / 64.835 | 3.965 [3.529, 4.628] | 68.731 / 64.781 |

Main full-nine NLL prediction/bootstrap parity: **exact_match**. Missing external references remain pending, never inferred.

| Matched target − source | Directed delta [95% CI] | Paper delta [95% CI] |
| --- | ---: | ---: |
| head | -4.807 [-5.270, -4.302] | -4.834 [-5.270, -4.186] |
| nll | -4.185 [-4.530, -3.835] | -3.877 [-4.500, -3.558] |

Paper AUROC folds every method/dataset as max(AUC,100−AUC) before equal weighting. Intervals condition on fixed selected programs, weights and these datasets. They exclude feature-discovery/search uncertainty; direction folding uses dataset labels. The feature pool and prior outer validation were selection-informed. This result is exploratory and is not matched evidence of beating the paper/Claude.
