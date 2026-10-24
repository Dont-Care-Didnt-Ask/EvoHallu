# Frozen screen-component logistic refit

Status: **complete_exploratory_report**. One exploratory fitted head; no primary search group is changed.

All eleven full-minus-drop components from all four screen winners plus `float64_total_nll` were retained. Coefficients use the source-only grouped 5,606/1,401 split, three-fold fit-only CV and five C values; the scaler stays inside CV. Source programs were already validation-selected, so outer validation is diagnostic. Benchmark results had already been viewed: this follow-up is exploratory.

| Scope | Verified datasets | Head directed | NLL directed | Gain | Head paper | NLL paper | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_nine | 9/9 | 71.359 | 65.644 | 5.715 | 73.633 | 68.711 | 4.922 |
| matched_source_six | 6/6 | 70.884 | 66.405 | 4.479 | 74.296 | 71.006 | 3.290 |
| target_six | 6/6 | 68.060 | 63.730 | 4.330 | 71.431 | 67.791 | 3.640 |

Paper AUROC folds each dataset using `max(AUC,100−AUC)` before the unweighted mean. The canonical NLL reference retains its original float16 accumulation; the extra fitted feature uses float64 accumulation. Six-dataset and nine-dataset means describe different scopes.

| Scope | Directed gain 95% CI | Paper gain 95% CI |
| --- | --- | --- |
| source_nine | [5.188, 6.266] | [4.193, 5.419] |
| matched_source_six | [3.719, 5.186] | [2.451, 4.138] |
| target_six | [3.662, 4.926] | [2.943, 4.280] |

Matched target minus source compares the same six datasets and rows. It never subtracts the source-nine average from target-six.

| Transfer quantity | Directed delta | Paper delta | Directed 95% CI | Paper 95% CI |
| --- | ---: | ---: | --- | --- |
| Frozen head | -2.824 | -2.865 | [-3.449, -2.144] | [-3.485, -2.182] |
| Canonical NLL | -2.675 | -3.214 | [-3.164, -2.194] | [-3.700, -2.760] |
| Head gain over NLL | -0.149 | 0.350 | [-0.595, 0.311] | [-0.128, 0.842] |

Source fit: C=0.1; training directed AUC=81.820; diagnostic outer-validation AUC=82.094.

Confidence intervals use the saved 500 shared claim-bootstrap replicates with fixed features and weights. They measure held-out claim uncertainty conditional on this selected head, not search-seed uncertainty. Only draws finite on every prescribed dataset enter macro intervals. No incomplete dataset subset is averaged. JSON/CSV retain per-dataset scores, strict duplicate-disjoint diagnostics, coverage, reference parity and artifact hashes.
