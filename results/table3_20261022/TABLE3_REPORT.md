# evohallu — Table 3: evolution versus logistic refitting

New exploratory CPU follow-up completed 22 October 2026. This reproduces the **comparison design** of [Table 3 in EvoHallu](https://aclanthology.org/2026.acl-srw.115.pdf#page=7): keep one generated formula's feature set fixed and compare its original weights with weights fitted by logistic regression. It uses our four frozen **Qwen3.5-9B fixed-context screen winners**, not the authors' Claude programs or their best/median candidate ranks.

Each logistic model uses only its own formula's two or three components. There is **no pooling across formulas and no extra NLL column**. The historical pooled twelve-column head scoring 73.633 is a different experiment and is unchanged.

Source probability/logit features come from **Llama3.1-8B-Instruct**. No LLM calls, GPU work, new candidate generation or target-model fitting were performed.

## Evaluation corresponding to Table 3

AUROC × 100. Each test-dataset cell uses `max(AUC, 100-AUC)`; Mean is the equal mean of the nine test cells, excluding PopQA fit and diagnostic validation. Evo uses the actual saved frozen-program predictions. LogReg is fitted only on 5,606 PopQA rows; the 1,401 grouped outer-validation rows are diagnostic. Scaling and C selection use three claim-grouped folds inside fitting rows only, with C ∈ {0.01, 0.1, 1, 10, 100}.

| Method | PopQA fit | PopQA test | AVeriTeC | X-Fact | UHEAD | Cities | Companies | Common Claims | CounterFact | WildHallucinations | Mean nine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evo / original | 79.11 | 77.96 | 57.16 | 55.31 | 55.41 | 94.29 | 86.93 | 71.02 | 74.29 | 57.40 | 69.97 |
| LogReg / original | 78.87 | 77.67 | 57.83 | 55.57 | 54.90 | 94.08 | 86.18 | 71.34 | 73.89 | 57.22 | 69.85 |
| Evo / neutral | 80.64 | 79.69 | 64.21 | 53.82 | 60.30 | 98.77 | 88.47 | 68.01 | 79.12 | 65.84 | 73.14 |
| LogReg / neutral | 80.62 | 79.72 | 64.08 | 54.19 | 59.84 | 99.02 | 88.89 | 68.33 | 80.03 | 65.21 | 73.26 |
| Evo / diverse | 79.17 | 78.44 | 63.89 | 51.30 | 62.94 | 99.35 | 86.85 | 65.15 | 80.32 | 68.83 | 73.01 |
| LogReg / diverse | 79.42 | 78.63 | 63.91 | 51.03 | 62.96 | 99.24 | 85.55 | 64.94 | 79.47 | 68.82 | 72.73 |
| Evo / compositional | 79.96 | 79.08 | 63.04 | 50.66 | 58.66 | 94.88 | 84.37 | 65.31 | 73.13 | 56.70 | 69.54 |
| LogReg / compositional | 80.27 | 79.26 | 62.77 | 50.11 | 58.49 | 94.86 | 87.85 | 66.83 | 74.95 | 57.78 | 70.32 |

| Formula | Evo mean | LogReg mean | Difference, pp | Chosen C | Evo diagnostic validation | LogReg diagnostic validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 69.973 | 69.855 | -0.118 | 100 | 78.727 | 78.464 |
| neutral | 73.137 | 73.257 | +0.121 | 0.01 | 80.962 | 81.132 |
| diverse | 73.007 | 72.728 | -0.280 | 10 | 80.417 | 80.387 |
| compositional | 69.538 | 70.323 | +0.785 | 1 | 78.910 | 79.374 |

## Coefficients on the same feature definitions

Here `φ` denotes the unweighted exported component. The original formula is approximately `Σ a_j φ_j`, with its original finite-precision output retained for evaluation. Logistic score is `b + Σ w_j φ_j`; sigmoid is unnecessary for AUROC. **The raw-input w values below are directly comparable in units with a.** They are not the standardized coefficients. Multiplying an entire score by a positive constant does not change AUROC, so absolute coefficient size alone is not importance.

### original

| Feature φ | Evolution a | Logistic raw-input w | Evolution, standardized basis | Logistic, standardized basis |
| --- | ---: | ---: | ---: | ---: |
| original/normalized_slope | 0.7 | 74.0404 | 0.0158216 | 1.67348 |
| original/inverse_tail_range | 0.3 | 27.7259 | 0.00775013 | 0.716266 |

Raw-input intercept **4.5451556**; standardized intercept **1.435182**.
Pearson correlation between the raw coefficient vectors: **not interpreted for a two-entry vector**.

### neutral

| Feature φ | Evolution a | Logistic raw-input w | Evolution, standardized basis | Logistic, standardized basis |
| --- | ---: | ---: | ---: | ---: |
| neutral/total_nll | 1 | 0.0201933 | 15.187 | 0.306675 |
| neutral/weighted_gap | 4 | 0.106827 | 29.5374 | 0.788844 |
| neutral/tail_gap_variance | 2 | 0.060672 | 17.2693 | 0.523883 |

Raw-input intercept **-5.568203**; standardized intercept **1.5372851**.
Pearson correlation between the raw coefficient vectors: **0.988422**.

### diverse

| Feature φ | Evolution a | Logistic raw-input w | Evolution, standardized basis | Logistic, standardized basis |
| --- | ---: | ---: | ---: | ---: |
| diverse/total_nll | 1 | 0.0614834 | 15.187 | 0.933749 |
| diverse/relative_gap_drift | 2 | 2.46544 | 0.188876 | 0.232831 |
| diverse/tail_gap_variance | 2.5 | 0.0879467 | 22.8548 | 0.804003 |

Raw-input intercept **-5.4408394**; standardized intercept **1.6341978**.
Pearson correlation between the raw coefficient vectors: **0.198387**.

### compositional

| Feature φ | Evolution a | Logistic raw-input w | Evolution, standardized basis | Logistic, standardized basis |
| --- | ---: | ---: | ---: | ---: |
| compositional/tail_surprisal_variance | 0.4 | -0.148517 | 0.747863 | -0.277675 |
| compositional/tail_surprisal_mean | 0.3 | 0.84516 | 0.133263 | 0.375429 |
| compositional/global_correlation | 3 | 24.2145 | 0.155261 | 1.25319 |

Raw-input intercept **15.37532**; standardized intercept **1.6167232**.
Pearson correlation between the raw coefficient vectors: **0.997640**.

## Feature definitions

- **original/normalized_slope**: Unweighted exported full-minus-drop component: 0.7 times negative Pearson(position, token log-probability), divided by mean absolute token log-probability plus 1e-8; divided by 0.7.
- **original/inverse_tail_range**: Unweighted exported full-minus-drop component: 0.3 / (1 + range(last 10% token log-probabilities)), divided by the same mean absolute token log-probability plus 1e-8; divided by 0.3.
- **neutral/total_nll**: Unweighted exported full-minus-drop component: -sum(token log-probabilities); coefficient 1; divided by 1.
- **neutral/weighted_gap**: Unweighted exported full-minus-drop component: 4 times sum_t [(t/(n-1)) * gap_t], with the original n=1 special case retained; divided by 4.
- **neutral/tail_gap_variance**: Unweighted exported full-minus-drop component: 2 times population variance of gaps from floor(0.75*n) to n; divided by 2.
- **diverse/total_nll**: Unweighted exported full-minus-drop component: -sum(token log-probabilities); coefficient 1, on the final-sum branch; divided by 1.
- **diverse/relative_gap_drift**: Unweighted exported full-minus-drop component: 2 times (mean(last-quarter gaps)-mean(first-quarter gaps))/max(mean(first-quarter gaps),1e-6), q=floor(n/4), last quarter starts at 3q; divided by 2.
- **diverse/tail_gap_variance**: Unweighted exported full-minus-drop component: 2.5 times population variance of gaps from 3*floor(n/4) to n; divided by 2.5.
- **compositional/tail_surprisal_variance**: Unweighted exported full-minus-drop component: 0.4 times population variance of negative log-probabilities in the last 10%; VAR_SCALE=1; divided by 0.4.
- **compositional/tail_surprisal_mean**: Unweighted exported full-minus-drop component: 0.3 times mean negative log-probability in the last 20%; MEAN_SCALE=1; divided by 0.3.
- **compositional/global_correlation**: Unweighted exported full-minus-drop component: 0.3 times [-10 * Pearson(position, token log-probability)], effective coefficient 3; divided by 3.

## Directed AUROC companion

These retain the intended hallucination-score direction, without dataset-specific folding. Fit and validation are the same in both presentations.

| Method | PopQA test | AVeriTeC | X-Fact | UHEAD | Cities | Companies | Common Claims | CounterFact | WildHallucinations | Mean nine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evo / original | 77.96 | 57.16 | 55.31 | 44.59 | 94.29 | 86.93 | 71.02 | 74.29 | 57.40 | 68.77 |
| LogReg / original | 77.67 | 57.83 | 55.57 | 45.10 | 94.08 | 86.18 | 71.34 | 73.89 | 57.22 | 68.77 |
| Evo / neutral | 79.69 | 64.21 | 53.82 | 39.70 | 98.77 | 88.47 | 68.01 | 79.12 | 65.84 | 70.85 |
| LogReg / neutral | 79.72 | 64.08 | 54.19 | 40.16 | 99.02 | 88.89 | 68.33 | 80.03 | 65.21 | 71.07 |
| Evo / diverse | 78.44 | 63.89 | 51.30 | 37.06 | 99.35 | 86.85 | 65.15 | 80.32 | 68.83 | 70.13 |
| LogReg / diverse | 78.63 | 63.91 | 51.03 | 37.04 | 99.24 | 85.55 | 64.94 | 79.47 | 68.82 | 69.85 |
| Evo / compositional | 79.08 | 63.04 | 49.34 | 41.34 | 94.88 | 84.37 | 65.31 | 73.13 | 56.70 | 67.47 |
| LogReg / compositional | 79.26 | 62.77 | 49.89 | 41.51 | 94.86 | 87.85 | 66.83 | 74.95 | 57.78 | 68.41 |

## Numerical and interpretation limits

The saved feature pool consists of **full-minus-drop** vectors, preserving each frozen program's output rounding and guards. For this follow-up each vector is divided by its original effective coefficient to define φ. This is a controlled numerical approximation to extracting each native arithmetic term, not a bitwise decomposition. It does not alter frozen sources.

Across all training/validation and nine-dataset comparisons, maximum reconstruction score error is **0.000183105469**, and maximum AUROC discrepancy is **0.000142068056 percentage points**. Both passed the predeclared limits of 0.001 score and 0.001 pp. Original Evo table cells use saved outputs, not this reconstruction.

All four formula families were retained. No held-out score selected features, C, a model, a coefficient sign or a replacement formula. Nevertheless the candidates were discovered using validation information and the benchmark had already been viewed, so this is a **post-benchmark exploratory follow-up**, not independent confirmation. Outer validation is diagnostic; the grouped refit split differs from the original evolutionary split. Claim-grouping is not entity/relation separation. Test overlap masks are retained in the exported predictions.

Coefficient correlations with only two features are uninformative; three-feature correlations are descriptive, not evidence of a general search/gradient distinction. Correlated predictors and regularization affect coefficients. AUROC does not measure calibration, and bootstrap/search-seed significance is not claimed here.

Runtime: **14.40 CPU wall-clock seconds**, excluding subsequent independent validation and report packaging. Python 3.12.3, NumPy 2.5.4, scikit-learn 1.9.1. This newly fitted local runtime is recorded separately from the historical GPU environment.
