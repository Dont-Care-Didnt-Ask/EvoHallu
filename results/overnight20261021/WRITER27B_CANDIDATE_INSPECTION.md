# Qwen3.8-27B validation-selected program inspection

**Both seed-2107 runs completed all sixty requests.** This audit reads their actual final sources, configuration/selection records and staged identities. Both staged sources match the runners byte-for-byte, and all **twelve pinned core/support files** match the execution manifest. No program was executed, repaired, refitted, rescored or selected using benchmark quality. Full hashes/provenance are in [JSON](WRITER27B_CANDIDATE_INSPECTION.json).

The code writer is **Qwen/Qwen3.8-27B**, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The feature model remains **Llama3.1-8b-instruct**, operating on the same supplied claims and cached features. These runs evolve deterministic detector formulas; they do not regenerate benchmark claims or train a language model.

| Policy / generation seed | Valid / requests | Directed PopQA validation | Selected candidate | Frozen source |
| --- | ---: | ---: | --- | --- |
| Original /2107 | 52 /60 | 79.5366 | `5a782bb1ff22f8fd0e9be3d5ce8ebf862b4f764a25fb805c8d686f101727c48d` | [source](frozen/next27b_evo_original_2107__primary/frozen_estimator.py) |
| Diverse /2107 | 56 /60 | 80.1645 | `4f1cce8a65a70070726f9d95e48f54c1fdd46fd99e83503570367d7a811dee97` | [source](frozen/next27b_evo_diverse_2107__primary/frozen_estimator.py) |

The diverse policy was selected by the predeclared **mean validation maximum over all three completed source-9B seeds**, among the non-original policies. Those means were neutral75.7504, diverse77.3867 and compositional77.2458. Original's78.2091 was retained as the reference and excluded from intervention selection. The new writer comparison uses **one generation seed**, rather than a new three-seed writer study. Validity fractions and formulas are descriptive; they do not establish a stable writer ranking.

## Original: global correlation plus local relative drops

Let `lp[t]` be the cached observed-token logprob, `n=len(lp)`, and
`W_j = lp[max(0,j−5):min(n,j+6)]`. For ordinary inputs the score is:

```text
r = PearsonCorrelation(arange(n), lp)
z_j = (lp[j] − mean(W_j)) / std(W_j)
score = −r + 0.5 * mean_j(max(0, −z_j))
```

Exact guards and implementation details:

- `n<2` returns zero. Global `std(lp)<1e−6`, or a NaN Pearson result, sets `r=0`.
- Windows have maximum width11, truncate at sequence edges and include the scored token itself. A window standard deviation below `1e−6` sets its `z_j=0`. Standard deviations use NumPy's population convention, `ddof=0`.
- **Every** below-local-mean token contributes through `max(0,−z_j)`; there is no additional statistical-significance threshold. The anomaly sum is divided by sequence length before its coefficient0.5 is applied.
- Observed logprobs, positions, window operations and reductions use **float64**. Only the final batch output is cast to **float32**.

The program's “LogProbSlope” is **negative Pearson correlation**, not a fitted regression slope. The second term measures relative local drops; it is not total NLL or a raw low-probability threshold. Both components are per-claim statistics, with no normalization across samples in the scoring batch. The docstring's proposed future top-k entropy feature is **not implemented**.

Source SHA-256: `f6977a70705aa15865a906e030ae09c9a2ea5b6ee1cdf2ebfd67b809811b1d89`.

## Diverse: quadratic later-position-weighted NLL

This winner implements **one aggregation family**, with no separate variance/entropy feature:

```text
n > 1: score = sum_t [−float32(lp[t]) * (float32(t)/(n−1))²]
n = 1: score = −float32(lp[0])
n = 0: score = 0
```

Logprobs, token indices, weights, multiplication and the sum use **float32**; the intermediate sum is converted to a Python float and the final output array is float32. For `n>1`, the first weight is0 and the final weight is1. The weights are bounded, unlike the overflowing exponential weights in the separate shortlist failure.

The score is a **sum**, not a weighted mean. Ignoring float32 rounding, weight mass is `n*(2*n−1)/(6*(n−1))`, approximately `n/3`. It therefore retains an accumulated-confidence/length contribution while emphasizing later tokens more strongly. It differs from the source9B diverse/4319 winner's linear weights `(t+1)/n`, including a zero first weight here. The docstring's future top-k-gap variance is **not calculated**.

Source SHA-256: `6a399c68edd3d47e2ad09a611d44204f378514103329a4205422f08a6cb336a3`.

## Alignment, branches and interpretation limits

Both classes declare and numerically consume **only `sequence_logprobs`**. Neither reads top-k logits, token-ID values, supplied labels or other claims in the batch. Consequently neither introduces an extra-top-k-row alignment problem or a batch-MAD dependency. Positions refer to the cached observed-token sequence, which includes its template context; they cannot automatically be interpreted as positions solely inside the raw claim.

No mathematically dead scoring term was identified. Short-sequence/constant-window guards are legitimate input-dependent branches; no new empirical test of their activation was performed. Non-executable “future hypothesis” text is not counted as an evolved component. The original winner has two implemented signals; the diverse winner is a simple weighted-NLL formula. Neither is a reconstruction of the many-component Claude example.

Selection remains the within-run **directed PopQA validation maximum including canonical-NLL fallback**, with fixed original/diverse policies and no test-best candidate or seed promotion. This source inspection makes no OOD-superiority or paper-SOTA claim. Benchmark reporting belongs to the separately saved evaluations and must retain the single-seed exploratory qualification.
