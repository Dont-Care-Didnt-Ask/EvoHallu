# Matched prompt-screen analysis

15 fixed historical parent contexts × four prompt policies × two seeded siblings = 120 requests.

**Status: complete_matched_screen**. Full paired comparisons require all four 30-request screens and complete provenance/probe diagnostics.

The primary outcome counts contexts with two valid, nonconstant, behaviorally different siblings (Spearman rho <0.995), divided by **all 15 contexts**. Conditional correlation is a separate descriptive measure among eligible pairs. No validation/test score magnitude is used.

| Policy | Saved requests / 30 | Completed contexts / 15 | Valid requests | Contexts with 0 / 1 / 2 valid | Joint diverse pairs / 15 | Conditional rho mean (pairs) | Truncated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| original | 30/30 | 15/15 | 19 | 3 / 5 / 7 | 6/15 | 0.0691 (6) | 0 |
| neutral | 30/30 | 15/15 | 19 | 2 / 7 / 6 | 4/15 | 0.4485 (4) | 0 |
| diverse | 30/30 | 15/15 | 23 | 0 / 7 / 8 | 7/15 | 0.1805 (7) | 0 |
| compositional | 30/30 | 15/15 | 11 | 7 / 5 / 3 | 3/15 | 0.3961 (3) | 0 |

## Paired context comparisons

Percentile 95% intervals from 2,000 paired resamples of the same 15 contexts; exploratory and unadjusted for multiple comparisons.

| Policy vs original | Valid-request change, pp [95% interval] | Joint-diverse-context change, pp [95% interval] |
| --- | ---: | ---: |
| neutral | 0.00 [-26.67, 26.75] | -13.33 [-46.67, 20.00] |
| diverse | 13.33 [-13.33, 36.67] | 6.67 [-26.67, 40.00] |
| compositional | -26.67 [-60.00, 10.00] | -20.00 [-53.33, 20.00] |

## Resource accounting

| Policy | Prompt tokens | Completion tokens | Total tokens | Sum request generation seconds | Sum round maximum generation seconds | Run elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 57066 | 55534 | 112600 | 675.7 | 390.3 | 1135.7848591829825 |
| neutral | 56644 | 40851 | 97495 | 505.8 | 288.1 | 845.7282882770232 |
| diverse | 57934 | 43947 | 101881 | 524.4 | 314.8 | 818.2512787549931 |
| compositional | 59344 | 57041 | 116385 | 672.0 | 380.0 | 1054.2282140599855 |

Generation requests overlap; their summed latency is not elapsed GPU time. Failure categories, candidate errors, per-context outcomes, prompt hashes and training-array hashes are retained in JSON.

## Interpretation limits

- Only 15 fixed contexts, one writer and one generation seed; bootstrap intervals describe context variability within this screen.
- Historical parent contexts come from one trajectory and need not be independent; percentile intervals are exploratory, without multiplicity adjustment.
- Diversity and interface validity are training-probe diagnostics; neither demonstrates better held-out AUROC or OOD generalization.
- Conditional sibling correlation excludes invalid/constant pairs and must be read alongside unconditional joint diversity.
- Latency includes server load/concurrency and does not measure isolated GPU throughput.
