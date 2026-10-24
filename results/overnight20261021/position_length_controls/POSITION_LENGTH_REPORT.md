# Frozen 27B quadratic-position and length controls

**complete_fixed_source9_position_length_diagnostic**. Exactly four predefined source9 CPU controls; all variants retained, no fitting or primary replacement.

The original quadratic-sum winner, canonical NLL, labels, normalized-claim group IDs and duplicate-disjoint masks matched all nine saved primary arrays byte-for-byte before any control scoring. All calculations use the complete existing cached LP span and the same 512-row scoring chunks; no top-k pack, GPU calls, new features or model calls.

| Method | Source9 directed | Source9 folded | Folded gain vs NLL [95% CI] | Strict folded |
| --- | ---: | ---: | ---: | ---: |
| canonical_nll | 65.644 | 68.711 | +0.000 [0.000, 0.000] | 68.685 |
| frozen_quadratic_sum | 70.297 | 72.306 | +3.595 [2.909, 4.058] | 72.229 |
| lp_total_length | 49.368 | 54.634 | -14.077 [-14.797, -13.031] | 54.623 |
| float32_nll_sum | 65.643 | 68.712 | +0.000 [-0.004, 0.005] | 68.686 |
| float32_nll_mean | 64.176 | 64.176 | -4.535 [-5.825, -3.087] | 64.086 |
| float32_quadratic_mean | 69.866 | 70.724 | +2.013 [0.952, 2.871] | 70.636 |

The quadratic mean divides the exact float32 weighted sum by the float32 sum of the same weights. It preserves n=0 score0 and n=1 weight1. Uniform sum/mean also use float32 reductions; canonical NLL preserves the historical float16 reduction. The length control measures total cached-LP length, whereas historical ClaimLength counts claim tokens.

The three predefined contrasts compare uniform sum versus uniform mean (accumulation), quadratic sum versus quadratic mean (weight mass), and quadratic mean versus uniform mean (positional emphasis). These are sensitivity contrasts: AUROC is nonlinear, so their effects are not a causal additive decomposition.

Intervals use 500 shared normalized-claim bootstrap multiplicities per dataset. Strict sensitivity restricts those same draws to claims absent from the entire PopQA training CSV. Fold each dataset within each draw, then average all nine equally; only replicates finite for every method and dataset enter intervals. No incomplete dataset macro is reported.

Historical context (already available; different spans/precision where stated):

| Historical method | Directed | Folded |
| --- | ---: | ---: |
| alignment/ClaimLength | 49.368 | 54.634 |
| alignment/ClaimOnlyMeanSurprisal | 66.339 | 66.937 |
| original/Perplexity | 64.175 | 64.175 |
| original/ExpWeightedNegLogProb | 69.797 | 70.606 |
| new/ablation_qwen_linear_position_nll | 69.793 | 72.000 |

Historical Perplexity implements -exp(mean(LP)); this is monotonic in mean NLL in real arithmetic, with float16 reduction/exponential ties. Historical exponential-weighted NLL normalizes weights to unit mass. The linear-position ablation retains accumulated length dependence. These are contextual results, not newly scored controls.

This diagnostic was designed after benchmark results were viewed and is exploratory. No highest-scoring variant is selected or promoted. Bootstrap uncertainty conditions on this one frozen validation-selected program; it does not estimate writer/search-seed uncertainty. No target1B claim is made. The original primary and all execution/evolution manifests remain unchanged.
