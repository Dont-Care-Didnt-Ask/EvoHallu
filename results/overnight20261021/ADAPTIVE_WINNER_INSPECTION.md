# Adaptive winner inspection

Snapshot **2026-10-21T22:14:25.954709+00:00**; all **12 adaptive runs completed 60/60 requests**,720 requests total. Writer **Qwen3.5-9B**; fixed feature model **Llama3.1-8b-instruct**. Each primary was already frozen by directed PopQA validation on 1,402 rows; canonical NLL fallback validation 75.7504. This inspection reads sources and saved summaries only: **no candidate execution, model calls, repairs or benchmark-based promotion**.

`lp[t]` denotes cached observed-token logprob, `n=len(lp)`, `NLL32=-sum(lp,dtype=float32)`. Cached baseline reduction uses float16 before output casting. Correlation means Pearson, not OLS slope. Exact source hashes and complete formulas/guards are recorded in [JSON](ADAPTIVE_WINNER_INSPECTION.json).

| Policy / seed | Valid /60 | Validation AUROC | Existing candidate | Effective feature family | Frozen source SHA |
| --- | ---: | ---: | --- | --- | --- |
| original /2107 | 30/60 | 76.9672 | 744d5e48aae6 | float32 total NLL; float32 token surprisal variance; conditional top-512 entropy ratio | [39c2607087c0](frozen/next_evo_original_2107__primary/frozen_estimator.py) |
| original /4319 | 36/60 | 79.2061 | 01ab33c6722e | global negative positional correlation; recent surprisal EMA relative to global mean | [eabfdd547677](frozen/next_evo_original_4319__primary/frozen_estimator.py) |
| original /8741 | 35/60 | 78.4540 | 74adf6f7b19d | global negative positional correlation; local/global standard-deviation ratio around minimum adjacent difference | [baf4ddab4136](frozen/next_evo_original_8741__primary/frozen_estimator.py) |
| neutral /2107 | 29/60 | 75.7504 | baseline | canonical SequenceLogProb/NLL baseline | [2441fedd58e6](frozen/next_evo_neutral_2107__primary/frozen_estimator.py) |
| neutral /4319 | 36/60 | 75.7504 | baseline | canonical SequenceLogProb/NLL baseline | [2441fedd58e6](frozen/next_evo_neutral_4319__primary/frozen_estimator.py) |
| neutral /8741 | 33/60 | 75.7504 | baseline | canonical SequenceLogProb/NLL baseline | [2441fedd58e6](frozen/next_evo_neutral_8741__primary/frozen_estimator.py) |
| diverse /2107 | 41/60 | 76.8771 | 231e2ec0f779 | strict low-logprob thresholded partial NLL | [8839fa3cd064](frozen/next_evo_diverse_2107__primary/frozen_estimator.py) |
| diverse /4319 | 25/60 | 79.5207 | 4edc60e6fc8a | linearly later-weighted sequence surprisal | [403f7b69e0ea](frozen/next_evo_diverse_4319__primary/frozen_estimator.py) |
| diverse /8741 | 37/60 | 75.7622 | 271d0a456048 | float32-accumulated total NLL | [66be09b1f829](frozen/next_evo_diverse_8741__primary/frozen_estimator.py) |
| compositional /2107 | 27/60 | 76.8237 | 697c9e4d9816 | tail-vs-head relative surprisal; tail-vs-head raw-logit spread contrast; tail-vs-head conditional top-512 entropy contrast | [0be32c5d6407](frozen/next_evo_compositional_2107__primary/frozen_estimator.py) |
| compositional /4319 | 29/60 | 76.6394 | 4803d0d3622d | mean sequence surprisal; last20% logprob variance; normalized top1/top2 logit gap complement; last30% mean adjacent absolute change | [71795a8c291b](frozen/next_evo_compositional_4319__primary/frozen_estimator.py) |
| compositional /8741 | 22/60 | 78.2742 | eac31045503b | canonical-accumulation total NLL; mean width5 stride2 local variance; mean largest/least-negative10% observed logprobs; maximum width5 stride2 local absolute deviation | [a96fdf5af49a](frozen/next_evo_compositional_8741__primary/frozen_estimator.py) |

## Exact effective formulas and caveats

**original /2107:** `NLL32 + 0.5*Var32(-lp) + mean_top512_entropy/(-max(lp)) if max(lp)<-1e-6, else NLL32 + 0.5*Var32(-lp)`.
Prior saved-data audit verified max(lp)==0 for every one of 7007 source training rows, so the entropy-ratio contribution is zero there; not a universal or OOD identity. Top-k indices are declared but not used numerically; entropy includes the extra final logits row.

**original /4319:** `-PearsonCorr(position,lp) + EMA32(-lp, alpha=0.5)[last]/mean32(-lp), ratio=0 if mean<=1e-6`.
EMA starts at the first surprisal; this is the final EMA, not a weighted sum over the whole sequence. Top-k arrays/indices are zipped but their numeric values do not enter the score. Constant or one-token sequences can make Pearson undefined; no new failure test was run.

**original /8741:** `-PearsonCorr(position,lp32) * (1 + std(lp32[W])/(std(lp32)+1e-8)); j=argmin(diff(lp32)); W=[max(0,j-2):min(n,j+3)]`.
The variable named local_var is a standard deviation, not variance. There is no division by sequence length; the normalization is by global standard deviation. Minimum adjacent difference need not be negative for every possible input; constant sequences can give NaN correlation.

**neutral /2107:** `-sum(lp) in native cached float16 precision, then float32 output`.
Baseline fallback won validation; no generated candidate exceeded its directed validation AUROC. This is the canonical unchanged baseline, not evidence of a newly discovered formula.

**neutral /4319:** `-sum(lp) in native cached float16 precision, then float32 output`.
Baseline fallback won validation; no generated candidate exceeded its directed validation AUROC. This is the canonical unchanged baseline, not evidence of a newly discovered formula.

**neutral /8741:** `-sum(lp) in native cached float16 precision, then float32 output`.
Baseline fallback won validation; no generated candidate exceeded its directed validation AUROC. This is the canonical unchanged baseline, not evidence of a newly discovered formula.

**diverse /2107:** `-sum(lp[lp < -5.0]), with native cached-array accumulation then float32 output`.
Threshold is fixed by generated code, not fitted afterward. This discards more confident tokens rather than adding a separate feature to NLL. Thresholding and reduction depend on stored float16 precision.

**diverse /4319:** `sum_t((-lp[t]) * (t+1)/n), weights float32`.
This is not a length-normalized mean; the weighting still sums over tokens. No top-k dependency or additional fitted coefficient.

**diverse /8741:** `-sum(lp,dtype=float32), output float32`.
This is an NLL precision variant, not a new uncertainty feature. Canonical baseline sums the cached float16 array before casting its output; the distinct summation can change ties. Prior rank-winner audit established the same precision-only family; no new candidate execution here.

**compositional /2107:** `max_{t>=floor(.7*n)} [0.4*(u_t-u_H)/u_H + 0.35*(F_H-F_t)/(F_H+1e-8) + 0.25*(H_t-H_H)/(H_H+1e-8)], with original denominator guards`.
u=-lp; H denotes the first floor(n/2) token positions; F_t=max(top512_logits[t])-mean(top512_logits[t]); H_t is conditional top-512 entropy. Head-empty and denominator-near-zero branches remain exactly as generated. Uses logits row t for observed-token logprob t and leaves final extra row unused. q25=sorted_logits[25//512] selects index0 but is unused, as is chosen_logit; neither affects the score. The score takes the maximum combined tail contrast, not an average of the tail.

**compositional /4319:** `0.25*(-mean(lp) + Var(lp[floor(.8*n):]) + 1-mean_rows(clip((top1-top2)/(abs(top1)+1e-9),0,1)) + mean(abs(diff(lp[floor(.7*n):]))))`.
Original empty/short-segment guards set corresponding terms to0. Gap averages all n+1 logits rows; LP-based terms use n predicted-token logprobs. This extra-row weighting is real; do not silently trim it. Dividing the gap by abs(raw top1 logit) breaks invariance to a constant logit shift and is fragile near zero; it is not a probability gap. Top-k indices are declared but unused numerically.

**compositional /8741:** `0.4*R_B(f1) + 0.25*R_B(f2) + 0.2*R_B(f3) + 0.15*R_B(f4)`.
R_B(f)=(f-median_B(f))/(1.4826*median_B(abs(f-median_B(f)))); if MAD<1e-8 the whole feature vector becomes0. B is the current sample batch in __call__. This is a batch-dependent, unsupervised set statistic. Changing chunk composition changes relative feature scales and can change ranking; not an independent pointwise score. f1=-sum(lp); f2=mean of width5 stride2 variances; f3=mean(sort(lp)[-floor(.1*n):]); f4=max over width5 stride2 windows of max(abs(window-mean(window))). f3 selects the largest, most confident logprobs, not bottom10% severity. When n<10, -0 slicing selects the entire sequence. It may be constant in some batches, but this inspection does not establish that empirically. For n<=5, f2 uses whole-sequence variance and f4 uses max(abs(lp)); empty sequences use0. For batch size1 every median absolute deviation is0, hence score0 analytically. No labels enter the batch normalization, but target/test-batch marginal statistics do. Retain original 512-row protocol and assess other chunks only as a separate diagnostic.

## What evolved

**Three neutral runs retain canonical NLL; diverse8741 retains a float32-NLL precision variant.** The other eight winners use simple thresholding, position weighting/correlation, local variation, or combinations of these with top-k spread/entropy. Original2107 has a third entropy branch that is verified inactive on all 7,007 source training rows; that simplification is data-specific. Only three programs numerically read top-k logits, and no winner uses token-ID values as a score feature. Declared dependencies and comments can therefore overstate effective novelty.

**Compositional8741 is the main new interpretation risk:** it estimates median/MAD across the current scoring chunk, so its weights are fixed but its effective feature scaling adapts to batch composition. Its advertised bottom-tail term actually selects the most confident logprobs. No term was corrected; the original 512-row source evaluations remain the primary record. **Training, validation and benchmark scoring all used 512-row chunks**, verified in the frozen trial configuration and task evaluator. There is no train-full versus test512 mismatch. The sensitivity concerns which samples share each chunk and the resulting normalization, not a failed primary reproduction. A separately declared chunk audit can characterize sensitivity without promoting another condition.

These are validation-selected executable detectors, not reconstructions of unavailable Claude programs or evidence that extra source lines improve generalization. Empirical inactivity is claimed only where a prior saved-data audit verified it; all other caveats above are static identities or possible numerical sensitivities. Sources, candidate identities and validation values are preserved exactly.

The separate fixed-case CPU [batch-sensitivity audit](batch_sensitivity/INTERPRETATION.md) has now completed. All nine original512 primary score arrays reproduce byte-for-byte; other conditions remain diagnostics and do not replace the frozen program or primary protocol.

A later label-free capture of the exact 5,605 fit rows verifies that compositional8741's f3/f4 each have **zero MAD in all 11 original 512-row training chunks**, so the existing guards deactivate their contributions throughout training. Raw feature values need not be constant; validation/OOD inactivity is not claimed. The separate train-only [normalization control](frozen_train_normalization/INTERPRETATION.md) preserves this rule and does not replace the primary.
