# Initial candidate inspection — partial overnight snapshot

Snapshot: **21 October 2026, approximately 18:43 UTC**. This inspects saved source, training predictions and PopQA validation predictions only. No nine-dataset test scores were read, and no candidate code was changed or repaired.

The writer is **Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Its Python detectors consume the existing **Llama3.1-8b-instruct** features. Labels remain `is_hallu`; training/validation contain 5,605/1,402 rows. The fixed NLL baseline is **74.3286 / 75.7504 AUROC** on those partitions. All numbers below are directed PopQA AUROC, **not the paper's nine-dataset macro**.

The completed neutral job is a **15-context × 2-sibling frozen-context prompt screen**, not a new 30-round adaptive evolutionary run. Other policies/searches are incomplete; these results cannot establish which prompt policy is better.

| Screen at inspection | Saved requests | Valid candidates | Status |
| --- | ---: | ---: | --- |
| Neutral | 30/30 | 19 | Finished |
| Original | 26/30 | 17 | Running; ordering can change |
| Diverse | First context available | Incomplete | Running |
| Compositional | No completed local records | — | Pending |

## Completed neutral screen: validation top three

Ranks use validation only, with no test selection. Context numbers are one-based; sibling indices are zero-based.

| Rank / candidate hash prefix | Context / sibling | Train AUROC | Validation AUROC | Gain over validation NLL |
| --- | --- | ---: | ---: | ---: |
| 1 — `0a747b101a0b` | 12 / 0 | 80.7778 | **80.4055** | +4.6551 pp |
| 2 — `f08d667084de` | 5 / 0 | 79.1275 | **79.7566** | +4.0062 pp |
| 3 — `385728e4b958` | 12 / 1 | 78.2910 | **78.9083** | +3.1579 pp |

Let `l[t]` be the observed token's cached log probability, `N = −sum(l)`, `w[t] = t/(n−1)`, and `g[t] = log(q_top1[t]) − l[t]`, where `q` normalizes **only the top 512 logits**.

1. **Tail-weighted gap plus tail instability:** `N + 4·sum(w·g) + 2·Var(g[last 25%])`. This combines sequence surprise, later-position confidence gaps and a tail variance. [Actual source](../../out/logs/research20261021/uncertainty_quantification_next_prompt_neutral_2107/validation_shortlist/rank_01_0a747b101a0b.py).
2. **Tail instability relative to the head:** `N + 5·weighted_var(l[last 40%]) / (mean(abs(l[first 60%])) + 1e−6)`. Tail variance uses increasing weights `1,…,tail_length`. This program uses sequence log probabilities only. [Actual source](../../out/logs/research20261021/uncertainty_quantification_next_prompt_neutral_2107/validation_shortlist/rank_02_f08d667084de.py).
3. **Gap mean plus second-moment contrast:** `N + 2.5·mu + 1.5·C`, with `mu = sum(w·g)/sum(w)` and `C = sum((w·g)^2)/sum(w) − mu^2`. The source calls `C` a “weighted variance”, but the numerator uses **squared weights**. It is not the usual weighted variance and can be negative. The measured detector is this exact formula, without correction. [Actual source](../../out/logs/research20261021/uncertainty_quantification_next_prompt_neutral_2107/validation_shortlist/rank_03_385728e4b958.py).

These are not aliases of NLL or the supplied `LogProbSlope`:

| Neutral rank | Spearman vs NLL, all 5,605 training rows | Spearman vs NLL, fixed 512-row training probe | Spearman vs supplied LogProbSlope, validation |
| --- | ---: | ---: | ---: |
| 1 | 0.773963 | 0.796233 | 0.872428 |
| 2 | 0.936243 | 0.931708 | 0.795209 |
| 3 | 0.918651 | 0.909139 | 0.824426 |

Their saved scores/rankings differ from the references. The top-three training ranking correlations with each other are 0.9051, 0.8942 and 0.9776: related feature families, with measurable behavioral differences. This is descriptive, not an independent quality confirmation.

## Feature semantics and inactive computations

**Top-512 normalization is conditional.** In candidates 1 and 3, the “top-1 log probability” is `max(logits) − logsumexp(top512_logits)`, not a full-vocabulary probability. Ideally, their gap equals the full-vocabulary top-1 gap **minus log(top-512 probability mass)**. It therefore also contains a normalization/mass term. These are valid numeric features, but their docstrings overstate the interpretation as the exact full-vocabulary confidence gap. Both gap candidates declare `top_k_indices` without using it; candidate 2 needs sequence log probabilities only.

**No `n` versus `n+1` dead branch occurs in these leaders.** The audited cache has `len(top_k_logits) = len(sequence_logprobs)+1`. By the precompute definition, `l[t]` scores the next input token using logits row `t`; the two gap candidates use that row and leave the final extra row unused. Candidate 2 does not use top-k features. None contains an equality guard disabling its added feature on this shape. This differs from the historical dead-branch example discussed earlier.

**The sequence includes the cached chat template.** Its tail is not necessarily the tail of claim text alone. This follows the unchanged extraction/evaluation representation; it is an interpretation limit, not a label change or a reopening of the resolved baseline reporting issue.

## What the unfinished original screen currently contains

- Current leader `8645811f2d67`, context 13/sibling 0: **76.6820 train / 76.0421 validation**. Formula: `0.7·[−corr(position, l)] + 0.3·[longest consecutive run below median(l) / n]`. It extends the supplied slope feature with a low-confidence streak. Validation rank correlation with `LogProbSlope` is **0.922269**, so it is not an exact alias.
- Runner-up `838acd05eba9`, context 8/sibling 1: **75.7545 validation**. It adds an approximate bottom-rank statistic to NLL, matching full-vocabulary observed log probabilities against conditionally normalized top-512 values. Its full-training rank correlation with NLL is **0.9999926**; the addition is almost a constant offset (approximately 2.45–2.53). Its nominally new feature barely changes the ranking.
- `03b82ca118be`, context 6/sibling 1: **75.7504 validation**, **exactly NLL on all 5,605 saved training rows**. It sets `chosen_id = top_k_indices[t,0]`, then finds that ID in the same row. The first match is necessarily column zero, so every added rank is zero. This is an inactive feature, not a new rank detector.

Two additional valid programs return constant training scores: neutral `01ab815d2e86` returns 12.7429428, and original `8728f6834551` returns zero; both have AUROC 50. The former weights positions as `0.1^t`, concentrating on the shared early template; the latter exponentiates raw logits without normalization. Constant outputs are confirmed from saved arrays; the explanations are static diagnoses, without code modification.

## Scope of this inspection

The neutral screen demonstrates that new combinations of positional gaps and instability can be produced, with a higher **selected validation** score in this run. It does not yet demonstrate an improvement on held-out datasets or superiority of a prompt policy. No complexity penalty, feature-count ceiling or preference for shorter formulas was applied.

The later inspection should list the validation-frozen **primary winner of every completed evolutionary/control run**, retaining all seeds and distinguishing frozen-context screening from adaptive search. The present screen candidates remain an explicitly separate diagnostic cohort.

Provenance: neutral `candidates.jsonl` SHA256 `a0c4d1a1352ab7594401096c4c772efddaa4863dc3816facf33cab5e84eebcba`; original 26-record snapshot SHA256 `5e6adbf762e6db98ea10d7ac8a43f07d4b884fbef1cd04b3b0fd79911c8045f3`. Comparisons used saved training arrays and `out/campaign/original_validation/stratified_val_scores.npz`; its NLL validation vector was verified identical to both new jobs' saved baseline vectors. The 1.1-GB raw cache was not loaded for this inspection.
