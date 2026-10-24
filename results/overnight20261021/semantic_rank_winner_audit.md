# Semantic audit of selected rank-sampling winners

Audit timestamp: **21 October 2026, 20:03 UTC**. This inspects already selected frozen programs and saved predictions. It does not repair programs, change selection, or evaluate new candidates. CPU verification used only the existing source-training cache's observed-token log-probabilities, all **5,605 fitting + 1,402 validation rows**, and the predefined **512-row training probe**. No top-k tensors were read for the data checks; no held-out raw data/labels, model API or GPU was used. Existing nine-dataset aggregate scores below are copied from the saved report, without rescoring.

| Selected run | What the executable computes | Source validation AUROC | Existing nine-dataset paper AUROC |
| --- | --- | ---: | ---: |
| Rank sampler, seed 2107 | NLL with float32 accumulation; alleged confidence-gap term is identically zero |75.7622 |68.7117 |
| Rank sampler, seed 4319 | NLL +0.5×token-logprob variance, using float16 arithmetic |76.9672 |69.3424 |
| Rank sampler, seed 8741 | Canonical NLL baseline selected as fallback |75.7504 |68.7112 |
| Original prompt evolution, seed 2107 | On every source row: float32 NLL +0.5×float32 variance; entropy-ratio term is gated off |76.9672 |69.3436 |

Canonical baseline source validation is **75.7504** and existing nine-dataset paper AUROC **68.7112**. These saved methods were selected within each run using source validation, not by the scores in this table. Paper AUROC applies `max(AUC,100−AUC)` separately within each dataset before the nine-dataset mean.

## Rank seed 2107: sorting the wrong axis makes the proposed term dead

The frozen code contains:

```python
sorted_logits = np.sort(logits, axis=1)[::-1]
gap = np.maximum(0, sorted_logits[:, 0] - sorted_logits[:, 1])
confidence_penalty = -np.sum(gap, dtype=np.float32)
base_score = -np.sum(logprobs, dtype=np.float32)
```

`np.sort(...,axis=1)` sorts **columns in ascending order**. The final `[::-1]` reverses the **token rows**, leaving each row's columns ascending. Thus column0 ≤ column1, the clipped difference is zero, and the whole confidence penalty is zero. For finite real logits with at least two columns, the effective detector is exactly **`-sum(logprobs,dtype=float32)`**. Nine synthetic array-operator checks across float16/32/64 and different row counts also confirm this identity. The comments' claims about a top1/top2 confidence or generated-token deviation signal are not realized by this code.

Saved scores match this LP-only arithmetic **exactly on all 5,605 train rows and all 1,402 validation rows**, including every fixed-probe row. No top-k feature calculation is needed to establish the effective program on these data.

The canonical baseline sums the stored float16 array without a dtype override, then returns float32. The selected program sums the **same stored float16 values** with float32 accumulation. It consequently changes many ties: validation has **418 unique baseline scores versus 1,390** selected scores, with Spearman **0.9999903**. Its validation change is only **+0.0117807 percentage points**; training changes slightly downward, **74.328583→74.326088**. This is a numerical NLL variant, not a new hallucination signal.

## Rank seed 4319 and original seed 2107: same basic family, different precision

Rank 4319 directly computes `-np.sum(lp) +0.5*np.var(lp)` and writes the result into a float32 output array. The input reductions and scalar arithmetic retain float16 precision under this environment. Saved predictions match that exact arithmetic on every train/validation row. This is a real simple variance feature, rather than the dead-gap variant.

The original-prompt winner additionally calculates conditional top-k entropy divided by `-max(lp)`, but only if `max(lp)<−1e-6`. **All 7,007 source sequences have maximum stored observed-token logprob exactly 0**, so that branch is false everywhere. Its actual source predictions are exactly float32 NLL +0.5×float32 variance. A zero in the quantized cache does not prove a full-precision probability exactly 1.

The two variance programs are **not bitwise identical**: maximum validation prediction difference is **0.132263**, Spearman **0.9999818**, and every saved validation value differs. Nevertheless their validation AUROC is **exactly 76.9672298678%**. Matching rounded AUROC alone would not have established executable equivalence. The original program can activate its entropy branch on another input with sufficiently negative maximum logprob, so its simplification is **source-data-specific**, not a global symbolic identity. No OOD branch-equivalence claim is made here.

## Interpretation

The saved three-seed rank-sampler mean is **68.9217 paper AUROC** (directed 65.8640), compared with canonical baseline 68.7112. Its selected winners consist of one baseline, one accumulation variant and one NLL-plus-variance detector. The mean gain is therefore mostly the simple variance winner; it does not demonstrate new complex-feature diversity or establish that rank-based parent sampling improves evolution. All completed matched ordinary-neutral seeds are needed for that comparison. This audit concerns selected winners, not the diversity of every generated candidate.

Keep the original frozen sources and reported numbers. Describe dead terms and precision variants explicitly rather than silently correcting programs after selection. `semantic_rank_winner_audit.json` records selected candidate/source/prediction hashes, complete arithmetic checks, probe results, branch counts and the saved-report timestamp.
