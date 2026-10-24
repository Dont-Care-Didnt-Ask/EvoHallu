# Interpretation: frozen source-training normalization

**The single declared control completed on all nine source datasets.** Four medians/MADs were fitted on the exact original **5,605 PopQA training rows**, with the 1,402 validation rows excluded. Labels entered only reproduction of the original stratified row split. Coefficients, feature formulas,1.4826 multiplier and zero-MAD rule were unchanged. No validation/test/target normalization statistics, coefficient tuning, new generation or GPU inference were used.

| Method | Directed AUROC | Folded AUROC |
| --- | ---: | ---: |
| Canonical NLL | 65.6439 | 68.7112 |
| Original C8741 primary,512-row batches | 66.5122 | 68.9655 |
| Separate frozen-training-stat control | 67.7833 | 70.5585 |

The control gains **1.593 folded points** over the original and **1.847** over NLL. These are exploratory point estimates; no interval or paper-SOTA claim is made.

**Batch independence is verified:** the transparent median proxy first reconstructs all nine original 512 primary/NLL/labels/groups/strict-mask arrays byte-for-byte. Then all **36 dataset×case** frozen scores and NLL checks match exactly across original-order chunks 1/128/512 and the fixed seed451 shuffle512, with positions restored. Original training, validation and benchmark scoring all used 512-row chunks; no train/test chunk-size mismatch is implicated.

| Actual feature | Training median | Training MAD | Frozen contribution |
| --- | ---: | ---: | --- |
| Total canonical NLL | 141.0 | 6.875 | retained with original weight |
| Mean width5/stride2 local logprob variance | 21.5625 | 2.734375 | retained with original weight |
| Mean largest/least-negative10% logprobs | 0.0 | 0.0 | zero by original MAD rule |
| Maximum width5/stride2 local absolute deviation | 13.359375 | 0.0 | zero by original MAD rule |

**Two components vanish by the existing rule, not by feature selection.** f3/f4 have zero fitting MAD. In the captured training features, their MAD is also zero in **all 11 original 512-row training chunks**; f1/f2 have nonzero MAD in all 11. This verifies training inactivity of these two original terms, without claiming validation/OOD inactivity of the original batch-normalized program. The frozen control always applies the stored zero-MAD rule to f3/f4.

Ignoring finite-precision roundoff, the remaining frozen score is an affine transformation of **`NLL + (11/7) * mean_local_variance`**: `(0.25/2.734375)/(0.4/6.875)=11/7`. This is an interpretation of the executed control; the original feature/normalization code was not rewritten to this algebraic form.

**CommonClaims drives the aggregate gain.** Its folded AUROC changes from 51.8898 to 66.8141 (**+14.9243 points**). Its supplied order is exactly 2,225 label-0 rows followed by 2,225 label-1 rows. Dividing this contribution across nine datasets gives +1.6583 macro points, exceeding the total gain because the other eight changes sum slightly negative. X-Fact drops 2.8943 points and Companies drops 1.2853 points. The control therefore improves this macro and gives a batch-independent score; it does not demonstrate uniform cross-dataset improvement.

This study was designed after the batch-sensitivity test outcomes were viewed, and its source program had already been selected on validation. It is a separately named exploratory control; **the original program and its 512-row primary remain unchanged**. No target-model evaluation, alternate normalization family, coefficient search, best-case selection or independent validation result is claimed.

Runtime **201.49seconds wall**, at most 4 single-thread CPU workers, CUDA hidden. Nine focused tests pass. Independent saved-array checks reproduce all training medians/MADs, the disjoint 5,605/1,402 row split indices, all 36 invariances and all source-nine macro values. [Full report](FROZEN_NORMALIZATION_REPORT.md), [JSON](FROZEN_NORMALIZATION_REPORT.json), [fit receipt](FIT_RECEIPT.json) and [prospective design](DESIGN.md).
