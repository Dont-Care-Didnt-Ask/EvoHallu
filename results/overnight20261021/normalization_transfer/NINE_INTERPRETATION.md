# Frozen normalization transfer: completed nine-dataset audit

The source-trained, batch-independent C8741 control transfers its advantage over NLL to the 1B feature model. This is a separate exploratory control; the original evolved program and every primary result remain unchanged. The same source parameters score all nine target datasets without target fitting, coefficient tuning or method selection.

| Frozen detector | Source 8B directed / folded | Target 1B directed / folded |
| --- | ---: | ---: |
| Original C8741, batches of 512 | 66.5122 / 68.9655 | 62.7792 / 64.9946 |
| Source-fixed C8741 normalization | 67.7833 / 70.5585 | 63.5637 / 66.5841 |
| Canonical NLL | 65.6439 / 68.7112 | 61.4593 / 64.8345 |

Scores are AUROC × 100, averaged equally over the same nine datasets. Folded scores use `max(AUROC, 100 − AUROC)` separately per dataset, matching the paper's reporting convention. This uses evaluation labels to choose orientation; the directed results are therefore also reported.

- Target fixed minus original: **+0.7846 directed** [0.4064, 1.1916] and **+1.5895 folded** [1.1328, 2.0179] percentage points.
- Target fixed minus NLL: **+2.1044 directed** [1.8575, 2.3520] and **+1.7496 folded** [1.4839, 2.0336].
- Exact-claim-disjoint target folded scores are fixed **66.5263**, original **64.9300**, NLL **64.7811**; fixed minus NLL is **+1.7452** [1.5082, 2.0086]. This removes 527 rows and is not an entity-disjoint evaluation.

Intervals are percentile 95% intervals from 500 paired claim resamples, shared across detectors and source/target models. They condition on the frozen source method, statistics and nine datasets; they do not include search, fitting, seed or dataset-selection uncertainty, and are not multiplicity adjusted.

## What changed from the sealed six-dataset result

All 18 historical target prediction/bootstrap/receipt files were copied byte for byte; all 21 sealed six-study artifacts remain exact. The nine-dataset result adds only CounterFact, PopQA and X-Fact. Target fixed folded AUROC changes from **69.4909 on six** to **66.5841 on nine** because coverage changes, not because the first six predictions changed. NLL changes from 67.7914 to 64.8345; the fixed-minus-NLL advantage is +1.6996 on six and +1.7496 on nine.

| Newly added target dataset | Original folded | Fixed folded | NLL folded |
| --- | ---: | ---: | ---: |
| CounterFact | 62.1187 | 61.4477 | 58.9972 |
| PopQA | 69.1438 | 68.2140 | 64.7614 |
| X-Fact | 50.9080 | 52.6497 | 53.0039 |

The fixed control loses to the original on CounterFact and PopQA. On X-Fact the apparent folded improvement reverses a directed decrease: original 49.0920, fixed 47.3503. Across all nine, CommonClaims contributes about 79% of the fixed-minus-original folded macro gain; the effect is not uniform across datasets.

## Transfer and mathematical limits

The fixed control loses **3.9743 folded points** from source to target [−4.6271, −3.3561], compared with NLL's −3.8766. Its NLL-relative advantage changes by **−0.0977** [−0.3201, +0.3758], so these results do **not** establish improved resistance to a change of feature model. The advantage over NLL is retained within this uncertainty.

The frozen source fit uses the original 5,605 PopQA training rows only; labels were not used to fit normalization statistics. Source zero-MAD rules permanently disable f3/f4 on the target. In real arithmetic the remaining ranking is equivalent to `NLL + (11/7) × mean_local_logprob_variance`; execution preserves the original finite-precision feature calculations. Medians add a constant offset, so AUROC cannot demonstrate calibration or centering transfer. This control was designed after inspecting source benchmark batch sensitivity and must not be promoted into the original primary result or a paper/SOTA claim.

## Verification and provenance

- All nine source predictions exactly match the previously saved source control, original C8741 and NLL predictions, including row order, labels, claim groups and strict masks. All nine target predictions share those row identities.
- For every target dataset, the fixed control's scores are byte identical for chunk sizes 1, 128 and 512, and fixed shuffle 451 with original row positions restored. The original remains evaluated at its configured chunk size 512. All NLL checks pass.
- External main-transfer NLL predictions and all 500 full claim-bootstrap values match exactly on all nine datasets; where present, original C8741 predictions also match. The study contains all 97,488 prescribed rows, with 96,961 retained by the strict mask.
- Independently rehashed all nine target raw caches (15,852,415,996 bytes), all 62 protected artifacts and the sealed six study. Verified 108 point AUROCs, 540 explicitly expanded bootstrap AUROCs and all 64 reported macro/difference/transfer intervals from saved predictions.
- The six historical caches are unchanged. The new three use the same pinned 1B weights, shared source tokenizer and float32 forward configuration. A fresh 427-row reference guard passed exact identity/shape/dtype checks, observed log-probability maximum difference 0.00390625 and NLL rank correlation 0.9999998844. This is numerical agreement, not bitwise equality between historical and current extraction environments.

Evidence: [full nine report](nine/TRANSFER_REPORT.json), [independent result audit](NINE_INDEPENDENT_RESULT_AUDIT.json), [independent raw audit](NINE_INDEPENDENT_RAW_AUDIT.json), [source-fixed parameters](../frozen_train_normalization/normalization.json), and [unchanged six result](six/TRANSFER_REPORT.json).
