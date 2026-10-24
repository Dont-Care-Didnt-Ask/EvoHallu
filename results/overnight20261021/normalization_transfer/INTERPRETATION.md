# Source-fixed normalization: matched-six feature-model transfer

**Completed six-dataset result, audited 21 October 2026 at 23:48 UTC.** The frozen source-statistics control improves on both the original batch-normalized C8741 and NLL on the prescribed six 1B-feature datasets. Its absolute quality still decreases when transferring from the 8B feature model. Full-nine results remain pending; the reviewed extension is waiting for formal cache completion and the passed 427-row numerical guard.

| Frozen detector | Source-six directed / folded | Target-six directed / folded |
| --- | ---: | ---: |
| Original C8741, original-order 512-row normalization | 65.979 / 69.659 | 64.110 / 67.130 |
| Source-fixed C8741 normalization | **68.371 / 72.534** | **65.844 / 69.491** |
| Canonical NLL | 66.405 / 71.006 | 63.730 / 67.791 |

These are means over exactly the same six datasets and rows: AVeriTeC, UHEAD, WildHallucinations, Cities, CommonClaims and Companies. They are not full-nine paper-comparison numbers. Labels remain `is_hallu`; folding is applied separately to each dataset before the unweighted mean. It uses evaluation labels to choose orientation, so directed scores are retained.

On target-six, source-fixed minus original is **+1.734 [1.203,2.326] directed / +2.361 [1.804,2.975] folded pp**. Source-fixed minus NLL is **+2.114 [1.739,2.474] directed / +1.700 [1.320,2.104] folded pp**. These are paired 95% claim-bootstrap intervals for fixed programs, using all 500 declared draws.

Absolute source-to-target transfer for the fixed control is **−3.043 [−3.647,−2.394] folded pp**; NLL changes by **−3.214 [−3.700,−2.760] pp**. The change in the control's gain over NLL is only **+0.171 [−0.135,0.463] pp**. Thus the control retains a positive target advantage, but the data do not resolve whether its relative advantage improves across models. It also loses more absolute quality than original C8741, whose folded change is −2.529 pp; a better target score is not the same as a smaller transfer loss.

## Dataset dependence

| Target dataset | Original folded | Source-fixed folded | NLL folded | Fixed − original, pp |
| --- | ---: | ---: | ---: | ---: |
| AVeriTeC | 60.887 | 62.067 | 61.627 | +1.180 |
| UHEAD | 59.061 | 60.942 | 62.185 | +1.881 |
| WildHallucinations | 67.482 | 69.970 | 71.528 | +2.489 |
| Cities | 90.193 | 90.113 | 87.876 | −0.080 |
| CommonClaims | 52.525 | 63.828 | 60.391 | **+11.303** |
| Companies | 72.633 | 70.026 | 63.141 | **−2.607** |

CommonClaims accounts for approximately **80% of the target macro gain over original C8741**. The gain is not uniform: Cities and Companies decrease. Relative to NLL, the fixed control wins four datasets and loses UHEAD/WildHallucinations. Companies supplies the largest contribution to its target gain over NLL. UHEAD's folded changes must also be read with its directed orientation; stronger folding need not mean a better direction-aware detector on that individual dataset.

Strict duplicate-disjoint target macro is **69.488 versus original 67.127 and NLL 67.788 folded**. This six-dataset mask removes only seven exact normalized-claim overlaps; it does not establish entity/relation-disjoint generalization. The additional PopQA overlap sensitivity belongs to the future nine-dataset scope.

## What was actually transferred

The code writer was Qwen3.5-9B. The selected detector used source features from Llama3.1-8b-instruct, then unchanged 1B target features with the original shared tokenizer. The target feature model is Llama3.2-1B-Instruct; no benchmark claims were regenerated. The comparison changes the feature model, not the writer or target training data.

The original class, four fixed coefficients, source-fitted medians/MADs, inference helper and source fit artifacts remain unchanged. Statistics came from the exact original 5,605 source training rows; the 1,402 validation rows and all target distributions/labels did not fit the statistics. The adapter never calls the source helper's fitting function.

Source zero-MAD rules permanently disable f3/f4. In real arithmetic the remaining ranking is equivalent to **NLL + (11/7) × mean local log-probability variance**; execution preserves the original finite-precision operations. This is evidence for transfer of a frozen feature/scale combination. Medians contribute a common score offset, so AUROC cannot demonstrate calibration or centering-statistic transfer. No coefficient, clipping rule or feature computation was repaired for target quality.

The original C8741 remains a separate batch-adaptive detector evaluated in its declared 512-row order. The fixed control's scores are byte-identical across chunks 1/128/512 and shuffle451/chunks512 on every target dataset. All four cases were retained; no highest-scoring case was selected.

## Integrity and uncertainty

- All nine source LP-only original/fixed/NLL predictions reproduce saved source arrays byte-for-byte, including labels, claim groups and strict masks. Original and proxy candidates compile into separate namespaces.
- All six target cache hashes, CSV order, architecture/float32-forward metadata, pinned model/shared-tokenizer provenance and source/target token IDs passed. A separate full-file rehash of all six target caches and all 62 protected artifacts also passed.
- Target NLL predictions and all 500 full-claim bootstrap values match the earlier target-six references exactly. All six target invariance receipts record the same fixed-score hash for all four cases, matching saved predictions.
- Independent audit recomputed **72 point AUROCs**, explicitly expanded **360 bootstrap AUROCs** from the first five claim draws per model/dataset/scope/method, and reproduced **64 macro/transfer intervals** from all 500 saved draws. No model inference was repeated for this audit.
- The original six artifacts are sealed and preserved byte-identically. The separate nine extension scores only CounterFact/PopQA/X-Fact after the formal guard, keeps the same state, and has no target-fitting or generation path.

This is an exploratory follow-up motivated by an observed source batch-sensitivity problem, after benchmark exposure. Intervals condition on the fixed program, statistics, rows and dataset set; they omit search/selection uncertainty and multiplicity correction. There is no primary replacement, new independent confirmation, native-tokenizer transfer or paper-SOTA claim.

Evidence: [sealed six report](six/TRANSFER_REPORT.json), [independent result audit](INDEPENDENT_RESULT_AUDIT.json), [source parity audit](SOURCE_PARITY_AUDIT.json), [independent raw audit](INDEPENDENT_RAW_AUDIT.json), [design](DESIGN.md), [nine launch receipt](nine_launch_receipt.json).
