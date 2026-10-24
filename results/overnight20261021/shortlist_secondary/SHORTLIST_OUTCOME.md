# Validation-shortlist outcome: execution complete, one OOD-invalid candidate

**All planned CPU dataset evaluations finished.** The [execution receipt](execution_receipt.json) records active execution from **22:08:55 to 22:17:16 UTC on 21 October 2026**; the [evaluation log](evaluation.log) confirms completion of all nine dataset processes. This was cached CPU evaluation, with no additional writer calls, feature extraction, fitting or primary-registry changes.

**35 of 36 validation-rank slots have complete source-nine results.** The remaining slot is `next_evo_original_8741`, validation rank **3**, candidate `09061240720756834dac5589a2f552001250b6bde3fcf7b4ec31df75b183127e`, selected at directed PopQA validation **76.0972**. Its frozen source SHA-256 is `1998aad097ac6435bc260da7e1396637d8f828c4875d77514775e9dca08971ca`.

This candidate fails on **WildHallucinations and X-Fact** with `ValueError: nonfinite scores or wrong output shape`; its other seven dataset results remain saved. Consequently `incomplete_secondary_shortlist` describes **incomplete valid prediction coverage**, rather than an unfinished or timed-out evaluation. The generated report's `PENDING` cells for this slot are unavailable metrics after candidate failure; no rerun is required to establish this outcome.

## Static explanation of the failure

The [unchanged frozen program](frozen/next_evo_original_8741__valrank03/frozen_estimator.py) computes exponentially increasing positional weights:

```python
positions = np.arange(n, dtype=np.float32)
alpha = 1.5
weights = np.power(alpha, positions)
```

It then forms weighted head/tail means and tail variance using these **unscaled** weights. In float32, `1.5**position` exceeds the representable range around position 219; weighted products and sums can overflow earlier. Infinite weighted numerators and denominators produce undefined arithmetic and nonfinite final scores. Both saved dataset logs contain **`overflow encountered in power` at line 46**, corresponding to this operation, followed by invalid/overflow multiplication warnings. The program appends one score for each input sequence, so its static control flow points toward nonfinite arithmetic as the failure mechanism, rather than an obvious output-length mismatch. The evaluator's saved error intentionally combines those two checks.

This explanation uses source and existing logs only. The exact failing claims were not re-executed or identified, and no stabilized replacement was introduced. Passing PopQA validation did not guarantee numerical robustness on OOD inputs.

## Retained-rank results

The table gives **paper-compatible source-nine AUROC, mean ± sample SD across all three predefined seeds**. Folding is applied separately within each method and dataset, followed by the unweighted mean of nine; labels remain `is_hallu`. Validation ranks were frozen before these benchmark scores.

| Policy | Validation rank 1 | Validation rank 2 | Validation rank 3 |
| --- | ---: | ---: | ---: |
| Original | 69.943 ± 1.059 | 69.663 ± 0.881 | Unavailable: seed 8741 fails two datasets |
| Neutral | 68.711 ± 0.000 | 68.517 ± 0.336 | 67.325 ± 2.400 |
| Diverse | 70.355 ± 1.648 | 68.711 ± 0.000 | 68.711 ± 0.000 |
| Compositional | 70.002 ± 0.911 | 69.342 ± 1.092 | 69.323 ± 1.060 |

For each run, the next diagnostic averages the **three individual detectors' AUROCs**, then summarizes those run-level collection means across seeds. It is not an ensemble, an average of score vectors, or a maximum selected on test quality.

| Complete policy collection | Directed source-nine mean ± SD | Folded source-nine mean ± SD | Duplicate-disjoint folded mean ± SD |
| --- | ---: | ---: | ---: |
| Neutral | 65.485 ± 0.275 | 68.185 ± 0.912 | 68.150 ± 0.926 |
| Diverse | 66.345 ± 0.702 | 69.259 ± 0.549 | 69.226 ± 0.542 |
| Compositional | 67.227 ± 1.508 | 69.556 ± 0.941 | 69.496 ± 0.909 |

The **original-policy three-seed collection mean is unavailable**. Its two complete seed collections remain individually recorded; the failing seed is not dropped to manufacture a complete-policy estimate. Likewise, the original rank-three group is not averaged over only its two successful seeds. All twelve rank-one primary results remain complete and unchanged. No missing scores are imputed, replaced by NLL, repaired or converted into a test-best winner.

## Aliases and interpretation

All **24 additional rank slots were staged as distinct within-run source files**: this realized cohort contains no within-run source-hash aliases to skip. Across the full 36 slots there are **29 unique source hashes**, including **eight explicit baseline selections** across runs. Distinct code is not necessarily distinct behaviour: several lower-ranked methods have NLL-level benchmark AUROC, and equal AUROC alone does not prove identical score vectors. Additional top-three training-probe correlation analysis was not implemented.

The [canonical baseline/row audit](SHORTLIST_REPORT.json) passed, and secondary labels, masks, claim-group IDs and NLL references agree with the primary evaluation. Complete per-run validation scores, directed/folded/duplicate-disjoint results and source hashes remain in [SHORTLIST_REPORT.json](SHORTLIST_REPORT.json); the [source gallery](SOURCE_GALLERY.md) preserves every original rank slot for inspection. The generated report, prediction files and frozen code were not modified for this explanation.

This study supports a concrete negative result: retaining three validation-ranked candidates did not automatically supply three strong, different and numerically robust detectors. Its collection means do not improve on the corresponding rank-one means in the complete policies. These are exploratory diagnostics after prior benchmark exposure, with the inherited split/duplicate limitations; they do not establish a matched comparison with the paper's curated collection or a new SOTA result.
