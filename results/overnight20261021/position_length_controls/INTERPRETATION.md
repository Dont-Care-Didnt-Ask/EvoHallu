# Interpretation: accumulated quadratic NLL versus length and normalization

The complete CPU diagnostic finished **2026-10-21 23:27:02 UTC**, after **59.16 seconds**. All nine frozen-program/NLL prediction arrays, CSV labels, claim-group IDs and duplicate-disjoint masks passed byte-exact primary-reference checks before control scoring. All **36 prescribed control/dataset combinations** completed across **97,488 benchmark rows**. No GPU/model call, fitting, new candidate, primary replacement or repair occurred.

| Fixed method | Directed source9 | Folded source9 | Folded duplicate-disjoint |
| --- | ---: | ---: | ---: |
| Canonical float16-reduction NLL | 65.644 | 68.711 | 68.685 |
| Frozen 27B quadratic weighted **sum** | 70.297 | **72.306** | **72.229** |
| Total cached LP length | 49.368 | 54.634 | 54.623 |
| Float32 uniform NLL sum | 65.643 | 68.712 | 68.686 |
| Float32 uniform NLL mean | 64.176 | 64.176 | 64.086 |
| Same-weight float32 quadratic **mean** | 69.866 | 70.724 | 70.636 |

The frozen reference gains **3.595 folded pp** over canonical NLL, paired claim-bootstrap 95% CI **[2.909, 4.058]**. Its normalized quadratic counterpart retains **2.013 pp [0.952, 2.871]**. Changing canonical NLL's reduction from float16 to float32 changes the folded macro by only **+0.000499 pp**; numerical precision cannot explain the quadratic winner's gain here.

The three predefined contrasts help describe the mechanism:

| Contrast, left minus right | Directed gain [95% CI] | Folded gain [95% CI] |
| --- | ---: | ---: |
| Uniform sum − uniform mean | +1.467 [0.086, 2.781] | +4.535 [3.089, 5.824] |
| Quadratic sum − quadratic mean | +0.431 [−0.225, 1.102] | +1.582 [0.977, 2.277] |
| Quadratic mean − uniform mean | +5.690 [4.894, 6.360] | +6.548 [5.534, 7.187] |

Positional weighting therefore helps compared with the matched uniform mean even after removing weight mass. Accumulation also matters to the paper's folded metric. Its additional benefit for quadratic weighting is less clear on the directed metric: the interval includes zero. The directed/folded difference is driven by UHEAD's reversed ranking: quadratic sum AUC **40.961**, quadratic mean **46.140**, reported as folded **59.039** and **53.860** respectively. Folding follows the author's reporting clarification; labels and primary training/validation selection directions were unchanged.

This is not a universal improvement. Relative to canonical NLL, the quadratic sum's folded UHEAD score falls **63.803 → 59.039**, and WildHallucinations falls **70.686 → 63.148**. Removing quadratic accumulation helps CommonClaims **69.286 → 72.016** and X-Fact **53.053 → 58.302**, despite reducing the overall folded macro. Uniform length normalization performs substantially worse overall. All these negative variants and dataset results remain in the report.

Length alone is weak. Saved historical claim-length vectors show that total cached LP length equals **claim-token length +18 for every row in every dataset**, so the two length controls have exactly the same ordering and AUROC. This is an observed assembly fact for these caches, not a new token-cropping rule. Because weighting acts over the complete cached LP span, this study does not isolate later-claim effects from downweighting fixed template positions. Nor does weak length-only AUROC rule out useful interaction between length and surprise.

Historical results already contained normalized exponential-weighted NLL (**70.606 folded**) and accumulated linear-position NLL (**72.000 folded**). The quadratic sum is numerically **+0.306 pp** above that historical linear ablation; this is context, not an independent model/policy comparison or proof of a newly discovered mechanism. Its **72.306** also does not exceed the paper's displayed **73.4** best result. AUROC is nonlinear: none of these contrasts forms a causal additive decomposition of the reference's gain.

[PLAN](PLAN.json) was frozen before this diagnostic executed, but after earlier benchmark results were viewed. The study is exploratory and retains every fixed variant. The 500 shared normalized-claim bootstrap draws condition on one validation-selected program, not search-seed uncertainty. Duplicate-disjoint sensitivity changes the reference gain over NLL only slightly (**3.545 folded pp**); it does not establish semantic/entity-disjoint generalization.

[BACKUP_VALIDATION](BACKUP_VALIDATION.json) independently verifies all compact remote artifact hashes, all nine prediction/identity byte matches, **108 saved-vector point metrics**, every macro/predefined contrast and all saved-bootstrap macro intervals. Same-pod independent arithmetic matches exactly. A supplementary local-runtime recomputation matches 100/108 scalar AUROCs exactly; the remaining differences are at most **2.84×10⁻¹⁴ pp**, while all underlying array bytes match. No byte-parity guard, helper or PLAN was changed. Seventy-one completed compact artifacts (**3,187,603 bytes**) were copied and SHA256-checked locally; large raw caches were not copied or reopened for this independent audit. See [the complete report](POSITION_LENGTH_REPORT.md) for all fixed methods, source9/strict results and retained limitations.
