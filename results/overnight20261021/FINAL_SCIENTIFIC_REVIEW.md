# Scientific review of the preliminary report draft

Read-only review at **21 October 2026, approximately 22:48 UTC**. Scope: completed Qwen3.5-9B source-nine results, retained validation ranks, batching/normalization controls, baseline reproduction and selection claims. Running/optional writer outcomes, eventual target-nine completion, resource totals and pod shutdown remain the root controller's responsibility.

Reviewed document hashes:

- `NARRATIVE_REPORT_DRAFT.md`: `a236438e295a4c540cea80f6542df34908cc4e71cc4f7e7b42ce78f921738da5`.
- `CODE_CHANGES_20261021.md`: `67b5ed407aec66f6632db2bc7cd95cf1112201f4bb0e571a5f51c1814b6cf042`.

No narrative, code, registry, prediction array, frozen program, fitted state or refit report was changed. No candidate execution, fitting, feature extraction or scoring was performed.

## Findings to correct before publication

| Priority | Location in the reviewed draft | Finding and precise correction |
| --- | --- | --- |
| Required | Narrative, extension-coverage table, line 102 | “Full adaptive table” remains listed as unfinished although all twelve adaptive runs and their source-nine results are already reported above. Mark the Qwen9B adaptive comparison completed; retain only genuinely pending optional writer outcomes and unmatched GPT-OSS cost comparison. |
| Required | Narrative, extension-coverage table, line 106 | The matched three-seed feedback result is still called pending. It is complete: neutral adaptive and neutral fixed-NLL-example selection both retain canonical NLL in all three seeds, with exactly **0.000 directed/folded pp difference**. Longer-run/restart and no-example-independent comparisons remain untested. |
| Required | Narrative, final paragraph, line 112 | “1,260 requests across 23 jobs” is described as planned rather than achieved. The saved **22:08:59 report** records **23/23 finished jobs and 1,260/1,260 issued requests** for this fixed Qwen9B cohort. Replace that qualification with its achieved count. Keep optional-writer totals, final GPU time, billing and shutdown separately pending. Requests are not valid/unique detectors. |
| Wording | Narrative, matched prompt-screen interpretation | “Useful diversity” overstates that screen's criterion. Its primary count requires two valid nonconstant siblings with signed Spearman **rho <0.995** on the fixed training probe; it does **not** require beating the baseline or demonstrating held-out usefulness. Call it “joint valid-and-rank-diverse pairs.” The separate temporal “novel-and-training-useful” diagnostic does explicitly require saved training fitness ≥ NLL. Do not conflate these definitions. |
| Documentation coverage | Code-changes document, scheduling/scoring/control sections | The document predates new [batch-sensitivity](../../scripts/overnight_batch_sensitivity.py), [frozen-normalization](../../scripts/overnight_frozen_normalization.py), [batch plot](../../scripts/plot_batch_normalization_diagnostics.py) and [Devstral](../../scripts/overnight_devstral.py) helpers. Add their separate-control/runtime-proxy scope and associated tests. The Devstral prefetch is real preparation, not a completed quality experiment. |
| Minor precision | Narrative, “exact” original-screen formula | The displayed algebra matches the ordinary branch, but the source returns zero for sequences shorter than five tokens and retains its actual Pearson/exception behavior. Say “on the ordinary n≥5 branch” if calling the formula exact; the tail starts at `floor(0.9*n)`. Do not replace Pearson correlation with regression slope. |

The cited extension-coverage audit is a legitimate **19:55 snapshot**, but its old pending labels and request counts must not be reused as final status. Either refresh the final coverage table or identify the linked audit as historical. This is a documentation-consistency issue, not evidence that the executed result ledger is wrong.

The **183-test receipt** remains a valid scoped earlier result, rather than an overall final test total. Additional suites now exist, including [36 combined late-scheduler/Devstral checks](devstral/combined_scheduler_validation.json). Do not sum overlapping runs into an inflated unique-test claim. Preserve each scope/receipt or run an explicitly defined final suite.

## Claims checked and supported

The final source-nine table agrees with [report.json](report.json):

| Completed adaptive policy | Three-seed directed mean | Three-seed folded mean ± sample SD |
| --- | ---: | ---: |
| Original | 67.823 | 69.943 ± 1.059 |
| Neutral | 65.644 | 68.711 ± 0.000 |
| Diverse | 67.748 | 70.355 ± 1.648 |
| Compositional | 68.297 | 70.002 ± 0.911 |

The draft's validity counts, fixed-example null result, sampler mean **68.922**, interface pilot null result and paired differences agree with the same artifact. Diverse's **+0.412 folded / −0.075 directed pp** versus original is accurately described as insufficient to establish a policy advantage. Three search seeds provide limited precision; their SD is different from claim-bootstrap uncertainty.

[Parent-pool diagnostics](parent_sampling_analysis/PARENT_SAMPLING.md) support the qualification that rank-plus-uniform weighting is **more concentrated on the observed pools** than the inherited temperature-40 softmax. This comparison conditions on pools produced by the rank arm. It is not an estimate of what every neutral-run population would have looked like under a different sampler, nor a guarantee of behavioural exploration.

The saved [baseline audit](baseline_exact_check.json) passes: **65.643882 directed / 68.711173 folded**. Per-method, per-dataset folding precedes the equal-dataset mean; `is_hallu` remains unchanged. The report correctly explains that folding uses evaluation labels to determine orientation and does not supply deployment calibration. The earlier UHEAD-label-error account is superseded.

The writer/feature-model distinction is correct: Qwen3.5-9B writes estimator code; Llama3.1-8b-instruct supplies cached source features for the existing claims. New Qwen27B/GLM/Devstral writer studies do not regenerate the benchmark claims or automatically become feature-model-transfer studies.

## Retained-rank failure is a completed negative result

[SHORTLIST_OUTCOME.md](shortlist_secondary/SHORTLIST_OUTCOME.md) correctly distinguishes **all nine CPU dataset jobs finished** from **35/36 valid full-nine rank slots**. Original/8741/rank3 fails finite-output validation on WildHallucinations and X-Fact. Static exponential-weight overflow is consistent with its source and both saved warning logs. No fresh execution is necessary to report the failure.

Keep original rank-three and original top-three collection **three-seed aggregates unavailable**. Do not omit the failing seed, average only seven datasets, impute NLL, repair the formula or substitute a lower-ranked program. All twelve primary rank-one results remain complete. Collection means average individual method AUROCs, not score vectors or ensembles; they do not show a gain from retaining more candidates. The gallery contains repeated baselines and distinct source files with potentially redundant behavior, so source uniqueness is not detector diversity.

## Batching and training-only normalization

The two controls are scientifically separated correctly in the narrative:

- [Batch sensitivity](batch_sensitivity/BATCH_SENSITIVITY.json) reproduces all nine original 512-row primary vectors exactly before changing conditions. The same immutable program gives **68.965497 folded** at original512, **70.847399** at full-dataset calls and **70.790171** after the fixed seed451 shuffle at512. CommonClaims contributes approximately 97% of the full-batch macro change. It is accurate to describe a row-membership/per-batch-normalization effect, not a train/test chunk-size mismatch: all original paths used512. “Accounts for” is preferable to wording implying that class ordering is proved to be the sole causal mechanism.
- [Frozen training normalization](frozen_train_normalization/FROZEN_NORMALIZATION_REPORT.json) fits four statistics pairs on exactly **5,605 original fitting rows**, excludes the 1,402 validation rows, uses labels only to reconstruct the inherited stratified split, and retains coefficients/feature formulas/zero-MAD handling. It gives **67.783265 directed /70.558451 folded**, with all36 saved case-invariance checks passing. Two terms vanish by the existing zero-MAD rule; this is not post-hoc feature selection.

Plan/source pointers match actual hashes for both studies, and the normalization report's fit-receipt hash matches its actual [FIT_RECEIPT.json](frozen_train_normalization/FIT_RECEIPT.json). Original primary reproduction, baseline parity and explicit exploratory-after-test-inspection flags are present. The controls must stay outside primary seed averages and outside claims of an independently validated new search improvement. Four datasets worsen under the frozen-training normalization; its macro gain is not uniform generalization.

The batch-dependent program is the **adaptive compositional/8741 winner**. The twelve-column logistic head uses the **four fixed-context screen winners** instead. Do not suggest that the batching finding automatically invalidates or changes the head's immutable **73.633** result; these are different source pools.

## Selection, supervision and paper comparison

The narrative correctly separates directed training fitness, directed PopQA validation selection including baseline, and later benchmark reporting. Baseline-first ties and all-seed retention must remain explicit. No best-by-test candidate or seed should be promoted when optional writer results arrive. The fixed-context screen also correctly acknowledges inherited historical parents and search effort.

The fourteen-feature head's **73.019** and component head's **73.633 folded /71.359 directed** agree with audited predictions. The component head's duplicate-disjoint folded result is **73.569710**. Its feature pool was discovered/selected with validation information before coefficient fitting; grouped coefficient CV cannot erase that history. Outer-validation **82.094** is therefore diagnostic, not an independent estimate. Exact-claim grouping does not imply semantic/entity/relation disjointness; the inherited writer split's88 shared normalized claims and test overlaps remain material limitations.

The draft appropriately declines to claim the paper beaten. **73.633 versus73.4** is a rounded numerical comparison across unmatched supervision, writers, search budgets, retained collections and selection procedures. Claim-bootstrap intervals condition on frozen methods; they are not a paired test against the unavailable Claude implementation, search-seed uncertainty or multiplicity adjustment.

Current source/core hashes still match all ten pins in the source execution manifest. Runtime interventions and separately named controls change declared behavior while leaving those core bytes fixed. New writer outcomes, full-nine transfer certification, actual cost and stop confirmation should be added only from their own terminal receipts, not from downloaded checkpoints or prepared plans.
