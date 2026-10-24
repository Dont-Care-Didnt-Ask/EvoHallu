# Experiment extensions: completed source study and remaining questions

**Results snapshot: 22 October 2026, approximately 00:42 UTC.** All **27 writer jobs** completed their planned budgets: **1,500 attempted candidates / 841 pipeline-valid**. Qwen3.5-9B contributes **23 jobs / 1,260 requests / 669 valid**, Qwen3.8-27B **120 requests / 108 valid**, and GLM-4.7-Flash **120 requests / 64 valid**. Source-nine results, all nine target feature caches and local archive backups are complete. Main target scoring retains **29/30 methods with nine-dataset coverage and one with eight**. Separate component-head and source-fixed-normalization nine reports are complete. [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) preserves the dated preparation plan; detailed evidence is in the [source report](results/overnight20261021/REPORT.md), [terminal writer receipt](results/overnight20261021/late_writer_decision.json) and [extension-coverage audit](results/overnight20261021/extension_coverage_audit.md).

**Devstral was skipped without serving** because only 65.51 minutes remained against its fixed 75-minute entry gate. GLM's second seed was also capacity-skipped. The unchanged formal preparer completed three additional target caches after the documented post-cleanup idle-check recovery; original logs and six-cache hashes are retained. **RunPod confirmed `EXITED` with no GPU runtime at 22 October 00:44:41 UTC.** The run used **6 h 28 min**, estimated **$23.16 compute** at **$3.58/hour total**, excluding storage/network and pre-task waiting. [Stop receipt](results/overnight20261021/pod_stop_confirmation.json), [resources](results/overnight20261021/resource_summary/RESOURCE_USAGE.md).

**Baseline reproduction is resolved.** Keep `is_hallu` unchanged. Fold each **method × dataset** AUROC as `max(AUC,100−AUC)`, then average the nine datasets equally. NLL reproduces **65.6439 directed / 68.7112 folded**, matching the paper's displayed **68.7**. Perplexity **64.1747** and mean token entropy **59.9043 folded** reproduce **64.2 / 59.9**. Exact historical NLL predictions, labels, dataset order and duplicate masks match. Folding chooses orientation using evaluation labels; directed scores remain reported. The former UHEAD label-error explanation is superseded.

The [historical extension answers](results/next_experiments/EXPERIMENT_EXTENSIONS_historical_2026-10-15.md) retain the earlier detailed campaign, including its superseded interpretation. The seven original research questions extending [EvoHallu](https://aclanthology.org/2026.acl-srw.115.pdf) remain below; completed studies do not exhaust their scope.

## 1. Can newer/smaller code writers improve quality and cost?

**Model roles:** Qwen3.5-9B writes Python detector code. **Llama3.1-8b-instruct supplies the fixed source features**; the supplied benchmark claims and labels are unchanged. The writer does not replace that feature model or generate a new evaluation dataset.

The main comparison uses four policies × three seeds × **60 requests per run (30 rounds, two attempts per round)**. Every run primary is frozen by directed PopQA validation, including canonical-NLL fallback; training AUROC governs evolutionary fitness. All twelve source-nine primaries are retained. Values are unweighted nine-dataset means; SD describes variation across the three seeds.

| Adaptive policy | Valid / attempts | Directed mean | Folded mean ± seed SD |
| --- | ---: | ---: | ---: |
| Original | 101 / 180 | 67.823 | 69.943 ± 1.059 |
| Neutral | 98 / 180 | 65.644 | 68.711 ± 0.000 |
| Diverse | 103 / 180 | 67.748 | 70.355 ± 1.648 |
| Compositional | 78 / 180 | 68.297 | 70.002 ± 0.911 |

Diverse minus original is **+0.412 folded / −0.075 directed pp**; compositional minus original is **+0.059 folded pp**. These small, variable outcomes do not establish a reliable policy advantage or superiority to the paper.

The completed **Qwen3.8-27B original/diverse confirmation** uses seed 2107, **60 requests per arm**, and **52/56 valid candidates**, respectively. Source-nine scores are **69.060 directed / 71.155 folded original** and **70.297 / 72.306 diverse**. Diverse was chosen by mean PopQA validation across all three completed non-original 9B seeds, not by benchmark scores. Both writer comparisons retain their within-run validation-selected program. This **one-seed** result cannot establish a stable writer/policy ranking; 72.306 remains below the paper's displayed Claude 73.4. [Exact programs and selection](results/overnight20261021/WRITER27B_CANDIDATE_INSPECTION.md).

The completed **GLM-4.7-Flash original/diverse pair**, also seed 2107 and **60 requests per arm**, produced **28/36 valid candidates**. Directed/folded source-nine scores are **68.533/70.671 original**, **65.644/68.711 diverse**; validation maxima are **78.7667 / 75.7504**. Original rediscovers inherited negative position/logprob Pearson correlation, with different empty/constant-input guards; diverse retains exact canonical NLL. These results do not establish a consistent benefit from the diverse prompt or a new feature family. GLM's second seed required **73.96 minutes** but only **65.54** remained. Devstral was independently resource-skipped, with no quality or serving result. [Exact GLM semantics](results/overnight20261021/GLM_CANDIDATE_INSPECTION.md), [terminal decisions](results/overnight20261021/late_writer_decision.json). Historical writer comparisons remain separate; matched GPT-OSS-120B cost/quality superiority and stable cross-writer ordering remain untested.

## 2. Does deliberate diversity improve useful proposals?

The completed **warm fixed-context screen** used the same fifteen saved historical parent contexts, two siblings per policy: **120 new requests**. It inherits information from an earlier search trajectory and is distinct from the adaptive matrix.

| Screen policy | Valid / 30 | Joint valid, nonconstant, diverse pairs / 15 contexts | Source-nine folded AUROC of validation winner |
| --- | ---: | ---: | ---: |
| Original | 19 | 6 | 69.973 |
| Neutral | 19 | 4 | 73.137 |
| Diverse | 23 | 7 | 73.007 |
| Compositional | 11 | 3 | 69.538 |

The joint valid-and-rank-diverse sibling diagnostic requires signed training-probe Spearman correlation below **0.995**, without requiring training-fitness improvement; the denominator includes **all fifteen contexts**, including failures. Every intervention-minus-original paired interval includes zero. The small screen supports neither a reliable diversity gain nor an independent from-scratch 30-request quality claim. See [paired analysis](results/overnight20261021/prompt_screen_analysis/PROMPT_SCREEN_ANALYSIS.md).

The three-seed adaptive table above is complete. Rank/uniform parent sampling and the interface contract were tested separately below. Two-stage sketch/implementation, temperature-only, repair and behavioural-archive interventions remain untested; their extra requests would need explicit budgets.

## 3. Why do Claude-like composite programs succeed or fail?

[All twelve adaptive winners](results/overnight20261021/ADAPTIVE_WINNER_INSPECTION.md) and the [four screen winners](results/overnight20261021/SCREEN_CANDIDATE_INSPECTION.md) have source-level formula/feature audits. Three neutral primaries retain canonical NLL; diverse seed 8741 selects float32-summed NLL. The other eight adaptive winners use thresholded surprise, positional weighting/correlation, local variation or combinations with top-k statistics. One original entropy contribution is verified inactive on all 7,007 source training rows. Complexity is descriptive; no new feature-count penalty was applied.

**Validation shortlists were also evaluated:** twelve runs × three retained ranks = **36 slots**, including 24 additional slots. All nine CPU dataset evaluations finished; **35/36 slots have complete source-nine predictions**. Original seed 8741, rank 3, fails WildHallucinations and X-Fact with nonfinite-score/output-shape errors. Existing logs show overflow in unscaled float32 `1.5**position`; its other seven results remain saved. This is an OOD numerical failure, rather than unfinished execution. No repair, replacement, imputation or successful-seeds-only collection mean was introduced. See [outcome](results/overnight20261021/shortlist_secondary/SHORTLIST_OUTCOME.md).

The historical Claude populations, executable feature definitions and matched Sonnet/Opus trajectories remain unavailable. The user's screenshot motivates composite features; it does not reconstruct those programs or explain their writer ranking.

## 4. Can evolution produce useful supervised estimators?

Completed conventional controls use fourteen documented features and a **claim-grouped 5,606 fit / 1,401 validation split**, with fit-only grouped inner CV/scaling. Source-nine directed/folded AUROC is **70.615 / 73.019 for logistic regression** and **69.690 / 72.166 for QDA**. This partition differs from the evolutionary **5,605 / 1,402 stratified split**.

A separate twelve-column head retains all eleven generated screen-component vectors plus float64 total NLL. Its frozen source-nine result is **71.359 directed / 73.633 folded**, gain over canonical NLL **+4.922 [4.193,5.419] pp**. The completed target-nine result is **66.552 directed / 68.800 folded**, gain **+3.965 [3.529,4.628]** over target NLL. The earlier source-six/target-six result **74.296 / 71.431 folded** is preserved separately. Its feature pool is validation-informed and the follow-up was designed after benchmark viewing; outer-validation quality is diagnostic. These are supervised fitted controls over generated features. **Joint supervised program/feature evolution remains untested.** See the [original frozen report](results/overnight20261021/screen_component_refit/REFIT_REPORT.md) and [separate nine transfer report](results/overnight20261021/screen_component_refit/transfer_nine/TRANSFER9_REPORT.md).

The head's 73.633 versus the paper's displayed Claude 73.4 is numerical context: supervision, search budgets, candidate pools and selection differ, and matched Claude predictions are unavailable. A paper-SOTA claim is unsupported.

On **target-nine**, conventional logistic-14 scores **68.939 folded**, versus the component head **68.800**. Head-minus-logistic is **−0.140 [−0.515,+0.458] folded pp**, unresolved; the directed difference **−0.842 [−1.322,−0.342] pp** favors logistic-14. The component head is therefore not the strongest target control, despite its higher source-nine point estimate.

## 5. Is evolutionary feedback necessary, and does parent selection matter?

**Three fixed-NLL-example controls** finished at 60 requests each, matched by seed to neutral adaptive evolution. Both arms retain NLL in every seed: **68.711 folded**, paired quality difference **0.000 pp**. This is a null result for this bounded neutral comparison; it does not establish that feedback is generally unnecessary.

**Three neutral rank/uniform-sampling controls** score **65.864 directed / 68.922 ± 0.364 folded**, versus neutral evolution 68.711 folded: **+0.211 pp**. One seed adds logprob variance, one is an NLL precision variant with a mathematically dead gap term, and one retains canonical NLL. There is no broad across-seed improvement.

The separate **one-seed interface/API/shape/alignment/metric-contract pilot** retains NLL, **0.000 pp** versus its neutral reference; both have **29/60 valid candidates**. It added no repair calls. A fresh no-example independent arm, long continuation versus restart portfolios, and an optimal evolutionary horizon remain untested. [Semantic audit](results/overnight20261021/semantic_rank_winner_audit.md) and [matched results](results/overnight20261021/REPORT.md) preserve these negative findings.

## 6. Do source-selected formulas transfer across feature models?

**All nine target feature caches are complete**, using **Llama3.2-1B with the shared source tokenizer** and unchanged supplied claims/labels. The three additions cover **41,253 rows**; all six historical raw-cache hashes are unchanged. The formal repeated **427-row** numerical guard passed (maximum observed-logprob difference **0.00390625**, canonical-NLL rank correlation **0.9999998844**), and all three new XZ archives are verified locally. [Completion](results/overnight20261021/remaining_transfer_completion.json), [backup](results/overnight20261021/cache_backup_receipt.json).

Main scoring completed all **18 dataset/kind tasks**, but compositional seed-2107's frozen primary fails on target UHEAD. Thus **29/30 methods have full-nine coverage and one has eight**, and no complete three-seed compositional target mean is claimed. Saved entropy warnings and unchanged float16 arithmetic suggest a `0 × log(0)` numerical cause; this is a static/log-supported explanation, not a new reproduction. The failed program remains unmodified and unreplaced. [Failure record](results/overnight20261021/transfer_llama32_1b/d00b4c0d58d795bf/new/UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot.json).

The separate frozen twelve-column head changes **73.633 source-nine → 68.800 target-nine folded** (−4.834 pp), versus target NLL **64.835**. Its gain over NLL is **+3.965 [3.529,4.628] pp**, but that gain **shrinks 0.957 pp**, signed change **−0.957 [−1.302,−0.149]**. This is weaker transfer evidence than the earlier six-dataset relative-gain comparison, whose interval included zero. The original six report remains unchanged. [Nine report](results/overnight20261021/screen_component_refit/transfer_nine/TRANSFER9_REPORT.md).

Source-fixed C8741 normalization scores **63.564 directed / 66.584 folded target-nine**, versus original **62.779/64.995** and NLL **61.459/64.835**. Its folded gains are **+1.5895 [1.1328,2.0179] pp** over original and **+1.7496 [1.4839,2.0336]** over NLL. Its source **70.558→66.584** loss is **3.974 pp**; NLL-relative transfer change **−0.0977 [−0.3201,+0.3758]** remains unresolved. Source statistics/coefficients remain frozen, no target fitting occurs, and the original six reports are sealed. [Nine interpretation and audits](results/overnight20261021/normalization_transfer/NINE_INTERPRETATION.md).

The historical component-drop study still covers **source-nine/target-six**, not target-nine; its coverage was not expanded by completing the main caches. Compare identical source/target dataset sets. Native-target-tokenizer transfer, another large feature model and target refitting remain untested. Frozen code can compute prediction-batch statistics; the separate source-fixed control does not replace the original batch-adaptive program or establish calibration transfer.

## 7. Which components help, and is apparent complexity real?

Completed **eleven fixed-coefficient drops plus four exact full-program controls** cover all four screen winners on source-nine/target-six. Full-control predictions reproduce exactly. Tail gap variance helps neutral/diverse formulas; diverse relative gap drift is negligible; the compositional tail-mean contribution hurts with its original fixed coefficient. These are conditional contributions inside correlated formulas, not independent causal importance. All drops remain reported; none is promoted using test quality. [Ablation report](results/overnight20261021/screen_component_ablations/ABLATION_REPORT.md).

A separate **C8741 batch-sensitivity study** found label-free median/MAD normalization across the current scoring batch. Training, validation and benchmark primary scoring all used **512-row chunks**, and all nine primary arrays reproduce byte-for-byte. Folded source-nine AUROC for predeclared **128 / 512 / 2,048 / full / shuffled-512** cases is **68.546 / 68.965 / 69.121 / 70.847 / 70.790**. CommonClaims' label-ordered input accounts for nearly all the full/shuffle change. These are diagnostics; 512 remains the primary. [Batch audit](results/overnight20261021/batch_sensitivity/INTERPRETATION.md).

One separately declared **train-only normalization control** freezes four medians/MADs from the exact original 5,605 training rows, retaining feature math and coefficients. It scores **67.783 directed / 70.558 folded**, with scores byte-identical across chunks 1/128/512 and fixed shuffle on all nine datasets. The original zero-MAD rule disables f3/f4; both were already inactive in all eleven original training chunks. The macro gain is CommonClaims-driven and four datasets lose folded AUROC. This exploratory control was designed after test viewing and replaces no primary. [Control report](results/overnight20261021/frozen_train_normalization/INTERPRETATION.md).

Shapley values, exhaustive feature-count/subset frontiers, feature permutation and ablations of unavailable original Evo-Anth/Evo-Gpt implementations remain untested. Fixed drops, coefficient fitting and frozen normalization answer different questions.

## Common limits and reproducibility

Training directed AUROC drives evolution; directed PopQA validation freezes primaries. Every full-budget seed and negative result is retained. Exact claim grouping protects the fitted-head split; the inherited evolutionary split still shares 88 normalized claims across train/validation. Duplicate-disjoint held-out sensitivity removes 527 rows, including 520 PopQA rows, without establishing entity/relation-disjoint generalization. Claim-bootstrap intervals condition on frozen programs and do not cover search/selection history or multiple comparisons.

The tested pod uses Python 3.12.3, Torch 2.13.0+cu130, vLLM 0.31.0 and Transformers 5.17.0; its 227-package dependency check passes. A second clean rebuild has not been performed. [Environment evidence](results/overnight20261021/ENVIRONMENT_REPRODUCTION.md) distinguishes the working runtime from historical setup files and the separate local CPU environment. Writer generation, all nine target caches, their archive backups and the scored-with-failure transfer scope above are terminal; resource accounting and pod shutdown are confirmed in the linked final receipts.
