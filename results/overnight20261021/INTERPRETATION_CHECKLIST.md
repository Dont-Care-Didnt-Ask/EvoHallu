# Interpretation checklist for the overnight report

This is an audit of the local evidence, not the final overnight result. The numeric snapshot below comes from `report.json` created at **2026-10-21 19:01:02 UTC**. That snapshot contains **3 finished jobs and 114 recorded requests out of 22 planned jobs / 1,200 requests**. It includes running jobs. The remote campaign continues; refresh final counts and scores before publication. No GPU work or new detector inference was performed for this audit.

## 1. Results already supported by saved predictions

All numbers are unweighted means over the same nine source-model datasets, in AUROC percentage points.

| Frozen method | Directed AUROC | Paper-compatible AUROC | Duplicate-disjoint paper AUROC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Sequence NLL | 65.6439 | 68.7112 | 68.6849 | Baseline; historical predictions reproduced exactly on all nine datasets |
| Grouped logistic, 14 documented features | 70.6154 | **73.0194** | **72.9486** | Supervised fitted control, frozen before test evaluation |
| Grouped QDA, same 14 features | 69.6898 | 72.1657 | 72.0836 | Separate predeclared fitted control |
| Neutral frozen-context screen, seed 2107 | 70.8482 | **73.1366** | 73.0601 | Secondary screen winner; 30 new requests conditioned on historical parent programs |
| Diverse frozen-context screen, seed 2107 | 70.1307 | 73.0072 | 72.9531 | Secondary screen winner; same 15 historical contexts and 30-request budget |
| Original frozen-context screen, seed 2107 | 68.7713 | 69.9729 | 69.8964 | Matched-budget secondary screen reference |

Sources: `report.json`, `methods.csv`, `frozen_manifest.json`, and immutable prediction files under `evaluations/`. The screen winners were frozen by directed PopQA validation, not chosen by these test scores. Their source hashes are in the frozen registry.

The screen is not an adaptive 30-round evolution run. Each arm makes 15 pairs using saved contexts from an earlier 9B trajectory. It inherits earlier search effort through those examples. Do not describe 73.1366 as a from-scratch 30-call search, the final three-seed evolution result, or evidence of improved evolutionary feedback.

The actual neutral-screen winner combines total NLL, a linearly position-weighted sum of observed-versus-top-1 gaps with coefficient 4, and final-quarter gap variance with coefficient 2. These fixed coefficients were proposed in code, not fitted afterward. Its top-1 logprob is normalized over top-512 logits, so the gap is approximate. Although `top_k_indices` is declared as a dependency, the code does not use it. This is a three-component generated formula, not the user's many-feature Claude example.

## 2. Is the grouped 73.02 control interpretable?

**Yes, as a fixed-feature supervised baseline with a documented limitation from test duplicates.** Its training/validation separation is sound at the normalized-claim level; the default test collection is not fully claim-disjoint from training. Both statements must be retained.

The fitting script performs the following steps:

1. Start from the same 7,007 supplied PopQA training rows. The first fold of five-fold `StratifiedGroupKFold`, with seed 441, creates **5,606 fitting rows / 1,401 outer-validation rows**. Groups use case folding and collapsed whitespace. There are **zero shared normalized claims** across this split.
2. Compute 14 deterministic, per-sample features from the existing Llama3.1-8b-instruct cache. Feature construction over all rows does not fit global statistics or read labels as features.
3. Inside the 5,606 fitting rows, use three-fold `StratifiedGroupKFold` to compare five logistic C values and five QDA regularization values. Every inner fold also has zero shared normalized claims.
4. `StandardScaler` is inside the sklearn pipeline, so its normalization is fitted separately within each inner training fold. The selected head is then fitted on the 5,606 rows only. Logistic selects **C=100**; QDA selects **reg_param=0.6**.
5. Outer validation is measured after fitting: **82.1672 logistic**, **80.3277 QDA**, and **73.7052 NLL** on that same partition. Outer-validation labels do not choose hyperparameters or enter final fitting. These validation numbers must not be directly compared with the different historical stratified validation partition.
6. The actual serialized models both have a 14-feature input and a scaler fitted on exactly **5,606 rows**. The test-evaluation model SHA-256 is `b2f27a12db5177f004fa81f5df521f59436e4df2fcdc7f632e0eba847ecd4bce`, matching the audited local artifact. Neither head was refitted on all 7,007 rows before test scoring.

The inner folds contain 3,736/1,870, 3,738/1,868 and 3,738/1,868 fit/fold-validation rows. Exact recomputation and artifact checks are saved in `grouped_control_interpretation_audit.json`.

The 14 features are sequence length, NLL sum/mean/std/max/quantiles, positional NLL slope, exponentially weighted NLL, top-512-normalized top-1 mass, top-1/top-2 raw-logit gap, top-512 entropy, and first/last-quarter NLL. The three full-vocabulary `entropy_*` features in the 17-feature pool are excluded. Top-512 entropy/mass refer to the truncated distribution, not known full-vocabulary mass.

This result demonstrates a strong conventional fitted control on fixed model features. It is **not supervised evolution**, a newly trained language model, or an unsupervised generated detector. Both logistic and QDA were retained; the report should not choose a new primary head by test performance.

## 3. Exact duplicate overlaps and their practical effect

The grouped split prevents duplicate claims from crossing fitting and outer validation. It does not clean the separate nine-dataset evaluation collection.

| Evaluation dataset | Test rows matching fitting claims | Test rows matching outer-validation claims | Total excluded by the existing full-training-CSV mask |
| --- | ---: | ---: | ---: |
| PopQA test, 6,974 rows | **393** | **127** | **520** |
| WildHallucinations, 46,605 rows | 5 | 1 | 6 |
| Cities, 1,496 rows | 1 | 0 | 1 |
| Other six evaluation datasets | 0 | 0 | 0 |

PopQA's 520 overlapping rows contain **412 unique normalized claims**: 323 matching fitting claims and 89 matching outer-validation claims. Do not interchange unique-claim counts with row counts. The existing disjoint mask excludes overlap with **all 7,007 training-CSV rows**, which is conservative for a head fitted on only 5,606 rows. It leaves **6,454 PopQA test rows**.

The logistic paper-compatible macro changes from **73.0194 to 72.9486**, a reduction of **0.0708 pp**. Its PopQA directed AUROC changes from **81.6578 to 81.0272**. This sensitivity result shows that exact duplicates explain little of the measured macro gain; it does not rule out semantic duplicates, shared entities/relations, or other dataset shortcuts.

The ordinary evolutionary split remains the inherited stratified **5,605/1,402** partition and has **88 shared normalized claims** between its training and validation portions. Do not transfer the grouped control's zero-overlap claim to those evolution runs. Claim grouping is also not entity- or relation-disjoint splitting.

The evaluation datasets share a few normalized claims with one another, as documented in `results/held_out_claim_overlap.json`. Current bootstrap groups are formed within each dataset, not across the entire benchmark; intervals do not model all cross-dataset dependence.

## 4. What can be compared with the paper?

[The paper, Sections 3/5/6 and Figure 2](https://aclanthology.org/2026.acl-srw.115.pdf), uses Llama3.1-8b-instruct features and PopQA fitness. Typical searches use 500 rounds with two proposals. It describes 20 GPT-OSS searches and 10 Claude searches, retaining three validation candidates per Claude run and a curated collection of 16 GPT candidates. Figure 2 shows selected collections, including Claude test ranks 1/8/15/22/30. Its displayed reference values are **68.7 NLL**, **64.2 perplexity**, **59.9 mean token entropy**, **70.8 best displayed GPT method**, and **73.4 best displayed Claude method**. The paper also describes an up-to-three-features prompt instruction; the supplied repository's active original prompt has no such explicit cap.

The author's later clarification, supplied by the user, resolves the baseline discrepancy: compute `max(AUC, 100-AUC)` for each method and dataset before averaging. This clarification is distinct from the paper's printed metric description. Labels remain **`is_hallu`**. All three baseline bars reproduce at displayed precision; there is no unresolved 65-versus-68.7 baseline error.

The grouped control is **+4.3082 pp over NLL** on the paper-compatible macro, numerically **2.2194 pp above 70.8** and **0.3806 pp below 73.4**. These are point comparisons. They do not establish a matched improvement over the paper's evolutionary search: estimator supervision, proposal budgets, writers, retained collections and selection procedures differ. The current screen winner is also below 73.4 and is a secondary fixed-context result. Do not say that the paper has been beaten on this evidence.

The current logistic's 500-replicate normalized-claim bootstrap gives a paper-compatible macro interval of **72.1908–74.0062** and a paired gain-over-NLL interval of **3.7933–4.9026 pp**. These condition on the frozen head and fixed collection of datasets. They are not search-seed uncertainty, a paired comparison against an unavailable paper method, or a multiplicity-adjusted family-wise result.

## 5. Metric language that must stay precise

- AUROC, ROC-AUC and AUC-ROC denote the same ranking metric here. Report AUROC, not accuracy, and use percentage points for differences on the ×100 scale.
- The directed score uses one fixed score direction with the supplied `is_hallu` labels. The paper-compatible score is direction-invariant separately on each dataset. It is not `max(mean_AUC, 100-mean_AUC)`.
- For the grouped logistic, UHEAD has directed AUROC **39.1820**, which becomes **60.8180** after folding. The nine-dataset mean therefore rises from 70.6154 to 73.0194. Do not imply that the same deployed score direction works above chance on every dataset.
- The fold uses evaluation labels to determine direction. It reproduces the reporting convention; it is not an extra learned calibration step available without labels in deployment. The report's `test_labels_used_for_selection=false` means candidate/seed identities were not selected on test scores, not that reporting itself is label-free.
- The macro gives each dataset equal weight. WildHallucinations has 46,605 rows and AVeriTeC 427, but each contributes one ninth. Do not call the macro a pooled 97,488-row AUROC.
- The logistic is near chance on X-Fact (**50.0185**) despite strong Cities (**98.9198**) and Companies (**87.5889**) results. Avoid claims of uniform improvement/generalization.

## 6. Extensions: implemented versus measured in this snapshot

| Requested extension | Implemented now | Evidence in the audited snapshot | Final reporting requirement |
| --- | --- | --- | --- |
| More code-writing models | Pinned 9B study; optional 27B confirmation and GLM setup exist | Three completed 9B screens; downloaded checkpoints are not inference results | Identify actual writers, full budgets and seeds; mark optional skips/partials |
| Prompt diversity | Four fixed-context policies and four-policy adaptive runs; matched context diagnostics | Three full screens; remaining one incomplete in this saved snapshot | Refresh all four screens; distinguish useful diversity, validity and OOD quality |
| Complex Claude-like detectors | Compositional prompt with no feature-count ceiling | The arm is running; no completed outcome in this snapshot | Report executed feature groups and validity; do not claim a Claude reconstruction |
| Supervised estimators | Fixed 14-feature logistic/QDA, nested grouped tuning | Completed nine-dataset results above | Separate fitted controls from generated unsupervised methods |
| Whether feedback helps | Neutral fixed-NLL-example versus neutral evolution, three seeds each | Prepared/running campaign; no complete matched comparison yet | Use the same writer, seed and 60-request budget; reference neutral evolution |
| Actual parent-selection tuning | Linear fitness ranks mixed with 20% uniform eligible-group probability | Additive wrapper and three seeds prepared; no completed sampler result yet | Compare with neutral evolution; preserve quantile gate/islands/within-group selection |
| Feature-model transfer | Six archived target caches; guarded three-cache completion and six/nine CPU scoring | Downloaded pinned 1B model/tokenizer; no completed transfer outcome in this snapshot | Freeze source methods; verify cache lineage; compare matching six or nine datasets |
| Selection and complexity | Unique frozen identities, train-best controls, optional top-three validation sources, failure/source inspection | Three validation-selected screen programs scored | Report selection-rule contrasts separately; ablate executed terms rather than explanations |

The planned core is **1,020 writer requests**, with **180 additional rank-sampler requests**, hence **1,200 across 22 jobs**. Optional writer confirmation/generalization requires separately recorded manifests and realized budgets. These totals count requests, not valid unique programs or independent scientific ideas.

Two-stage sketch/implementation calls, automatic repair, a behaviour-aware novelty archive, fitted-parameter program evolution, and a matched long-trajectory-versus-restart experiment remain untested designs. Three separate 30-round seeds alone do not establish that restarts outperform one equal-cost longer run.

## 7. Documentation updates needed after the campaign finishes

`EXPERIMENT_EXTENSIONS.md` and `NEXT_EXPERIMENTS.md` remain preparation-era documents. Their historical timestamp is legitimate, but they must not be presented as a final run ledger.

- [ ] Replace “prepared/no writer requests” with a link to the final, timestamped campaign status; retain the original proposed matrix for provenance.
- [ ] Add the implemented rank-sampling arm. Remove any blanket statement that rank sampling remains design-only; behaviour-aware novelty archives remain unimplemented.
- [ ] Add the new nine-dataset grouped fitted-control results; its old “held_out_test_scored=false” preparation artifact is historical and should not be read as current state.
- [ ] Distinguish fixed-context screen outcomes from adaptive evolution and from feedback controls. The screen's inherited parent trajectories are part of its information budget.
- [ ] Replace old single-A100 runtime estimates with measured two-A100 wall time, actual GPU count, model-serving/runtime metadata, issued requests, tokens, and confirmed stop time. Summed overlapping request latency is not GPU-hours.
- [ ] Update optional 27B/GLM and feature-transfer status only from real run receipts and prediction coverage, not checkpoint downloads or generated configs.
- [ ] Preserve both ordinary and folded AUROC, all seed results, full-budget indicators, fit/test duplicate diagnostics, and all frozen source hashes.
- [ ] State that the nine datasets were already inspected during previous work. Additional policy changes after seeing test scores would be adaptive exploratory work; no new blind benchmark was created.

## 8. Selection/comparison checks for the final tables

- [ ] Training fitness controls the evolutionary population. Directed PopQA validation chooses the ordinary run winner, with baseline fallback. Neither the nine test scores nor target-1B labels choose a candidate or seed.
- [ ] Report means/spread across the three completed evolution seeds. A best-by-validation seed is an additional labelled selection; never promote the best test seed.
- [ ] For feedback and rank-sampler effects, use **neutral evolution** as the primary comparator. Original evolution remains a separately named broad reference. The CPU report helper has been extended accordingly.
- [ ] Pair by writer, seed, request budget and full completion; exclude mismatched/partial runs from matched claims. Keep screen30 comparisons within the screen phase and writer confirmations within their writer/phase.
- [ ] Keep finished-but-partial source identities as individual exploratory records. Exclude them from primary seed averages and target transfer; the CPU helpers now enforce `full_budget=true` and record the excluded identities explicitly.
- [ ] Inspect validity and diversity jointly. In this snapshot both original and neutral screens have 19/30 valid candidates; distinct nonconstant probe rankings are 18 and 17. This alone does not demonstrate increased diversity from neutral guidance.
- [ ] Recompute the paired context analysis after all screens finish. Its current Markdown is an older partial snapshot than `report.json`; do not combine its counts with the later report as though simultaneous.
- [ ] Save actual failure categories, truncation flags and constant vectors. Invalid output is part of request cost; syntax failure without a `length` finish reason is not proven truncation.
- [ ] Keep the 512-row training probe diagnostic-only. Source-hash uniqueness, rank diversity and detector accuracy are separate observations.
- [ ] Inspect actual top-k alignment and operations. Cached top-k arrays have one extra row; observed logprob index t corresponds to logits row t. Top-512 softmax is conditional. Full cached chat/template tokens make “late claim token” interpretations approximate.
- [ ] For six-target transfer, retain a fixed source-selected cohort and compare source/target on exactly those same six datasets. Early transfer must not guide later writer/prompt choice. Final nine-target scoring may add new source-selected methods but must not select by target quality.
- [ ] Verify that the final report's nine-dataset method count equals the frozen cohort, all selected sources have matching hashes, raw score arrays are saved, and archival success is separate from pod-stop success.

## 9. Additional evidence produced by this audit

- `grouped_control_interpretation_audit.json`: exact outer/inner grouped splits, test-overlap rows/unique claims, scaler fit counts and fitted-artifact SHA.
- `scripts/overnight_report.py`: explicit full-budget neutral-evolution comparisons; partial/unverified rows stay outside primary seed groups and remain labelled exploratory in individual tables.
- `scripts/overnight_transfer_eval.py`: only full-budget primary identities transfer, and adding a partial identity leaves the full cohort unchanged. Metadata now records the actual `/workspace/mod1-models/llama31-tokenizer` directory plus the pinned source tokenizer ID/revision. Cached evaluation does not load a tokenizer, so this corrects provenance without changing score calculations.
- Five combined matching, partial-budget and tokenizer-provenance regression tests passed in 7.35 seconds. They do not execute model inference.

No execution configuration, pinned search implementation or candidate source was modified for this interpretation audit.
