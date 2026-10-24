# EvoHallu Preliminary Report

Experiment campaign: 15 October 2026. Code and selection audit: 19 October 2026.

Four new language models were tested as authors of Python hallucination detectors. The strongest primary directed-AUROC evolution result selected on validation was **Qwen3.8-27B: 69.80 macro AUROC**, compared with **65.64 for Sequence NLL**, a gain of **4.15 percentage points**. Evolutionary feedback improved Laguna's results, but an advantage over generation without feedback was not consistent across writers.

All primary results use the original **`is_hallu`** target labels. The metric is **ROC-AUC / AUROC**, expressed on a 0–100 scale and averaged equally across nine datasets. Alternative label diagnostics are excluded from these results and do not replace the supplied target labels.

**Reporting reconciliation, 21 October:** the author clarified that paper reporting applies `max(AUC, 100-AUC)` separately to each method/dataset before averaging. With that rule, Sequence NLL is **68.71**, reproducing the published **68.7**. The baseline discrepancy is resolved without changing `is_hallu`. For the existing validation-selected evolution programs, paper-compatible macro scores are **Qwen9B 72.45, Qwen27B 72.00, Laguna 71.27, Qwen4B 70.10**. Directed AUROC remains the historical training/validation selection metric and is retained throughout this report. Full paired reporting columns for all 58 methods: [paper_reporting.csv](results/next_experiments/paper_reporting.csv). These are reporting transforms of existing results, not new searches or a matched-budget superiority claim.

## 1. Models and their roles

| Component | Model or method |
| --- | --- |
| Original code writer configured in EvoHallu | GPT-OSS-120B; not rerun in this campaign |
| New code writers | Qwen3.8-27B, Laguna XS 2.1, Qwen3.5-9B, Qwen3.5-4B |
| Fixed feature model | **Llama3.1-8b-instruct**, shared across the primary experiments |
| Evaluated detector | A generated Python class, `HallucinationDetector(Estimator)` |
| Baseline detector | `SequenceLogProb`: `score = -sum(sequence_logprobs)`; **65.643882 AUROC** |

Qwen and Laguna replaced the **code-writing model**. Llama3.1-8b-instruct remained the source of token-level features. Evaluation executes the generated formula; it does not call the writer model for each claim. Language-model weights were not fine-tuned. Separate logistic regression and QDA controls were fitted on fixed features.

The feature checkpoint was inherited from the supplied EvoHallu repository. Its exact identifier and pinned revision are recorded in [feature_model_provenance.json](results/feature_model_provenance.json). A subsequent metadata check found that all four Safetensors weight files have the same published SHA-256 hashes as the corresponding [official Meta Llama-3.1-8B-Instruct checkpoint](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Some configuration files differ, so this does not establish identity of the entire model package. The check compared published file metadata; it did not rerun inference. Details are in [feature_checkpoint_identity_audit.json](results/feature_checkpoint_identity_audit.json).

## 2. Input data and feature extraction

The supplied data comprise **10 CSV files: one PopQA training file and nine evaluation datasets**.

- **PopQA training file:** 7,007 rows, split into **5,605 training / 1,402 validation** rows using a stratified split with `split_seed=441`.
- **Evaluation datasets:** AVeriTeC, UHEAD, WildHallucinations, Cities, Companies, CommonClaim, CounterFact, PopQA test, and X-Fact: **97,488 claims** in total.
- The main feature-extraction path uses `claim` as input text and `is_hallu` as the target. It does not use `primed_text`. The `10shot` filename suffix does not itself introduce ten demonstrations into the model input.
- Llama computes features for the supplied claim text: token log-probabilities, top-512 logits, and their token indices. It does not generate new answers for verification. The extraction template contains the instruction `Generate true statement` and the supplied claim.

Primary results retain the existing feature-array preparation convention. Attention and hidden-state features were not extracted into the primary caches. Token alignment and template-suffix effects were investigated separately. These runs therefore establish results on our cached inputs; byte-for-byte agreement with the paper's historical caches has not been verified.

## 3. Parent selection inside evolution

Actual entry point: `src/experiments/main.py` → `extension_trial=true` → `src/experiments/extension_trial.py`. The runner reuses the repository's original `ProgramsDatabase` and prompt builder.

1. Initialize **two islands**, each containing the Sequence NLL baseline. Randomly choose an island for each round.
2. Evaluate candidate fitness using **training AUROC on 5,605 rows**. The stored fitness is `round(100 * roc_auc_score(...), 4)`. Validation scores are not used as population fitness or shown as the scores attached to parent examples in the prompt.
3. Within each island, group programs by **identical training fitness**. These are score groups, not semantic or code-similarity clusters.
4. Apply the schedule `p(r) = 0.6 + (r / 30) * (0.2 - 0.6)`, with `r=0..29`, to narrow sampling toward the top fraction of score groups: from 60% to approximately 21.33%. The implementation uses a `nearest` quantile and scores strictly above its threshold. Filtering applies only when more groups remain above the threshold than the number of required examples; it may be skipped for small populations.
5. Sample up to **two distinct score groups**, weighted by `softmax(train_score / 40)`. Within each group, sample a program with a modest preference for shorter source code. Parent selection is probabilistic.
6. Pass parent code and its **training fitness** to the writer. Generate **two candidates from the same prompt**, with temperature **0.7**, `top_p=0.8`, `top_k=20`, and at most **4,096 output tokens**. The serving context limit is 16K.
7. Evaluate candidates in separate processes, first on training and then on validation data. Enforce time/memory limits, finite-score checks, and exact-source deduplication by SHA-256. Insert programs with successful training evaluations into the selected island in response order.

The insertion condition permits a training-successful candidate to enter the population even if validation fails. **No such cases occurred in the recorded runs:** all 463 candidates with successful training evaluations also received validation scores. Island resets are disabled in the reused database, and older programs are not automatically removed.

## 4. Final program and seed selection

The selection stages are separate:

```text
PopQA training   → population fitness and parent sampling
PopQA validation → best program across all candidates in a run
PopQA validation → best evolution seed for each writer
Nine test sets   → evaluation of frozen programs
```

- **Run winner:** the highest stored validation AUROC among all successfully evaluated candidates generated during that run.
- Ties retain the first candidate in ledger order. If baseline validation AUROC is at least as high as the best generated candidate's score, retain Sequence NLL. Selection uses training/validation scores rounded to four decimal places on the percentage scale.
- Save the chosen source as **`frozen_estimator.py`**. The remaining candidates retain their source and validation predictions in the logs; they are not all evaluated on the nine test datasets.
- **Primary writer seed:** the highest PopQA validation score among that writer's completed primary `evolution` runs. Pilot runs, protocol-excluded attempts, `independent`, and `baseline_seeded` runs are excluded from this seed selection.
- No validation splits from the other eight datasets are used for selection; those datasets serve as test sets.
- The complete test table contains **58 methods**: 25 supplied formulas, 6 diagnostic baselines, 6 fitted control models, and 21 frozen programs. It includes both seeds, control arms, and the pilot; inclusion in the table does not make a run the primary result for its writer.

For example, Laguna seed 1047 achieves **69.43** on the test sets, while seed 1048 achieves **69.23**. The primary result is **1048**, because its validation score is higher: **79.5008 versus 79.2575**. The test result did not override validation selection. The Qwen27B pilot, with test AUROC **70.12**, is also excluded from the primary comparison because it used a different generation budget.

## 5. Completed primary evolution runs

| Writer | Seed | Rounds / requests | Valid candidates | Selected round¹ | Selected train AUROC | Selected validation AUROC | Macro test AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3.8-27B** | **1047** | **30 / 60** | **57** | **5** | **79.9031** | **79.6135** | **69.80** |
| Qwen3.8-27B | 1048 | 30 / 60 | 53 | 14 | 79.8256 | 78.7667 | 68.53 |
| Laguna XS 2.1 | 1047 | 30 / 60 | 46 | 28 | 80.1655 | 79.2575 | 69.43 |
| **Laguna XS 2.1** | **1048** | **30 / 60** | **43** | **28** | **80.5794** | **79.5008** | **69.23** |
| Qwen3.5-9B | 1047 | 30 / 60 | 32 | 12 | 77.5949 | 77.8208 | 68.02 |
| **Qwen3.5-9B** | **1048** | **30 / 60** | **36** | **16** | **79.3900** | **79.5752** | **69.54** |
| **Qwen3.5-4B** | **1048** | **21 / 42** | **16** | **10** | **79.4759** | **78.0317** | **68.37** |

¹ Round numbers are one-based here and zero-based in the JSONL logs. Bold rows are the primary runs selected by validation.

Primary evolution totals: **7 runs, 201 rounds, 402 generation requests, and 283 valid candidates**. Qwen4B stopped after three consecutive rounds without a successful validation evaluation. These are seven separate searches, not one continuous evolution trajectory.

Additional comparisons using the same `is_hallu` targets:

| Method | Macro test AUROC | Gain over NLL, percentage points |
| --- | ---: | ---: |
| Sequence NLL | **65.64** | — |
| Qwen27B, primary evolution | **69.80** | **+4.15** |
| Qwen9B, primary evolution | **69.54** | **+3.90** |
| Laguna, primary evolution | **69.23** | **+3.59** |
| Qwen4B, primary evolution | **68.37** | **+2.73** |
| Supplied `LogProbSlope_BottomMassAvg`, selected by validation | **68.62** | **+2.98** |
| Logistic regression, 14 fixed features | **70.28** | **+4.63** |
| QDA, the same 14 features | **69.95** | **+4.30** |

The supplied formulas were evaluated separately on validation data; `LogProbSlope_BottomMassAvg` had the highest validation score. The highest test score across all 25 supplied formulas is **69.88**, but that is a descriptive maximum selected on test performance. The new 69.80 result therefore does not outperform every supplied formula.

## 6. Controls: does evolutionary feedback help?

`independent` generates each program from scratch without population examples. `baseline_seeded` includes the same original NLL example and its training score in every prompt; improvements are never fed back. Both controls use the same final validation-selection rule as evolution.

Matched comparisons at **seed 1048, with 60 requests in each arm**:

| Writer / control | Evolution | Control | Evolution − control, percentage points | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Laguna / independent | 69.23 | 68.31 | **+0.93** | +0.67 … +1.19 |
| Laguna / baseline_seeded | 69.23 | 68.53 | **+0.70** | +0.42 … +0.99 |
| Qwen27B / independent | 68.53 | 69.47 | **−0.94** | −1.25 … −0.64 |
| Qwen9B / baseline_seeded | 69.54 | 69.80 | **−0.26** | −0.69 … +0.11 |

The Qwen27B comparison uses evolution seed 1048; its 68.53 must not be replaced by the validation-selected seed-1047 result of 69.80. Qwen9B independent stopped after 12 requests with one valid candidate and retained NLL. That incomplete search is not a matched 60-request control.

Intervals use 500 paired bootstrap replicates over groups of identical normalized claims. They describe the fixed nine-dataset collection and frozen programs. They do not capture variation across search seeds and are not adjusted for multiple comparisons.

The evidence supports a benefit from feedback for Laguna in these runs. It does not establish a universal advantage of evolution across writers. The fixed-example control more directly isolates feedback; the independent control removes both feedback and code examples.

## 7. Properties of the discovered formulas

- **Qwen27B:** position-weighted NLL that emphasizes later tokens. Algebraically, the main expression reduces to `n * weighted_mean(NLL, weights=1..n)`, subject to floating-point differences. A simplified float64 implementation achieves **69.7926**, compared with **69.7969** for the original: **−0.0043 percentage points**.
- **Laguna:** negative correlation between position and log-probability, a normalized range of top-512 entropy, and variance of top-1 probability. Removing the variance component gives **69.2192**, compared with **69.2343**: **−0.0151 percentage points**.
- **Qwen9B:** total NLL plus the change in mean gap between the first and last quarters and gap variance near the end. Alternative-token probabilities are normalized within top-512, so this is not an exact full-vocabulary probability gap.
- **Qwen4B:** a combination of transformed position/log-probability correlation and second differences of log-probabilities.

The writer proposed the coefficients in the primary formulas, and selection operated on whole programs. Those coefficients were not fitted by gradient descent. Comments mentioning “learned weights” in generated code do not imply that a separate fitting step occurred.

The top five saved candidates per run have also been exported by validation score. No manual diversity curation of a top-k collection was used to change the primary results.

## 8. Coverage of the proposed extensions

| Idea from `EXPERIMENT_EXTENSIONS.md` | Completed work |
| --- | --- |
| New code-writing models | Four writers, seven primary evolution runs, all nine test datasets |
| Deliberately increase diversity | Measured candidate-prediction diversity; two-stage generation and a completed structured-family intervention were not tested |
| Explain differences between Claude models | Historical Claude populations were unavailable; no explanation was established |
| Supervised evolution | Fitted fixed-feature logistic regression and QDA; full evolution of programs with fitted parameters was not implemented |
| Test whether evolution is necessary | Completed independent/fixed-example controls and separate CPU coefficient searches; results depend on the writer |
| Transfer to another feature model | Preserved six Llama-3.2-1B feature caches; no complete transfer-quality comparison |
| Complexity and useful features | Completed four formula ablations and feature diagnostics; historical Claude formulas were unavailable |

## 9. Differences from the paper and original runner

The [paper, Sections 4–5 and Figure 2](https://aclanthology.org/2026.acl-srw.115.pdf) describes typical searches of **500 rounds × 2 candidates**, followed by selection of the top three candidates per run. For GPT-OSS, the authors manually curated a diverse collection of 16 high-validation-scoring methods. For Claude, Figure 2 displays selected test-rank positions among the top 30 methods selected by validation. Our one-winner-per-run protocol does not reproduce that collection-selection procedure.

In the supplied original UQ branch of `main.py`, the candidate with the highest **training AUC in the current processed batch** is evaluated and logged on train/validation/testset. That choice uses `extra_metrics["AUC"]`, rounded to two decimal places, while population insertion uses the four-decimal `score`. That block does not implement our final `argmax(validation)` over all candidates from the run.

Our runner uses **30 × 2**, explicit resource limits and stopping conditions, validation evaluation for each successfully executed candidate, one frozen winner per run, and a separate nine-dataset evaluation. Each full primary search uses **6% of the paper's typical round budget**. Multiple writers and control arms do not substitute for a longer evolution trajectory.

The author's reporting clarification resolves the baseline comparison:

| Baseline | Directed macro AUROC | Paper-compatible macro AUROC | Published |
| --- | ---: | ---: | ---: |
| Sequence Probability / NLL | 65.6439 | **68.7112** | 68.7 |
| Perplexity | 64.1747 | **64.1747** | 64.2 |
| Mean Token Entropy | 58.4200 | **59.9043** | 59.9 |

All three reproduce the paper at its displayed precision. The reporting rule is applied per dataset before averaging; it does not complement any labels. Mean Token Entropy changes direction on both UHEAD and Cities, illustrating why flipping UHEAD labels alone was not the correct general explanation. Training fitness, validation-selected candidates, and frozen predictions remain unchanged. A certified mapping from repository classes to every `Evo-Gpt-*` alias remains unavailable; baseline agreement does not establish a matched writer comparison. The paper's displayed best Evo-Gpt and Evo-Anth values are 70.8 and 73.4. [Figure 2](https://aclanthology.org/2026.acl-srw.115.pdf#page=4)

## 10. Code changes

The archived diff from the **supplied ZIP to the audited experiment snapshot** includes **57 code/configuration/test files: 5,660 insertions and 50 deletions**. The complete repository diff, including documentation, result tables, and binary figures, covers **112 files**. These counts describe the experiment snapshot before this English report and the additional checkpoint metadata audit.

| File or component | Change |
| --- | --- |
| `src/experiments/main.py` | Dispatches bounded experiments to `extension_trial`; removes an unused import |
| `src/experiments/extension_trial.py` | Local generation, control modes, resource limits, isolated evaluation, deduplication, candidate ledger, validation selection, and frozen source export |
| `src/packing/evaluate/uncertainty_quantification/task_uncertainty_quantification.py` | Opens the separate test file only for `testset`; fixes the imports/class execution namespace; saves predictions; adds a claim-group split option; clarifies the log-probability input description |
| `src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py` | Supports probability-only feature extraction, dynamic GPU microbatches, dtype/attention-backend settings, and metadata |
| `supervised_baselines.py`, `alignment_baselines.py` | Adds diagnostic detectors, fixed 14/17-feature pools, and fitted logistic/QDA controls |
| `scripts/evaluate_cached_claims.py` and related evaluators | Reuses shared caches, evaluates frozen programs, computes bootstrap comparisons, checks overlap, and verifies additions without changing existing predictions |
| `scripts/fit_supervised_baselines.py` | Tunes logistic/QDA hyperparameters with three-fold CV inside the 5,605 training rows; validation/test do not participate in fitting or hyperparameter selection |
| Other `scripts/` | Campaign scheduling, local serving, provenance, diagnostics, cache archiving, and pod shutdown |
| `configs/`, `requirements-trial.txt`, `tests/test_extension_trial.py` | Model/budget configurations, dependencies, and protocol checks |

The original **`programs_database.py`, `packing/model/prompt.py`, `features.py`, and `found_formulas.py` were unchanged**. Orchestration and final selection changed; islands and parent sampling were inherited from the supplied implementation.

The separate 21 October preparation adds opt-in neutral/diverse/compositional prompts, frozen-context trials, saved training predictions, and top-three validation exports. These additions preserve the original default prompt and are not included in the historical 57-file experiment patch. Their prepared studies, completed CPU control, and validation are documented in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md).

## 11. Limitations and resource use

- Primary evolution uses the original stratified split. An audit found **88 shared normalized claims between train and validation**, and **412 shared unique claims between PopQA train and PopQA test**. Duplicate-disjoint evaluations and grouped fitted controls are available, but full evolution was not rerun on a cleaned/grouped split.
- Each writer has only one or two primary seeds. This is insufficient for a stable ranking of writer capability. Models were not matched on elapsed time or token budgets.
- Some additional controls and ablations were introduced after inspecting early results. The campaign is not a single preregistered blind confirmatory experiment. Primary candidate/seed selection nevertheless uses PopQA validation rather than test scores.
- Devstral's six initial requests failed with HTTP400 before generation. This is a client-protocol failure, not a model-quality result. Six initial Qwen27B independent requests with an incomplete evaluator-interface description were also excluded; the corrected run is separate.
- Across controls, pilot, and excluded attempts: **676 LLM requests, 463 valid candidates**, and approximately **1.785 million reported input-plus-output tokens**. Requests and valid programs should not be interpreted as distinct scientific ideas.
- A100 80 GB rental: **10.18 hours**, estimated GPU cost **$18.23**, external model API spending **$0**. This includes feature caching, model changes, and evaluation. The pod was stopped after the campaign.

## 12. Audit evidence and full diff

Preparing this report required no new GPU runs, model generations, or searches. The experiment pipeline was not modified during the reporting audit.

All **15 LLM runs / 676 requests** were checked against their ledgers: selected candidates match the validation-maximum rule with baseline fallback, and frozen source files match the selected code. The original population algorithm reproduced **335 saved prompts and island selections with no mismatches**. Three prompts from the excluded independent run using the old interface description were not replayed. This replay did not regenerate model responses or re-execute candidate programs.

All nine test tables contain the same **58 methods, with zero scoring failures**. Of 23 frozen source files, 21 are included; the two protocol-excluded attempts are absent from the final cohort. Primary writer seeds were independently reconstructed from validation scores and match the reported selections.

Available review artifacts:

- [Full code/configuration diff](results/technical_review/full_code.patch).
- [Full repository diff, including documents, results, and binary figures](results/technical_review/full_repository.patch).
- [Existing uncommitted edits to the two earlier reports](results/technical_review/uncommitted_user_documents.patch).
- [Per-run selection audit, CSV](results/technical_review/selection_audit.csv) and [JSON](results/technical_review/selection_audit.json).
- [Top five saved validation candidates per run](results/technical_review/top_validation_candidates.csv).
- [Diff boundaries, SHA-256 hashes, and application checks](results/technical_review/diff_manifest.json).

The main patches compare the actual supplied ZIP with the audited working files. Import commit `49504d5` had already extended `.gitignore`; two original sweep-example CSVs were ignored by git but remained on disk and are preserved in the ZIP comparison. The experiment snapshot is HEAD `45dff51361a6f252a17ff71f479bbfb9ea4c5431`, plus the existing edits to `REPORT_RU.md` and `EXPERIMENT_RESULTS_RU.md`.

Both ZIP-to-snapshot patches were applied to an isolated copy of the original archive and verified byte-for-byte for every affected file. Datasets, weights, caches, environments, and credentials are excluded. All 23 frozen programs, including the exclusion markers, are exported separately under `frozen_estimators/` in the review package. New reporting and audit artifacts are outside the historical diff.

Primary result sources: `results/run_table.csv`, `results/method_table.csv`, `results/campaign_summary.json`, `results/search_budget_audit.json`, and `results/validation_controls/original_formulas.json`. Detailed run records: `out/logs/mod1/*/{trial_config.json,candidates.jsonl,summary.json,frozen_estimator.py}`.
