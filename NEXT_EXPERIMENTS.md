> **Completed execution update — 22 October 2026, 00:44:41 UTC:** all **27 writer jobs** completed their planned budgets, **1,500 requests / 841 pipeline-valid candidates**. Qwen3.5-9B contributes **23 jobs / 1,260 requests / 669 valid**, including all twelve adaptive runs. The Qwen3.8-27B original/diverse seed-2107 pair adds **120/120 requests / 108 valid**, scoring **71.155 / 72.306 folded source-nine AUROC**. GLM-4.7-Flash original/diverse adds **120/120 requests / 64 valid**, scoring **70.671 / 68.711**. Each new writer pair has **one matched seed**, with the diverse policy chosen from source validation. GLM original rediscovers negative position/logprob correlation; diverse retains canonical NLL. [27B formulas](results/overnight20261021/WRITER27B_CANDIDATE_INSPECTION.md), [GLM formulas](results/overnight20261021/GLM_CANDIDATE_INSPECTION.md), [terminal writer decision](results/overnight20261021/late_writer_decision.json).
>
> Full target-nine caches and scoring are now available. The component head scores **68.800 folded**, versus NLL **64.835**, gain **+3.965 [3.529,4.628] pp**; its relative advantage shrinks **0.957 pp [0.149,1.302]** across feature models. Conventional logistic-14 scores **68.939**; head-minus-logistic folded **−0.140 [−0.515,+0.458] pp** is unresolved, while directed quality favors logistic-14. Source-fixed normalization scores **66.584**, versus original **64.995**, gain over NLL **+1.750 [1.484,2.034]**; its NLL-relative transfer change is unresolved. Original six-dataset reports remain preserved. [Head-nine report](results/overnight20261021/screen_component_refit/transfer_nine/TRANSFER9_REPORT.md), [normalization-nine interpretation](results/overnight20261021/normalization_transfer/NINE_INTERPRETATION.md).
>
> **Coverage failure retained:** all 18 main evaluator tasks finished, but one compositional primary fails the finite-output check on target UHEAD, with saved entropy warnings. **29/30 methods have nine-dataset coverage; one has eight**, and no complete three-seed compositional target mean is claimed. The likely float16 entropy issue is inferred from frozen code and saved warnings, without repair or new reproduction. The formal three-cache guard passed; **41,253 new rows** were extracted with all six historical caches unchanged. All three new XZ archives, **762,346,556 bytes**, are verified locally. [Completion](results/overnight20261021/remaining_transfer_completion.json), [backup](results/overnight20261021/cache_backup_receipt.json).
>
> **Devstral was skipped without serving**, and GLM's second seed was capacity-skipped. **RunPod confirmed `EXITED` with no GPU runtime at 22 October 00:44:41 UTC.** The run used **6 h 28 min**, estimated **$23.16 compute** at **$3.58/hour total**, excluding storage/network and pre-task waiting. [Stop receipt](results/overnight20261021/pod_stop_confirmation.json), [resources](results/overnight20261021/resource_summary/RESOURCE_USAGE.md). Extraction/scoring used the unchanged formal helpers after the documented post-cleanup idle-check recovery; no scientific threshold or source core was changed.
>
> **The body below is preserved as the dated preparation plan.** Its “prepared/pending” statements and preparation-environment counts describe that earlier snapshot, rather than live execution status. Completed findings and remaining questions are in [EXPERIMENT_EXTENSIONS.md](EXPERIMENT_EXTENSIONS.md), the [coverage audit](results/overnight20261021/extension_coverage_audit.md) and [source results](results/overnight20261021/REPORT.md). Current tested-runtime evidence is in [ENVIRONMENT_REPRODUCTION.md](results/overnight20261021/ENVIRONMENT_REPRODUCTION.md); no second clean rebuild is claimed.

# EvoHallu: Issues, Research Decisions, and Prepared Experiments

Updated 21 October 2026. This document implements the earlier checklist: reproduce the baseline, verify dataset assembly and the environment, inspect candidate selection, inspect top validation programs, and test targeted improvements to evolution.

**Current decision:** baseline reproduction is resolved. Keep `is_hallu`, examine useful diversity and executable feature combinations, and compare several controlled searches before committing to a single long trajectory. Four prompt policies and their study configurations are prepared. No new writer inference was run during this preparation.

## 1. What the earlier checklist asked us to settle

| Issue | Current evidence and answer | Next action |
| --- | --- | --- |
| “68.7 in the paper versus our 65” | Resolved by the author's clarified reporting transform; all three baseline bars reproduce at one decimal | Report both directed AUROC and paper-compatible AUROC |
| “10 datasets / did the datasets drift?” | Ten CSV files mean one training file plus nine evaluation datasets. All ten are byte-identical to the supplied attachments. The 7,007-row PopQA cache matches the CSV's claims and `is_hallu` in order | Pin these hashes for future studies; no label conversion |
| “How do we select during evolution?” | Training AUROC controls population fitness and parent examples. The inherited sampler uses score groups, a quantile gate, softmax weighting, and two islands | Hold this mechanism fixed during the prompt study; test archive/sampling changes separately |
| “60 candidates, validation selection, inspect the top programs” | A complete run requests 30 pairs; some requests fail. Historical winners are selected by PopQA validation. New runs preserve the primary winner and export the top three validation entries | Inspect the 27 exported historical programs and the new shortlists; no winner selection on nine-dataset test scores |
| “Long runs versus better candidate selection” | Some Qwen runs peak early; Laguna improves late. Neither proves that 500 rounds is the right next expenditure | Screen prompt policies first; compare a longer trajectory and restart portfolios later at equal total requests |
| “Environment assembled correctly?” | CPU dependency check passes for 98 packages; baseline validation predictions reproduce exactly; protocol tests cover the actual optional hooks | GPU server/model loading and throughput still need smoke validation on the execution host |
| “Work on my branch, with my identity” | Working branch is `vladimir/evohallu-preliminary`; configured identity is Vladimir Urvantsev `<urvantsevve@gmail.com>` | Current changes are uncommitted; `main` is unchanged; no remote publication |

Evidence: [readiness.json](results/next_experiments/readiness.json), [historical selection audit](results/technical_review/selection_audit.json), [candidate shortlist](results/next_experiments/CANDIDATE_SHORTLIST.md).

Setup validation: **26 protocol tests passed**, all **19 Hydra configurations** compose with their intended budgets and checkpoint revision, and deterministic replay still matches all **201 primary historical prompts/island choices**. Local HTTP stubs in protocol tests are synthetic interface checks, not experimental writer outputs. [Validation record](results/next_experiments/setup_validation.json).

## 2. Baseline reporting is now reconciled

The author explanation supplied by the user states that reporting used:

```text
dataset_reported_auc = max(dataset_auc, 100 - dataset_auc)
reported_macro = mean(dataset_reported_auc over the nine datasets)
```

Apply this separately to each **method × dataset**, before averaging. It is not `max(macro_AUC, 100-macro_AUC)`. Labels remain `is_hallu`. The former explanation alleging a UHEAD label-conversion error is superseded.

| Baseline | Directed macro AUROC | Paper-compatible macro AUROC | Published precision |
| --- | ---: | ---: | ---: |
| Sequence NLL | 65.6439 | **68.7112** | 68.7 |
| Perplexity | 64.1747 | **64.1747** | 64.2 |
| Mean Token Entropy | 58.4200 | **59.9043** | 59.9 |

All three match. Mean Token Entropy changes direction on UHEAD and Cities; NLL changes on UHEAD only. This explains why the historical UHEAD-only label-flip analysis was not the correct general reporting rule. Published baseline values: [Figure 2](https://aclanthology.org/2026.acl-srw.115.pdf#page=4).

Training fitness and validation selection remain directed AUROC. The clarification concerns reporting; it does not establish that the original search used folded fitness. A practical detector's direction and the paper's direction-invariant score are different quantities, so both columns are retained. No confidence intervals are newly inferred for the transformed scores.

## 3. What our existing results look like under that rule

These are the same frozen candidates and validation-selected seeds as before:

| Method | Directed macro AUROC | Paper-compatible macro AUROC |
| --- | ---: | ---: |
| Sequence NLL | 65.64 | **68.71** |
| Qwen3.8-27B evolution | 69.80 | **72.00** |
| Qwen3.5-9B evolution | 69.54 | **72.45** |
| Laguna XS 2.1 evolution | 69.23 | **71.27** |
| Qwen3.5-4B evolution | 68.37 | **70.10** |
| Existing logistic-14 control | 70.28 | **73.11** |

The Qwen4B transform also changes direction on X-Fact. Do not copy the historical UHEAD-only alternate-label column as its new paper-compatible score. All **58 methods** were recomputed directly from saved score vectors: [CSV](results/next_experiments/paper_reporting.csv), [JSON](results/next_experiments/paper_reporting.json).

No model was rerun, no seed was reselected, and no test result chose a new program. Numerical comparison with published evolved methods still involves different budgets, writers, and selection procedures; reproducing baseline reporting alone does not create a matched model comparison.

## 4. Data and candidate inspection findings

The raw PopQA cache has **5,605 training / 1,402 validation rows** under the historical stratified split. Its Sequence NLL is **74.3286 training / 75.7504 validation**, and the new validation vector matches the saved campaign baseline exactly. Stored sequence log-probabilities are float16; the reference implementation preserves their original summation behaviour. A float64 variant is a separate ablation.

All **7,007 samples** have one more top-k row than sequence-log-probability entries, with top-k width 512. A Qwen27B candidate checks for equal lengths, disables its added margin term when they differ, and consequently reproduces LogProbSlope rankings exactly. This is an implementation failure of the extra feature, not evidence that confidence margins never help.

The stratified train/validation split shares **88 normalized claims**. A claim-grouped split shares zero, with **5,606 / 1,401 rows**. The previously measured PopQA train/test overlap is 412 unique normalized claims. Keep historical results intact; use grouped sensitivity studies and the saved duplicate-disjoint masks for additional diagnostics. Grouping identical text does not guarantee entity- or relation-disjoint evaluation.

The [inspection shortlist](results/next_experiments/CANDIDATE_SHORTLIST.md) exports **27 source files**: the union of ordinary top-three validation entries and three entries spread by prediction behaviour per primary run. The spread rule greedily retains candidates with positive Spearman correlation below 0.995 to earlier retained entries. This is a retrospective validation shortlist for inspection, not a training-time archive or a new test-selection rule.

Examples checked in actual code:

- **Qwen27B, seed 1047:** positional NLL, approximately `n * weighted_mean(NLL, weights=1..n)`, plus numerical rounding effects.
- **Laguna, seed 1048:** negative position/log-probability correlation plus normalized top-512 entropy range and top-1 probability variance.
- **Qwen9B, seed 1048:** NLL plus first-to-last-quarter confidence-gap drift and final-quarter gap variance. Top-512 normalization and token alignment need explicit inspection before interpreting this as a full-vocabulary probability gap.
- **Qwen27B, seed 1048:** its top three validation programs all have the same ranking. The spread shortlist reaches ranks four and five to obtain different behaviours.

## 5. Preserve room for complex, useful formulas

The Claude example supplied by the user combines several families: lower log-probability quantiles, positional weighting, final-segment statistics, severity counts, local windows, early/late degradation, streaks, low-confidence probability mass, and a global mean term. That is a reason to test expressive composite proposals, rather than impose a small feature limit or treat extra complexity as failure.

The image does not specify complete executable feature definitions, so the new compositional policy is inspired by that structure; it is not labelled as the original Claude implementation. Historical Claude population histories remain necessary to explain differences between Claude writers.

Our experiment introduces **no complexity penalty or feature-count ceiling**. Record feature use, code complexity, execution cost, and validation quality, then ablate actual components. The inherited database has a shorter-source preference inside equal-training-score groups; it stays unchanged in this study and can be ablated later.

## 6. Prepared experiment matrix

Writer: pinned **Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Feature model remains **Llama3.1-8b-instruct**. This study changes code-writing prompts, not answer generation or feature-model weights.

### Prompt screen: 120 requests

Use 15 uniformly spaced saved parent contexts, selected independently of scores. Each context gets two requests under each policy, using matched request seeds and the same parent code/training scores.

| Policy | Intervention | What it isolates |
| --- | --- | --- |
| A — `original` | Historical baseline-plus-signal prescription; shared sibling messages | Reference |
| B — `neutral` | Open formula choice; parent code and interface unchanged | Explicit architectural anchoring |
| C — `diverse` | B plus a local-refinement role and a separate exploration role per sibling | Deliberate proposal-role diversification |
| D — `compositional` | B plus explicit guidance to combine complementary feature families; shared sibling messages | Composite-detector guidance, separate from sibling roles |

C does not guarantee mathematically different outputs; it requests distinct roles whose actual behaviour must be measured. D changes composite guidance without also changing sibling-role assignment, allowing a cleaner comparison with B.

The frozen-context mode never samples evolved examples. This isolates proposal behaviour conditional on fixed examples; it is not an end-to-end test of long-term evolution.

### Evolution and feedback confirmation

Fresh seeds: **2107, 4319, 8741**. Each run requests **30 rounds × 2 = 60 candidates**.

| Prepared phase | Arms | Requests | Status |
| --- | --- | ---: | --- |
| `prompt_screen` | A/B/C/D on fixed contexts | **120** | Prepared |
| `evolution_screen` | Original / neutral / diverse, three seeds each | **540** | Prepared |
| `compositional_control` | Compositional evolution, three seeds | **180** | Prepared |
| `feedback_control` | Neutral fixed-NLL-example generation, three seeds | **180** | Prepared |
| **All phases** | **19 runs** | **1,020** | No writer requests made |

The two evolution phases together form a four-policy, three-seed comparison. Neutral evolution versus neutral fixed examples tests the value of evolutionary examples under the same prompt policy. Independent generation is a different information-access control and is not silently substituted for it.

Across the suite, keep temperature 0.7, top-p 0.8, top-k 20, 4,096 output tokens, the initial NLL, both islands, parent selection, preprocessing, and final validation rule fixed. All arms use a common failure watchdog that permits the full candidate budget; this is a prospective reference, not an exact reproduction of the historical failure-based early stop. Repairs and additional sketch calls are absent in this first study.

New artifacts include each actual candidate prompt and request seed, saved training predictions, the ordinary validation winner, and a top-three validation shortlist. A fixed **512-row training probe** supports behavioural diagnostics. Validation predictions do not enter generation, novelty feedback, or parent fitness.

Report all seed results and their mean/spread. Do not report the best test seed as the primary result. Freeze candidate identities before nine-dataset scoring, preserve both directed and paper-compatible AUROC, and identify repeatedly inspected datasets as exploratory evidence.

## 7. CPU experiment completed during this preparation

The fixed **14-feature documented-input** pool was fitted with a claim-grouped outer split and three-fold claim-grouped inner tuning. Each head considers five hyperparameter settings; outer validation and test labels do not fit or tune the head.

| Method | PopQA grouped validation AUROC |
| --- | ---: |
| Sequence NLL, same partition | **73.7052** |
| Logistic-14 | **82.1672** |
| Regularized QDA-14 | **80.3277** |

The logistic choice was `C=100`; QDA used `reg_param=0.6`. These are validation results on a different partition from the historical writer runs, not a nine-dataset macro result or evidence of a superiority margin over those runs. They support including strong fitted controls in a future search comparison. Artifacts: [grouped_supervised14.json](results/next_experiments/grouped_supervised14.json), fitted state in `out/research20261021/supervised_grouped/models.joblib`.

## 8. Other extensions: feasible, but distinguish their setup state

| Experiment | Concrete design | Availability |
| --- | --- | --- |
| Reliability | Fixed class wrapper, mutable scoring body, shape/finite smoke checks, one budgeted repair | Design only; requires implementation before claiming a runnable repair arm |
| Archive and parent sampling | Training-probe behaviour archive; rank sampling with a uniform reserve; isolate quantile-gate changes | Design only; current sampler is intentionally preserved |
| Two-stage generator | Several distinct sketches followed by implementations; count every call/token | Design only; no completed result |
| Continuation versus restarts | One 90-round search versus three independent 30-round searches at 180 requests, repeated over seeds | Design only; decouple the percentile schedule from total rounds first |
| Supervised evolution | Evolve feature structure, fit inside training folds, serialize fit-once state, compare with fixed-feature heads | Existing supervised baselines are runnable; generated supervised-program support is not implemented |
| Partial feature-model transfer | Frozen source-selected formulas on six completed Llama-3.2-1B caches; source/target compared on the same six datasets | CPU scoring is possible with more scratch/RAM; three target datasets still need inference for a nine-dataset result |
| Complexity/ablation | Disable one actual feature group at a time; compare frozen scores, optionally refit coefficients in training folds | Existing evaluator and four ablations available; exact historical Claude code still absent |

The source pipeline is EvoTune-based, as described by [Surina et al.](https://arxiv.org/abs/2504.05108) and the EvoHallu method section. That parent work also studies RL updates to the generator; our searches keep writer weights fixed. A directory named `funsearch` does not make this a full reproduction of every FunSearch policy.

Useful outside mechanisms are explicit hypotheses for our next implementation: [ShinkaEvolve](https://arxiv.org/html/2509.19349v1#S3) offers ranked parent sampling and novelty assessment; we can adapt these using inexpensive training-probe prediction behaviour. [AlphaEvolve](https://arxiv.org/html/2506.13131v1#S2.S4) motivates small-scale correctness screening before expensive evaluation. [Evolution of Heuristics](https://proceedings.mlr.press/v235/liu24bs.html) motivates evolving an idea description alongside executable code. None establishes an AUROC gain on this benchmark without our own comparison.

## 9. Runbook and resource limits

From the repository root:

```bash
# Reconcile paper reporting and verify the CPU/data baseline.
.venv/bin/python scripts/reconcile_paper_reporting.py
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_next_experiment_readiness.py

# Prepare configs, fixed contexts, training probe, and inspection shortlists.
.venv/bin/python scripts/prepare_next_experiments.py

# List the first phase; this makes no endpoint requests.
.venv/bin/python scripts/run_prepared_experiments.py --phase prompt_screen
```

The completed CPU fitted-control experiment can be reproduced with:

```bash
PYTHONPATH=src:.:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/fit_supervised_baselines.py \
  --feature-set documented --split-strategy claim_group \
  --output out/research20261021/supervised_grouped
```

On a GPU host with the pinned writer already served at `127.0.0.1:8000`, explicit execution is:

```bash
.venv/bin/python scripts/run_prepared_experiments.py \
  --phase prompt_screen --execute --max-hours 2

# Or execute the predeclared full suite with one total client-run deadline:
.venv/bin/python scripts/run_prepared_experiments.py \
  --phase all --execute --max-hours 8

.venv/bin/python scripts/summarize_next_experiments.py
```

The executor checks audited training-cache/CSV hashes, study source/configuration hashes, and the served model ID. Completed full-budget jobs are skipped; interrupted jobs require inspection before explicit resumption. A time cap can prevent completion of the full suite, so partial budgets are recorded and must not be presented as matched runs. The deadline applies to one invocation; separate invocations need a shared resource schedule. The controller does not provision or stop provider pods.

The historical 60-request Qwen9B runs averaged about **20 minutes**, making 1,020 requests approximately **5.7 hours at that old end-to-end rate**. Prepared configs conservatively use one evaluation worker rather than the previous parallel pair; actual wall time can be longer. Allow setup, a measured pilot, and evaluation overhead before committing the remaining GPU window. Within eight hours, prioritize the prompt screen and completed matched evolution seeds over the optional feedback phase. These are estimates, not GPU-hours already spent or guaranteed completion times.

The local machine has CPU-only PyTorch, no CUDA/vLLM runtime, approximately 2 GiB RAM and 3 GiB free disk. CPU audits and the grouped fitted-control experiment completed here. It cannot serve the writer or restore the largest approximately 7 GiB cache. A GPU host also needs adequate checkpoint and scratch storage. Historical GPU package versions and the actual successful Qwen9B server arguments are preserved in [software_runtime.json](results/software_runtime.json) and `configs/campaign/phase2.json`; validate the new host before generation.

After freezing new programs, the existing cached evaluator can score their nine-dataset predictions in CPU `--mode evaluate`, with restored source caches and a new output directory. Pass the research-run frozen-program glob and omit `--primary-only` when reporting all seeds. Restore one cache at a time where storage is constrained. No new source-model feature extraction is required for these prompt studies.

The [study status](results/next_experiments/STUDY_STATUS.md) distinguishes prepared jobs from results. Current completed work consists of the reporting reconciliation, data/environment audits, the grouped 14-feature fitted control, and protocol validation; the prepared LLM arms have no new quality results yet.
