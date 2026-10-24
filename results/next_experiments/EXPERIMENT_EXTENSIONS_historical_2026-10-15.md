> Historical campaign document. Its interpretation of a UHEAD label error was superseded by the author clarification supplied on 21 October 2026: reporting uses per-dataset max(AUC, 100-AUC). Keep is_hallu unchanged. See NEXT_EXPERIMENTS.md for current decisions.

# Experiment extension ideas

Extensions to [the original paper](https://arxiv.org/abs/2604.03473), prioritizing small changes to the existing EvoHallu pipeline and potentially informative results. The first four tasks reflect the main proposed directions; the remainder are complementary experiments.

**Answers added after the completed campaign, 15 October 2026.** Each idea below
now has an answer tied to actual artifacts and the corresponding paper section.
The numerical paper comparison is in [Comparison with the paper](#comparison-with-the-paper).

**Two label conventions must be distinguished.** “Supplied” uses the provided
CSVs exactly as received, as in the original campaign report. “UHEAD verified”
uses the retained UHEAD annotation field following the upstream false=1
convention; its `is_hallu` export is inverted on all 2,057 rows. This newly
identified conversion issue affects model rankings. Both conventions are
reported below. The source-consistent analysis is a separately recorded,
posthoc audit, pending confirmation of the original CSV conversion history.
No model, formula, training split or validation-selected seed was changed.

| Idea | What was actually tested | Answer status |
| --- | --- | --- |
| 1. New writer models | Four writers; seven primary evolution runs; all nine test datasets | Bounded model comparison completed; controlled superiority over the paper unresolved |
| 2. Deliberate diversity | Candidate validity and prediction diversity measured | Proposed two-stage/structured intervention not run |
| 3. Claude ranking | Local-writer complexity and similarity diagnostics | Original Claude explanation unresolved; histories unavailable |
| 4. Supervised evolution | Training-only logistic and QDA controls; template refits | Baseline component completed; supervised program evolution not run |
| 5. Need for evolution | Independent and fixed-example controls; coefficient searches | Direct, model-dependent evidence; limited seeds and unequal realized budgets |
| 6. Feature-model transfer | Six target-model feature caches preserved | Incomplete; no quality result |
| 7. Complexity and useful features | Four formula ablations and validation permutation controls | Partial answer for our formulas; requested historical Claude ablations unavailable |

**Common protocol:** keep the Llama-3.1-8B feature model, cached inputs, data splits and baseline set fixed unless explicitly testing them. Select final estimators on validation data and freeze them before held-out evaluation. Use multiple evolution seeds where affordable. Report both improvement over baseline in ROC-AUC percentage points and search cost (generated/evaluated candidates, generation tokens, GPU-hours or API cost); plot best validation AUC against cumulative cost rather than relying only on a gain/cost ratio.

## 1. Benchmark newer open-weight code-generating models

Test whether newer open-weight models can design better uncertainty estimators than GPT-OSS-120B at similar or smaller scale. Shortlist 2–3 currently available coding/reasoning models after checking release dates, licensing and local serving compatibility; include a substantially smaller model to test cost efficiency. For mixture-of-experts models, report both total and active parameters.

Start with short, equal-round-budget runs using the existing prompts, then extend the most promising models. Change the **estimator-writing model**, not the model producing uncertainty inputs. Most changes should be confined to model configuration and serving/model-name handling.

**Potential result:** a newer or smaller open model outperforms GPT-OSS-120B, or achieves comparable gains at much lower cost.

**Answer — the four-writer experiment completed; beating the original writer
under a matched protocol has not been established.** We retained the Llama-3.1-8B
feature model and used the bounded trial runner reached through
`src/experiments/main.py`. It retains training fitness, population examples,
program insertion and validation selection. It is a modified, shorter
implementation of the evolutionary structure, rather than a rerun of every
paper setting.

| Writer | Evolution seeds | Selected seed | Selected-run requests / valid candidates | Supplied macro AUC (%) | UHEAD verified macro AUC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 1047, 1048 | 1047 | 60 / 57 | 69.80 | 72.00 |
| Laguna XS 2.1 | 1047, 1048 | 1048 | 60 / 43 | 69.23 | 71.27 |
| Qwen3.5-9B | 1047, 1048 | 1048 | 60 / 36 | 69.54 | 72.45 |
| Qwen3.5-4B | 1048 | 1048 | 42 / 16 | 68.37 | 69.83 |

The selected seed is always the source-validation winner, under both label
conventions. Requested main budgets are 30 pairs; 4B stops early after three
consecutive failed rounds. Its lower functional validity is an experimental
outcome, not a reason to remove it. Devstral loaded, but all six initial requests
failed the native-tokenizer request protocol; the corrected retry was cancelled.
It supplies no fifth writer-quality result.

On the source-consistent convention, the selected 9B method gains 3.74 points
over NLL, with paired 95% interval 3.42–4.08. The corresponding 27B gain is
3.29 points, interval 2.84–3.74. Two seeds per main writer are insufficient for
a stable model-capability ranking. Selected-run wall times are 18.81 minutes
for 9B and 74.28 for 27B; serving profiles and CPU scoring overhead differ,
and these times exclude shared feature extraction. They are not comparable
GPU-efficiency measurements against the paper.

**Paper match:** §§3/5.1 and Figure 2. The published scores and the exact
limitations on comparison are tabulated below. Evidence:
[`run_table.csv`](results/run_table.csv),
[`paper_alignment.json`](results/paper_alignment.json), and the immutable
`out/campaign/evaluation_with_4b/` prediction cohort in the results archive.

## 2. Improve candidate diversity and search efficiency

Test a two-stage generator: one LLM call proposes `num_outputs_per_prompt` explicitly different method sketches, followed by one implementation call per sketch. Require different signal families or aggregation principles—not merely different coefficients.

Compare this against direct generation and a cheaper intervention: randomized, structured prompt variants. Note: the current `system_prompt_fabric()` already randomizes a baseline with an auxiliary idea, but this approach did not provide significant improvements. Apparently, this was insufficient for improving diversity.

**Potential result:** deliberate diversity improves AUC per unit compute, or reveals that additional syntactic diversity does not translate into useful prediction diversity.

**Answer — the proposed diversity intervention was not tested.** Neither the
sketch-then-implementation generator nor a completed structured-family arm
is present in the results. The planned 4B structured arm was cancelled.
Independent sampling and fixed-example generation test feedback/example
access; they do not implement the proposed two-stage diversity mechanism.

We did measure prediction diversity among valid candidates on source validation.
For the selected evolution runs, mean pairwise prediction Spearman correlation
is 0.494 for Qwen27B, 0.343 for Laguna and 0.257 for Qwen9B. Fractions of pairs
with correlation at least 0.995 are 3.13%, 11.41% and 2.06%, respectively.
These statistics use 57, 43 and 36 nonconstant saved prediction vectors.
They describe the generated populations, without isolating a causal effect
of a prompt intervention or proving novelty over existing formulas.

**Paper match:** §3, §5.1 and §6.3, Table 2. Our recorded populations make a
future matched intervention measurable; the current evidence cannot answer
whether deliberate diversity improves AUC per unit compute. Evidence:
`runs[*].mean_prediction_spearman` and
`prediction_pairs_above_0995_fraction` in
[`campaign_summary.json`](results/campaign_summary.json).

## 3. Explain the Claude model ranking through candidate analysis

Analyze the reported **Sonnet 4.5 > Opus 4.5 > Opus 4.6** ranking once candidates from those runs become accessible. This task is currently **blocked on candidate access**; winning formulas alone would support only a limited analysis, not conclusions about the full search distribution.

 Manually inspect a small matched sample of successful and failed candidates, and track how each population changes over evolution. Check budgets, prompts and seeds before attributing differences to model capability. Compare method-family diversity, prediction similarity, code complexity, invalid-program rates, runtime and train–validation gaps.

**Potential result:** stronger general-purpose models may converge on similar or unnecessarily elaborate estimators, while a simpler, more diverse search generalizes better. Treat this as a hypothesis, not an assumed explanation.

**Answer — unresolved for Claude.** The original Claude populations and the
historical candidate implementations referenced by the evaluator are absent.
Commented model/round names do not recover those programs or their trajectories.
We therefore cannot explain the proposed Sonnet/Opus ordering from these runs.

The wording of the idea also needs qualification: Figure 4 concerns the
relationship between complexity and performance, rather than establishing a
universal ordering of mean detector quality. In addition, §6.4 names Spearman
correlation while the figure axis says Pearson. This reporting ambiguity should
be resolved before attempting a numerical replication. [Paper, §6.4 and Figures 4–5](https://aclanthology.org/2026.acl-srw.115.pdf#page=7)

For our selected evolution populations, AST-node count versus **validation**
AUC has Spearman correlations −0.166 for Qwen27B, +0.417 for Laguna and +0.159
for Qwen9B. These are descriptive local-writer diagnostics. Their metric,
complexity proxy and evaluation partition do not duplicate the paper's figure,
and they cannot establish why any Claude model behaves differently.
Qwen27B seed 1048 reproduces the validation ranking of supplied `LogProbSlope`
exactly; this is evidence that new program text need not mean a new detector.
Evidence: `complexity_validation_spearman` and
`nearest_supplied_formula_on_validation` in
[`campaign_summary.json`](results/campaign_summary.json).

## 4. Evolve supervised uncertainty estimators

Extend evolution to methods that fit labelled reference data. The code already has `task.supervised` support and training-statistics plumbing, but the active prompt describes unsupervised methods; audit the estimator interface and update the prompt and evaluation path before treating the flag as a complete supervised setup. Code changes might be required for supporting supervised methods correctly. For supervised methods, evolution run still must only store the candidate code. However, additional effort is required to ensure reproducibility between evolution run and evaluation. For evaluation, supervised methods still must only use PopQA train subset, and be trained once (evaluation code will also require some changes).

Give each candidate a fixed small tuning budget, such as five randomly sampled hyperparameter configurations. Choose hyperparameters by in-sample training AUC - it should be okay for small tuning budget. Keep the outer validation split for final candidate selection and test labels unavailable to fitting or tuning.

Compare against manually designed supervised baselines—e.g. logistic regression or Quadratic Discriminant Analysis — with comparable tuning budgets, as well as the unsupervised methods.

**Potential result:** evolution discovers a useful learned estimator, or establishes how much apparent progress disappears against properly tuned supervised baselines.

**Answer — supervised baselines completed; supervised evolutionary search did
not.** The active generated-program contract remains unsupervised at inference.
We fitted logistic regression and regularized QDA once using 5,605 PopQA training
rows. Each uses five hyperparameter settings with three-fold training-only CV;
the 1,402-row outer validation split is not used to fit or tune those heads.
This intentionally replaces the idea's in-sample tuning suggestion with inner
CV. No test labels enter fitting.

| Fitted control | Supplied macro AUC (%) | UHEAD verified macro AUC (%) |
| --- | ---: | ---: |
| Logistic, 14 documented-input features | 70.28 | 72.61 |
| QDA, 14 documented-input features | 69.95 | 72.35 |
| Logistic, 17 features including full-vocabulary entropy | 69.72 | 71.88 |
| QDA, 17 features including full-vocabulary entropy | 69.63 | 72.10 |

These controls show that a cheap fitted comparator is competitive with every
new writer's selected method. The 14-feature head uses likelihood and top-512
logit-derived signals accessible in the documented input contract. The full
17-feature pool adds three full-vocabulary entropy summaries. The 14-feature
result is therefore the more appropriate matched-input reference.

**Paper match:** §6.2 and Table 3 already contain a logistic-regression weight
comparison. Our heads use a different, manually fixed 14/17-feature basis;
they do not refit the exact Evo-Anth-1/15 features. Thus this completes the
baseline part of the extension, not a replication of Table 3 or a demonstration
of supervised program evolution. The per-dataset context table below makes
that distinction explicit.

An exploratory header-trimmed training refit scores 71.36 supplied / 73.61
UHEAD-verified macro AUC for logistic-14. Its design followed observation of a
held-out perturbation result, although fitting/tuning still use training only.
It is a separate posthoc preprocessing result and is not the primary writer
comparison. Evidence: [`supplementary_method_table.csv`](results/supplementary_method_table.csv),
`results/validation_controls/`, and `paper_alignment.json`.

## 5. Test whether evolution is necessary

Compare the existing evolutionary search with independent sampling from the same task prompt and a baseline-seeded, non-evolutionary generator. Match candidate or compute budgets and apply the same validation-selection procedure. Reuse cached features and evaluation code; only the prompt/example-selection loop needs to change.

**Potential result:** quantify the value of evolutionary feedback over simply asking a model for many methods, and identify when evolution starts paying off.

**Answer — evolutionary feedback helps some writers, but is not uniformly
necessary.** The following comparisons use seed 1048 for both arms and retain
the selected frozen programs. Intervals are paired 95% claim-bootstrap intervals
conditional on this fixed collection of nine datasets; they do not include
variation across search seeds or adjustment for multiple comparisons.

| Writer and control | Evolution minus control, supplied (points; 95% interval) | Evolution minus control, UHEAD verified (points; 95% interval) |
| --- | ---: | ---: |
| Laguna vs independent | +0.93 [0.67, 1.19] | +1.04 [0.81, 1.28] |
| Laguna vs fixed example | +0.70 [0.42, 0.99] | +0.91 [0.60, 1.19] |
| Qwen27B vs independent | −0.94 [−1.25, −0.64] | −1.07 [−1.37, −0.75] |
| Qwen9B vs fixed example | −0.26 [−0.69, 0.11] | +0.43 [0.04, 0.85] |

Each tabled arm makes 60 requests. Independent generation also removes example
implementations, so it changes the information supplied along with feedback.
The fixed-example control more closely isolates the value of replacing a
constant implementation example with evolved population examples. The 9B
independent arm is excluded from this matched-budget table, but retained in
the full results: it stops at 12 requests, with one valid candidate and the
baseline retained. Its apparent gap primarily reflects functional failures.

The 9B feedback conclusion changes with the UHEAD convention. This is a concrete
reason to resolve annotation provenance before publishing a significance claim.
Laguna's advantage and Qwen27B's disadvantage persist under both conventions;
one search seed per control still limits generalization.

As a cheap complementary experiment, 50 fresh seeds per restricted coefficient
search mode yield a +0.52-point mean validation advantage for evolution
(paired-seed interval 0.21–0.84). The two winners selected across 3,000 candidates
per mode score 69.81 versus 70.15 supplied macro AUC, or 72.23 versus 72.42
UHEAD-verified, for evolution and independent sampling respectively. Better
validation search did not guarantee the same held-out ordering. This restricted
linear search is not a substitute for the LLM comparison.

**Paper match:** §3/§5.1 describe the feedback loop; our no-feedback controls
address its added value directly. Evidence: `feedback_comparisons` in
[`paper_alignment.json`](results/paper_alignment.json), plus the saved requests
and `validation_linear_seed_sweep/` / `linear_best50/` cohorts.

## 6. Test transfer across feature models

Freeze a small set of selected formulas and evaluate them using uncertainty inputs from one additional, affordable feature model. Recompute inputs with consistent preprocessing and retain comparable manually designed baselines. Do not retune the formulas on the target model's test results.

**Potential result:** distinguish broadly useful uncertainty principles from formulas specialized to Llama-3.1-8B. This complements the existing cross-dataset evaluation without requiring another full evolution campaign.

**Answer — incomplete; no transfer-quality conclusion.** The target checkpoint
was pinned `unsloth/Llama-3.2-1B-Instruct`, with frozen source-model formulas and
fitted weights. Feature extraction completed AVeriTeC, UHEAD, WildHallucinations,
Cities, Companies and Common Claims: six datasets and 56,235 claims. CounterFact
was interrupted and PopQA/XFact were not processed when the user requested
closure. All six completed caches are preserved locally and integrity checked.

The shared source tokenizer has the same token-ID vocabulary, but is not a
demonstration of identical native tokenization. No nine-dataset target scores
or target-fitted weights are reported. Testing six selected feature caches is
not equivalent to completing the planned nine-dataset transfer benchmark.

**Paper match:** the single-feature-model limitation, p. 9. The paper's Table 1
concerns cross-task selective prediction; it is not a feature-model transfer
result and uses PRR, not the AUROC reported here. Evidence:
[`campaign_closeout.json`](results/campaign_closeout.json),
[`transfer_model_provenance.json`](results/transfer_model_provenance.json), and
[`feature_cache_validation.json`](results/feature_cache_validation.json).

## 7. Test whether complexity actually helps

For several strong evolved formulas, analyze feature importance by dropping them or using Shapley values (either only shuffling feature values and keeping a single pre-fitted model, or re-fitting the model if it is cheap). Measure AUC vs kept feature count. Formulas for analysis should likely be taken from the original paper, Table 3 (Evo-Anth-1, Evo-Anth-15, or Evo-Gpt-1, Evo-Gpt-3).

**Potential result:** obtain a compact, interpretable estimator with similar performance, or isolate a genuinely useful new ingredient. This also provides concrete evidence for the Claude complexity/diversity analysis.

**Answer — some new winners simplify substantially; the requested historical
Claude/GPT formula ablations were not completed.** Four untuned ablations were
frozen using training/validation evidence before held-out scoring. They alter
our Qwen/Laguna winners, not the unavailable Evo-Anth implementations.

| Frozen formula | Supplied macro AUC (%) | UHEAD verified macro AUC (%) |
| --- | ---: | ---: |
| Qwen27B seed-1047 parent | 69.7969 | 72.0020 |
| Its float64 linear position-weighted NLL simplification | 69.7926 | 72.0003 |
| Laguna seed-1048 parent | 69.2343 | 71.2668 |
| Slope plus entropy range, top-1 variance term removed | 69.2192 | 71.2777 |
| Slope only | 68.5347 | 70.3605 |

The Qwen score is approximately `length * weighted_mean(NLL, weights=1..length)`;
the original expression adds a small half-precision rounding residual. Its
simplification changes macro AUC by −0.0043 supplied / −0.0017 source-consistent
points, with both paired intervals spanning zero. Removing Laguna's variance
term changes performance by −0.0151 / +0.0109 points, again with intervals
spanning zero. Removing entropy range as well loses 0.70 / 0.91 points. Thus the
extra variance term is dispensable in these measurements, while entropy range
contributes in this particular frozen formula. These are observed ablation
effects, not a formal equivalence test or a claim that complexity never helps.

Validation-only permutation diagnostics for four fitted heads identify
likelihood signals as important. They are marginal permutations with correlated
features, not Shapley values or causal importance. Refit 14-versus-17-feature
controls also change fitted weights and selected hyperparameters, so their
score gap cannot be attributed solely to removing three features.

**Paper match:** §6.2/RQ2 for positional signals, §6.4/RQ4 for complexity,
§6.5/RQ5 for prediction similarity, and Table 3 for fitted comparators.
This is support for simple positional likelihood scores within our runs; it
does not overturn the paper's observations about Claude populations. Evidence:
`formula_ablations` in [`paper_alignment.json`](results/paper_alignment.json),
[`formula_ablation_preparation.json`](results/validation_controls/formula_ablation_preparation.json),
and [`fitted_feature_sensitivity.json`](results/validation_controls/fitted_feature_sensitivity.json).

## Comparison with the paper

This section uses the [published ACL PDF](https://aclanthology.org/2026.acl-srw.115.pdf),
with PDF page numbers in links. All local values come from frozen predictions;
this comparison required no GPU restart or additional model generations.

### UHEAD annotation provenance and the baseline discrepancy

The supplied UHEAD CSV has `verified == is_correct == 1 - is_hallu` on all
2,057 rows. The original evaluator followed `is_hallu` exactly, as documented.
The [pinned upstream fact checker](https://github.com/IINemo/lm-polygraph/blob/4d635e81128411d792acd584f78b0f849a21cd0b/src/lm_polygraph/generation_metrics/openai_fact_check.py#L65)
returns **0 for true and 1 for false**. The
[pinned UHEAD annotation exporter](https://github.com/IINemo/llm-uncertainty-head/blob/fc213edc26526b446e76e8253f7fe3b964a54be7/luh/cli/generate_dataset/run_extract_verify_claims.py#L211)
saves that annotation as `verified` and uses label 1 for hallucinated tokens.
Consequently the supplied CSV appears to have complemented an already
hallucination-oriented annotation while constructing `is_hallu`.

This is supported by upstream semantics and an exact 2,057-row cross-check,
not merely by the fact that scores improve after changing a label direction.
The exact historical CSV conversion script is unavailable, so the secondary
analysis is labelled **source-consistent UHEAD convention**, not a certified
reproduction of the authors' original evaluation.

There are 1,515 positives under supplied `is_hallu` and 542 under `verified`.
UHEAD NLL AUC changes from 36.1972 to 63.8028. For a fixed score vector,
complementing binary labels gives `AUC_new = 100 - AUC_old`, including ties.
With nine equally weighted datasets, the new macro is therefore
`old_macro + (100 - 2 * old_UHEAD_AUC) / 9`.
Saved UHEAD bootstrap draws are complemented in the same way; draws for the
other eight datasets remain unchanged. The script directly checks both label
orientations against every one of the 58 saved UHEAD score vectors, and checks
the identity with nonuniform sample weights as well.

**Original CSVs, cached statistics, predictions, model weights, primary tables
and validation-selected seeds remain unchanged.** Checksums and full secondary
results are in [`paper_alignment.json`](results/paper_alignment.json).

### Matched baseline results: Figure 2 and Table 2

| Same named/ranking-equivalent baseline | Paper Figure 2 (%) | Supplied labels (%) | UHEAD verified (%) |
| --- | ---: | ---: | ---: |
| Sequence Probability / sequence NLL | 68.7 | 65.6439 | 68.7112 |
| Perplexity | 64.2 | 64.1747 | 64.1676 |
| Mean Token Entropy | 59.9 | 58.4200 | 59.8007 |

Published values are from [Figure 2, PDF p. 4](https://aclanthology.org/2026.acl-srw.115.pdf#page=4).
The UHEAD convention almost entirely explains the NLL macro discrepancy.
Perplexity is already close under both conventions; entropy retains a 0.10-point
residual after the change. Thus the audit narrows the discrepancy without
claiming all preprocessing and numerical details have been reproduced.

For the single-dataset anchor, the paper's Table 2 gives 74.1 for PopQA NLL;
ours is 74.05. This agreement does not depend on the UHEAD convention.
Table 2's logits-only evolution result is 78.3; our selected Qwen27B method
scores 78.97 on PopQA. These are different searches, so the latter is contextual
rather than a paired model comparison. [Table 2, PDF p. 6](https://aclanthology.org/2026.acl-srw.115.pdf#page=6)

### Which evolved methods can actually be identified?

Figure 2 reports Evo-Gpt-1 at **70.8** and Evo-Anth-1 at **73.4** macro AUROC.
Those are the published headline anchors, rather than results newly reproduced
by this campaign. [Figure 2](https://aclanthology.org/2026.acl-srw.115.pdf#page=4)

Our best descriptive score among the 25 supplied classes is
`ExpWeightedNegLogProbWithTop1`: 69.88 supplied / 70.78 UHEAD-verified. Its
source-consistent number is close to 70.8, but neither numeric agreement nor a
similar position-weighting mechanism establishes that it is the exact
published Evo-Gpt-1. We have no certified mapping from every paper alias to
the supplied class names. Claude candidate classes referenced by model/round
names are unavailable in the checkout's installed dependencies.

Accordingly, [`paper_figure2_comparison.csv`](results/paper_figure2_comparison.csv)
lists all 18 published bars and marks historical identities or missing
evaluations explicitly. RAUQ/AttentionScore were not evaluated in this campaign
because their required attention statistics were not cached. “25 supplied
formulas” must not be read as “all published GPT/Claude winners reproduced.”

The new source-consistent Qwen9B and Qwen27B results, 72.45 and 72.00, lie
numerically between the two published headline anchors. This is promising
context, not a controlled claim to beat GPT-OSS-120B or to match Claude: the
original writer was not rerun with our budget, population and selection rule.

### Per-dataset context for supervised controls: Table 3

| Dataset | Paper Evo-Anth-1 (%) | Paper LogReg-1 (%) | Our Qwen27B (%) | Our logistic-14 (%) |
| --- | ---: | ---: | ---: | ---: |
| PopQA test | 83 | 84 | 78.97 | 81.67 |
| AVeriTeC | 68 | 65 | 65.60 | 65.75 |
| X-Fact | 53 | 57 | 54.52 | 47.75 |
| UHEAD | 68 | 69 | 59.92 | 60.48 |
| Cities | 99 | 98 | 96.11 | 98.99 |
| Companies | 89 | 87 | 86.48 | 87.47 |
| Common Claims | 69 | 67 | 68.99 | 69.47 |
| CounterFact | 78 | 77 | 72.11 | 78.40 |
| WildHallucinations | 65 | 65 | 65.31 | 63.49 |

Paper cells are rounded values transcribed from
[Table 3, PDF p. 7](https://aclanthology.org/2026.acl-srw.115.pdf#page=7).
Local columns use the source-consistent UHEAD convention; the other eight
datasets are unchanged. Paper LogReg-1 uses the Evo-Anth-1 feature basis;
our logistic-14 uses a different fixed basis. The table therefore shows where
performance differs without treating the heads as the same estimator.
The exact local values and both label conventions are in
[`paper_table3_context.csv`](results/paper_table3_context.csv).

A further consistency check matters: averaging the nine displayed Table 3
Evo-1 test cells gives **74.67%**, whereas Figure 2 gives **73.4%** for
Evo-Anth-1. This exceeds the effect of rounding those cells alone. The PDF does
not explicitly reconcile the figures; this report preserves both as published
instead of constructing a new paper macro from Table 3.

### Remaining limits on comparability

| Dimension | Paper reference | This campaign |
| --- | --- | --- |
| Search budget | §3: typically 500 rounds, two candidates per round | 30 requested pairs; early stops and wall caps recorded |
| Writer coverage | §5.1: 20 GPT-OSS runs and 10 Claude runs | Seven primary evolution runs across four new writers |
| Candidate selection | §5.1/Figure 2: validation shortlist; displayed Claude examples ordered by test rank | One seed per writer selected only by source validation, with all completed seeds retained |
| Fitted controls | §6.2/Table 3: Evo-Anth feature sets | Fixed 14/17-feature pools; distinct training-only tuning |
| Complexity analysis | §6.4/Figures 4–5 | AST nodes versus validation AUC; different proxy/partition |
| Selective prediction | Table 1: PRR | Not evaluated; AUROC cannot be substituted for PRR |
| Feature-model transfer | Limitations, p. 9 | Six target caches only; no completed quality evaluation |

Even after resolving label polarity, checkpoint/tokenizer details, historical
feature caches, exact paper candidate identities and selection differences
limit a head-to-head claim. The established results are the within-campaign
comparisons with fixed cached inputs and frozen methods. The original supplied
convention is an audit trail; source-consistent results are separately
available for confirmation and further reproduction work.

To reproduce this CPU-only comparison, run
`python scripts/compare_paper_results.py` from the repository root with the
provided input CSVs and the preserved compact evaluation artifacts available.
It verifies primary artifact hashes and writes the three comparison CSVs plus
`paper_alignment.json`; it does not rewrite a primary benchmark.

---

## Execution assessment — 15 October 2026

**Archived campaign record: supplied CSV label convention.** The numerical
results and interpretations in this section retain the original evaluation.
The answers and paper comparison above add the subsequent UHEAD provenance
audit and its separate source-consistent scores. In particular, writer ranking
and the 9B feedback contrast depend on that convention; use the paired columns
above when discussing those claims.

The authorized resources are one A100 80 GB, at most 12 hours, with at least an
eight-hour campaign requested. There is no hosted model API budget. The first
writer model is **Qwen3.8-27B**; Llama-3.1-8B remains the feature model.
The paper's search uses cached model states and training fitness, followed by
validation selection and held-out evaluation. Its results motivate testing both
new writer models and the value of evolutionary feedback. [ACL paper](https://aclanthology.org/2026.acl-srw.115.pdf)

### Completed benchmark — first four writer models

**Campaign closed on request at 12:26 Moscow time (09:26 UTC) on 15 October.** The pod management
API confirmed `EXITED` and a null GPU runtime after the completed artifacts
were copied locally. Provider uptime immediately before shutdown was 36,654
seconds (10.18 hours), approximately **$18.23 in GPU charges** at $1.79/hour;
this estimate excludes storage charges. Hosted model API spending was $0.
The final code and compact results are packaged as `mod1_results.zip`.

The expanded cohort finished all **97,488 claims across nine datasets**, with
58 methods and no scoring failures. These are results under this checkout's
cached-input protocol, not a reproduction of the paper's reported table.
Writer seeds and the supplied comparator below were chosen using source
validation only. Completed writer models are **Qwen3.8-27B, Laguna XS 2.1,
Qwen3.5-9B and Qwen3.5-4B**. The same evolutionary pipeline uses training fitness,
population examples and validation selection, with a bounded 30-pair budget
and the fixed Llama-3.1-8B feature model. This is a shorter search than the
paper's 500-round configuration.

| Frozen method | Nine-dataset macro AUC (%) | Gain over sequence NLL (points) |
| --- | ---: | ---: |
| Sequence NLL | 65.64 | — |
| Claim-only sequence NLL | 65.69 | +0.04 |
| Supplied formula selected by validation | 68.62 | +2.98 |
| Qwen3.8-27B evolution, validation-selected seed 1047 | 69.80 | +4.15 |
| Laguna XS 2.1 evolution, validation-selected seed 1048 | 69.23 | +3.59 |
| Qwen3.5-9B evolution, validation-selected seed 1048 | 69.54 | +3.90 |
| Qwen3.5-9B fixed-example generation, seed 1048 | 69.80 | +4.16 |
| Qwen3.5-4B evolution, seed 1048 | 68.37 | +2.73 |
| Logistic regression, 14 documented-input features | 70.28 | +4.63 |
| Logistic regression, 17 features including full entropy | 69.72 | +4.08 |

The Qwen selected winner's paired gain interval over NLL is **+3.73 to +4.58
points**; the 14-feature fitted control's interval is **+4.03 to +5.22**.
These intervals concern the fixed dataset collection, not variability across
evolution seeds. The two Qwen evolution seeds score 68.53 and 69.80 macro AUC;
the two Laguna seeds score 69.43 and 69.23. This run variation cautions against
interpreting the selected winners as a definitive ranking of writer models.
All 25 supplied formulas are retained in the full tables; the table's supplied
reference is selected by validation, rather than by its test score.

The Qwen3.8 winner exceeds that supplied comparator by **1.18 points**
(paired 95% interval 0.64–1.77). The 14-feature logistic comparator exceeds
Qwen3.8 by 0.48 points (0.01–0.94). These exploratory intervals do not adjust
for multiple comparisons or include search-seed variability.

For 9B, the two evolution seeds score 68.02 and 69.54 macro AUC. The fixed-example
control scores 69.80; evolution minus fixed-example generation is −0.26 points
(interval −0.69 to +0.11). Independent generation stops after 12 requests with
one valid method and retains the baseline. A fixed implementation example
therefore matters for this writer's functional validity; these runs give no
clear evidence that evolutionary feedback improves over a fixed example.
The fixed-example control independently discovers a linear position-weighted
NLL score, closely resembling the simplified 27B winner.

Matched seed-1048 results make the feedback comparison explicit:

| Writer | Evolution | Independent | Fixed example |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 68.53 | 69.47 | Not run |
| Laguna XS 2.1 | 69.23 | 68.31 | 68.53 |
| Qwen3.5-9B | 69.54 | 65.64 (12 requests, baseline retained) | 69.80 |

Laguna evolution exceeds its fixed-example control by 0.70 points (paired
interval 0.42–0.99), whereas Qwen3.8 evolution seed 1048 trails independent
sampling by 0.94 points (−1.25 to −0.64). Together these runs show model-dependent
feedback effects. From-scratch sampling also changes the example information;
the fixed-example comparison more closely isolates population feedback.

For context, the highest descriptive test macro among the 25 supplied formulas
is 69.88 (`ExpWeightedNegLogProbWithTop1`). Its test ranking is reported only as
a descriptive envelope; the main 68.62 comparator was chosen by validation.
The current writer results do not establish improvement over the entire
supplied set or over a freshly rerun GPT-OSS-120B search.

The predeclared removal of four assistant-header tokens also finished all nine
datasets. Frozen Qwen3.8 and Laguna evolution winners change by +0.24 and +0.54
points, while the 9B evolution winner changes by −1.74 and the 14-feature fitted
head by −2.30. The latter models are not refitted after changing their inputs.
These are input-template sensitivity results; they are not a new selection
protocol or a correction of the paper's reported results. The full-template
likelihood baseline changes by only +0.003 points. Paired intervals and complete
measurements are included in `results/campaign_summary.json`.

An exploratory recovery control, added after observing that perturbation,
refits the same four classifiers using header-trimmed **training** features.
The split, five-setting grids, three-fold inner CV and training labels stay
fixed. Outer validation and held-out labels never enter fitting or tuning.

| Classifier | Original inputs | Trimmed inputs, original weights | Trimmed inputs, refitted weights |
| --- | ---: | ---: | ---: |
| Logistic, 14 features | 70.28 | 67.98 | 71.36 |
| Logistic, 17 features | 69.72 | 67.93 | 70.89 |
| QDA, 14 features | 69.95 | 66.75 | 69.02 |
| QDA, 17 features | 69.63 | 67.08 | 69.30 |

All entries are nine-dataset macro AUC percentages. The 14-feature logistic
refit recovers 3.39 points over its frozen perturbation (paired interval
2.44–4.34). This separates frozen input sensitivity from adaptation to the
changed representation; it is an exploratory follow-up rather than the
primary writer ranking. Removing four tokens retains the original prefix and
end-of-turn context, so this condition is called **header-trimmed**, rather
than claim-only statistics. Its separate expanded cohort verifies all 153
previous prediction/label/mask/group arrays unchanged. The original models and
source-protocol results remain available.

The untuned Laguna slope-plus-entropy-range ablation differs from its full
parent by −0.015 points (interval −0.048 to +0.016). Slope alone loses 0.70 points.
The Qwen float64 position-weighted simplification differs from its parent by
−0.004 points (−0.012 to +0.002). These results favor testing simple interpretable
scores alongside larger formula searches, without selecting ablations by test
performance.

A supplementary validation-only sweep adds 50 fresh seeds per restricted
search mode (seeds 1049–1098), with 60 candidates per seed and at most three
nonzero coefficients from the fixed 17-feature pool. It exactly reproduces all
four original coefficient-control candidate records first. Evolution averages
79.90% validation AUC (seed SD 0.73), versus 79.38% (SD 1.02) for independent
sampling; the mean difference is +0.52 points, with a paired-seed bootstrap
interval of 0.21–0.84. All 100 searches, including feature extraction, take
28.49 CPU seconds on the pod. This concerns the restricted search space and
validation distribution, rather than a general conclusion about LLM feedback.

An exploratory larger-budget follow-up freezes the highest-validation winner
across those 3,000 candidates per mode. Its held-out macro AUC is 70.15 for
independent sampling and 69.81 for evolution, with no scoring failures. This
separate cohort preserves the original 60-candidate primary controls. The
follow-up was added after the initial benchmark; coefficients and mode winners
are selected exclusively using the saved training/validation records. Both
winners were also registered for the 1B feature-transfer test before target
quality was computed. These controls demonstrate why larger validation gains
need not preserve their ordering across held-out datasets.

The optional Llama-3.2-1B feature pass was stopped when the user requested an
immediate finish. Six completed datasets (56,235 claims) are preserved as
compressed raw-feature caches: AVeriTeC, UHEAD, WildHallucinations, Cities,
Companies and Common Claims. CounterFact was interrupted; PopQA test and XFact
were not processed. **No complete transfer benchmark or transfer quality claim
is reported.** The corrected Devstral generation retry, the 4B structured-prompt
arm and the target-addition scoring watcher were cancelled. Completed primary,
formula-ablation, header-perturbation/refit and coefficient-search results are
retained. The controller's final `KeyboardInterrupt` records this requested
stop, rather than a failure of the completed benchmark.

There were 676 writer requests, 463 valid candidates and 1,784,975 reported
input/output tokens, including pilots and failed protocol attempts. These
counts include unequal actual budgets caused by the declared early-stop rule.
The portable archive includes source, frozen formulas, candidate histories,
compact prediction/bootstrap arrays, fitted heads, provenance and tables.
Large raw feature archives remain locally under `data/feature_cache_archive/`
and are excluded from the portable ZIP. Allocation and closure evidence are
in `results/campaign_allocation.json` and `results/campaign_closeout.json`.

### Which models fit the allocation?

These are checkpoint weight sizes, verified against the publishers' HF file
metadata, **before** activation memory, serving buffers and KV cache. A fit is a
planning estimate until the local server passes a generation test. Full revision
IDs, checkpoint creation timestamps and licenses are recorded in
`results/model_inventory.json`; creation timestamps are not release dates.

| Writer model | Parameters | Weight GiB | License | One A100 80 GB |
| --- | --- | ---: | --- | --- |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense language model, plus vision components | 51.75 | Apache 2.0 | First trial; BF16, text only, 16K context |
| [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1) | 33B total / 3B active | 62.30 | OpenMDW 1.1 | Plausible; custom architecture must pass a serving check |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 80B total / 3B active | 148.41 | Apache 2.0 | BF16 does not fit; a separately reported quantized experiment would be needed |
| [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) | 24B dense language model, plus vision components | 24.02 per format, published FP8 | Apache 2.0 | A100 weight-only FP8/Marlin comparison; GPU check required |
| [Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) | approximately 30B dense, including perception encoder | 55.46 | Apache 2.0 | Plausible; confirm architecture support before allocating a long run |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 30B total / 3B active | 58.16 | MIT | Plausible BF16 MoE comparison |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B total / 3B active | 66.97 | Apache 2.0 | Tight; reduce context and concurrency and verify memory |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B dense language model, plus vision components | 17.98 | Apache 2.0 | Preferred smaller comparator |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B dense language model, plus vision components | 8.68 | Apache 2.0 | Fast fallback if the larger model's throughput is poor |

Confirmed announcement dates include Devstral Small 2 on 9 December 2025,
Qwen3-Coder-Next on 3 February 2026, Laguna XS 2.1 on 2 July 2026, and
Muse Glimmer on 10 August 2026. Other exact announcement dates are not inferred
from repository creation or update dates.
[Mistral announcement](https://mistral.ai/news/devstral-2-vibe-cli/),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-Coder-Next),
[Poolside announcement](https://poolside.ai/blog/introducing-laguna-xs-2-1),
[NVIDIA model card](https://build.nvidia.com/meta/muse-glimmer-30b/modelcard).

The actual pod has a 30 GiB root filesystem and 55 GiB shared-memory filesystem.
The Qwen checkpoint is staged in shared memory, without a duplicate download
cache. Checkpoint files are pinned and checked against published LFS hashes.
Weights must be rotated between larger models; the original allocation does not
have room for every checkpoint simultaneously.

Devstral's publisher stores equivalent native Mistral and Hugging Face layouts.
The initial 48.04 GiB repository sum counted both; one format is 24.02 GiB and
its configuration explicitly specifies FP8 weights. It is therefore a separate
precision condition, not a BF16 writer. The optional download selects a single
layout and checks its shard list against the published index. vLLM documents
weight-only FP8 with Marlin on Ampere; native configuration and tokenizer checks
have passed on this pod, while actual GPU serving remains conditional.
[Pinned model configuration](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/blob/55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128/config.json),
[vLLM FP8 documentation](https://docs.vllm.ai/en/latest/features/quantization/fp8/).

The pod runtime uses PyTorch 2.13.0+cu130, Transformers 5.17.0 and vLLM 0.31.0.
Its host driver required NVIDIA's CUDA forward-compatibility userspace package;
the campaign exports that library path for GPU processes. A real CUDA operation
and the long writer runs verify this setup. Package versions, the compatibility
package URL and its SHA256 are saved in `results/software_runtime.json` and the
full package inventory in the campaign artifacts. This is a runtime detail of
this pod, not an estimator modification.
[NVIDIA forward compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)

For Laguna, the pinned checkpoint's bundled custom configuration loader fails
under Transformers 5.17.0 with a per-layer rotary-parameter validation error.
The installed native `LagunaConfig` and vLLM configuration loader pass the CPU
preparation check, so the launch omits `--trust-remote-code`. Checkpoint files
and weights are unchanged. GPU serving passed and the first four candidate
methods evaluated successfully; the preparation record is saved with the logs.

### Experiments prioritized for this campaign

1. Run a five-round Qwen3.8 pilot, recording generation speed, tokens, validity,
   duplicate rate and evaluation time. Determine feasible budgets from this pilot.
2. Compare evolution with independent sampling and fixed-baseline sampling,
   using the same model, interface specification, seeds, decoding settings and
   requested candidate budgets. Train scores alone enter evolutionary prompts. Every valid
   unique candidate is evaluated on validation data; the final winner is frozen.
3. Compare the large writer with Qwen3.5-9B or the 4B fallback. Use matched
   budgets and multiple seeds before extending the strongest configurations.
4. Test structured signal-family prompts as a cheap diversity intervention.
   Defer the two-stage sketch/implementation generator until direct generation
   costs and redundancy are measured.
5. Run a claim-group split robustness experiment. Report original-split and
   duplicate-disjoint held-out results separately.
6. Fit logistic regression and regularized QDA on 17 fixed numerical features,
   with five hyperparameter settings each and inner training-only CV. Freeze the
   fitted models; outer validation and held-out labels never enter fitting.
7. Evaluate all frozen winners, supervised comparators and 25 supplied formulas
   on the same nine feature caches. Report AUC, gains in percentage points,
   paired bootstrap intervals, failure rates and cost trajectories.

The from-scratch independent arm receives the stochastic task prompt, including
its baseline and signal ideas, but no baseline implementation, population
examples or fitness feedback. Its explicit constructor/call contract contains
no scoring implementation. It therefore changes prior information as well as removing evolution. The
`baseline_seeded` arm is a closer prompt control: it uses the original creative
prompt factory and auxiliary ideas while retaining the fixed NLL example and
its training score instead of updating the population. These distinctions are
retained when interpreting any advantage from feedback. Each primary run
requests 30 pairs, with model-specific time caps to protect the held-out stage;
actual requests and generation times are reported rather than assuming every
run completed the same budget.

The initial Qwen from-scratch attempt exposed an incomplete interface prompt:
the required `__call__` method was not specified without the example classes.
It is retained as a protocol failure, excluded before test evaluation, and
superseded by `qwen38_independent_contract_1048`. The corrected prompt states
the constructor and callable signatures explicitly without providing a detector
implementation. No test data informed this correction. All request costs,
including this failed attempt, remain in the campaign accounting.

The 9B independent arm used that corrected interface but stopped after 12
attempts, with one valid program and the NLL baseline retained. Eight responses
omitted the required inheritance, causing `object.__init__` errors; three
imported `entropy` from the wrong SciPy module. Raw and parsed class declarations
agree, so these are generated-program failures. A fixed-baseline-example arm
is added for this writer if 135 minutes remain, matching the 30-pair and
30-minute search caps. This is motivated by training-side validity, before
held-out scoring of the 9B methods, and tests whether a fixed example suffices.
The independent arm remains in the results, with its actual request count.

The 4B evolution run stops after 21 pairs (42 requests, 16 valid candidates)
because of three consecutive failed rounds; its frozen winner scores 78.03%
validation AUC and 68.37% held-out macro AUC. This is a functional generation
outcome under the specified early-stop rule, not a protocol exclusion.

Devstral's published FP8 model loads successfully with native Mistral serving
and Marlin on the A100. Its first six requests are rejected before generating
any tokens: vLLM's native Mistral validator forbids `chat_template_kwargs`,
including the otherwise common `enable_thinking=False` parameter. That attempt
is retained as a request-protocol failure, with its baseline-only cohort
archived separately and excluded from writer ranking. The corrected client
omits that optional field for the native tokenizer; actual installed-server
request validation and end-to-end HTTP tests pass. The separately named retry
was cancelled on the user's request to finish immediately. Devstral therefore
has **no valid generated-candidate result** in the final writer comparison;
loading the checkpoint is not counted as completing an experiment.

The current runner is `src/experiments/main.py --config-name config_uq_trial`.
`scripts/campaign.py` executes a resumable plan with a hard wall-clock deadline.
Its deadline stops experiment processes. Budget guards had been scheduled for
10:50 and 10:55 UTC. The user requested an earlier finish, so processes were
stopped, the final artifacts backed up, and the external GraphQL client stopped
this specific pod at approximately 09:26 UTC. The returned state was `EXITED`
with `runtime: null`, confirming release. The first REST interface had returned
HTTP 403; the successful stop used GraphQL.
Runpod reports $1.79/hour for this allocation, or $21.48 for twelve GPU hours
before any storage charges. `scripts/evaluate_cached_claims.py` separates GPU microbatches from
fixed estimator scoring chunks.
Feature extraction requests 16 rows per GPU forward, reducing that microbatch
when padding would exceed 16,384 tokens. This retains long claims and their
order; estimator scoring still uses fixed 512-row chunks. CPU checks preserve
features and labels, and an A100 float32 toy-model check found identical token
likelihoods/entropies and at most 0.000031 difference in retained top-k logits.
The check and actual extraction settings are recorded with the artifacts.
Feature extraction uses 12 CPU threads after a local 4/8/12-thread probe reduced
representative post-processing from 0.177 to 0.109 seconds, with identical
checked outputs. Estimator-scoring workers still use one thread each.
CPU scoring workers cap virtual address space at 32 GiB and each method call
at 120 seconds per 512-row chunk. A method that fails is reported as unavailable
for that dataset; it is never replaced using test performance. These bounds do
not apply to GPU feature extraction.

### Findings already established before model search

The supplied files contain **7,007 PopQA training rows** and **97,488 evaluation
rows across nine datasets**. The default seed-441 split has 5,605 training rows
and 1,402 validation rows. Normalized claim text reveals:

- 88 shared claims between default training and validation partitions;
- 412 shared claims between the supplied PopQA train and test files;
- smaller overlaps with Cities and WildHallucinations;
- repeated identical claims with inconsistent labels in several files.

The original files are retained. The additional `claim_group` split keeps a
normalized claim entirely in one partition. Evaluation also reports the subset
excluding claims seen anywhere in the supplied PopQA training CSV. Full counts,
class balance and file hashes are in `results/dataset_audit.json`.

Synthetic invariance checks pass for finite outputs on all 25 supplied formulas.
Five formulas depend on the composition of the scoring batch; scoring chunks
must therefore remain fixed at 512 throughout a comparison. Thirteen method
pairs have absolute prediction-rank correlation at least 0.995 on the synthetic
probe. These are engineering diagnostics, **not evidence of hallucination
detection quality**. See `results/formula_diagnostics.json`.

### Directions deferred or limited

- Full Claude ranking analysis remains blocked: the archive contains selected
  formulas, not the populations from the Sonnet/Opus runs.
- Supervised evolutionary search is incomplete: the active prompt is
  unsupervised, and the supplied feature builder rejects supervised logits-only
  mode. Supervised baselines can run now; the flag alone is insufficient.
- The multi-dataset task's train/test split problem is documented in
  `configs/TRAIN_TEST_REMINDER.md`; that path is excluded from this campaign.
- Feature-model transfer was a lower-priority optional feature pass after the
  completed writer comparisons. It started with the required 60-minute reserve
  and was stopped on the user's early-finish request after six datasets. It used
  the pinned public
  [Unsloth Llama-3.2-1B checkpoint](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct),
  with the same reference tokenizer after verifying token-to-ID vocabulary
  equality. Best source-validation evolution seeds, supervised weights and
  fixed formulas remain frozen; no target-model fitting or selection occurs.
- Existing preprocessing and the original end-to-end evaluator use different
  likelihood crops for some baselines. The campaign keeps the cached-input
  convention fixed across methods and does not describe its numbers as an
  exact reproduction of paper tables without an alignment audit.

Search runs use zero bootstrap resamples; uncertainty intervals are computed
only after estimators are frozen. Test files are not loaded by training or
validation evaluation. Multiple exploratory comparisons are reported as such,
without selecting or retuning methods using held-out results.
The default final intervals use 500 paired resamples of normalized claim groups
within each dataset, keeping duplicate rows together. Point AUC still weights
the original rows. Row resampling is available with `--bootstrap-unit row`;
the recorded unit identifies the calculation used. Macro intervals condition
on this fixed collection of nine datasets.
Resampling is independent between datasets. Five held-out dataset pairs share
between one and five normalized claims each; these small cross-dataset overlaps
are not coupled in the macro resampling. Counts are recorded separately in
`results/held_out_claim_overlap.json`.

CPU evaluation was optimized during the campaign: after the first Qwen main
seed, each already generated pair is scored concurrently in independent spawned
processes, with fresh estimators and identical split-specific random seeds.
Population insertion retains request order. Numerical scoring is unchanged;
raw run wall times include different CPU startup overheads. The generation-time
curves exclude this evaluation overhead. Saved validation predictions also
identify methods that reproduce the ranking of a supplied formula, rather than
assuming different source text establishes a different detector.

Qwen3.8 and Laguna used eager serving. The smaller Qwen3.5-9B stage used
vLLM's standard compilation/graph setup to reduce decoding overhead.
Search controls within each writer share its serving profile, and profiles
are recorded with the runs. Cross-writer runtime comparisons describe these
specific configurations; they do not isolate an intrinsic model speed ranking.

The predeclared Qwen3.5-4B size comparison completed after the primary benchmark
using the same cached Llama inputs and serving profile as the 9B writer. Its
separately saved expanded cohort is the final primary report, after verification
that all previous predictions are identical. The header ablation was frozen
earlier and includes only the first three writers. The grouped fitted controls
and original-formula checks completed; the extra Laguna structured-prompt and
grouped-generator arms did not run because their six-hour reserve was not met.

The optional 4B structured-family search had seed 1048, 30 requested pairs and
a 20-minute search cap. It was cancelled before execution on the request to
finish immediately. Devstral's planned retry had seed 1048, 30 requested pairs
and a 30-minute cap, using published FP8 weights, BF16 activations and eager
Marlin serving. It was also cancelled before execution. Configuration files
preserve these plans for resumption, without presenting them as completed work.

Later writer additions score only their newly frozen method plus the shared
NLL reference on all nine caches. Before merging with the completed cohort,
labels, normalized claim group IDs, overlap masks, reference predictions and
reference bootstrap draws must match exactly. Existing frozen predictions are
reused; none are refitted or reselected. This avoids repeatedly calculating
all earlier methods while retaining identical paired bootstrap samples. Each
expanded cohort has its own verification marker before the report uses it.

The initial large-writer cohort is copied to a fixed snapshot and scored on
CPU as individual feature caches finish on the GPU. Its nine-dataset aggregate
is saved as `evaluation_before_9b`. The completed 9B search runs are then added
with the same reference checks, producing the primary `evaluation` cohort.
This overlaps independent computation without changing validation selection.
The fitted controls share fixed feature calculations within each scoring chunk;
a 512-training-claim audit found bit-identical predictions for all six heads,
reducing that block from 1.27 seconds to 0.19 seconds on the pod.

A 256-training-claim timing probe for the optional 1B feature model runs after
the source feature pass, without quality tuning. It also compares garbage
collection every batch against every 32 batches. The transfer stage can use
the less frequent schedule only if the probe's saved token IDs, likelihoods,
entropies and top-k values/indices are identical. This changes memory cleanup,
not feature calculations; the source pass keeps its original cleanup schedule.
An additional training-only probe compares device allocator cleanup every
batch with every 32 batches. The optional transfer stage can adopt it only
when all seven saved feature fields are bit-identical and measured runtime is
lower. This probe runs after the writer server stops. Completed source caches
and frozen estimator scoring are unaffected.

### Verified token-boundary issue and early controls

For the pinned repository-default feature checkpoint, all 7,007 training inputs
contain 14 tokens before the claim and **five tokens after it**, including an
additional assistant header. Exact token subsequence matching is unambiguous on
every training row. The historical tail crop `[-n_claim-1:-1]` therefore drops
four claim tokens and includes template tokens. This finding concerns the
provided checkpoint and software path; it does not by itself establish how the
paper's cached features were built.

A frozen-method suffix ablation is predeclared before held-out scoring: remove
the four added assistant-header tokens from cached statistics, preserving all
causal likelihoods for claim tokens. This isolates dependence on the accidental
suffix without another GPU forward pass or estimator retuning.

On the standard validation split, exact claim negative log-likelihood scores
75.75% AUC; the historical crop scores 72.85%. The template-included likelihood
scores 75.76% when accumulated in float64. The existing baseline's float16
accumulation yields 75.75%. Boundary controls are evaluated separately; the
writer-model comparisons retain the same full-template cache throughout.
See `results/token_alignment_audit.json`.

Training-only fitted controls currently score 81.64% validation AUC for logistic
regression and 80.24% for QDA. The claim-group protocol has a different validation
partition and scores 82.51% and 80.41%, respectively; these are not paired
estimates of the effect of removing overlap. The strongest of the 25 supplied
formulas scores 78.81% on standard validation. The five-round Qwen3.8 pilot
produced nine valid candidates from ten requests and selected a 78.48% validation
estimator. These early validation measurements are retained for context; the
completed held-out results appear in the execution assessment above.

The 17-feature pool includes three full-vocabulary entropy statistics beyond
the writer prompt's documented inputs. A matched 14-feature version removes
those values entirely, using sequence likelihoods and top-512 logits only.
With the same training-only tuning, it scores 81.60% validation AUC for logistic
regression and 80.21% for QDA. Both feature sets are frozen for testing; results
identify which inputs each fitted comparator uses. The strongest fixed-linear
validation winner has nonzero coefficients only for NLL sum, NLL standard
deviation and position slope, although its search considered the full pool.

The fitted classifiers are scored by their raw decision function for AUC.
Probabilities can round to exactly 1 on long out-of-domain claims, losing
ranking information despite distinct finite decision values. This scoring
representation was fixed before held-out evaluation; model weights and tuning
choices are unchanged. Standard validation AUCs are recorded for both score
representations in the fitted-feature audit.

Validation-only permutation diagnostics have also completed for the four
standard-split fitted heads, with 40 permutations per individual feature or
signal group. No models were refitted and no held-out datasets were read.
Likelihood features have the largest grouped sensitivity. Marginal permutations
break correlations: the large drops for individual correlated likelihood
summaries must not be interpreted as causal importance or as predictions of
performance after removing a feature and refitting. In particular, the almost
identical 14- and 17-feature validation AUCs show why a high marginal permutation
importance does not establish that an input adds unique information. Full
measurements and permutation standard deviations (not confidence intervals)
are in `results/validation_controls/fitted_feature_sensitivity.json`.

Four untuned formula ablations were frozen before held-out scoring. For the
Laguna seed-1048 winner, retain just the slope, slope plus the entropy-range
term, or slope plus top-1 variance; coefficients of retained terms stay fixed.
The Qwen seed-1047 expression simplifies algebraically to token count times a
linearly position-weighted mean NLL. Its float64 simplification removes the
original half-precision cancellation residuals and scores 79.6146% validation
AUC versus 79.6135% for the generated program (prediction Spearman 0.999989).
All variants enter the held-out benchmark, without choosing among them using
test performance. The generated programs remain unchanged. The other Qwen
evolution seed's top-k margin branch is skipped on all 7,007 training rows
because top-k arrays have one more position than token likelihood arrays; its
winner consequently reproduces a supplied slope ranking. See
`results/validation_controls/formula_ablation_preparation.json`.
