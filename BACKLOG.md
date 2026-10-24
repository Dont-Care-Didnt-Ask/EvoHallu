# EvoHallu research backlog

**Status: proposed future work, not an execution instruction.** This roadmap follows the completed Qwen3.5-9B source/CPU snapshot of 21 October 2026. It authorizes no additional paid compute, model downloads, API spending or overnight jobs. Optional-writer and full-nine-target outcomes must first be reconciled with the final campaign report; this document does not infer their completion from preparation or partial logs.

The main next question is whether a **strong open-weight code writer above 100B total parameters**, given a paper-scale search budget, discovers better detectors than our smaller writers. It remains unanswered by the 30-round comparisons. A second question is whether explicit changes to the evolutionary algorithm improve *useful behavioural diversity*, beyond changing prompt wording.

## Evidence that determines the priorities

The fixed source feature model is **Llama3.1-8b-instruct**. **Qwen3.5-9B is the code writer**: it proposes Python uncertainty estimators over cached features. A future larger writer replaces that proposal model, not the feature model, benchmark claims, or labels.

| Completed finding | Consequence for the next study |
| --- | --- |
| NLL is **65.6439 directed / 68.7112 paper-compatible AUROC**. The 65 versus 68.7 discrepancy is resolved by per-dataset folding; `is_hallu` stays unchanged. | Preserve exact baseline/row parity. Do not spend another study rediscovering a supposed UHEAD label error. |
| Four policies × three fresh adaptive seeds × 60 requests: folded means **69.943 original, 68.711 neutral, 70.355 diverse, 70.002 compositional**. Diverse minus original is +0.412 folded but −0.075 directed pp, with substantial seed variation. | Prompt changes alone have not established a reliable improvement. Do not select a new universal prompt using these test means. |
| The 15-context warm prompt screen found 6/4/7/3 jointly valid, nonconstant, diverse sibling pairs out of 15 contexts; all paired intervention-minus-original intervals include zero. | Revisit diversity with more independent contexts and distinguish proposal diversity from useful end-to-end search. The screen inherits historical search information. |
| Three fixed-NLL-example runs and three neutral adaptive runs all select NLL; the interface-only pilot also selects NLL and has the same 29/60 validity as its matched neutral seed. | These are bounded null controls, not proof that feedback or interface design never helps. Mere prompt clarification is not a demonstrated repair mechanism. |
| Rank/uniform sampling gives **68.922 ± 0.364 folded**, versus neutral 68.711. On observed eligible pools it raises expected two-parent training fitness by **0.456 pp** and lowers selection entropy. | This tested **stronger exploitation**, despite its 20% uniform component. It is not evidence that a behavioural novelty archive or broader eligibility was tested. |
| Top-three validation shortlists yield **35/36 complete source-nine slots**. One frozen candidate overflows on two datasets; several distinct sources are NLL aliases or close relatives. | Distinct code and a larger shortlist do not guarantee different, reliable detectors. Add train-only semantic/numerical diagnostics before the next search. |
| The 12-column supervised component head reaches **71.359 directed / 73.633 folded**, versus logistic-14 **70.615 / 73.019**. | Evolved feature discovery plus fitting is promising, but feature discovery and tuning must be nested for a clean new comparison. This is a different estimator class from fixed-coefficient programs. |
| C8741 uses normalization across each scoring batch. Predeclared chunk/order changes move folded macro **68.965 → 70.847**; a separate train-fixed control gives **70.558**, mainly through CommonClaims. | Batch invariance and fit-once state need explicit contracts. The follow-up is exploratory and does not replace the original 512-row primary. |
| Frozen formulas lose quality on matched source-six/target-six feature-model transfer; the 12-column head changes **74.296 → 71.431 folded**. | Cross-domain performance on one feature model does not establish cross-model or native-tokenizer transfer. |

Evidence: [source results](results/overnight20261021/REPORT.md), [prompt screen](results/overnight20261021/prompt_screen_analysis/PROMPT_SCREEN_ANALYSIS.md), [parent-selection audit](results/overnight20261021/parent_sampling_analysis/PARENT_SAMPLING.md), [shortlist outcome](results/overnight20261021/shortlist_secondary/SHORTLIST_OUTCOME.md), [component-head audit](results/overnight20261021/component_head_interpretation_audit.md), [batch audit](results/overnight20261021/batch_sensitivity/INTERPRETATION.md), [train-fixed control](results/overnight20261021/frozen_train_normalization/INTERPRETATION.md). These results motivate hypotheses; they do not select future test winners.

## P0 — Larger writer and a credible paper-style search

### P0.1 Verify the reference and choose a genuinely larger writer

**Primary-source check:** the paper explicitly uses **Gpt-oss-120B and several Claude writers**, usually **500 rounds × two proposals**. Its claim-verification study describes 20 GPT-OSS runs and 10 Claude runs, validation-based shortlisting, and additional manual diversity curation for GPT-OSS. Therefore one 1,000-request run is paper-scale per trajectory, not a reproduction of its complete search campaign. [EvoHallu, §§3 and 5.1](https://aclanthology.org/2026.acl-srw.115.pdf#page=3).

The repository also defaults to GPT-OSS-120B and 500 rounds in [config_uq.yaml](configs/config_uq.yaml). These are separate pieces of evidence: a config default does not reconstruct all published models, prompts, revisions or selection decisions.

**Setup.** Build two explicit candidate roles: (a) GPT-OSS-120B as the verified paper/repository reference; (b) an independently shortlisted competitive open-weight code writer above 100B **total** parameters. If the same checkpoint fills both roles, state that and avoid counting it twice. Verify the current official checkpoint, immutable revision, weight availability, license, native prompt template, reasoning/output format, total and active parameters, quantization, and supported serving backend. Record why the proposed model is competitive using current primary evidence before selecting it; no model is declared current SOTA here. Recover the paper's prompts, sampler settings and executable reference formulas where available. List unrecoverable details rather than inventing an exact replication.

**Measure.** First use a bounded **20-request engineering pilot per distinct writer**, using training-only synthetic/real feature fixtures and no benchmark scoring. Record load time, peak VRAM/RAM, prompt/output tokens, request latency, end-to-end candidate throughput, validity, numeric failures, template parsing and actual precision. Pilot outputs are excluded from confirmatory seed populations.

**Acceptance and cost gate.** Proceed only with correct model/template provenance, finite outputs on declared fixtures, successful independent requests, and a measured resource projection that fits an explicitly approved future cap. Memory estimates must include all resident weights, KV cache and runtime workspace: active MoE parameters are not a weight-residency estimate. Evaluate single-card, multi-card or quantized serving only after checking the actual checkpoint. Quantization/backend changes are named interventions. No assumption that any >100B writer fits one 80GB A100; no new hardware is authorized by this backlog.

### P0.2 Matched smaller-versus-larger writer comparison

**Hypothesis.** Larger-writer capability improves the rate of valid, useful, behaviourally distinct programs and frozen detector quality at the same attempt budget; it may still be less cost efficient.

**Setup.** Keep source features, source tokenizer, CSV/cache hashes, NLL start, fitness, parent rules, prompts, candidate interface, validation rule and frozen OOD scorer fixed. Use one predeclared original-style prompt as the primary writer comparison; a second prompt is a separate factorial arm, not silently chosen from tonight's test ranking. Pair at least three search seeds between Qwen3.5-9B and each larger writer. Normalize native serving templates correctly while logging the exact rendered messages. Retain all attempted proposals, including invalid and constant candidates.

Start with a **60-attempt/30-round matched study** if a short comparison is needed; the actual extension is **1,000 attempts/500 rounds per writer × seed**, conditional on the pilot. Two writers × three seeds cost **6,000 main attempts** at the long budget; a third distinct writer adds 3,000. An independently executed short matrix adds **360 attempts** for two writers. A 60-request prefix can substitute for a short run only when its schedule, RNG/resumption and eligibility semantics are exactly the same; otherwise keep it separate.

**Measure.** Save checkpoints at declared attempt counts, best training fitness, validation-selected identities, fallback frequency, train-only behavioural novelty, valid useful programs per 100 attempts, tokens, wall time, CPU evaluation time and GPU-hours. Freeze one primary and a predeclared top-three inspection set per seed using PopQA validation; deterministic ties prefer the earlier candidate. Inspect actual formula behaviour and NLL correlation. Publish all seed outcomes, directed and paper-compatible nine-dataset macros, per-dataset results and paired fixed-program claim intervals; separate search-seed variation from claim uncertainty.

**Acceptance.** Engineering success means complete matched budgets and provenance/score parity, even if quality is worse. A larger-writer quality claim requires a predeclared across-seed advantage over the matched smaller writer on the designated primary metric, uncertainty compatible with that claim, and a defensible held-out evaluation. A paper-compatible numerical gain alone, a best seed, or a tiny accumulation-induced NLL gain is insufficient. Fix the minimum useful effect and inference procedure before confirmation; do not choose them from the observed test table. Beating the paper is a hypothesis, not a deliverable that can be promised.

**Cost gate.** Scale the actual pilot's prompt-length-aware end-to-end time, not model size or output tokens alone. Reserve load/download time, CPU evaluation, final backup and safe stop time. Authorize the complete matched seed block before starting; a partial model/seed remains labelled partial. If resources cannot support 6,000 attempts, report a smaller complete comparison rather than a paper-budget claim.

### P0.3 Long trajectories versus restart portfolios

**Hypothesis.** At a fixed proposal budget, independent restarts can outperform a long trajectory whose population has converged; longer exploitation can also be necessary for some writers.

**Setup.** First compare **one 90-round trajectory versus three independent 30-round restarts**, both 180 attempts, across three paired portfolio seeds. This costs **1,080 attempts** for the two arms. Pool restart candidates and select one final primary by the same PopQA validation rule; do not average the three winners and call it a portfolio selector. Retain all restart candidates and failures. At paper scale, a separate comparison can use **one 500-round run versus ten 50-round restarts**, both 1,000 attempts per portfolio.

Decouple the inherited percentile/temperature schedule from the configured total rounds, or predeclare exactly how schedules differ. Otherwise changing the horizon also changes parent eligibility. Keep a literal original-schedule replication as a separate reference. Resumption must preserve database, RNG and schedule state; an interrupted rerun is not a new independent seed.

**Measure/acceptance.** Record training improvement and novelty over time, validation-selected portfolio quality, early/late failure rates, time to a predeclared training target, and final all-seed frozen quality. A novelty decline against a growing archive alone does not prove collapse. Accept the portfolio hypothesis only through the predeclared matched comparison; preserve null outcomes. **Cost gate:** start at 180 attempts per arm/seed after the writer pilot; expand to 1,000 only if the approved cap permits complete paired portfolios, without inspecting OOD winners to allocate compute.

## P1 — Change the evolutionary process, not just its wording

### P1.1 Training-only behavioural archive and novelty-aware parents

**Hypothesis.** Avoiding repeated *rank behaviours* increases useful proposal yield. It may increase novelty without improving quality, which must remain an admissible outcome.

**Setup.** Build an opt-in archive from the fixed training probe, with no validation/test scores, labels or predictions used for novelty. Explicitly separate constant outputs, exact source duplicates, score-vector duplicates, rank aliases and reversed rankings. Predeclare signed correlation and the absolute-correlation supplement; a score sign flip alone should not count as a newly useful mechanism. Use a frozen threshold/grid chosen in training-only pilots. Test one change at a time:

| Arm | Intervention | What remains fixed |
| --- | --- | --- |
| A | Inherited sampler/reference | Prompts, islands, gate and all request budgets |
| B | Behaviour archive used only to diversify selected parent examples | Same fitness and eligibility gate |
| C | Behaviour-aware archive admission/retention | Same generation count; rejected duplicates still count as attempts |
| D, later | Novelty/fitness trade-off or novelty-aware eligibility | Explicitly separate from B/C; no bundled claim |

**Measure.** Joint valid, nonconstant, novel **and training-useful** yield per *issued request*, not only correlations conditional on two successful siblings. Preserve unconditional and conditional denominators, archive size, nearest-neighbour correlation, NLL similarity, parent entropy/inclusion, fitness and runtime. Track which novelty survives final validation selection.

**Acceptance/cost gate.** First prove that the intervention changes actual parent examples/admission on deterministic fixtures. Then use three paired 60-attempt seeds per arm: **180 attempts per added arm**, plus a fresh matched reference unless it is prospectively shared. Advance only on train-only engineering/useful-yield evidence under a fixed rule; hold OOD evaluation until policies and primaries are frozen. Novelty alone is not quality success. CPU archive cost and probe overfitting must be reported; use a separate training-only audit probe for generalization of the diagnostic.

### P1.2 Eligibility, parent temperature, source-length preference and injections

**Setup.** The completed rank control retained the percentile gate, islands and shorter-source preference within score groups. Future trials separately vary: (a) eligibility-gate breadth; (b) candidate-parent temperature in the correct AUROC units; (c) within-group source-length preference; (d) a fixed-rate fresh baseline/random-parent injection; (e) island restart/migration policy. Generation temperature is a different parameter. A 20% uniform mixture **inside** an existing gate cannot explore excluded candidates.

**Measure.** On identical observed eligible pools, replay exact two-parent inclusion probabilities and set entropy; then measure actual generated training-useful diversity and final validation selection in matched searches. Keep complex multi-feature candidates eligible; feature count or source length is not itself a quality penalty.

**Acceptance/cost gate.** Replay must demonstrate the intended pressure change; call it increased exploration only if measured coverage/entropy supports that description. No inference is required for the replay. Choose a small, predeclared training-only shortlist, then spend **180 requests per added 3-seed arm**, with matched controls. Do not launch a large combinatorial grid or select gate settings on the nine test datasets.

### P1.3 A cleaner prompt/diversity study

**Setup.** Reproduce fixed-context comparisons on more independent training-generated contexts, then test separately: sibling roles, feature-family guidance, writer generation temperature, and two-stage idea/sketch-to-code. Include a genuine **no-evolved-example independent generation** arm, distinct from the already tested fixed-NLL-example arm. Historical warm contexts remain a separate cohort. Keep large composite formulas available; the Claude screenshot is not an executable method.

**Measure/acceptance.** Primary proposal diagnostic remains jointly valid, nonconstant, behaviourally different siblings per all issued contexts. Use paired context intervals and show effective independent context counts. Downstream validation-selected quality and all-seed variation are necessary to claim better evolution. The existing 15 contexts did not resolve prompt diversity; neither do their high-quality isolated winners establish a from-scratch 30-request advantage.

**Cost gate.** A proposed first screen of **50 contexts × two siblings × three arms = 300 requests**, with contexts and contrasts frozen in advance, precedes any adaptive confirmation. For sketch/implementation or repair, count **every** LLM call/token in the same budget; two calls are not one free candidate. Increase the screen only after a prospective precision/power calculation using the existing paired diagnostic, not by repeatedly checking significance.

### P1.4 Reliability, semantic validity and fit-once normalization

**Setup.** Add a new future candidate contract/fixture suite before expensive training evaluation: correct Estimator constructor/imports, ragged inputs, token/top-k lengths **n versus n+1**, known alignment, finite output, empty/short/long sequences, extreme log-probabilities and resource caps. Distinguish top-512 conditional softmax from full-vocabulary probabilities. Test observed-token positions and chat/template contributions explicitly. Comments such as “bottom 10%” must be checked against executable indexing.

Make the default detector **sample-wise or fit-once**: prohibit undocumented statistics over the current prediction batch, or label batch-adaptive estimators as a separate class. A scaler/median/MAD may be fitted on training rows only and serialized. Predeclare zero-MAD behaviour, storage/accumulation precision, safe exponential weights, chunk-1/128/512/shuffle invariance and repeated-call state checks. Existing frozen failures remain unchanged; the new contract applies to future runs.

The observed target-UHEAD entropy failure motivates **training-only or unlabelled synthetic sharp/zero-probability and long-sequence fixtures before validation selection or benchmark evaluation**. Require explicit float32/64 accumulation and entropy-safe zero handling; a float16 epsilon can vanish and permit `0 × log(0)`. This is a future contract intervention. Keep tonight's frozen failure in the reported evaluation, without post-test repair.

**Measure.** Separate generation transport/context errors, parse/syntax errors, API/import failures, shape/alignment/numeric failures, timeouts, valid constants, semantic aliases and actual truncation finish reasons. Audit runtime/branch activation and full training/validation NLL correlation for frozen shortlists. Treat float16 rounding and dead terms as semantic findings, not new feature discoveries.

**Acceptance/cost gate.** CPU fixtures and parity checks precede GPU time. For a new contract or one-budgeted-repair arm, use three matched 60-attempt seeds (**180 attempts per arm**), including repair calls in the attempt/token budget. Require intended invariance and correct feature semantics; separately test whether validity and quality improve. The interface-only null pilot does not justify claiming that a stronger contract works. Do not repair candidates based on OOD failures and then reuse those OOD results as fresh confirmation.

## P2 — Supervised evolved features and explanations

### P2.1 A real fit/predict API inside evolution

**Hypothesis.** Evolution may be more productive discovering complementary feature maps whose coefficients are fitted than hand-writing fixed sums.

**Setup.** Add a separate generated `fit(train)` / `predict(features)` API with serialized scaler/weights and no prediction-batch refitting. Candidate feature construction and coefficient tuning occur entirely inside claim-grouped training folds. Use grouped inner validation or nested cross-fitting for evolutionary fitness; keep the outer PopQA validation split for final candidate selection. Include all fitting choices in the candidate's effective search budget. Compare fixed-feature logistic-14/QDA, fixed-coefficient evolved programs, and evolved-feature fitted heads under matched data access.

Audit near-identical NLL columns; compare a predeclared canonicalized feature pool against the full pool as separate arms, because duplicate columns alter L2 regularization. Do not interpret coefficients of correlated features as independent importance. No complexity ceiling is imposed merely because Claude-like programs are long.

**Measure/acceptance.** Publish discovery and coefficient-fit provenance, fold/entity leakage checks, validation selection, all-seed frozen scores, strict-overlap sensitivity, calibration if declared, and inference cost. Reproduce all reference heads before new discovery. Supervised gains are reported as supervised; the existing **73.633 versus paper Claude 73.4** comparison is not a matched SOTA result. A clean claim needs a prospectively held-out confirmation and nested discovery/tuning.

**Cost gate.** First replay the fit/predict API with existing fixed components using CPU only. Proceed to a three-seed, 60-attempt generated-feature pilot (**180 requests**) only after deterministic serialization/parity and bounded per-candidate CV cost are established. Allocate CPU/RAM separately from writer GPU-hours; no new feature extraction is needed for fixed source modalities.

### P2.2 Component utility, real complexity and the Claude question

**Setup.** Preserve the completed eleven fixed-coefficient drops and four exact controls. For future frozen populations, predeclare component definitions before benchmarks and compare fixed drops versus train-only coefficient refits. If affordable, examine a small fixed subset/feature-count frontier and permutation controls; correlated features make exhaustive/Shapley interpretations conditional. Recover official original Claude/GPT estimator code, selection pools and trajectories before explaining Sonnet/Opus differences or claiming literal original-method ablations.

**Measure/acceptance.** Verify exact full-program reconstruction, score/branch identities, feature correlations, effective feature count, CPU cost and OOD robustness. Report every declared subset; no test-best drop is promoted. Agreement between static and executable definitions is required. A feature's conditional contribution is not an independent causal effect.

**Cost gate.** Existing saved predictions and CPU-only fixed-control recomputation first; no writer calls unless missing definitions require a separately approved new generation study. Set a bounded component count before any exhaustive subset job. Historical artifacts may remain unavailable; document that limit instead of rebuilding an invented “Claude method” from the screenshot.

## P2 — Stronger transfer and data validity

### P2.3 Feature-model and tokenizer transfer

**Setup.** Reconcile the final target-nine completion and cache-backup receipts first. Carry the **same frozen source-validation-selected programs and head weights** to target features; do not select/refit using target labels. Compare source/target only on identical prescribed datasets and rows. Separate a factorial study of (a) feature model with the shared source tokenizer, (b) feature model with its native tokenizer, and, later, (c) source-model size/family. Native tokenizer changes tokenization and may require a separate extraction/template protocol; a shared tokenizer is not automatically native deployment. Target refitting is another supervised adaptation arm.

**Measure/acceptance.** Require per-cache pinned model/tokenizer assets, dtype, row/label/order hashes, token-ID alignment, top-k shape, metadata, sample counts, NLL parity where applicable and a current-runtime numerical reference. Use fixed raw-logit/log-probability tolerances before looking at AUROC; quantized storage zeros are not exact full-precision probabilities. Paired source-target deltas and NLL-relative transfer use the same dataset coverage, row groups and bootstrap draws. Six-dataset means remain explicitly six until all nine pass.

**Cost gate.** CPU scoring of completed caches first; new extraction needs a small bounded numerical/memory pilot, protected historical hashes, estimated output/archive sizes and approval for the full model/tokenizer dataset block. Never overwrite the original caches. A failed numerical guard stops extraction and leaves an explicit incomplete result, without loosening tolerances to rescue quality.

### P2.4 Claim-, entity- and relation-disjoint confirmation

**Setup.** Resolve the inherited **88 normalized-claim overlaps** across the 5,605/1,402 evolutionary split. Keep historical experiments intact; build a newly versioned grouped split for all future arms. Pin all ten CSVs: one training file plus **nine** evaluation datasets. Audit duplicate entities/relations and near-duplicate text with rules frozen without benchmark quality. Introduce entity/relation-disjoint or leave-domain-out training/validation where metadata permits.

The existing nine datasets have already informed follow-up hypotheses. They can support a transparent exploratory replication, but must not be described as untouched confirmation. Reserve an inaccessible new test partition/dataset or a controlled external evaluation for the final scientific claim. Register hypotheses, metrics, program/weight hashes and all seed identities before opening it.

**Measure/acceptance.** Report row/group/entity counts and retained class balance, exact overlap removal, all-data versus strict-disjoint performance and source-only orientation selection. Label-aware AUROC folding is paper-compatible reporting, not a deployable direction-selection rule. Use a predeclared source-selected direction for a practical detector and show directed scores alongside folding. Prevent validation labels from entering novelty, parent selection, model fitting or repair feedback.

**Cost gate.** Metadata and split construction are CPU tasks; first reproduce canonical baselines and grouped-head fits. Any rerun caused by a new split is a new matched study, with its own writer attempt cap. Do not “correct” the current split retroactively and merge scores across partitions.

## P3 — Cross-task extension after the core result is credible

**Setup.** Study longer generations and selective prediction separately from atomic claims: frozen detector transfer first, then in-domain evolution with its own fitness/data split. Recover the original task/evaluator inputs before claiming a paper reproduction. Attention/hidden-state access, verbalized confidence, writer fine-tuning/RL and model ensembles are further independent interventions, not free additions to the current cached-logit experiment.

**Measure/acceptance.** Use the task's declared quality/rejection metric, strong fixed baselines, length/numerical robustness and all-seed selection rules. Cross-task transfer and in-domain search answer different questions. No AUROC result establishes selective-prediction quality.

**Cost gate.** Inventory existing caches and required generation/modalities first; new benchmark generation or hidden-state extraction needs a separately approved pilot and storage/compute budget. Defer this until the larger-writer, algorithm and data controls above are interpretable.

## Execution and reporting gates for every future study

1. **Freeze the question.** Record one primary contrast, fixed arm order, writer/source revisions, seed count, training/validation/test access, failure handling, minimum useful effect, metrics and uncertainty method. Existing OOD observations may motivate a study; they cannot become a new untouched test set.
2. **Verify the environment.** Rebuild once in a fresh environment from the tested GPU requirements and documented install steps; verify the Torch CUDA build, dependency check, model load, one bounded generation and canonical CPU/cache parity. The current pod proved a working runtime, not a second clean rebuild. See [environment reproduction](results/overnight20261021/ENVIRONMENT_REPRODUCTION.md).
3. **Run a capped engineering pilot.** Count it separately unless its inclusion was frozen in advance. Verify model/template/fit contracts and numeric invariance before any scale-up. Rejection, repair, sketch and failed requests consume budget.
4. **Quote resources from measurements.** For each model/hardware candidate, record current provider price and availability at booking; estimate `GPU cost = hourly rate × GPU count × billed hours`, plus CPU/storage/transfer charges. Project time from real end-to-end requests and evolving context lengths. Compare A100/H100/H200 or multiple cards using measured throughput and fit, not an assumed architecture speed ratio. Preserve stop/backup reserve; no present price or 8-hour capacity is promised here.
5. **Authorize a complete block.** Finish matching seeds/arms within a cap; do not use test quality to decide which model gets more attempts. CPU evaluators must not contend with generation beyond the measured budget. Record partial budgets, final failures and all unsuccessful seeds rather than dropping them.
6. **Freeze before scoring.** Save exact programs, fit state, prompts, request ledgers and validation ranking. Score all predeclared primaries; no benchmark-based promotion. Program bootstraps condition on fixed discoveries; seed variation, selection history and multiple comparisons require separate reporting. Nine datasets are a fixed benchmark set, not nine independent search seeds.
7. **Make the result portable.** Back up source, validation predictions, full ledgers, hashes, model/cache receipts and all result scopes before releasing paid hardware. Keep the user's separate branch/author identity and provide a reviewable repository diff/ZIP; publication remains a separate action.

## Open user questions mapped to concrete deliverables

| Question | Next deliverable | Priority |
| --- | --- | --- |
| “Only 30 rounds — can we do something like the paper with a >100B writer?” | Verified checkpoint/hardware pilot, matched 60/1,000-attempt design, all-seed paper-style comparison; distinguish one trajectory from the full original campaign. | P0.1–P0.2 |
| “Would better candidate selection beat one long run?” | Equal-attempt trajectory/restart portfolios, ordinary validation winner plus frozen inspection shortlist, measured selector semantics. | P0.3 |
| “Do current prompts cause low diversity?” | Larger paired screen plus training-only useful behavioural novelty; separate prompt roles, generation temperature, archive, eligibility and parent pressure. | P1.1–P1.3 |
| “How exactly does evolution select, and is it a baseline from another framework?” | Versioned EvoTune/sampler provenance, exact eligibility/island/within-group probabilities, isolated interventions and deterministic replay. Current mechanism evidence is [here](results/overnight20261021/parent_sampling_analysis/PARENT_SAMPLING.md). | P1.2 |
| “What formulas actually evolve, and are complex Claude-like methods useful?” | Effective-formula/branch/correlation audit, original artifact request, predeclared ablations and fitted-feature comparison. Start from the [12 winners](results/overnight20261021/ADAPTIVE_WINNER_INSPECTION.md) and [screen sources](results/overnight20261021/SCREEN_CANDIDATE_INSPECTION.md). | P1.4, P2.1–P2.2 |
| “Were models used for generation or as detectors?” | Every run manifest names writer, fixed feature model/tokenizer, estimator/fit state and input claim provenance separately. | P0.2, P2.3 |
| “Did datasets drift; what was passed in; was `is_hallu` correct?” | Pinned ten-CSV/row/cache hashes, feature schema, grouped/entity splits and canonical baseline parity. Baseline orientation discrepancy is already resolved. | P2.3–P2.4 |
| “Is the environment reproducible and which code changed?” | One fresh rebuild receipt, exact setup/runtime comparison, scoped diff, data/code/model SHA manifest, branch/author/archive checks. | All-study gates |
| “Which GPU is most economical and how much quota is needed?” | A measured pilot and current quoted cost for a complete matched seed block, explicit maximum spend/deadline, backup and stop reserve. | P0 cost gate |
| “Can we beat the paper?” | A matched reference comparison plus prospective confirmation. Until then publish improvements and negative results with their exact supervision, budget and dataset scope; promise no SOTA outcome. | After P0 and validity gates |

The original seven extension questions and completed coverage remain in [EXPERIMENT_EXTENSIONS.md](EXPERIMENT_EXTENSIONS.md) and the [coverage audit](results/overnight20261021/extension_coverage_audit.md). The [dated preparation plan](NEXT_EXPERIMENTS.md) is historical context. This backlog adds prospective experiments without changing any executed program, frozen primary, existing result, or active job.

## Engineering addendum: GPT-OSS-120B on A100

A [read-only feasibility audit](results/overnight20261021/BIG_WRITER_FEASIBILITY.md) verifies the public pinned checkpoint, exact 65.249 GB weight-file footprint and the installed vLLM 0.31 SM80-compatible MXFP4/Marlin path. One 80GB A100 is plausible with bounded context; TP2 provides more memory margin. This is static engineering evidence, with no weight download, model load, quality measurement, speed forecast or additional compute authorization. The future pilot and matched P0 comparison remain required.
