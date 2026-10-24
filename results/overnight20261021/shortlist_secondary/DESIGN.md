# Optional validation-shortlist secondary analysis

**Prepared only; no real scoring, GPU work or writer calls launched by this preparation.** This follows the user's request to inspect the highest validation candidates and the paper's retention of three candidates. It adds a retained-collection diagnostic, not another test-driven winner selection.

The cohort is fixed by the exact twelve original Qwen3.5-9B adaptive job IDs: original, neutral, diverse and compositional policies × seeds 2107, 4319, 8741. The source manifest labels the compositional phase `compositional_control`; these are still its intended adaptive evolution jobs. Screens, feedback/sampler/interface controls and larger writers are excluded. Every run must finish all 60 requests, with its original rank-one source and predictions frozen on all nine source datasets, before any secondary staging occurs. The separate registry, source copies, evaluation outputs and report remain under this directory. The primary cohort is read-only.

The existing generator sorts baseline plus valid proposals by descending directed PopQA validation AUROC, stably preserving baseline first on ties and proposal order afterward. The helper validates this exported order and exact rank-one identity; it does not reselect candidates. All 36 original rank slots are retained. Source-identical ranks alias one prediction vector, including a possible baseline alias; they are not silently relabelled as lower ranks. Fewer than three available valid candidates is explicit. Existing staging is reused and writes only this cohort. Only unique additional roles `valrank02,valrank03` are evaluated; at most 24 additional sources require scoring.

The exact existing cached CPU evaluator uses four workers and 500 claim-bootstrap draws, with no feature extraction. The wrapper hides CUDA devices and gives the evaluator a process-group timeout: at most 1,800 seconds by default, optionally 3,600, **starting only after all twelve runs and original primary-nine predictions are ready**. Waiting polls every twenty seconds against the independent absolute gate; it does not consume the active allowance. Both stages always end before **2026-10-22 02:06 UTC**, and an explicitly supplied deadline must be timezone-aware UTC and cannot exceed that cap. Thirty seconds are reserved for report generation. An alarm also bounds active CPU reporting. Cancellation markers `watchdog.cancel` or `shortlist_secondary.cancel` in the campaign directory, or `shortlist_secondary.cancel` here, stop waiting/evaluation; additional markers can be passed with `--cancel-file`. SIGTERM and KeyboardInterrupt unwind through the owned process-group cleanup, terminating all evaluator descendants; evaluation checks markers at least once per second. This is an optional CPU-capacity consumer; do not start active evaluation while another CPU-heavy follow-up needs those workers or if it would compromise archival/shutdown deadlines. No promise of complete coverage follows from the prepared plan.

Reports show validation score, directed/folded nine-dataset AUROC and duplicate-disjoint sensitivity for every retained rank. Policy/rank means and sample SD require all three seeds and all nine datasets. A separate top-three collection diagnostic averages the three **individual method AUROCs** within each run, followed by all-seed summaries. It neither averages predictions nor reports a test-best/oracle maximum. Missing datasets suppress macros. All saved secondary labels, claim-group IDs, masks and canonical NLL predictions must agree exactly with the original source reference. Baseline-selected primary vectors must equal canonical NLL. The historical baseline/CSV identity audit is reused. The gallery links every actual frozen source and SHA-256, preserving aliases for inspection.

This remains exploratory: the original stratified split has normalized-claim overlaps, benchmarks have previously been inspected, and three retained methods are not independent samples. Collection maxima cannot establish improved SOTA or a matched paper comparison. Original primary predictions are reused rather than recomputed; source hashes, original primary identities and source-file hashes prove which frozen rank-one method they represent.

Prepare/read readiness only:

```bash
.venv/bin/python scripts/overnight_shortlist_secondary.py
```

Parent-approved optional execution, after other CPU work permits:

```bash
.venv/bin/python scripts/overnight_shortlist_secondary.py --execute --max-active-seconds 3600 --cache-dir data/restored/mod1-eval-cache
```

The exact evaluator command is recorded in [PLAN.json](PLAN.json). Regenerate the plan on the pod so its Python interpreter/cache paths reflect the execution environment. Reporting existing outputs only uses `--report-only`. Failed/partial evaluation remains isolated and cannot alter primary results. A timeout will not claim successful completion.

Five focused checks cover refusal to stage one incomplete member of the twelve-run cohort; baseline tie priority plus source-identical rank aliases with byte-preserved primary registry; suppression of partial-nine-dataset macros; an independent waiting clock with an active budget beginning after readiness and validated UTC hard cap; and real tiny-child cancellation/process-group cleanup plus SIGTERM handler restoration. These tests execute actual existing staging and a tiny sleeping child, never model inference.

Optional top-three correlations on the saved 512-row training probe are not implemented in this helper. Source-identical alias counts are reported; distinct source hashes alone must not be interpreted as behavioural diversity. The separate campaign semantic and training-probe audits remain applicable.
