# Overnight evaluation protocol

The generator is Qwen3.5-9B unless a separately named writer cohort is added. Existing Llama3.1-8b-instruct feature caches and original `is_hallu` labels remain fixed. No held-out label is used to choose a program, a search seed, a prompt policy, or a coefficient.

1. Run the prepared prompt screen and evolutionary/control jobs. Prompt-screen runs are diversity diagnostics, not primary quality-search arms.
2. `overnight_eval.py` stages the validation-selected frozen winner of **each** finished non-screen run. It preserves all seeds, including baseline winners and runs that stopped short of their planned request budget. Optional `--shortlist 3` also stages validation ranks two and three; these remain secondary diagnostics. Optional `--include-train-selected` stages the candidate with maximum directed training fitness, including the baseline; ties keep the earliest candidate and do not consult validation or test quality. Immutable names include the job ID and selection role, avoiding the shared evaluator's parent-folder naming collisions. A changed cohort receives a new evaluation directory rather than reusing stale dataset results.
3. Existing CPU scoring uses fixed chunks of 512 samples and separate estimator state for every dataset. `--method-prefix new/` retains the shared sequence-log-probability/NLL reference. A separate supervised cohort loads the new documented-14-feature claim-grouped logistic and QDA heads. Existing fitted controls may also be scored by that cohort and remain explicitly named.
4. Report directed AUROC and the author's paper-compatible `max(AUC,100-AUC)` **within each dataset**, followed by the unweighted mean over nine datasets. The clarified reporting convention resolves the historical NLL 65.64 versus paper 68.7 discrepancy; no label inversion is applied. Duplicate-disjoint results are additional sensitivity analyses.
5. Use 500 shared normalized-claim bootstrap resamples per dataset for paired comparisons to NLL. Reapply the paper transform within every bootstrap replicate, then average over the nine independently resampled datasets. These intervals condition on a frozen program; report writer/search-seed variability separately as all-seed mean, sample standard deviation, minimum and maximum. Avoid claims of superiority to the paper from a single selected seed.
6. Diversity uses the prespecified 512-row **training** probe, following `summarize_next_experiments.py`: valid fraction, constant candidates, distinct rank vectors, near-previous/baseline count and sibling Spearman correlation ≥0.995. Missing or incomplete jobs remain pending; no synthetic test numbers are supplied.

Stage and inspect the exact commands without running evaluation:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/overnight_eval.py --shortlist 1
```

Run the nine-dataset CPU evaluation on a host with restored caches:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/overnight_eval.py --shortlist 1 --roles primary --execute --workers 2 --bootstrap 500 --cache-dir /dev/shm/mod1-eval-cache
```

After primary scoring, freeze and score the training-fitness diagnostic with `--include-train-selected --roles trainbest --kind new`. This uses a separate cohort and does not repeat primary scoring. Optional prespecified shortlist scoring uses `--shortlist 3 --roles valrank02,valrank03 --kind new`; schedule it when CPU time and memory allow. `--kind new` or `--kind supervised` runs an individual method family. No GPU inference occurs in these scripts.

Create JSON, CSV and Markdown reports, including honest pending status:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/overnight_report.py
```

The primary table includes every frozen winner, never a test-best seed. Corresponding all-seed groups and matched-seed differences against the original prompt policy are saved to `report.json`. Historical paper scores (NLL 68.7; best GPT 70.8; best Claude 73.4) are rounded comparison context, not new matched statistical tests.

## Incremental CPU scoring and exact baseline checks

CPU scoring can overlap later GPU searches. `overnight_eval.py --jobs job_id_1,job_id_2 --roles primary --kind new` scores only those finished jobs, using explicit individual frozen file paths. Later calls score different jobs in separate cohorts without repeating earlier methods. Shared baseline predictions must agree exactly when reports merge these additive cohorts. Pass the combined non-executable analysis manifest with `--manifest` to keep sampler controls and separate writer-confirmation phases distinct.

`overnight_report.py` includes `baseline_exact_check.json`: each dataset's NLL predictions and AUROC must match `out/campaign/evaluation_with_4b`, along with exact rowwise `is_hallu` labels, claim-group IDs, train-disjoint masks, CSV label/group order and historical CSV byte hashes. The unchanged nine-dataset source NLL macro should be 65.643882 directed and 68.711173 paper-compatible, rounding to 68.7. Missing new predictions remain pending; drift is reported as failed.

## Source-selected feature-model transfer

`overnight_transfer_eval.py` transfers the exact same immutable source-validation primary winners, preserving every seed, plus the fixed documented-14-feature grouped logistic and QDA heads. No target labels are used for selection or fitting. The default `--coverage six` uses only the six preexisting target caches in `data/cache_backup_manifest.json`. Its source comparison uses exactly those same six datasets and is explicitly distinguished from the full nine-dataset benchmark.

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/overnight_transfer_eval.py --execute --workers 2 --bootstrap 500 --cache-dir data/restored/mod1-transfer-cache
```

If the separately guarded `prepare_remaining_transfer.py` finishes the missing CounterFact, PopQA and X-Fact caches, run full-nine transfer:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/overnight_transfer_eval.py --coverage nine --execute --workers 2 --bootstrap 500 --max-seconds 1800
```

Full-nine execution requires all nine target caches and a completed pinned extraction receipt. Six- and nine-dataset cohorts have separate immutable output directories under `transfer_llama32_1b/<cohort-id>/`. Generic `source_matched_subset_*` and `target_matched_subset_*` metric fields describe the chosen coverage; existing six-dataset JSON aliases are retained. Coverage, incomplete methods, execution failures and estimator failures are explicit. Paired claim-bootstrap source/target differences reuse matching deterministic resamples after verifying rowwise labels, claim groups and disjoint masks. Every seed remains separate, and all-seed summaries are reported without selecting a test-best seed.

Before scoring, each target cache's architecture, logits modality and model forward dtype must match hidden size 2048, 16 layers, 32 attention heads, vocabulary 128256 and `torch.float32`. Stored features can still be float16. CSV claims/labels must match in order; streamed raw SHA hashes are checked against restoration/completion receipts. When corresponding source raw caches are available, **all** chat and raw-claim token IDs must also match.

Historical embedded metadata contains neither checkpoint revision nor tokenizer identity, so these are recorded as external provenance rather than falsely attributed to embedded metadata. The pinned target is `unsloth/Llama-3.2-1B-Instruct` revision `5a8abab4a5d6f164389b1079fb721cfab8d7126c`; the shared original Llama3.1 tokenizer revision is `247cb7c3baebabc46cff1fb97a3a5913e4c4511d`. Full-nine execution requires receipt evidence of verified pinned model/tokenizer file hashes, historical tokenizer-input parity, and unchanged original six caches. No source-selected formula or fitted head is refit on target labels.
