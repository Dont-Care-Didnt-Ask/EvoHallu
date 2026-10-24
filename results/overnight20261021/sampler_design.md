# Rank-parent sampling control

This is a separate, opt-in comparison with neutral-prompt evolution. It changes score-group sampling pressure while preserving the original percentile gate, two-island choice, within-group source-length preference, candidate budgets, training fitness, and validation winner rule.

For the `n` eligible groups, ascending fitness rank is `1` for the worst and `n` for the best. The probabilities are:

```text
p_i = 0.8 * rank_i / sum(rank) + 0.2 / n
```

Score-group fitnesses are unique in the inherited database; the implementation also gives average ranks to ties. The 20% uniform mixture reserves exploration across the eligible groups. It is not a behavioural novelty archive.

The historical parent weights are `softmax(training_AUROC_percent / 40)`, which the earlier audit found almost uniform within eligible groups. This experiment tests whether stronger selection pressure helps; an improvement is not guaranteed.

The wrapper intercepts `_softmax` only when its immediate caller is the original `Island.get_prompt` code object. The same `_softmax` function is used inside `Cluster.sample_program_and_metrics`; that call is explicitly delegated unchanged. No pinned search source is edited. The wrapper writes exact eligible fitnesses and probabilities per round, plus its own and the inherited database's SHA-256 hashes.

Prepared manifest: `results/next_experiments/sampler_manifest.json`. It retains the original training/source pins and adds the wrapper hash. Three configurations use neutral prompts, seeds 2107/4319/8741, and 30 rounds × two requests. The main 19-job manifest is unchanged. For compatibility with the existing controller, job phase is `evolution_screen`; manifest `study_phase` and per-job `intervention` distinguish the sampler control.

Dry run:

```bash
.venv/bin/python scripts/run_overnight_parallel.py \
  --manifest results/next_experiments/sampler_manifest.json --phase all \
  --deadline-utc 2026-10-22T02:16:28Z
```

Only the parent orchestrator launches the paid GPU experiment; this design task did not issue model requests. Real execution adds the controller's `--execute` option after servers are ready. Fresh run names are required; the wrapper deliberately refuses unreviewed resumption or replacement of its provenance files.

Report paired policy-seed comparisons against the ordinary neutral-prompt evolution runs. Do not pool all four policies as this arm's comparator, and do not count the sampler arm as an independent test of prompt diversity.

Validation includes exact probability checks, ties/singleton/invalid-input handling, invariance to fitness rescaling, active and inactive quantile gates, seeded repeatability, unchanged island choice, and the unchanged within-group softmax. An end-to-end synthetic HTTP test exercises the real wrapper, task and spawned candidate evaluator across three rounds, including one completely invalid round; it is an interface check, not a writer-quality experiment.
