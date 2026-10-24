# Prospective optional Devstral breadth allocation

Recorded before any GLM or Devstral candidate results, on 2026-10-21 at approximately 22:32 UTC. Preparing or downloading a checkpoint does not mean that an experiment executed. The execution receipts and complete candidate ledgers determine actual coverage.

The user prioritized breadth across local code-writing models. With `--prefer-devstral-breadth`, the late controller first attempts the existing GLM-4.7-Flash seed-2107 original/chosen-policy pair. After this pilot, it reserves the otherwise optional GLM seed-4319 allocation for a new Devstral writer only if all of the following hold:

- The native prefetch receipt has status `ready_native_checkpoint`, model `mistralai/Devstral-Small-2-24B-Instruct-2512`, pinned revision `55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128`, and an absolute checkpoint path.
- The separate Devstral launcher exists.
- At least 75 minutes remain before the unchanged 2026-10-22 01:10 UTC writer cutoff. Equality at 75 minutes is eligible.

The reservation depends on identity, readiness, and time. It does not inspect GLM validation quality, held-out AUROC, or candidate validity. Thus a failed GLM startup or pilot can still be followed by Devstral when the same capacity gate holds. If the Devstral reservation is unavailable, the existing GLM seed-4319 rule remains unchanged: both seed-2107 jobs must finish all 60 requests, each must produce a valid candidate, and the time remaining must be at least `max(45 minutes, 1.3 × slower pilot runtime + 5 minutes)`.

GLM servers are stopped before the Devstral launcher is called. The launcher rechecks the native assets, GPU idleness, free ports, and the 75-minute entry gate. It allows at most 10 minutes for startup and requires at least 65 minutes remaining after startup. The two fixed Devstral arms are original and diverse prompts, seed 2107, 60 requests each, at most 60 minutes per job. The diverse policy was selected using mean PopQA validation AUROC across all three complete Qwen3.5-9B source seeds; target or held-out benchmark scores do not select it. This is a writer comparison; the cached source answer/feature model remains Llama3.1-8b-instruct.

The late controller withholds `late_writer_decision.json` until GLM and optional Devstral have reached terminal outcomes and their owned-server cleanup has been attempted. It records Devstral separately from the GLM decision. The persistent CPU driver therefore sees the additional writer manifest before its final scoring flush. Feature extraction starts only afterward and independently requires idle GPUs. Cleanup failures are recorded; a failed GLM cleanup prevents a Devstral launch. If cleanup consumes the remaining reservation window, the Devstral launcher can skip rather than exceed the cutoff.

All existing deadlines remain: writer cutoff 01:10 UTC, feature-transfer cutoff 02:01:28 UTC, CPU finish 02:06:28 UTC, and hard pod deadline 02:16:28 UTC. There is no additional GPU budget. A single seed per new writer is exploratory breadth, not a three-seed prompt-policy estimate. Partial-budget jobs are retained individually and excluded from full-budget aggregate comparisons.

Default invocation without the new flag preserves the previous GLM pilot/extension allocation and GLM-only late-writer marker schema. The opt-in command is:

```bash
.venv/bin/python scripts/overnight_late_extensions.py --execute --prefer-devstral-breadth
```

This document and scheduler preparation do not authorize a duplicate controller. The root controller will verify that the existing late process is still waiting before replacing it and will retain the earlier waiting-process state.

## Prospective GLM completion cap, 22:46 UTC

Before any GLM candidate results, the controller gained a separate optional `--glm-max-job-minutes 60` setting. The default remains 45 minutes. This changes the per-job wall-time ceiling for both possible GLM pairs, while retaining 60 requested candidates, the same immutable configurations/manifests, decoding, search process, selection rules, 01:10 UTC writer cutoff, and Devstral 75-minute entry gate. The existing startup and minimum-entry gates remain unchanged; the cap cannot guarantee completion when the absolute cutoff leaves less time.

The generated GLM configuration has `trial.max_seconds=5400`; the existing parallel controller appends the runtime override `trial.max_seconds=min(job cap, seconds until the writer cutoff minus 45 seconds)`. The actual command and resolved trial configuration record this override. No configuration or manifest rewrite is needed.

The reason is completion coverage, independent of model quality. With the then-current forecast of GLM readiness around 23:15 UTC, a forced 45-minute partial run would end around 00:00, already after Devstral's latest eligible entry of 23:55. Allowing up to 60 minutes can instead yield a full 60-candidate pair around 00:15, while preserving approximately two hours before the hard deadline for feature extraction, CPU scoring, and backup. A fast GLM pair retains the same Devstral opportunity. No GLM throughput has yet been measured. For a given GLM pair that would otherwise reach the 45-minute ceiling, the extra cap is 15 minutes across the existing two GPUs, approximately $0.90 at $3.58 per hour total. Completion can also affect eligibility for the already conditional second pair; that remains governed by measured runtime and the unchanged absolute cutoff. There is no increase to the eight-hour limit.

Opt-in command for both prospective allocation settings:

```bash
.venv/bin/python scripts/overnight_late_extensions.py --execute --prefer-devstral-breadth --glm-max-job-minutes 60
```
