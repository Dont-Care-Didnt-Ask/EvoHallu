# Move existing feedback controls earlier

This is a scheduling change only: execute the **same three original fixed-NLL feedback jobs** while the remaining adaptive jobs evaluate candidates on CPU. It creates no new requests, prompts, model settings, data splits, selections, job IDs or scientific arm. The unchanged source manifest still owns those three jobs in analysis. **Do not merge this subset as new analysis identities.**

`results/next_experiments/feedback_overlap_manifest.json` contains byte-equivalent original job dictionaries for seeds 2107/4319/8741, their original configuration hashes/log paths, and the original source-manifest SHA. Each job still uses 30 rounds × 2 requests, neutral prompt policy, fixed-NLL context, one evaluation worker, 180-second local request timeout and 4,096 output cap. Only the time at which these 180 already-budgeted requests run changes.

The additive launcher uses a separate `feedback_overlap_state.json`. Before execution it verifies the source controller is active on ports 18000/18001, has not reached feedback, and still has at least two unstarted adaptive jobs. Rank and interface overlap controllers must be terminal with matching manifest hashes. Thus at most two source + two feedback search jobs share the same two pinned vLLM replicas; each server remains at `max-num-seqs=2`. Historical overlapping rank controls already exercised this resource arrangement. Existing request timeout 180 s is unchanged; doubled queue pressure must remain visible in the original response timing/error ledgers.

## Prevent duplicate execution

Reserve **all three original output directories**, including the queued third job, with `.feedback_overlap_owner.json`. The unchanged main controller sees marker-only or running directories as partial and refuses to launch them. The additive launcher applies a scoped runtime hook only to its own controller's `inspect_job`: an exactly matching owner marker in an otherwise empty directory is pending. Unknown files/changed ownership and real partial runs keep their normal refusal. The trial runner permits metadata markers and still refuses an existing candidate ledger; it is unchanged.

The normal case is that all three feedback jobs finish long before the adaptive batch. The main controller then finds complete summaries/ledgers and skips their launch. If it reaches still-owned feedback earlier, its refusal may stop orchestration; `overnight_followups.wait_source` now observes the separately pinned overlap controller's terminal evidence **before** checking all 19 original ledgers or rotating models. The completed union is accepted only if every original job has its exact full budget and frozen estimator. Partial results remain partial. Recorded source-state orchestration errors are retained; no error or artifact is erased.

Merely preparing/uploading the subset does not block followups when no overlap execution state or active receipt exists. Failed-before-start marker-only reservations are removed; real partial/completed runs retain owner metadata. The controller and pinned core files are not edited. Runtime hooks restore in `finally`.

## Expected scheduling effect

The previous central source forecast finished adaptive work around 22:13 and three feedback jobs around 23:14, because the third feedback run occupied an extra wave. Completing those jobs earlier could remove roughly 45–60 minutes of the late tail. Queue contention may slow adaptive generation, so the net benefit is not guaranteed. No source run is shortened or dropped, and the 45-minute full-nine feature reservation remains. The user’s novelty preference flag and the existing 100-minute 27B launch gate remain unchanged.

## Prepared commands and activation

Preparation and controller dry plans were checked without launching inference:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_feedback_overlap.py --prepare
```

The orchestrator may explicitly execute:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_feedback_overlap.py --execute
```

Before activation, restart only the **verified waiting followups process** so it loads the new terminal reconciliation, preserving `--reserve-transfer-minutes 45` and `--prioritize-novel-writer-after-utc 2026-10-21T22:45:00Z` as separate flag/value arguments. Do not restart the active source controller or change its manifest. This preparation task did not launch or restart anything. Per-run metadata remains in the original folders; scheduling receipt is `feedback_overlap_receipt.json`.

Validation includes exact-subset/idempotence checks, all-three reservation/queued-job duplicate protection, restored hook and unchanged partial protections, an actual unchanged controller completing three synthetic standard-library child jobs, source 19 reconciliation only after aux termination, prepared-but-unlaunched behavior, and refusal when a prior overlap owner remains active. Machine-readable results and hashes are in `validation.json`; synthetic child results are not model experiments.

**29 tests passed in 6.59 seconds.** All three Hydra configurations resolve to the original60-request budgets,180-second request timeout and one evaluation worker; pinned source/config hash mismatches:0. The parallel controller accepts the exact subset in dry-run mode.
