# Overnight scheduling audit

Read-only remote snapshot: **21 October 2026, 19:55:35 UTC**, with storage/process checks through 19:56:59. No inference, process changes, execution-manifest edits or configuration changes were made. Runtime evidence and historical ledger hashes are in `schedule_audit_evidence.json`.

## Current capacity and remaining source work

- Main source suite: **6/19 jobs complete**, two running, eleven unstarted. All four 30-request screens and original/neutral seed 2107 evolution are complete; diverse is 52/60 and compositional 22/60. **706 of 1,020 source requests remain.** The complete suite retains all twelve evolution runs and all three fixed-feedback controls.
- All **three rank-sampler runs are complete** at 60 requests each. The separate interface pilot is 52/60; its shell is scheduled to restart followups after its controller exits. The old followups PID is intentionally stopped, not evidence of an accidental crash.
- Across source/rank/interface: **546 persisted requests, 1,654,371 reported tokens** at the snapshot. Elapsed rental since 18:16:28 is approximately 1.65 h, or **$5.91** at the recorded total $3.58/h. The eight-hour compute ceiling remains $28.64; storage/network are excluded.
- Both actual servers are pinned 9B BF16 replicas with 16K context and `max-num-seqs=2`. Snapshot utilization was 100%/0%; this is a single sample. Serial candidate evaluation causes compute/generation bursts, so allocated VRAM alone does not establish GPU utilization. CPU work has no recorded driver failures; five jobs had completed source evaluation. Earlier sampled GPU utilization and the source's overlapping rank/interface work make naive linear extrapolation unreliable.
- Both optional checkpoints are already present: 27B: 18 safetensor shards and GLM: 48 shards, with matching model/revision download manifests. Available space was approximately 27 GiB on `/workspace` and 176 GiB in `/dev/shm`. The missing features need roughly 6.75 GB plus a disposable AVeriTeC numerical-reference cache; checkpoint download is not on the late critical path.

Using actual two-port scheduling, the current active jobs' recent round times, original 40.54 min / neutral 30.30 min / diverse 24.20 min / compositional 28.62 min for unstarted 60-request jobs, and neutral 30.30 min as the unobserved fixed-feedback estimate:

| Scenario for unstarted jobs | Full source suite finishes |
| --- | --- |
| 20% faster |22:38 UTC |
| Observed policy times |**23:14 UTC** |
| 25% slower |00:03 UTC |

These are scheduling scenarios, not statistical confidence bounds. They include the controller's evolution/feedback batch barrier and the final third feedback job running alone. No further optional 9B arm is assumed. Source completion remains comfortably before 01:46:28 in these scenarios, but both additional writers are not guaranteed.

## What the gates actually permit

27B generation must end by **01:01:28**, preserving the 45-minute feature reservation relative to the main 01:46:28 cutoff. The four-job 170-minute gate requires servers ready by **22:11:28**; the two-job 100-minute gate requires ready by **23:21:28**. Both the pre-start and post-start checks apply. Under the central source forecast, only about seven minutes remain for preparation/startup before that latter gate closes.

The closest historical evidence is materially slower than 9B:

| Historical 27B run | Requests | Total wall time | Median request generation | Sum of per-round maximum generation |
| --- | ---: | ---: | ---: | ---: |
| evolution 1047 |60 |**74.28min** |112.49s |64.67min |
| evolution 1048 |60 |**81.25min** |110.24s |61.54min |
| independent with interface contract 1048 |60 |47.79min |67.46s |40.33min |

All used the same 4,096 output cap as tonight. The independent run is a different prompt regime and should not determine the evolution launch floor. Evolution 1047 already used two evaluation workers; tonight uses one. Historical 27B used **eager BF16, 16K, max-num-seqs=2** on one A100-SXM4-80GB. Its recorded startup took **135 s**. Tonight's compiled 9B startup also took approximately **137 s**, with engine warmup 98 s and reported compilation only 0.46–0.48 s. There is no measured compiled 27B throughput or evidence that eager-first saves meaningful wall time; retaining compiled-first with the existing startup-failure fallback is justified.

**Keep the current 100-minute 27B gate for now.** A 65-minute floor is below the historical generation-plus-evaluation evidence, and 75 minutes already fails the slower completed evolution run. A 95-minute ready floor would be a narrow capacity-only relaxation consistent with the configured 90-minute job cap plus five minutes of margin, but does not solve the main breadth constraint and is not recommended without need.

GLM can start only while at least 55 minutes remain before **01:10**, so its entry gate closes at **00:15**. Startup must still leave 45 minutes. Its first pair consists of original/chosen policies at seed 2107, **60 requests each**, concurrently; a second seed is allowed only by recorded full-budget pilot validity and throughput. No test-quality threshold controls extension. With central source timing, a historical-speed 27B pair would finish around 00:38 and thereby eliminate the GLM pilot. Lowering the 27B gate cannot create the missing time.

## Breadth recommendation and dependencies

If source completion is later than approximately **22:45–22:50**, prefer the **novel GLM pair** over repeating 27B, while keeping the complete source 12 + feedback 3 + rank 3 suite unchanged. Earlier completion can leave room for historical-speed 27B plus the 55-minute GLM entry window. This prospective allocation rule uses time and model novelty only. It must not depend on benchmark quality, and should not shorten either writer's 60-request protocol. No extra 30-request 27B arm is needed.

GLM has **no execution dependency on 27B**: its preparer imports the shared `inspect_discovery` function, reads the original 9B manifest and all twelve full-budget source ledgers, and selects the non-original policy by the predeclared three-seed mean source-validation rule. It does not require a 27B confirmation manifest or 27B results. A skipped 27B stage can publish `generation_complete.json`, after which the existing late controller can prepare/run GLM normally. If a breadth-first skip rule is implemented, prospectively correct the GLM preparation metadata `optional_schedule.only_after_27b=true`; it currently describes order rather than an enforced completed 27B dependency. No GLM manifest existed at the remote snapshot.

Preserve at least 45 GPU minutes for full-nine feature work. GLM's 01:10 stop leaves 51 min 28 s before the 02:01:28 feature cutoff, including the new ≤180 s reference pass inside the shared 35-minute extraction cap. CPU completion is 02:06:28 and the hard rental cutoff 02:16:28. Required checks before handoff: interface shell really restarts followups; source/rank controllers terminate with matching manifest hashes and full budgets; managed writer PIDs/ports are cleared before GLM/1B; the 427-row numerical pass succeeds before missing caches; nine-coverage evaluation checks its embedded pass and falls back to six on failure. The completion marker certifies controller termination, so its outcome/full-budget evidence must accompany claims about source success. `coverage_from_receipt` is less strict than the downstream nine-cache audit; the latter still rejects a missing/failed numerical pass. No current such receipt exists.

Increasing source evaluator concurrency would require explicit provenance and an implementation change; active pinned configurations should not be silently edited. The existing CPU driver intentionally waits for both the generation and late-writer markers before final completion. A missing marker can therefore suppress late feature work even if GPUs are idle: verify those handoffs rather than weakening identity, safety, or full-budget gates.
