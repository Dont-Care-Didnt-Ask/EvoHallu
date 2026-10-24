# Prospective frozen-normalization feature-model transfer

Prepared only. Execution requires root review and `--execute`; no writer/GPU
calls, feature extraction, target fitting, or source-artifact changes occur.

## Fixed question and detectors

Carry the already selected C8741 program and already fitted source-normalization
state to existing Llama3.2-1B features, using the shared original source tokenizer.
Retain three detectors: original C8741 on original-order 512-row scoring chunks;
source-fixed median/MAD C8741; canonical SequenceLogProb NLL. No variant is chosen
using target quality. Report negative results equally.

Original class SHA: `a96fdf5af49a1a6f0f6426d79e2ef60b596be1d7ffd38a672afb6338523cc884`.
Reusable inference helper SHA: `f2f92a85857801c301e1383fabde5b4903c68a401f46d7a6dbb1a968a7fdc89d`.
Saved normalization SHA: `e83b7b1e83b6a2e347f0d080bf50bb5d50d67b6ae7f578ce40251f5e94340341`.

The adapter imports `ProxiedEstimator` and `parent.score_case`, never the source
helper's `fit()` or CLI. The source helper/PLAN/report/fit receipt, exact source
training parameters, original program, source predictions, core evaluator and
feature builder are SHA-pinned. Original, fixed-proxy and transparent-proxy
candidates are compiled into separate namespaces: swapping `np` for the proxy
cannot affect the original detector's globals.

## LP-only gate and target integrity

Static AST inspection verifies that the frozen class reads only
`sequence_logprobs`. Before any target scoring, all nine source datasets are
scored through the existing LP dependency path. Original512, fixed-control512
and canonical NLL predictions, labels, claim groups and strict masks must equal
the previously saved source arrays byte-for-byte. A transparent proxy must also
reproduce the original output exactly. Failure stops the study; no source state
is repaired or refitted.

The six target caches must pass the existing provenance auditor: every raw SHA,
CSV claim/label/order, pinned target model/shared tokenizer, float32-forward
architecture metadata and source/target chat/raw token IDs match. Original512
and transparent-proxy target predictions must agree exactly. The fixed control
is scored in all four prespecified cases: chunks1/128/512 and shuffle451/chunks512
with original row positions restored. All four fixed-score arrays and all NLL
arrays must be byte-identical. Only the original512 is used for original C8741.

Target-six NLL predictions and all 500 full-claim bootstrap values must match the
completed earlier cohort exactly. No approximate parity is accepted. The source
zero-MAD rule keeps f3/f4 disabled even if the target feature distribution differs.
No target medians, MADs, coefficients, thresholds or sign selection are fitted.

## Six first, three-only extension

Run the historical six-dataset cohort first. Its predictions, bootstraps, audit
and report are sealed in `SIX_IMMUTABLE.json`. The nine-dataset extension copies
the six prediction/result/bootstraps byte-identically into a new folder and
scores **only indices6/7/8: CounterFact, PopQA, X-Fact**. Its gate requires the
formal `complete_three_missing_caches` receipt, passed full427-row numerical
reference, all six historical caches unchanged, pinned model/tokenizer hashes
and tokenization parity, all nine cache files and exact source/target row IDs.
The earlier standalone numerical pilot alone does not satisfy this gate.

If the main full-nine transfer NLL reference is still being written, new scores
may finish but the target-nine macro remains pending. `--report-only` can later
verify those references without rescoring. An external NLL/original-C8741
mismatch invalidates the study. Partial coverage never supplies a shorter macro.
The original six report is never refreshed or overwritten after sealing.

## Statistics and limits

Use 500 deterministic claim-group resamples shared across methods and source/
target. Full draws use the existing dataset seed `441+SHA256(name)[:8]`; strict
draws add1000003 and remap the retained groups contiguously. Report directed and
per-dataset-folded AUROC, full/strict matched-six and matched-nine macros, paired
fixed-minus-original/NLL intervals, source-target deltas and NLL-relative gain
changes. Missing a prescribed dataset blocks that macro. Intervals condition on
the fixed program, training statistics and dataset set, not search history.

This is exploratory after source benchmark viewing; no original primary is
replaced. The frozen program effectively combines NLL/local variance while the
other branches remain zero. Since medians shift an offset, AUROC does not prove
calibration transfer. Shared-tokenizer transfer is not native-tokenizer transfer.

## Execution bounds and review commands

Four CPU subprocess workers, one thread each, GPU visibility empty. A shared
**1,200-second cumulative active-work budget** covers six and nine invocations;
waiting for formal completion does not spend active-work budget. Hard cutoff:
**22 October2026 02:03 UTC**. Watchdog/manual cancellation stops only owned CPU
process groups. No pod/server shutdown, GPU setup or automatic extraction exists.

Preparation only:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/overnight_normalization_transfer.py
```

After review, on the pod:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/overnight_normalization_transfer.py --execute --coverage six
PYTHONPATH=src:. .venv/bin/python scripts/overnight_normalization_transfer.py --execute --coverage nine --wait
PYTHONPATH=src:. .venv/bin/python scripts/overnight_normalization_transfer.py --report-only --coverage nine
```

These are proposed commands, not evidence of execution. Default preparation
never launches scoring. Nine without `--wait` returns a waiting status immediately
until the formal numeric/completion receipt is available.
