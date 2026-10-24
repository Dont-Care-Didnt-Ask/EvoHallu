# Completing the Llama-3.2-1B transfer benchmark

The six historical target caches cover 56,235 rows. Only the following three datasets need new feature inference:

| Sorted dataset index | Dataset | Rows | Cache file |
| --- | --- | ---: | --- |
| 6 | CounterFact | 31,964 | `geometry__counterfact_true_false.pt` |
| 7 | PopQA test | 6,974 | `popqa__popqa_test_filtered_10shot.pt` |
| 8 | X-Fact | 2,315 | `x-fact__test_filtered_10shot.pt` |
| Total missing | | **41,253** | |

Historical CounterFact inference reached 28,896 rows in 590 seconds before the earlier campaign ended; its final cache was never saved. This projects approximately 11 minutes for CounterFact. The earlier small-model probe measured approximately 42 rows/s, projecting 16.4 minutes for all missing rows. Allow **15–30 minutes on one GPU**, or approximately **11–20 minutes on two GPUs**, with additional time for model download, serialization, CPU scoring and backup. Long X-Fact claims and contention can change these timings. The wrapper has one shared absolute cutoff; its maximum runtime is 45 minutes in serial mode and 35 minutes in two-GPU mode.

## Exact inference protocol

- Target model: `unsloth/Llama-3.2-1B-Instruct`, revision `5a8abab4a5d6f164389b1079fb721cfab8d7126c`.
- Shared source tokenizer: `VityaVitalich/Llama3.1-8b-instruct`, revision `247cb7c3baebabc46cff1fb97a3a5913e4c4511d`. Only its tokenizer/configuration files are downloaded; no 8B model weights are needed.
- Load target weights as **float32**, attention backend **SDPA**. Stored logits/log-probability features remain **float16**, matching the historical precompute code.
- Chat template mode `generate_true_statement`: user message `Generate true statement`, followed by the supplied claim as the assistant message; render with `add_generation_prompt=False`, then apply ordinary tokenization. Preserve the historical formatting, including its observed double BOS and terminal assistant header.
- Right padding; padding token falls back to the source tokenizer's EOS token.
- Forward batch 16; maximum padded forward tokens 16,384; top-k 512; logits-only extraction; 12 CPU feature threads; CPU GC and CUDA cache-cleanup intervals both 32 batches.
- Preserve claim stripping, `is_hallu`, row order, full sequence features, and zero suffix trimming.

Actual historical metadata and the first cached AVeriTeC sample's input token IDs were extracted on CPU into `remaining_transfer_reference_metadata.json`. The script verifies exact chat/raw token IDs against that sample before GPU inference. Historical cache metadata does **not** embed the model revision or tokenizer identity; checkpoint-file hashes and the external protocol supply that evidence.

## Commands

All commands run from the repository root. Default invocation writes a plan and makes no network or GPU requests:

```bash
.venv/bin/python scripts/prepare_remaining_transfer.py
```

Prefetch the pinned 1B model (approximately 2.5 GB) and shared tokenizer (approximately 9 MB), while writers are still running:

```bash
.venv/bin/python scripts/prepare_remaining_transfer.py --download
```

After the orchestrator assigns GPUs with sufficient free memory, run only the missing three caches. The preferred two-GPU mode fixes CounterFact on GPU 0 and PopQA followed by X-Fact on GPU 1. Each GPU runs at most one inference process at a time. No dataset is partitioned, so batches and row order remain unchanged. The example leaves 15 minutes before the hard pod cutoff for scoring/backup; the orchestrator may choose an earlier deadline:

```bash
.venv/bin/python scripts/prepare_remaining_transfer.py \
  --cache-dir /dev/shm/mod1-transfer-cache \
  --execute --gpu-indices 0 1 --max-minutes 35 \
  --deadline-utc 2026-10-22T02:01:28Z
```

The original single-GPU interface is preserved:

```bash
.venv/bin/python scripts/prepare_remaining_transfer.py \
  --cache-dir /dev/shm/mod1-transfer-cache \
  --execute --gpu-index 1 --max-minutes 30 \
  --deadline-utc 2026-10-22T02:01:28Z
```

`--gpu-index` and `--gpu-indices` are mutually exclusive. Both GPU modes share the supplied deadline; parallel lanes do not each receive a separate time budget. A child failure or deadline cancels the other active child and prevents queued work from launching. Workers always reap their own process groups. Completed caches are recorded as they finish, using a single main-thread receipt writer and atomic replacement. The receipt schema and three-cache completion status are unchanged; `gpu_indices` and `execution_mode` are additional provenance fields.

Model and tokenizer directories can be changed with `--model-dir` and `--tokenizer-dir`. The model's path string in new cache metadata can differ from the historical path while the verified checkpoint bytes remain identical. Writer servers on the selected GPUs may need to be stopped to release their reserved VRAM; this script does not stop servers or pods.

The script calls the existing `evaluate_cached_claims.py --mode cache --dataset-index ...` separately for indices 6, 7 and 8. It does not call the historical `transfer_feature_model.py` wrapper, rotate any model directory, overwrite the six completed caches, or select formulas using target quality.

## Historical/current runtime numerical preflight

Before any missing-cache job starts, execution now recomputes **all 427 existing AVeriTeC rows** with the same pinned checkpoint/tokenizer, float32 SDPA forward pass, batch settings, source formatting, and float16 feature storage. It uses the first assigned GPU and a separate uniquely named scratch directory beside the target cache directory. The reference GPU child is capped at **180 seconds**, inside the existing shared 35-minute two-GPU (or 45-minute serial) cutoff. It does not replace or modify the historical AVeriTeC cache. The new scratch cache is disposable and excluded from the three-cache archive helper; only its numerical JSON receipt needs backup.

This is necessary because the historical six caches and newly generated three may use different PyTorch/backend versions despite identical model bytes and settings. Exact identity, shapes, dtypes and finite values are required for every sample, full/raw token IDs, attention masks, observed log-probabilities, greedy log-probabilities, entropy, and top-k logits/IDs. Labels are compared only as a row-identity field. No AUROC or other label-quality metric is computed for this guard.

The **predeclared gross-drift limits**, chosen before this real preflight, are observed-token log-probability maximum absolute difference **≤0.1** and canonical NLL Spearman rank correlation **≥0.999** across all 427 rows. Canonical NLL preserves the historical float16 accumulation followed by float32 output. The tolerance allows small changes from stored float16 quantization and backend/kernel arithmetic; it is a coarse corruption/backend-drift check, not proof of bitwise equivalence or comprehensive cross-version equivalence. Any missing, nonfinite, misaligned or incorrectly typed field fails regardless of those two limits.

The receipt additionally describes maximum/mean/99th/99.9th-percentile differences in raw top-k logits by sorted rank, observed/greedy log-probabilities, entropy, and float64-recomputed top-512-conditional top1 mass, top10 mass and entropy. Top-k rank-slot/top1 ID matches and complete-ID-row matches are descriptive because near ties can reorder IDs; they are not an equality gate. Package versions, exact commands, cache hashes, GPU, runtime and unchanged historical-cache evidence are recorded. No threshold is changed using test quality.

A failed numerical guard or child timeout stops the missing-three sequence through the existing failure path; the orchestrator can retain the completed historical six-dataset transfer result. The full-nine transfer evaluator now requires a passed 427-row numerical-reference receipt, verified identity/finite checks, and both measured numerical bounds in addition to all previous checkpoint/tokenizer/cache pins. Six-dataset coverage keeps its existing requirements.

## Artifacts and execution safeguards

- `remaining_transfer_plan.json`: exact checkpoints, download hashes, preserved cache identities, commands and estimates.
- `remaining_transfer_download.json`: verified model/tokenizer asset hashes and tokenization parity; no inference claim.
- `remaining_transfer_completion.json`: incremental receipt with each completed cache's SHA-256, rows, CSV hash, metadata, shape/dtype checks and command timing. All six historical raw hashes are checked before and after execution.
- `remaining_transfer_numerical_reference.json`: bounded historical/current-runtime preflight and numerical comparison; its complete pass/fail record and SHA are embedded in the completion receipt.
- `numerical_reference/<attempt>/transfer_completion_logs/dataset_0.log`: isolated AVeriTeC recomputation log; no GPU inference runs during planning or download-only mode.
- `transfer_completion_logs/dataset_{6,7,8}.log`: actual inference logs.

All six restored historical target caches must match `results/feature_cache_validation.json` before execution. Existing newly completed caches are reused only when their own receipt and raw hash match; unknown cache files and partial files cause a refusal rather than replacement. A timeout records partial completion. Do not report a nine-dataset target macro until all nine caches have been validated and scored using the same frozen source-selected formulas.

Estimated additional uncompressed feature storage is approximately **6.75 GB**, beyond the existing six caches and the model download. The model must run alone or with enough unreserved VRAM on its assigned GPU; two 9B vLLM replicas can otherwise reserve most of both devices.

## Validation state

Twelve CPU tests passed in 10.58 seconds. They cover pin/file selection, absence of source-model weight downloads, exact missing-index command settings, existing-file hash protection, actual mmap cache metadata/row checks, and default plans that cannot download or start a GPU process. Real synthetic subprocesses additionally verify simultaneous GPU routing, one process per GPU, PopQA-before-X-Fact ordering, main-thread completion callbacks, shared-deadline cleanup, failed-lane cleanup, and refusal of conflicting/duplicate GPU options or a two-GPU cap above 35 minutes. These subprocesses use only Python's standard library and do not perform GPU inference.

The added numerical-reference and full-nine completion-gate regression suite now totals **21 passing CPU tests in 15.50 seconds** across the preparation and transfer-provenance test files. Synthetic cache fixtures verify small quantization differences, descriptive top-k ID changes, gross-drift rejection, exact input/shape/dtype/finite checks, isolated first-GPU routing with the shared cutoff, pass/fail receipts, missing/failed reference rejection for full-nine coverage, and unchanged six-dataset compatibility. The real 427-row GPU preflight has not been run during preparation.
