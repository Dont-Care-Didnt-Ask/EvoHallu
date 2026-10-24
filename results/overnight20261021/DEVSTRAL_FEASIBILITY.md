# Optional Devstral writer breadth: read-only feasibility audit

**Recommendation: conditionally feasible without installation or core changes.** A single original/diverse pair, seed 2107, sixty attempted candidates per arm, would add a writer family with no previous successful detector-generation result in this repository. Prefer it to a second GLM seed only if GLM's first pair completes with sufficient time. No model weights were downloaded, GPU/inference process started, package installed or active protocol changed for this audit.

## Verified checkpoint and installed support

- Official model: [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512). Its unauthenticated [official model API](https://huggingface.co/api/models/mistralai/Devstral-Small-2-24B-Instruct-2512?blobs=true) reports `private=false`, `gated=false`, current immutable revision **`55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128`**, matching the existing model configuration and historical provenance.
- Download **one weight format only**. Native Mistral consolidated weights are **25,793,035,888 bytes across two shards**; the prior pinned native-format download including supporting files totals **25,827,009,376 bytes**. The alternative six HF shards total **25,793,059,408 bytes**. Downloading both duplicates the weights. Index `total_size` excludes header differences and is not the actual shard-byte total.
- The [pinned configuration](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/blob/55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128/config.json) declares `Mistral3ForConditionalGeneration`, a Ministral3 text decoder, BF16 activation dtype and static FP8 weight quantization. This is not a new BF16-weight conversion.
- Read-only checks on the active pod confirm **vLLM 0.31.0, transformers 5.17.0 and mistral-common 1.12.0**. Installed source registers `Mistral3ForConditionalGeneration`, `Ministral3ForCausalLM` and `MistralTokenizer`. Its FP8 configuration accepts BF16 activations and minimum capability 7.5; its Marlin path explicitly supports weight-only FP8 compression on GPUs without native FP8 computation. A100 therefore uses the existing Marlin fallback, not native FP8 arithmetic.
- Native `tekken.json`/Mistral-common tokenization and the supplied chat format are supported. The official [model card](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) requests mistral-common ≥1.8.6. The pinned separate Jinja template has system/user/assistant support and no `enable_thinking` switch. Avoid applying Qwen template/thinking kwargs.

Installed-source checks read package metadata and source files only. Historical server logs also show successful model initialization with `MarlinFP8ScaledMMLinearKernel`; actual new native-chat serving still requires a bounded readiness/API check before a full search is trusted.

## Why the earlier attempt failed

The historical Devstral summary contains **six attempted requests, zero valid candidates and approximately eleven seconds elapsed**. Every request failed with HTTP 400: **`chat_template is not supported for Mistral tokenizers`**. This is a request-format failure, not evidence of poor generated detector quality or measured decoding throughput.

The current pinned [runner](../../src/experiments/extension_trial.py) already omits template kwargs when `supports_chat_template_kwargs=false`; the existing [Devstral config](../../configs/model/devstral_small2.yaml) sets that value. No pinned-core fix is required. Do not infer actual request speed from those six failed calls.

## Proposed bounded execution, not launched

Use the inherited hash-verifying downloader with `--revision 55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128 --weight-format mistral`. Prefetch into a separate directory while other writers run, once authorized, without removing their checkpoints or consuming reserved transfer/cache storage. Verify both published shard hashes, native index membership, supporting-file hashes and available storage.

After both GLM pilot arms are terminal and its two owned servers are stopped, require idle GPUs and free loopback ports. Use two independent replicas, one per A100, with the historical supported settings:

```text
--dtype bfloat16 --tokenizer-mode mistral --config-format mistral
--load-format mistral --language-model-only --linear-backend marlin
--max-model-len 16384 --max-num-seqs 2 --enforce-eager
```

Keep Qwen/GLM-specific chat-template flags absent. Preserve the inherited decoder settings, 4,096 output-token cap, original/diverse prompt policies, source Llama3.1 features, training fitness, validation selection, baseline fallback and source hashes. Freeze one separate manifest before either trial, with two fresh run IDs and the native-serving provenance. Use the existing controller/CPU evaluation pattern; do not edit or add methods to an already frozen cohort silently. Any readiness health calls must be explicitly accounted separately from the **120 candidate requests**.

**Capacity gate:** require at least **75 minutes before the fixed 01:10 UTC writer cutoff** when starting the servers, including at most roughly ten minutes for startup and at least sixty-five minutes for the simultaneous pair. A 23:45 completion of the GLM pilot would leave eighty-five minutes. Start servers by **23:55** and require readiness by **00:05** with the full sixty-five-minute search allowance remaining. Preserve the independent full-nine transfer, archive/evaluation and pod-stop windows. A failed startup or insufficient remaining time means skip, not install dependencies or displace another workload.

The sixty-five-minute search estimate is plausible but **unbenchmarked for valid Devstral requests**. Dense FP8/Marlin latency can differ from the current MoE/BF16 writers; the old failed requests provide no usable timing evidence. Apply the absolute controller deadline regardless of estimate, and label any truncated arm exploratory rather than a completed matched pair. This is a proposed breadth allocation based on capacity and user preference, not target/test quality.
