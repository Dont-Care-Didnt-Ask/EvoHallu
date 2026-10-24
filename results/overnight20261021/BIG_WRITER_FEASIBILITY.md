# GPT-OSS-120B engineering feasibility on two A100 80GB GPUs

**Read-only audit, 21 October 2026. Future allocation only.** No model weights were downloaded, no model or GPU kernel was loaded, and no environment or running server was changed. This establishes a plausible serving path, not measured speed, code-generation quality or a successful model load.

**Conclusion:** the installed **vLLM 0.31.0 has an A100-compatible path for the official MXFP4 checkpoint**. One A100 80GB is plausible for a bounded-context pilot; **tensor parallelism across two cards provides substantially more memory margin**. Neither configuration needs a full BF16 conversion merely because A100 lacks native FP4 arithmetic. The initial future comparison should measure TP1 and TP2, then choose on successful correctness checks and actual cost per completed candidate. There is no basis here to claim that TP2 is twice as fast or more economical.

## Checkpoint and architecture

The official model is **openai/gpt-oss-120b**, about **117B total / 5.1B active parameters**, with an Apache 2.0 license. All experts remain part of the model's resident weights: active parameters describe routing computation, not the memory needed to store the model. The checkpoint uses the model's native MXFP4 expert weights; this is the released representation, not a new third-party quantization. [Official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-oss-120b), [official model card](https://huggingface.co/openai/gpt-oss-120b).

| Verified metadata | Value |
| --- | --- |
| Public checkpoint revision at inspection | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| Access | Public; API reports `private=false`, `gated=false` |
| Architecture | `GptOssForCausalLM`; 36 layers; hidden size 2,880 |
| Experts | 128 per layer; four routed per token |
| Attention | 64 query heads, eight KV heads, head dimension 64; alternating full/sliding layers; sliding window 128 |
| Configured maximum positions | 131,072; a pilot need not allocate that context |
| Quantization config | `mxfp4`; attention, router, embedding and output head excluded |
| Indexed HF weight files | **15** root safetensors shards, named `00000` through `00014` |
| Actual weight-file bytes | **65,248,893,184 bytes = 65.249 GB = 60.768 GiB** |
| Indexed tensor payload | 65,248,815,744 bytes; file headers account for the small difference |
| Other repository weight layouts | `original/` and `metal/` are alternative copies, not additional tensors required by the root HF index |
| Full repository size | Approximately **195.764 GB / 182.319 GiB**; avoid an unfiltered snapshot download |

The revision, required shard list, publisher LFS SHA-256 values, sizes, configuration and tokenizer/template metadata are retained in [checkpoint_summary.json](big_writer_feasibility/checkpoint_summary.json), [API response](big_writer_feasibility/checkpoint_api.json) and [retrieval receipt](big_writer_feasibility/retrieval.json). Required shard identities come from the pinned [index](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/model.safetensors.index.json), matched against [publisher file metadata](https://huggingface.co/api/models/openai/gpt-oss-120b?blobs=true). **These are publisher-provided weight hashes, not locally verified weight downloads.** Configuration details come from the pinned [config](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json).

## A100 support: the backend distinction matters

The current official vLLM recipe explicitly describes A100 operation using **TRITON_ATTN plus Marlin MXFP4 MoE**, including GPT-OSS-120B on one 80GB A100 and TP2/TP4 alternatives. This is weight-only FP4 serving with supported arithmetic on Ampere. It should not be confused with hardware-native FP4 tensor operations on newer GPUs, or a different inference backend's hardware restrictions. [vLLM GPT-OSS recipe, A100 section](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md#A100).

Static inspection of the **actual installed** package confirms the relevant route:

| Installed code | Evidence |
| --- | --- |
| `quantization/mxfp4.py`, lines 49–143 | Minimum capability **80**; supported activation dtype BF16; GPT-OSS checkpoint `mxfp4` mapped to `gpt_oss_mxfp4`. |
| `fused_moe/oracle/mxfp4.py`, lines 342–364 and 534–650 | GPT-OSS backend list includes Marlin; candidates are filtered by their device/config support. The code also permits explicit backend selection. |
| `fused_moe/experts/marlin_moe.py`, lines 560–645 | Marlin supports `kMxfp4Static`, W4A16 operation and CUDA capability compatible with SM80. |
| `quantization/utils/marlin_utils_fp4.py`, lines 560–657 | Expert weights are repacked for Marlin; block size is 32. Scales are temporarily converted during transformation, then returned to float8 by `mxfp4_marlin_process_scales`. No full-model BF16 weight conversion is required by this route. |
| `models/gpt_oss.py`, lines 800–817 | Explicit per-rank MXFP4 block-aligned loading exists. |
| CUDA MoE extension | Installed `_moe_C_stable_libtorch.abi3.so`, 45,448,928 bytes; SHA `65be3121d8ae4008b52e958622dff7f9fdf2747652d4eb9527b08faefb825015`; contains an `sm_80` string. |

Exact installed-source hashes and line excerpts are in [installed_backend.json](big_writer_feasibility/installed_backend.json) and [hardware_and_padding.json](big_writer_feasibility/hardware_and_padding.json). The binary presence/string check is limited static evidence: it does not execute or prove correctness of a particular kernel. `cuobjdump` is unavailable, and no binary/kernel smoke test was attempted. The live documentation can change and differs in backend-priority details from this installed release; the installed files are the audit reference. See also the upstream [MXFP4 configuration API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/mxfp4/).

## Memory estimates: TP1 plausible, TP2 more forgiving

Both current devices report **NVIDIA A100-SXM4-80GB, 81,920 MiB, capability 8.0**. Their topology reports **NV12** between the cards. That is favorable connectivity for a TP2 pilot; no collective bandwidth or generation throughput was measured.

The compressed checkpoint is 60.768 GiB, but **file size is not peak VRAM**. Installed Marlin code rounds the hidden dimension to a multiple of 256 and each rank's expert intermediate dimension to a multiple of 128. Using the checkpoint dimensions, the following are illustrative weight-plus-scale calculations:

| Serving case | Padded expert dimensions per rank | Approximate resident weight/scale estimate | Gross space remaining inside a 90% × 80GiB budget |
| --- | --- | ---: | ---: |
| One card, TP1 | hidden 3,072; intermediate 2,944 | **65.894 GiB** | **6.106 GiB** |
| Two cards, TP2 | hidden 3,072; intermediate 1,536 | **34.292 GiB per card** | **37.708 GiB per card** |

These calculations assume the documented padding route and evenly divide non-expert checkpoint payload for TP2. They exclude replicated/padded small tensors, allocator overhead, CUDA contexts, activations, graph buffers, scratch and transient loading/repacking allocations. They are **not a measured VRAM requirement or a guarantee that TP1 initializes at 90% utilization**. The exact arithmetic and exclusions are saved in [memory_estimates.json](big_writer_feasibility/memory_estimates.json).

For scale, a conservative BF16 KV calculation retaining all 36 layers as full attention uses **73,728 bytes per cached token**, or **1.125 GiB for one 16K sequence**, before TP sharding. Two such sequences are 2.25 GiB. Hybrid sliding-window management can reduce retained KV; engine allocation and padding still need measurement. Begin with one sequence and a bounded context rather than the checkpoint's 131K maximum.

A hypothetical uncompressed BF16 representation of the reported logical parameter count is roughly **217.611 GiB for weights alone**. That exceeds the two cards' combined 160 GiB. Therefore “fall back to all BF16” is not a viable fully resident substitute on this pair; offloading would be a separate, unmeasured configuration. The native Marlin path preserves the reason this model is feasible here.

## Existing environment and remaining engineering work

The inspected environment has **vLLM 0.31.0, Torch 2.13.0, Transformers 5.17.0, Triton 3.7.1**, Python 3.12, and driver **595.91.07**. The separately audited runtime establishes Torch's CUDA 13.0 build. Existing successful 9B/other serving does not establish GPT-OSS correctness. [Environment reproduction record](ENVIRONMENT_REPRODUCTION.md).

The `openai_harmony` module and compiled extension **are present**, supplied by distribution **`oss-harmony==0.0.11`**. A lookup for a distribution literally named `openai-harmony` returns absent; that is a packaging-name difference, not evidence that the Harmony module is missing. No parser/encoding call was run in this audit.

Future work still required before a quality experiment:

1. **Authorized selective download:** fetch the 15 indexed root shards plus required config/tokenizer/template files at the pinned revision, verify actual shard hashes, and reserve disk/RAM for loading. Do not download the alternative `original/` and `metal/` copies unnecessarily. No weights or tokenizer payload were fetched here; only small metadata/template/license files were saved.
2. **Fresh bounded server pilot:** use a separate allocation/clean environment or a formally idle approved slot, pin the native MXFP4/Marlin path and TRITON_ATTN, BF16 activations, one sequence and bounded 16K context. Record actual chosen attention/MoE backends, initialization memory, peak memory and logs. TP1 first establishes minimum-card feasibility; TP2 tests the more spacious configuration. Neither command was launched here.
3. **Correct Harmony handling:** preserve native rendering and separate reasoning from final code. Fix reasoning effort explicitly rather than inheriting an unknown default, record stop/finish reasons and total reasoning/output tokens, and ensure a reasoning-only truncation is not treated as a completed code candidate. Start from [OpenAI's implementation-verification guidance](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations). Reasoning and repair calls consume the experiment's budget.
4. **Contract correctness:** complete a small capped set of independent writer requests, verify final-code extraction and the existing Estimator fixtures, then one real training evaluation. Keep source Llama features/tokenizer and `is_hallu` fixed. No benchmark labels are needed for this engineering gate.
5. **Measure economics:** compare TP1 versus TP2 using the same actual EvoHallu prompt/context distribution, end-to-end valid candidates/hour, output tokens, CPU evaluation time and current rental prices. Also compare a newer GPU only after the same checks. There is no speed, cost or eight-hour 1,000-request completion forecast from this static audit.

## Decision for the backlog

**Proceed to a future, separately authorized engineering pilot; do not allocate or launch tonight.** GPT-OSS-120B is a credible >100B reference for the paper-style extension, and current A100 support removes a blanket hardware-compatibility objection. Single-card success, TP2 benefit, writer quality and budget-to-completion remain unmeasured. Use [BACKLOG P0](../../BACKLOG.md#p0--larger-writer-and-a-credible-paper-style-search) for the matched search/selection design after the pilot passes.
