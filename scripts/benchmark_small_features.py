"""Time the optional feature model using training claims, without quality tuning."""
import gc
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packing.evaluate.uncertainty_quantification.precompute_raw_stats_separated import _precompute_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended-cleanup", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("out/campaign/small_feature_probe.json"))
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(12)
    model_id = "/dev/shm/mod1-models/Llama3.2-1B"
    tokenizer_id = "/workspace/mod1-models/llama31"
    source = AutoTokenizer.from_pretrained(tokenizer_id)
    target = AutoTokenizer.from_pretrained(model_id)
    if source.get_vocab() != target.get_vocab():
        raise ValueError("The shared-tokenizer vocabularies differ")
    source.padding_side = "right"
    source.pad_token = source.pad_token or source.eos_token
    df = pd.read_csv("data/claims/popqa/popqa_train_filtered_10shot.csv", usecols=["claim", "is_hallu"])
    indices = np.random.default_rng(441).choice(len(df), 256, replace=False)
    df = df.iloc[indices].copy()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="sdpa").eval().to("cuda:0")
    padded_tokens = []
    hook = model.register_forward_pre_hook(lambda module, inputs, kwargs: padded_tokens.append(kwargs["input_ids"].numel()), with_kwargs=True)
    original_collect = gc.collect
    gc_seconds = 0.

    def collect(*args, **kwargs):
        nonlocal gc_seconds
        started = time.monotonic()
        result = original_collect(*args, **kwargs)
        gc_seconds += time.monotonic()-started
        return result

    gc.collect = collect
    started = time.monotonic()
    try:
        raw = _precompute_split(df, source, model, "claim", "is_hallu", None, 16,
                                "generate_true_statement", 512, 1, modalities="logits", max_batch_tokens=16384)["logits_struct"]
    finally:
        gc.collect = original_collect
        hook.remove()
    elapsed = time.monotonic()-started
    started = time.monotonic()
    optimized = _precompute_split(df, source, model, "claim", "is_hallu", None, 16,
                                "generate_true_statement", 512, 1, modalities="logits",
                                max_batch_tokens=16384, gc_interval=32)["logits_struct"]
    optimized_elapsed = time.monotonic()-started
    identical = all(np.array_equal(np.asarray(a[key]), np.asarray(b[key]))
                    for a, b in zip(raw["samples"], optimized["samples"])
                    for key in ["input_ids", "sequence_logprobs", "greedy_logprobs", "token_entropies", "top_k_logits", "top_k_indices"])
    report = {"model": "unsloth/Llama-3.2-1B-Instruct", "revision": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
              "shared_token_to_id_vocabulary_identical": True, "rows": len(df), "seconds": elapsed,
              "claims_per_second": len(df)/elapsed, "cpu_gc_seconds": gc_seconds,
              "padded_forward_tokens": sum(padded_tokens), "forwards": len(padded_tokens),
              "row_rate_projection_for_97488_minutes": elapsed/len(df)*97488/60,
              "projection_limitation": "Short training claims only; longer held-out claims can take more time",
              "metadata": raw["meta"], "held_out_quality_metrics_computed": False}
    report.update(gc32_seconds=optimized_elapsed, gc32_claims_per_second=len(df)/optimized_elapsed,
                  gc_schedule_features_identical=identical,
                  gc32_row_rate_projection_minutes=optimized_elapsed/len(df)*97488/60,
                  timing_limitation="The first pass also includes CUDA warmup; this is an operational timing probe")
    if args.extended_cleanup:
        started = time.monotonic()
        pooled = _precompute_split(df, source, model, "claim", "is_hallu", None, 16,
                                   "generate_true_statement", 512, 1, modalities="logits",
                                   max_batch_tokens=16384, gc_interval=32, cuda_cache_interval=32)["logits_struct"]
        pooled_elapsed = time.monotonic()-started
        same = all(np.array_equal(np.asarray(a[key]), np.asarray(b[key]))
                   for a, b in zip(raw["samples"], pooled["samples"])
                   for key in ["input_ids", "orig_input_ids", "sequence_logprobs", "greedy_logprobs", "token_entropies", "top_k_logits", "top_k_indices"])
        report.update(cuda32_seconds=pooled_elapsed, cuda32_features_identical=same,
                      cuda32_row_rate_projection_minutes=pooled_elapsed/len(df)*97488/60)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
