from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import os
from pathlib import Path
import platform
import time

from .cache import DatasetCache, write_shard
from .config import DATASET_ORDER, MODELS
from .data import load_datasets
from .features import features, render
from .io import sha256, write_json


def infer_shard(model, tokenizer, tokens, batch_size, device="cuda:0"):
    import torch

    order = sorted(range(len(tokens)), key=lambda index: len(tokens[index][0]))
    values = [None] * len(tokens)
    cursor, reductions, shapes = 0, 0, []
    started = time.monotonic()
    while cursor < len(order):
        chosen = order[cursor:cursor + batch_size]
        inputs = outputs = None
        batch_values = []
        did_oom = False
        try:
            inputs = tokenizer.pad(
                {"input_ids": [tokens[index][0] for index in chosen]},
                padding=True, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = model(
                    **inputs, use_cache=False, output_hidden_states=False, output_attentions=False
                )
                for row, index in enumerate(chosen):
                    ids, targets = tokens[index]
                    batch_values.append(features(
                        outputs.logits[row, :len(ids)], inputs["input_ids"][row, :len(ids)], targets
                    ))
            for index, value in zip(chosen, batch_values):
                values[index] = value
            shapes.append([len(chosen), int(inputs["input_ids"].shape[1])])
            cursor += len(chosen)
        except torch.cuda.OutOfMemoryError:
            did_oom = True
            batch_values.clear()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            reductions += 1
        finally:
            del inputs, outputs, batch_values
        if did_oom:
            gc.collect()
            torch.cuda.empty_cache()
    return values, {
        "compute_seconds": time.monotonic() - started,
        "effective_batch_size": batch_size, "oom_reductions": reductions,
        "batch_shapes": shapes,
    }, batch_size


def extract_dataset(cache, model, tokenizer, tokens, batch_size, shard_size, writer_threads):
    pending = deque()

    def commit():
        cache.commit(pending.popleft().result())
        print(f"{cache.directory}: {cache.cursor}/{len(cache.rows)}", flush=True)

    with ThreadPoolExecutor(max_workers=writer_threads) as writer:
        try:
            for start in range(cache.cursor, len(cache.rows), shard_size):
                stop = min(start + shard_size, len(cache.rows))
                values, compute, batch_size = infer_shard(
                    model, tokenizer, tokens[start:stop], batch_size
                )
                pending.append(writer.submit(
                    write_shard, cache.directory, cache.rows[start:stop], values, start, compute
                ))
                if len(pending) >= 2 * writer_threads:
                    commit()
                while pending and pending[0].done():
                    commit()
        finally:
            while pending:
                commit()
    return batch_size


def extract(args):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import tokenizers
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Extraction requires one visible CUDA GPU")
    datasets = load_datasets(args.data_archive, args.datasets, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "dtype": args.dtype, "batch_size": args.batch_size, "shard_size": args.shard_size,
        "pilot_only": args.limit is not None or args.dtype != "bf16",
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "sources": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("features.py", "extract.py", "cache.py", "data.py", "protocol.json")
        },
    }
    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "model_dtype": args.dtype,
        "probability_reductions_dtype": "float32", "attention_backend": "eager",
        "top_k_values_dtype": "float16", "generation_calls": 0, "training": False,
    }
    write_json(args.output_dir / "environment.json", environment)
    for model_key in args.models:
        model_id, revision = MODELS[model_key]
        snapshot = snapshot_download(model_id, revision=revision, allow_patterns=[
            "*.json", "*.jinja", "*.model", "*.txt",
        ])
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokens = {
            key: [render(tokenizer, row["claim"], model_key)[1:] for row in dataset["rows"]]
            for key, dataset in datasets.items()
        }
        template = tokenizer.get_chat_template()
        model_dir = args.output_dir / model_key
        model_dir.mkdir(exist_ok=True)
        (model_dir / "chat_template.jinja").write_text(template)
        metadata = {
            **environment, "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        }
        caches = {
            key: DatasetCache(model_dir / key, model_key, key, dataset, metadata, settings)
            for key, dataset in datasets.items()
        }
        pending = [key for key, cache in caches.items() if cache.cursor < len(cache.rows)]
        if not pending:
            continue
        snapshot = snapshot_download(model_id, revision=revision, allow_patterns=[
            "*.json", "*.jinja", "*.model", "*.txt", "*.safetensors",
        ])
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
            device_map={"": "cuda:0"}, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, attn_implementation="eager",
        ).eval()
        metadata = {
            **metadata, "model_class": type(model).__name__,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        try:
            batch_size = args.batch_size
            for key in pending:
                cache = caches[key]
                cache.manifest["meta"] = metadata
                batch_size = extract_dataset(
                    cache, model, tokenizer, tokens[key], batch_size,
                    args.shard_size, args.writer_threads,
                )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract frozen-model features for EvoHallu transfer")
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["llama", "gemma", "olmo"])
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--writer-threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    sizes = [args.batch_size, args.shard_size] + ([] if args.limit is None else [args.limit])
    if min(sizes) < 1:
        parser.error("Batch size, shard size and limit must be positive")
    extract(args)


if __name__ == "__main__":
    main()
