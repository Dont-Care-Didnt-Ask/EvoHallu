"""Precompute raw model statistics for LM-Polygraph datasets in chunks.

This script iterates over CSV files in a directory (like precompute_raw_stats_polygraph.py),
computes generation-token-only raw statistics, and saves a single struct per chunk:

- <stem>_raw_stats.pt
- <stem>_raw_stats_chunk_<idx>.pt

Each saved file contains:
- samples: list of per-example dictionaries
- meta: model metadata
- chunk_idx, num_chunks, chunk_start, chunk_end
"""

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from packing.evaluate.uncertainty_quantification.utils import (
    detect_dataset_type_from_filename,
    get_metric_for_dataset_type,
    select_best_available_metric,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _format_input(inp: str, gen_text: str, tokenizer, instruct: bool) -> str:
    if not instruct:
        return inp + gen_text
    chat = [{"role": "user", "content": inp}]
    if gen_text:
        chat.append({"role": "assistant", "content": gen_text})

    # If no generation text is provided, we add the generation prompt to the chat template
    add_generation_prompt = (gen_text == "")
    return tokenizer.apply_chat_template(chat, add_generation_prompt=add_generation_prompt, tokenize=False)


def _chunk_outputs_exist(output_dir: Path, base_name: str) -> bool:
    base_path = output_dir / base_name
    single = Path(f"{base_path}.pt")
    chunked = list(output_dir.glob(f"{base_name}_chunk_*.pt"))
    return single.exists() or len(chunked) > 0


def _build_meta(model: torch.nn.Module) -> Dict[str, Any]:
    config = model.config
    return {
        "model_id": getattr(config, "_name_or_path", None),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
    }

@torch.inference_mode()
def _extract_polygraph_sample(
    inp: str,
    gen_text: str,
    gt_score: float,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    instruct: bool,
    top_k: int,
    layer_stride: int,
) -> Optional[Dict[str, Any]]:
    raw_generation_ids = tokenizer(gen_text, return_tensors="pt", add_special_tokens=False)
    # first dimension is batch size (==1 in this case), second dimension is number of tokens
    n_orig_tokens = int(raw_generation_ids["input_ids"].shape[1])
    if n_orig_tokens <= 0:
        return None

    formatted_full_text = _format_input(inp, gen_text, tokenizer, instruct)
    full_enc = tokenizer(
        formatted_full_text,
        return_tensors="pt",
        add_special_tokens=not instruct,
    ).to(device)
    total_tokens = int(full_enc["input_ids"].shape[1])
    assert total_tokens > n_orig_tokens, "Total amount of tokens (chat template + input + generation + special tokens) must be greater than amount of generation tokens"
    n_tokens = n_orig_tokens + 5  # +1 for eot and +4 assistant header tokens in Llama-3 instruct template
    assert total_tokens >= n_tokens, f"Expected at least {n_tokens} tokens in formatted input, got {total_tokens}"

    gen_tokens_list = full_enc["input_ids"][0, -n_tokens:].tolist()
    formatted_generation_ids = torch.tensor(gen_tokens_list, dtype=torch.int32)
    attention_mask = torch.ones(n_orig_tokens, dtype=torch.int8)
    orig_input_ids = raw_generation_ids["input_ids"][0].to(dtype=torch.int32, device="cpu")
    assert orig_input_ids.ndim == 1, f"orig_input_ids must be 1d, got shape {tuple(orig_input_ids.shape)}"
    assert int(formatted_generation_ids.shape[0]) == n_tokens, "formatted_generation_ids length mismatch"
    assert int(orig_input_ids.shape[0]) == n_orig_tokens, "orig_input_ids length mismatch"

    outputs = model(
        **full_enc,
        output_hidden_states=True,
        output_attentions=True,
    )

    logits = outputs.logits.cpu()
    log_probs = torch.log_softmax(logits, dim=-1)

    gen_log_probs = (
        log_probs[0, -n_tokens:].float().numpy()
    )
    gen_logits = logits[0, -n_tokens:].float()

    sequence_logprobs = np.array(
        [float(gen_log_probs[j - 1, gen_tokens_list[j]]) for j in range(1, n_tokens)],
        dtype=np.float32,
    )
    greedy_logprobs = np.max(gen_log_probs, axis=1).astype(np.float32)
    probs_np = np.exp(gen_log_probs)
    token_entropies = (-np.sum(probs_np * gen_log_probs, axis=1)).astype(np.float32)

    top_k_actual = min(top_k, gen_logits.shape[-1])
    top_k_vals, top_k_inds = torch.topk(gen_logits, k=top_k_actual, dim=-1)
    top_k_vals = top_k_vals.numpy().astype(np.float32)
    top_k_inds = top_k_inds.numpy().astype(np.int64)

    hs_layers: List[torch.Tensor] = []
    for layer_idx, hs in enumerate(outputs.hidden_states):
        if layer_idx % layer_stride != 0:
            continue
        gen_hs = hs[0, -n_tokens:].to(
            dtype=torch.float16, device="cpu"
        )
        hs_layers.append(gen_hs)
    hidden_i = torch.stack(hs_layers, dim=0)
    assert int(hidden_i.shape[1]) == n_tokens, f"hidden_states token dimension mismatch: expected {n_tokens}, got {int(hidden_i.shape[1])}"

    attn_layers: List[torch.Tensor] = []
    for attn in outputs.attentions:
        slice_start_q = -n_tokens
        slice_end_q = total_tokens
        slice_start_k = -n_tokens
        slice_end_k = total_tokens

        attn_slice = attn[0, :, slice_start_q:slice_end_q, slice_start_k:slice_end_k]
        attn_layers.append(attn_slice.to(dtype=torch.float16))
        del attn_slice

    attn_final = torch.stack(attn_layers, dim=0).cpu()
    assert int(attn_final.shape[2]) == n_tokens and int(attn_final.shape[3]) == n_tokens, (
        f"attention shape mismatch: expected (..., {n_tokens}, {n_tokens}), got {tuple(attn_final.shape)}"
    )

    sample = {
        "claim": gen_text,
        "input": inp,
        "ground_truth_score": float(gt_score),
        "input_ids": formatted_generation_ids,
        "attention_mask": attention_mask,
        "orig_input_ids": orig_input_ids,
        "sequence_logprobs": sequence_logprobs,
        "greedy_logprobs": greedy_logprobs,
        "token_entropies": token_entropies,
        "hidden_states": hidden_i,
        "attentions": attn_final,
        "top_k_logits": top_k_vals,
        "top_k_indices": top_k_inds,
    }

    del outputs
    del logits
    del log_probs

    return sample


@torch.inference_mode()
def _precompute_file_chunked(
    df: pd.DataFrame,
    tokenizer,
    model: torch.nn.Module,
    metric_col: str,
    instruct: bool,
    top_k: int,
    layer_stride: int,
    chunk_size: Optional[int],
    max_rows: Optional[int],
    output_dir: Path,
    output_base_name: str,
) -> None:
    assert top_k > 0, "top_k must be positive"
    assert layer_stride > 0, "layer_stride must be positive"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = df["input"].fillna("").astype(str).tolist()
    generations = df["model_generation"].fillna("").astype(str).tolist()
    scores = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0).astype(float).tolist()

    if max_rows is not None:
        assert max_rows > 0, "max_rows_per_dataset must be positive when provided"
        inputs = inputs[:max_rows]
        generations = generations[:max_rows]
        scores = scores[:max_rows]

    total_samples = len(inputs)
    assert total_samples > 0, "No samples to process after preprocessing"

    effective_chunk_size = total_samples if not chunk_size else chunk_size
    assert effective_chunk_size > 0, "chunk_size must be positive when provided"
    num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
    meta = _build_meta(model)

    output_base = output_dir / output_base_name
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * effective_chunk_size
        chunk_end = min(chunk_start + effective_chunk_size, total_samples)
        logger.info(
            "Processing chunk %s/%s for %s (rows %s:%s)",
            chunk_idx + 1,
            num_chunks,
            output_base_name,
            chunk_start,
            chunk_end,
        )

        chunk_samples: List[Dict[str, Any]] = []
        for idx in tqdm(range(chunk_start, chunk_end), desc=f"{output_base_name} chunk {chunk_idx + 1}"):
            sample = _extract_polygraph_sample(
                inp=inputs[idx],
                gen_text=generations[idx],
                gt_score=float(scores[idx]),
                tokenizer=tokenizer,
                model=model,
                device=device,
                instruct=instruct,
                top_k=top_k,
                layer_stride=layer_stride,
            )
            if sample is not None:
                chunk_samples.append(sample)

        if num_chunks > 1:
            chunk_path = Path(f"{output_base}_chunk_{chunk_idx}.pt")
        else:
            chunk_path = Path(f"{output_base}.pt")

        chunk_struct = {
            "samples": chunk_samples,
            "meta": meta,
            "chunk_idx": chunk_idx,
            "num_chunks": num_chunks,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        }
        torch.save(chunk_struct, chunk_path)
        logger.info("Saved %s samples to %s", len(chunk_samples), chunk_path)

        del chunk_samples
        gc.collect()
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute chunked raw model statistics for LM-Polygraph datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/uncertainty_quantification/polygraph_generations_metrics",
        help="Directory containing CSV files with input, model_generation, output, and metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .pt files. Default: same as input-dir.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="VityaVitalich/Llama3.1-8b-instruct",
        help="HF model ID or local path (must match generation model).",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="HF tokenizer ID. We use a different tokenizer since the tokenizer for our default model does not handle `add_generation_prompt` correctly.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/extremessd10tb/seleznev/.cache/huggingface/hub",
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Use chat template for inputs (required for instruct models).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=512,
        help="Number of top logits to keep per token (default: 512).",
    )
    parser.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        help="Stride for sampling layers (default: 1).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum number of rows per output chunk (default: all rows in one file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without processing.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Maximum number of rows per dataset to process (default: None).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert args.top_k > 0, "--top-k must be positive"
    assert args.layer_stride > 0, "--layer-stride must be positive"
    if args.chunk_size is not None:
        assert args.chunk_size > 0, "--chunk-size must be positive when provided"
    if args.max_rows_per_dataset is not None:
        assert (
            args.max_rows_per_dataset > 0
        ), "--max-rows-per-dataset must be positive when provided"

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"Input directory not found: {input_dir}"

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob(args.pattern))
    if not csv_files:
        logger.warning("No CSV files found matching '%s' in %s", args.pattern, input_dir)
        return
    logger.info("Found %s CSV files", len(csv_files))

    tokenizer = None
    model = None
    if not args.dry_run:
        if args.tokenizer_id != args.model_id:
            logger.warning("ATTENTION: Using different tokenizer (%s) for model (%s)", args.tokenizer_id, args.model_id)

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_id,
            cache_dir=args.cache_dir,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float32,
        ).eval().to("cuda:0" if torch.cuda.is_available() else "cpu")

    processed = 0
    skipped = 0
    for csv_path in csv_files:
        filename = csv_path.name
        output_base_name = f"{csv_path.stem}_raw_stats"
        dataset_type = detect_dataset_type_from_filename(filename)

        if _chunk_outputs_exist(output_dir, output_base_name):
            logger.warning("Skipping %s: outputs already exist for %s", filename, output_base_name)
            skipped += 1
            continue

        if dataset_type is None:
            logger.warning("Skipping %s: unknown dataset type", filename)
            skipped += 1
            continue

        preferred_metric = get_metric_for_dataset_type(dataset_type)
        if preferred_metric is None:
            logger.warning("Skipping %s: no metric mapping for dataset type %s", filename, dataset_type)
            skipped += 1
            continue

        df = pd.read_csv(csv_path)
        required = ["input", "model_generation", "output"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("Skipping %s: missing columns %s", filename, missing)
            skipped += 1
            continue

        metric_col = select_best_available_metric(df, [preferred_metric])
        if metric_col is None:
            logger.warning(
                "Skipping %s: required metric '%s' not found. "
                "Run compute_polygraph_metrics first. Columns: %s",
                filename,
                preferred_metric,
                list(df.columns),
            )
            skipped += 1
            continue

        if args.dry_run:
            expected = f"{output_base_name}.pt or {output_base_name}_chunk_<idx>.pt"
            logger.info("[DRY RUN] Would process %s with metric %s -> %s", filename, metric_col, expected)
            processed += 1
            continue

        df["input"] = df["input"].fillna("").astype(str)
        df["model_generation"] = df["model_generation"].fillna("").astype(str)
        df = df[df["model_generation"] != ""].reset_index(drop=True)
        if len(df) == 0:
            logger.warning("Skipping %s: no valid rows after cleaning", filename)
            skipped += 1
            continue

        logger.info("Processing %s (%s rows) with metric %s", filename, len(df), metric_col)
        _precompute_file_chunked(
            df=df,
            tokenizer=tokenizer,
            model=model,
            metric_col=metric_col,
            instruct=args.instruct,
            top_k=args.top_k,
            layer_stride=args.layer_stride,
            chunk_size=args.chunk_size,
            max_rows=args.max_rows_per_dataset,
            output_dir=output_dir,
            output_base_name=output_base_name,
        )
        processed += 1

    logger.info("Done: %s processed, %s skipped", processed, skipped)


if __name__ == "__main__":
    main()