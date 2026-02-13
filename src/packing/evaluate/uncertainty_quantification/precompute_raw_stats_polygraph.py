"""Precompute raw model statistics for LM-Polygraph datasets.

For each CSV with input, model_generation, output, and precomputed metrics,
runs a forward pass on [input | model_generation] and extracts statistics
for generation tokens only. Saves to *_hidden_states.pt, *_attentions.pt, *_logits.pt.

Requires CSV to have metrics columns (run compute_polygraph_metrics first).
"""

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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


def _format_input(inp: str, tokenizer, instruct: bool) -> str:
    if not instruct:
        return inp
    chat = [{"role": "user", "content": inp}]
    return tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )


def _precompute_file(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    metric_col: str,
    instruct: bool,
    top_k: int,
    layer_stride: int,
    max_rows: Optional[int],
) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    inputs = df["input"].fillna("").astype(str).tolist()
    generations = df["model_generation"].fillna("").astype(str).tolist()
    ground_truth_scores = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0).tolist()
    if max_rows is not None:
        inputs = inputs[:max_rows]
        generations = generations[:max_rows]
        ground_truth_scores = ground_truth_scores[:max_rows]

    hidden_states_samples: List[Dict[str, Any]] = []
    attentions_samples: List[Dict[str, Any]] = []
    logits_samples: List[Dict[str, Any]] = []

    for i in tqdm(range(len(inputs)), desc="Precomputing raw stats"):
        inp = inputs[i]
        gen_text = generations[i]
        gt_score = float(ground_truth_scores[i])

        formatted_input = _format_input(inp, tokenizer, instruct)

        input_enc = tokenizer(
            formatted_input,
            return_tensors="pt",
            add_special_tokens=not instruct,
        )
        prompt_len = input_enc["input_ids"].shape[1]

        full_text = formatted_input + gen_text
        full_enc = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=not instruct,
        ).to(device)

        full_len = full_enc["input_ids"].shape[1]
        gen_len = full_len - prompt_len

        if gen_len <= 0:
            logger.warning(f"Sample {i}: empty generation, skipping")
            continue

        gen_tokens_list = full_enc["input_ids"][0, prompt_len:].tolist()
        orig_input_ids = torch.tensor(gen_tokens_list, dtype=torch.int32)
        attention_mask = torch.ones(gen_len, dtype=torch.int8)

        with torch.no_grad():
            outputs = model(
                **full_enc,
                output_hidden_states=True,
                output_attentions=True,
            )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        gen_log_probs = log_probs[0, prompt_len - 1 : prompt_len - 1 + gen_len].float().cpu().numpy()
        gen_logits = logits[0, prompt_len - 1 : prompt_len - 1 + gen_len].float().cpu()

        ll = [float(gen_log_probs[j, gen_tokens_list[j]]) for j in range(gen_len)]
        sequence_logprobs = np.array(ll, dtype=np.float32)
        greedy_logprobs = np.max(gen_log_probs, axis=1).astype(np.float32)
        probs_np = np.exp(gen_log_probs)
        token_entropies = (-np.sum(probs_np * gen_log_probs, axis=1)).astype(np.float32)

        top_k_actual = min(top_k, gen_logits.shape[-1])
        top_k_vals, top_k_inds = torch.topk(gen_logits, k=top_k_actual, dim=-1)
        top_k_vals = top_k_vals.numpy().astype(np.float32)
        top_k_inds = top_k_inds.numpy().astype(np.int64)

        common_fields = {
            "claim": gen_text,
            "input": inp,
            "label": 0.0,
            "ground_truth_score": gt_score,
            "input_ids": orig_input_ids,
            "attention_mask": attention_mask,
            "orig_input_ids": orig_input_ids,
            "sequence_logprobs": sequence_logprobs,
            "greedy_logprobs": greedy_logprobs,
            "token_entropies": token_entropies,
        }

        hs_layers: List[torch.Tensor] = []
        for j, h in enumerate(outputs.hidden_states):
            if j % layer_stride != 0:
                continue
            gen_h = h[0, prompt_len : prompt_len + gen_len].to(dtype=torch.float16, device="cpu")
            hs_layers.append(gen_h)
        hidden_i = torch.stack(hs_layers, dim=0)

        attentions = outputs.attentions
        attn_stack = torch.stack(
            [a[0, :, :full_len, :full_len].cpu().float() for a in attentions],
            dim=0,
        )
        attn_reshape = attn_stack.reshape(num_layers * num_heads, full_len, full_len)

        if prompt_len > 0:
            attn_shifted = attn_reshape[
                :, prompt_len - 1 : prompt_len + gen_len - 1, prompt_len : prompt_len + gen_len
            ].clone()
        else:
            attn_shifted = attn_reshape[:, : gen_len - 1, :gen_len].clone()

        for j in range(attn_shifted.shape[1]):
            attn_shifted[:, j, j:] = 0.0

        attn_final = attn_shifted.reshape(
            num_layers, num_heads, attn_shifted.shape[1], attn_shifted.shape[2]
        ).to(dtype=torch.float16)  # (num_layers, num_heads, gen_len-1, gen_len)

        hidden_states_sample = common_fields.copy()
        hidden_states_sample["hidden_states"] = hidden_i
        hidden_states_samples.append(hidden_states_sample)

        attentions_sample = common_fields.copy()
        attentions_sample["attentions"] = attn_final
        attentions_samples.append(attentions_sample)

        logits_sample = common_fields.copy()
        logits_sample["top_k_logits"] = top_k_vals
        logits_sample["top_k_indices"] = top_k_inds
        logits_samples.append(logits_sample)

        del outputs
        torch.cuda.empty_cache()
        gc.collect()

    config = model.config
    meta = {
        "model_id": getattr(config, "_name_or_path", None),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
    }

    return {
        "hidden_states_struct": {"samples": hidden_states_samples, "meta": meta},
        "attentions_struct": {"samples": attentions_samples, "meta": meta},
        "logits_struct": {"samples": logits_samples, "meta": meta},
    }


def _save_outputs(output_dir: Path, base_name: str, output_structs: Dict[str, Any]) -> None:
    base_path = output_dir / base_name
    torch.save(output_structs["hidden_states_struct"], f"{base_path}_hidden_states.pt")
    torch.save(output_structs["attentions_struct"], f"{base_path}_attentions.pt")
    torch.save(output_structs["logits_struct"], f"{base_path}_logits.pt")
    logger.info(f"Saved to {base_path}_*.pt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute raw model statistics for LM-Polygraph datasets."
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

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"Input directory not found: {input_dir}"

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob(args.pattern))
    if not csv_files:
        logger.warning(f"No CSV files found matching '{args.pattern}' in {input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files")

    tokenizer = None
    model = None
    if not args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
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
        dataset_type = detect_dataset_type_from_filename(filename)

        if dataset_type is None:
            logger.warning(f"Skipping {filename}: unknown dataset type")
            skipped += 1
            continue

        preferred_metric = get_metric_for_dataset_type(dataset_type)
        if preferred_metric is None:
            logger.warning(f"Skipping {filename}: no metric for dataset type {dataset_type}")
            skipped += 1
            continue

        df = pd.read_csv(csv_path)
        required = ["input", "model_generation", "output"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            logger.warning(f"Skipping {filename}: missing columns {missing}")
            skipped += 1
            continue

        metric_col = select_best_available_metric(df, [preferred_metric])
        if metric_col is None:

            logger.warning(
                f"Skipping {filename}: required metric '{preferred_metric}' not found. "
                f"Run compute_polygraph_metrics first. Columns: {list(df.columns)}"
            )
            skipped += 1
            continue

        if args.dry_run:
            logger.info(f"[DRY RUN] Would process {filename} with metric {metric_col}")
            processed += 1
            continue

        df["input"] = df["input"].fillna("").astype(str)
        df["model_generation"] = df["model_generation"].fillna("").astype(str)
        df = df[df["model_generation"] != ""].reset_index(drop=True)

        if len(df) == 0:
            logger.warning(f"Skipping {filename}: no valid rows after cleaning")
            skipped += 1
            continue

        logger.info(f"Processing {filename} ({len(df)} rows) with metric {metric_col}")

        output_structs = _precompute_file(
            df=df,
            tokenizer=tokenizer,
            model=model,
            metric_col=metric_col,
            instruct=args.instruct,
            top_k=args.top_k,
            layer_stride=args.layer_stride,
            max_rows=args.max_rows_per_dataset,
        )

        base_name = csv_path.stem + "_raw_stats"
        _save_outputs(output_dir, base_name, output_structs)
        processed += 1

    logger.info(f"Done: {processed} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
