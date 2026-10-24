import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import gc

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax, softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm


def _normalize_dataframe(
    csv_path: str,
    text_col: str = "claim",
    label_col: str = "is_hallu",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV at {csv_path} must contain '{text_col}' and '{label_col}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    # Raise an error if there are any non-numeric labels
    non_numeric_rows = ~pd.to_numeric(df[label_col], errors="coerce").notnull()
    if non_numeric_rows.any():
        raise ValueError(
            f"CSV at {csv_path} contains non-numeric values in '{label_col}' column at rows: {df[non_numeric_rows].index.tolist()}."
        )
    df[label_col] = df[label_col].astype(float)

    if len(df) == 0:
        raise ValueError(f"No valid rows left after cleaning CSV at {csv_path}")

    return df.reset_index(drop=True)


def _apply_chat_template(tokenizer, texts: List[str], template: Optional[str]) -> List[str]:
    if not template:
        return texts

    if template == "assistant-only":
        # Simple assistant-only wrapper by default
        build_prompt = lambda t: [
            {"role": "assistant", "content": t},
        ]
    elif "generate_true_statement" in template:
        build_prompt = lambda t: [
            {"role": "user", "content": "Generate true statement"},
            {"role": "assistant", "content": t},
        ]
    else:
        raise ValueError(f"Unknown value for --chat-template: {template}")

    rendered: List[str] = []
    for t in texts:
        rendered.append(
            tokenizer.apply_chat_template(
                build_prompt(t),
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return rendered


def _precompute_split(
    df: pd.DataFrame,
    tokenizer,
    model,
    text_col: str,
    label_col: str,
    id_col: Optional[str],
    batch_size: int,
    chat_template: Optional[str],
    top_k: int,
    layer_stride: int,
    modalities: str = "all",
    max_batch_tokens: int = 16384,
    gc_interval: int = 1,
    cuda_cache_interval: int = 1,
) -> Dict[str, Any]:
    if batch_size <= 0 or max_batch_tokens < 0 or gc_interval < 0 or cuda_cache_interval < 0:
        raise ValueError("Batch size must be positive and token budget nonnegative")
    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.cuda.device_count()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_states_samples: List[Dict[str, Any]] = []
    attentions_samples: List[Dict[str, Any]] = []
    logits_samples: List[Dict[str, Any]] = []

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(float).tolist()
    ids = df[id_col].tolist() if id_col and id_col in df.columns else [None] * len(df)

    # Chat-form texts (may be identical to raw texts if template is not used)
    rendered_texts = _apply_chat_template(tokenizer, texts, chat_template)

    progress = tqdm(total=len(texts), desc="Precomputing raw stats", unit="claims")
    start = 0
    batches_completed = 0
    while start < len(texts):
        end = min(start + batch_size, len(texts))
        batch_chat = rendered_texts[start:end]

        # Tokenize chat-form inputs for the LM forward pass
        chat_inputs = tokenizer(
            batch_chat,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # A single long claim otherwise pads every row in its GPU batch.
        # Keep row order and all tokens, shrinking only the forward microbatch.
        while (modalities == "logits" and max_batch_tokens and end-start > 1
               and chat_inputs["input_ids"].numel() > max_batch_tokens):
            end = start + max(1, (end-start)//2)
            batch_chat = rendered_texts[start:end]
            chat_inputs = tokenizer(batch_chat, return_tensors="pt", padding=True, truncation=True)
        chat_inputs = chat_inputs.to(device)
        batch_raw = texts[start:end]
        batch_labels = labels[start:end]
        batch_ids = ids[start:end]

        # Tokenize raw claims without any special tokens for later alignment
        raw_inputs = tokenizer(
            batch_raw,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **chat_inputs,
                output_hidden_states=modalities == "all",
                output_attentions=modalities == "all",
            )

        hidden_states = outputs.hidden_states if modalities == "all" else ()
        attentions = outputs.attentions if modalities == "all" else ()
        logits = outputs.logits  # [bs, seq_len, vocab]

        # Move vocab-related tensors to CPU / float16 to save space
        logits = logits.to(dtype=torch.float16, device="cpu")

        # For each sequence in the batch, crop by attention mask length to avoid padding
        attn_mask = chat_inputs["attention_mask"]
        for i in range(end - start):
            length = int(attn_mask[i].sum().item())
            if length <= 0:
                continue

            # Input IDs and attention mask (chat form)
            input_ids_i = chat_inputs["input_ids"][i, :length].to("cpu", dtype=torch.int32)
            attention_mask_i = attn_mask[i, :length].to("cpu", dtype=torch.int8)

            # Raw (non-chat) input IDs for the original claim
            raw_attn_mask_i = raw_inputs["attention_mask"][i]
            raw_length = int(raw_attn_mask_i.sum().item())
            orig_input_ids_i = raw_inputs["input_ids"][i, :raw_length].to("cpu", dtype=torch.int32)

            # Stack hidden states into [num_layers + 1, length, hidden_size]
            hs_layers: List[torch.Tensor] = []
            for j, h in enumerate(hidden_states):
                if j % layer_stride != 0:
                    continue

                hs_layers.append(
                    h[i, :length].to(dtype=torch.float16, device="cpu")
                )
            hidden_i = torch.stack(hs_layers, dim=0) if hs_layers else None

            # Stack attentions into [num_layers, num_heads, length, length]
            attn_layers: List[torch.Tensor] = []
            for a in attentions:
                attn_layers.append(
                    a[i, :, :length, :length].to(dtype=torch.float16, device="cpu")
                )
            attn_i = torch.stack(attn_layers, dim=0) if attn_layers else None

            raw_logits_i = logits[i, :length]  # already on CPU, float16
            top_k_vals, top_k_inds = torch.topk(raw_logits_i, k=top_k, dim=-1)
            logits_i = log_softmax(raw_logits_i, dim=-1)
            probs_i = softmax(raw_logits_i, dim=-1)
            del raw_logits_i

            # Log probabilities of each token in the sequence (excluding the first token)
            sequence_logprobs_i = logits_i[:-1][torch.arange(length - 1), input_ids_i[1:]]
            # Top-1 logit for each token in the sequence
            greedy_logprobs_i = logits_i.max(dim=-1).values
            # Entropy of probability distribution at each token
            token_entropies_i = -probs_i.mul(logits_i).sum(dim=-1).to(dtype=torch.float16)

            # Common fields for all outputs
            common_fields = {
                "claim": batch_raw[i],
                "label": float(batch_labels[i]),
                "id": batch_ids[i],
                "input_ids": input_ids_i,
                "attention_mask": attention_mask_i,
                "orig_input_ids": orig_input_ids_i,
                "sequence_logprobs": sequence_logprobs_i.numpy(), # length - 1
                "greedy_logprobs": greedy_logprobs_i.numpy(), # length
                "token_entropies": token_entropies_i.numpy(), # length
            }

            # Create separate samples for each output type
            if modalities == "all":
                hidden_states_sample = common_fields.copy()
                hidden_states_sample["hidden_states"] = hidden_i
                hidden_states_samples.append(hidden_states_sample)
                attentions_sample = common_fields.copy()
                attentions_sample["attentions"] = attn_i
                attentions_samples.append(attentions_sample)

            logits_sample = common_fields.copy()
            logits_sample["top_k_logits"] = top_k_vals
            logits_sample["top_k_indices"] = top_k_inds
            logits_samples.append(logits_sample)

        # Free CUDA memory between batches
        del outputs
        batches_completed += 1
        if cuda_cache_interval and batches_completed % cuda_cache_interval == 0:
            torch.cuda.empty_cache()
        # Explicit collection scans the growing retained CPU cache. It does not
        # change feature calculations; default 1 preserves historical behavior.
        if gc_interval and batches_completed % gc_interval == 0:
            gc.collect()
        progress.update(end-start)
        start = end
    progress.close()

    config = model.config
    meta = {
        "model_id": getattr(config, "_name_or_path", None),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "modalities": modalities,
        "model_dtype": str(getattr(model, "dtype", "unknown")),
        "requested_forward_batch_size": batch_size,
        "max_padded_forward_tokens": max_batch_tokens if modalities == "logits" else None,
        "cpu_feature_threads": torch.get_num_threads(),
        "cpu_gc_interval_batches": gc_interval,
        "cuda_cache_cleanup_interval_batches": cuda_cache_interval,
    }

    return {
        "hidden_states_struct": {
            "samples": hidden_states_samples,
            "meta": meta,
        },
        "attentions_struct": {
            "samples": attentions_samples,
            "meta": meta,
        },
        "logits_struct": {
            "samples": logits_samples,
            "meta": meta,
        },
    }


def save_outputs(output_path: Path, output_structs: Dict[str, Any]) -> None:
    """Save each output structure to a separate file with appropriate suffix."""
    base_path = output_path.parent / output_path.stem
    suffix = output_path.suffix
    
    # Save hidden states
    hidden_states_path = f"{base_path}_hidden_states{suffix}"
    if output_structs["hidden_states_struct"]["samples"]:
        torch.save(output_structs["hidden_states_struct"], hidden_states_path)
        print(f"Saved hidden states to: {hidden_states_path}")
    
    # Save attentions
    attentions_path = f"{base_path}_attentions{suffix}"
    if output_structs["attentions_struct"]["samples"]:
        torch.save(output_structs["attentions_struct"], attentions_path)
        print(f"Saved attentions to: {attentions_path}")
    
    # Save logits
    logits_path = f"{base_path}_logits{suffix}"
    torch.save(output_structs["logits_struct"], logits_path)
    print(f"Saved logits to: {logits_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Precompute raw model outputs (hidden states, attentions, logits) "
            "for a hallucination dataset split."
        )
    )
    ap.add_argument("--data", required=True, help="Path to CSV with claims and labels.")
    ap.add_argument(
        "--text-col",
        default="claim",
        help="Column name containing the claim text (default: 'claim').",
    )
    ap.add_argument(
        "--label-col",
        default="is_hallu",
        help="Column name containing the hallucination label (default: 'is_hallu').",
    )
    ap.add_argument(
        "--id-col",
        default=None,
        help="Optional column name with unique IDs (stored as metadata).",
    )
    ap.add_argument(
        "--output-path",
        required=True,
        help="Path to save the raw statistics (e.g. 'train_raw.pt').",
    )
    ap.add_argument(
        "--model-id",
        default="VityaVitalich/Llama3.1-8b-instruct",
        help="HF model ID or local path to the hallucination model.",
    )
    ap.add_argument(
        "--cache-dir",
        default="../.cache/huggingface/hub",
        help="Optional Hugging Face cache directory.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for model forward passes.",
    )
    ap.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help=(
            "Optional chat template tag. If provided, texts are rendered via "
            "tokenizer.apply_chat_template; 'generate_true_statement' reproduces the pattern used "
            "in HalluDetect."
        ),
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to precompute (default: None).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=512,
        help="Number of top logits to keep for each token (default: 512).",
    )
    ap.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        help="Stride for sampling layers. Larger stride means less layers are sampled.",
    )

    ap.add_argument("--modalities", choices=["all", "logits"], default="all",
                    help="Use logits to avoid retaining attention and hidden-state tensors.")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="Shrink logits-only forward batches above this padded token count; 0 disables.")
    args = ap.parse_args()

    df = _normalize_dataframe(args.data, text_col=args.text_col, label_col=args.label_col)
    if args.max_samples is not None:
        # Take first `max_samples` samples
        df = df.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager" if args.modalities == "all" else "sdpa",
    ).eval().to("cuda:0" if torch.cuda.is_available() else "cpu")

    output_structs = _precompute_split(
        df=df,
        tokenizer=tokenizer,
        model=model,
        text_col=args.text_col,
        label_col=args.label_col,
        id_col=args.id_col,
        batch_size=args.batch_size,
        chat_template=args.chat_template,
        top_k=args.top_k,
        layer_stride=args.layer_stride,
        modalities=args.modalities,
        max_batch_tokens=args.max_batch_tokens,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_outputs(output_path, output_structs)


if __name__ == "__main__":
    main()
