from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
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
    else:
        build_prompt = lambda t: [
            {"role": "user", "content": "Generate true statement"},
            {"role": "assistant", "content": t},
        ]

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
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_samples: List[Dict[str, Any]] = []

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(float).tolist()
    ids = df[id_col].tolist() if id_col and id_col in df.columns else [None] * len(df)

    # Chat-form texts (may be identical to raw texts if template is not used)
    rendered_texts = _apply_chat_template(tokenizer, texts, chat_template)

    for start in tqdm(range(0, len(texts), batch_size), desc="Precomputing raw stats"):
        end = min(start + batch_size, len(texts))
        batch_raw = texts[start:end]
        batch_chat = rendered_texts[start:end]
        batch_labels = labels[start:end]
        batch_ids = ids[start:end]

        # Tokenize chat-form inputs for the LM forward pass
        chat_inputs = tokenizer(
            batch_chat,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

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
                output_hidden_states=True,
                output_attentions=True,
            )

        hidden_states = outputs.hidden_states  # tuple (layers + 1)
        attentions = outputs.attentions  # tuple (layers)
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
            input_ids_i = chat_inputs["input_ids"][i, :length].to("cpu")
            attention_mask_i = attn_mask[i, :length].to("cpu")

            # Raw (non-chat) input IDs for the original claim
            raw_attn_mask_i = raw_inputs["attention_mask"][i]
            raw_length = int(raw_attn_mask_i.sum().item())
            orig_input_ids_i = raw_inputs["input_ids"][i, :raw_length].to("cpu")

            # Stack hidden states into [num_layers + 1, length, hidden_size]
            hs_layers: List[torch.Tensor] = []
            for h in hidden_states:
                hs_layers.append(
                    h[i, :length].to(dtype=torch.float16, device="cpu")
                )
            hidden_i = torch.stack(hs_layers, dim=0)

            # Stack attentions into [num_layers, num_heads, length, length]
            attn_layers: List[torch.Tensor] = []
            for a in attentions:
                attn_layers.append(
                    a[i, :, :length, :length].to(dtype=torch.float16, device="cpu")
                )
            attn_i = torch.stack(attn_layers, dim=0)

            logits_i = logits[i, :length]  # already on CPU, float16

            sample: Dict[str, Any] = {
                "claim": batch_raw[i],
                "label": float(batch_labels[i]),
                "id": batch_ids[i],
                "input_ids": input_ids_i,
                "attention_mask": attention_mask_i,
                "orig_input_ids": orig_input_ids_i,
                "hidden_states": hidden_i,
                "attentions": attn_i,
                "logits": logits_i,
            }
            raw_samples.append(sample)

        # Free CUDA memory between batches
        del outputs
        torch.cuda.empty_cache()

    config = model.config
    meta = {
        "model_id": getattr(config, "_name_or_path", None),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
    }

    return {
        "samples": raw_samples,
        "meta": meta,
    }


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
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for model forward passes (default: 1).",
    )
    ap.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help=(
            "Optional chat template tag. If provided, texts are rendered via "
            "tokenizer.apply_chat_template; 'chat' reproduces the pattern used "
            "in HalluDetect."
        ),
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to precompute (default: None).",
    )

    args = ap.parse_args()

    df = _normalize_dataframe(args.data, text_col=args.text_col, label_col=args.label_col)
    if args.max_samples is not None:
        # Take first `max_samples` samples
        df = df.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda" if torch.cuda.is_available() else "cpu")

    raw_struct = _precompute_split(
        df=df,
        tokenizer=tokenizer,
        model=model,
        text_col=args.text_col,
        label_col=args.label_col,
        id_col=args.id_col,
        batch_size=args.batch_size,
        chat_template=args.chat_template,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw_struct, str(output_path))


if __name__ == "__main__":
    main()


