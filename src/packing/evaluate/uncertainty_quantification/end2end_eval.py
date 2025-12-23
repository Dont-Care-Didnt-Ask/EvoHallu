import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import gc
import time
import logging
import json

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax, softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from src.packing.evaluate.uncertainty_quantification.features import make_all_stats
from src.packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _bootstrap_auc_ci, _delong_auc_ci
from src.packing.evaluate.uncertainty_quantification.estimator_base import Estimator

from src.packing.evaluate.uncertainty_quantification.found_formulas import (
    HallucinationEstimator,
    QuartileEntropyDiff,
    QuartileEntropyDiffNormalized,
    PositionWeightedGap,
    PositionWeightedGapNormalized,
    Gap,
)

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

def pack_single_sample_raw_states(
    i: int,
    batch_raw: List[str],
    batch_labels: List[float],
    batch_ids: List[str],
    chat_inputs: Dict[str, Any],
    raw_inputs: Dict[str, Any],
    hidden_states: List[torch.Tensor],
    attentions: List[torch.Tensor],
    attn_mask: torch.Tensor,
    logits: torch.Tensor,
    layer_stride: int,
    top_k: int,
) -> Dict[str, Any]:
    length = int(attn_mask[i].sum().item())
    if length <= 0:
        raise ValueError("Empty attention mask")

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
    hidden_i = torch.stack(hs_layers, dim=0)

    # Stack attentions into [num_layers, num_heads, length, length]
    attn_layers: List[torch.Tensor] = []
    for a in attentions:
        attn_layers.append(
            a[i, :, :length, :length].to(dtype=torch.float16, device="cpu")
        )
    attn_i = torch.stack(attn_layers, dim=0)

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
    raw_internal_states = {
        "claim": batch_raw[i],
        "label": float(batch_labels[i]),
        "id": batch_ids[i],
        "input_ids": input_ids_i,
        "attention_mask": attention_mask_i,
        "orig_input_ids": orig_input_ids_i,
        "sequence_logprobs": sequence_logprobs_i.numpy(), # length - 1
        "greedy_logprobs": greedy_logprobs_i.numpy(), # length
        "token_entropies": token_entropies_i.numpy(), # length
        "hidden_states": hidden_i,
        "attentions": attn_i,
        "top_k_logits": top_k_vals,
        "top_k_indices": top_k_inds,
    }
    return raw_internal_states

def eval_estimators_on_split(
    df: pd.DataFrame,
    tokenizer,
    model,
    model_metadata: Dict[str, Any],
    supervised: bool,
    text_col: str,
    label_col: str,
    id_col: Optional[str],
    batch_size: int,
    chat_template: Optional[str],
    top_k: int,
    layer_stride: int,
    estimators: List[Estimator],
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(float).tolist()
    ids = df[id_col].tolist() if id_col and id_col in df.columns else [None] * len(df)

    # Chat-form texts (may be identical to raw texts if template is not used)
    rendered_texts = _apply_chat_template(tokenizer, texts, chat_template)
    
    estimators_to_scores: Dict[str, List[np.ndarray]] = {str(estimator): [] for estimator in estimators}
    labels_list: List[np.ndarray] = []
    
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

        samples: List[Dict[str, Any]] = []
        for i in range(end - start):
            raw_internal_states = pack_single_sample_raw_states(
                i,
                batch_raw=batch_raw,
                batch_labels=batch_labels,
                batch_ids=batch_ids,
                chat_inputs=chat_inputs,
                raw_inputs=raw_inputs,
                hidden_states=hidden_states,
                attn_mask=attn_mask,
                attentions=attentions,
                logits=logits,
                layer_stride=layer_stride,
                top_k=top_k,
            )
            samples.append(raw_internal_states)

        # Free CUDA memory between batches
        del outputs
        torch.cuda.empty_cache()
        gc.collect()

        # Compute features
        features_start_time = time.perf_counter()
        chunk_raw = {
            "samples": samples,
            "meta": model_metadata,
        }
        end = start + batch_size
        logging.info(f"[UQ evaluate_func] Chunk {end // batch_size}: samples [{start}:{end}], calling make_all_stats...")
        
        stats_start_time = time.perf_counter()
        (
            all_stats_chunk,
            labels_chunk,
            train_stats_cache,
            background_stats_cache,
        ) = make_all_stats(
            eval_raw=chunk_raw,
            train_raw=None,
            background_raw=None,
            supervised=supervised,
            train_stats_cache=None,
            background_stats_cache=None,
            input_regime="full",
        )
        stats_elapsed = time.perf_counter() - stats_start_time
        logging.info(f"[UQ evaluate_func] Chunk {end // batch_size}: make_all_stats took {stats_elapsed:.2f}s, got {len(all_stats_chunk)} stat keys")

        chunk_start_time = time.perf_counter()
        logging.info(f"[UQ evaluate_func] Chunk {end // batch_size}: calling estimator...")
        for estimator in estimators:
            estimator_start_time = time.perf_counter()
            chunk_scores = estimator(all_stats_chunk)
            estimator_elapsed = time.perf_counter() - estimator_start_time
            logging.info(f"[UQ evaluate_func] Chunk {end // batch_size}: estimator {str(estimator)} took {estimator_elapsed:.2f}s")

            if chunk_scores.ndim != 1 or chunk_scores.shape[0] != labels_chunk.shape[0]:
                logging.warning(
                    f"[UQ evaluate_func] Estimator returned scores of shape {chunk_scores.shape}, expected ({labels_chunk.shape[0]},). Marking as failed. Populating scores with -1.0."
                )
                function_class.score = 0.0
                function_class.true_score = 0.0
                function_class.fail_flag = 1
                function_class.correct_flag = 0
                function_class.extra_metrics = {
                    "error": f"Shape mismatch: got {chunk_scores.shape}, expected ({labels_chunk.shape[0]},)",
                }
                chunk_scores = np.full(labels_chunk.shape[0], fill_value=-1.0, dtype=np.float32)

            chunk_scores = np.asarray(chunk_scores, dtype=np.float32)
            estimators_to_scores[str(estimator)].append(chunk_scores)
        
        labels_chunk = np.asarray(labels_chunk, dtype=np.float32)
        labels_list.append(labels_chunk)
        
        chunk_elapsed = time.perf_counter() - chunk_start_time
        logging.info(f"[UQ evaluate_func] Chunk {end // batch_size}: completed in {chunk_elapsed:.2f}s total")

    failed_estimators: List[str] = []

    for name, scores in estimators_to_scores.items():
        if len(scores) == 0:
            logging.warning(f"No scores found for estimator {name}")
            failed_estimators.append(name)
            continue

    logging.info(f"[UQ evaluate_func] All chunks processed. Concatenating results...")
    estimators_to_scores = {key: np.concatenate(scores) for key, scores in estimators_to_scores.items() if key not in failed_estimators}
    labels = np.concatenate(labels_list)
    logging.info(f"[UQ evaluate_func] Concatenated: {len(labels)} labels")
    logging.info(f"[UQ evaluate_func] Concatenated:\n")
    for key, scores in estimators_to_scores.items():
        logging.info(f"{key}: {scores.shape}")

    estimators_to_metrics: Dict[str, Dict[str, Any]] = {key: {} for key in estimators_to_scores.keys()}

    # ROC-AUC objective; higher scores should mean higher hallucination prob.
    logging.info(f"[UQ evaluate_func] Computing ROC-AUC...")
    for key, scores in estimators_to_scores.items():
        auc = float(roc_auc_score(labels, scores))
        score = round(100.0 * auc, 2)
        estimators_to_metrics[key]["ROC-AUC"] = score
        logging.info(f"[UQ evaluate_func] {key}: ROC-AUC = {auc:.2f}")

        # Bootstrap confidence interval for AUC
        B = 1000
        bootstrap_start = time.time()
        auc_ci = _bootstrap_auc_ci(labels, scores, B)
        delong_auc_ci = _delong_auc_ci(labels, scores)
        bootstrap_elapsed = time.time() - bootstrap_start
        logging.info(f"[UQ evaluate_func] Bootstrap CI took {bootstrap_elapsed:.2f}s. Bootstrap CI: {auc_ci}, DeLong CI: {delong_auc_ci}")
        estimators_to_metrics[key]["Bootstrap CI"] = auc_ci
        estimators_to_metrics[key]["DeLong CI"] = delong_auc_ci
        print("-" * 100)


    return estimators_to_metrics

def save_json(output_path, data):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


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
        torch_dtype=torch.float32,
    ).eval().to("cuda" if torch.cuda.is_available() else "cpu")

    config = model.config
    model_metadata = {
        "model_id": getattr(config, "_name_or_path", None),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
    }

    context = {
        "supervised": False,
        "meta": model_metadata,
    }

    estimators = [
        HallucinationEstimator(context=context),
        QuartileEntropyDiff(context=context),
        QuartileEntropyDiffNormalized(context=context),
        PositionWeightedGap(context=context),
        PositionWeightedGapNormalized(context=context),
        Gap(context=context),
    ]

    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    estimators_to_metrics = eval_estimators_on_split(
        df=df,
        tokenizer=tokenizer,
        model=model,
        model_metadata=model_metadata,
        supervised=context["supervised"],
        text_col=args.text_col,
        label_col=args.label_col,
        id_col=args.id_col,
        batch_size=args.batch_size,
        chat_template=args.chat_template,
        top_k=args.top_k,
        layer_stride=args.layer_stride,
        estimators=estimators,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, estimators_to_metrics)


if __name__ == "__main__":
    main()