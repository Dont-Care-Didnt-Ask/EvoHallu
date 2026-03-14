import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    SequenceLogProb,
    SequenceLogProbBottom,
    MinimalLogProb,
    MeanTokenEntropy,
    Perplexity,
    StdTokenEntropy,
    MeanStdTokenEntropy,
    QuartileEntropyDiff,
    QuartileEntropyDiffNormalized,
    PositionWeightedGap,
    PositionWeightedGapNormalized,
    Gap,
    EntropySlope,
    QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay,
    QuartileEntropyDiff_PositionWeightedGap,
    PositionWeightedGap_EntropySlope_TopKMassInv,
    LogProbSlope,
    ExpWeightedNegLogProb,
    ExpWeightedNegLogProbWithTop1,
    # Evolved estimators from EvoTune (Rounds 289, 144, 177)
    LogProbSlope_OutlierScore,
    SequenceLogProb_OutlierScore,
    LogProbSlope_TopBottomRatio,
    SequenceLogProb_TopBottomRatio,
    LogProbSlope_BottomMassAvg,
    SequenceLogProb_BottomMassAvg,
)

from lm_polygraph.estimators import (
    Opus46_Round358_Function1_DynamicPrompt,
    Opus46_Round450_Function1_DynamicPrompt,
    Opus46_Round330_Function2_DynamicPrompt,
    Sonnet40_Round419_Function2_StaticPrompt,
    Sonnet40_Round497_Function2_StaticPrompt,
    Sonnet40_Round406_Function2_StaticPrompt,
    Sonnet45_Round19_Function2_DynamicPrompt,
    Sonnet45_Round32_Function2_DynamicPrompt,
    Sonnet45_Round249_Function1_DynamicPrompt,
    Sonnet40_Round493_Function1_DynamicPrompt,
    Sonnet40_Round412_Function1_DynamicPrompt,
    Sonnet40_Round400_Function1_DynamicPrompt,
    Sonnet45_Round476_Function2_StaticPrompt,
    Sonnet45_Round181_Function2_StaticPrompt,
    Sonnet45_Round467_Function2_StaticPrompt,
    Opus45_Round476_Function2_StaticPrompt,
    Opus45_Round453_Function1_StaticPrompt,
    Opus45_Round441_Function1_StaticPrompt,
    Opus45_Round351_Function2_DynamicPrompt,
    Opus45_Round272_Function1_DynamicPrompt,
    Opus45_Round385_Function2_DynamicPrompt,
    Haiku45_Round36_Function1_StaticPrompt,
    Haiku45_Round30_Function1_StaticPrompt,
    Haiku45_Round348_Function1_StaticPrompt,
    Haiku45_Round490_Function1_DynamicPrompt,
    Haiku45_Round486_Function2_DynamicPrompt,
    Haiku45_Round404_Function1_DynamicPrompt,
    Opus46_Round476_Function1_StaticPrompt,
    Opus46_Round484_Function1_StaticPrompt,
    Opus46_Round200_Function2_StaticPrompt,
    # Sonnet45_Round133_Function1_StaticPrompt_3datasettrainmix,
    # Sonnet45_Round238_Function1_StaticPrompt_3datasettrainmix,
    # Sonnet45_Round476_Function2_StaticPrompt_3datasettrainmix,
    # Opus46_Round495_Function1_StaticPrompt_3datasettrainmix,
    # Opus46_Round461_Function1_StaticPrompt_3datasettrainmix,
    # Opus46_Round480_Function2_StaticPrompt_3datasettrainmix,
    # Opus45_Round454_Function2_StaticPrompt_3datasettrainmix,
    # Opus45_Round456_Function2_StaticPrompt_3datasettrainmix,
    # Opus45_Round325_Function1_StaticPrompt_3datasettrainmix,
    # Sonnet40_Round340_Function2_StaticPrompt_3datasettrainmix,
    # Sonnet40_Round196_Function1_StaticPrompt_3datasettrainmix,
    # Sonnet40_Round315_Function2_StaticPrompt_3datasettrainmix,
    # Haiku45_Round309_Function1_StaticPrompt_3datasettrainmix,
    # Haiku45_Round427_Function1_StaticPrompt_3datasettrainmix,
    # Haiku45_Round275_Function1_StaticPrompt_3datasettrainmix,
)

from lm_polygraph.estimators import (
    RAUQ,
    AttentionScore,
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
    # Truncate text to 300 characters to avoid long sequences that might cause OOMs
    df[text_col] = df[text_col].str.slice(start=0, stop=300)
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

def pack_batch_raw_states_optimized(
    batch_raw,
    batch_labels,
    batch_ids,
    chat_inputs,
    raw_inputs,
    hidden_states,
    attentions,
    logits,
    layer_stride,
    top_k,
):
    """
    Vectorized packing implementation.
    Performs heavy operations once on GPU and minimizes CPU transfers.
    """

    device = logits.device
    bs, seq_len, vocab = logits.shape
    attn_mask = chat_inputs["attention_mask"]

    # ----------------------------
    # GPU batch computations
    # ----------------------------

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    token_entropies = -(probs * log_probs).sum(-1)

    greedy_logprobs = log_probs.max(-1).values

    topk_vals, topk_inds = torch.topk(logits, k=top_k, dim=-1)

    # gather sequence token logprobs
    input_ids = chat_inputs["input_ids"]

    seq_token_logprobs = log_probs[:, :-1].gather(
        -1,
        input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    # ----------------------------
    # stack hidden states once
    # ----------------------------

    selected_layers = hidden_states[::layer_stride]
    hidden_tensor = torch.stack(selected_layers, dim=0)  # [L, B, T, H]
    hidden_tensor = hidden_tensor.permute(1, 0, 2, 3)    # [B, L, T, H]

    # ----------------------------
    # stack attentions once
    # ----------------------------

    attn_tensor = torch.stack(attentions, dim=0)  # [L, B, H, T, T]
    attn_tensor = attn_tensor.permute(1, 0, 2, 3, 4)  # [B, L, H, T, T]

    # ----------------------------
    # move tensors to CPU once
    # ----------------------------

    input_ids = input_ids.cpu()
    attn_mask = attn_mask.cpu()

    raw_input_ids = raw_inputs["input_ids"].cpu()
    raw_attn_mask = raw_inputs["attention_mask"].cpu()

    seq_token_logprobs = seq_token_logprobs.cpu().float()
    greedy_logprobs = greedy_logprobs.cpu().float()
    token_entropies = token_entropies.cpu().float()

    hidden_tensor = hidden_tensor.cpu().to(torch.float16)
    attn_tensor = attn_tensor.cpu().to(torch.float16)

    topk_vals = topk_vals.cpu()
    topk_inds = topk_inds.cpu()

    # ----------------------------
    # construct samples
    # ----------------------------

    samples = []

    assert token_entropies.isfinite().all(), "Token entropies are not finite"
    assert seq_token_logprobs.isfinite().all(), "Sequence log probabilities are not finite"
    assert greedy_logprobs.isfinite().all(), "Greedy log probabilities are not finite"
    assert hidden_tensor.isfinite().all(), "Hidden states are not finite"
    assert attn_tensor.isfinite().all(), "Attentions are not finite"
    assert topk_vals.isfinite().all(), "Top-k logits are not finite"
    assert attn_mask.isfinite().all(), "Attention mask is not finite"
    assert raw_attn_mask.isfinite().all(), "Raw attention mask is not finite"

    for i in range(bs):

        length = int(attn_mask[i].sum())
        raw_length = int(raw_attn_mask[i].sum())

        sample = {
            "claim": batch_raw[i],
            "label": float(batch_labels[i]),
            "id": batch_ids[i],

            "input_ids": input_ids[i, :length].to(torch.int32),
            "attention_mask": attn_mask[i, :length].to(torch.int8),
            "orig_input_ids": raw_input_ids[i, :raw_length].to(torch.int32),

            "sequence_logprobs": seq_token_logprobs[i, :length-1].numpy(),
            "greedy_logprobs": greedy_logprobs[i, :length].numpy(),
            "token_entropies": token_entropies[i, :length].numpy(),

            "hidden_states": hidden_tensor[i, :, :length],
            "attentions": attn_tensor[i, :, :, :length, :length],

            "top_k_logits": topk_vals[i, :length].to(torch.float16),
            "top_k_indices": topk_inds[i, :length],
        }

        samples.append(sample)

    return samples

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

    raw_logits_i = logits[i, :length]
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
        "sequence_logprobs": sequence_logprobs_i.cpu().numpy(), # length - 1
        "greedy_logprobs": greedy_logprobs_i.cpu().numpy(), # length
        "token_entropies": token_entropies_i.cpu().numpy(), # length
        "hidden_states": hidden_i,
        "attentions": attn_i,
        "top_k_logits": top_k_vals.cpu(),
        "top_k_indices": top_k_inds.cpu(),
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
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
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

        forward_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                **chat_inputs,
                output_hidden_states=True,
                output_attentions=True,
            )
        forward_elapsed = time.perf_counter() - forward_start
        logging.info(f"[UQ evaluate_func] Forward pass took {forward_elapsed:.2f}s")

        hidden_states = outputs.hidden_states  # tuple (layers + 1)
        attentions = outputs.attentions  # tuple (layers)
        logits = outputs.logits  # [bs, seq_len, vocab]

        # For each sequence in the batch, crop by attention mask length to avoid padding
        attn_mask = chat_inputs["attention_mask"]

        pack_start = time.perf_counter()
        samples = pack_batch_raw_states_optimized(
            batch_raw=batch_raw,
            batch_labels=batch_labels,
            batch_ids=batch_ids,
            chat_inputs=chat_inputs,
            raw_inputs=raw_inputs,
            hidden_states=hidden_states,
            attentions=attentions,
            logits=logits,
            layer_stride=layer_stride,
            top_k=top_k,
        )
        # samples: List[Dict[str, Any]] = []
        # for i in range(end - start):
        #     raw_internal_states = pack_single_sample_raw_states(
        #         i,
        #         batch_raw=batch_raw,
        #         batch_labels=batch_labels,
        #         batch_ids=batch_ids,
        #         chat_inputs=chat_inputs,
        #         raw_inputs=raw_inputs,
        #         hidden_states=hidden_states,
        #         attn_mask=attn_mask,
        #         attentions=attentions,
        #         logits=logits,
        #         layer_stride=layer_stride,
        #         top_k=top_k,
        #     )
        #     samples.append(raw_internal_states)
        pack_elapsed = time.perf_counter() - pack_start
        logging.info(f"[UQ evaluate_func] Packing took {pack_elapsed:.2f}s")

        # Free CUDA memory between batches
        del outputs

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
            if not np.isfinite(chunk_scores).all():
                print("Non-finite scores:", chunk_scores[~np.isfinite(chunk_scores)])
                print("="*100)
                for key, value in all_stats_chunk.items():
                    if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                        print(key, value.shape)
                print("="*100)
            assert np.isfinite(chunk_scores).all(), f"Estimator {str(estimator)} returned scores that are not finite"
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
    estimators_to_bootstrap_aucs: Dict[str, List[float]] = {}

    # ROC-AUC objective; higher scores should mean higher hallucination prob.
    logging.info(f"[UQ evaluate_func] Computing ROC-AUC...")

    # Pre-generate bootstrap indices with fixed seed for reproducibility
    # This ensures aligned samples across all methods for pairwise comparisons
    B = 1000
    BOOTSTRAP_SEED = 31
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_samples = len(labels)
    bootstrap_indices = rng.choice(n_samples, size=(B, n_samples), replace=True)
    logging.info(f"[UQ evaluate_func] Generated {B} bootstrap samples with seed {BOOTSTRAP_SEED}")

    for key, scores in estimators_to_scores.items():
        auc = float(roc_auc_score(labels, scores))
        score = round(100.0 * auc, 2)
        estimators_to_metrics[key]["ROC-AUC"] = score
        logging.info(f"[UQ evaluate_func] {key}: ROC-AUC = {auc:.2f}")

        # Bootstrap confidence interval for AUC (using pre-generated indices)
        bootstrap_start = time.time()
        auc_ci, bootstrap_aucs = _bootstrap_auc_ci(labels, scores, B, bootstrap_indices)
        delong_auc_ci = _delong_auc_ci(labels, scores)
        bootstrap_elapsed = time.time() - bootstrap_start
        logging.info(f"[UQ evaluate_func] Bootstrap CI took {bootstrap_elapsed:.2f}s. Bootstrap CI: {auc_ci}, DeLong CI: {delong_auc_ci}")
        estimators_to_metrics[key]["Bootstrap CI"] = auc_ci
        estimators_to_metrics[key]["DeLong CI"] = delong_auc_ci
        estimators_to_bootstrap_aucs[key] = bootstrap_aucs
        print("-" * 100)


    return estimators_to_metrics, estimators_to_bootstrap_aucs

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

    if Path(args.output_path).exists():
        logging.info(f"[UQ evaluate_func] Output path {args.output_path} already exists. Exiting.")
        return

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
    ).eval().to("cuda:0" if torch.cuda.is_available() else "cpu")

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
        SequenceLogProb(context=context),
        # SequenceLogProbBottom(context=context),
        # MinimalLogProb(context=context),
        MeanTokenEntropy(context=context),
        # Perplexity(context=context),
        # StdTokenEntropy(context=context),
        # MeanStdTokenEntropy(context=context),
        # QuartileEntropyDiff(context=context),
        # QuartileEntropyDiffNormalized(context=context),
        # PositionWeightedGap(context=context),
        # PositionWeightedGapNormalized(context=context),
        # Gap(context=context),
        # EntropySlope(context=context),
        # QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay(context=context),
        # QuartileEntropyDiff_PositionWeightedGap(context=context),
        # PositionWeightedGap_EntropySlope_TopKMassInv(context=context),
        # LogProbSlope(context=context),
        # ExpWeightedNegLogProb(context=context),
        # ExpWeightedNegLogProbWithTop1(context=context),
        # # Evolved estimators from EvoTune (Rounds 289, 144, 177)
        # # Round 289: Best performer (Test AUC: 74.35%)
        # LogProbSlope_OutlierScore(context=context),
        # SequenceLogProb_OutlierScore(context=context),
        # # Round 144: Top-bottom ratio (Test AUC: 73.20%)
        # LogProbSlope_TopBottomRatio(context=context),
        # SequenceLogProb_TopBottomRatio(context=context),
        # # Round 177: Bottom-mass average (Test AUC: 73.19%)
        # LogProbSlope_BottomMassAvg(context=context),
        # SequenceLogProb_BottomMassAvg(context=context),
        # Opus46_Round358_Function1_DynamicPrompt(),
        # Opus46_Round450_Function1_DynamicPrompt(),
        # Opus46_Round330_Function2_DynamicPrompt(),
        # Sonnet40_Round419_Function2_StaticPrompt(),
        # Sonnet40_Round497_Function2_StaticPrompt(),
        # Sonnet40_Round406_Function2_StaticPrompt(),
        # Sonnet45_Round19_Function2_DynamicPrompt(),
        # Sonnet45_Round32_Function2_DynamicPrompt(),
        # Sonnet45_Round249_Function1_DynamicPrompt(),
        # Sonnet40_Round493_Function1_DynamicPrompt(),
        # Sonnet40_Round412_Function1_DynamicPrompt(),
        # Sonnet40_Round400_Function1_DynamicPrompt(),
        # Sonnet45_Round476_Function2_StaticPrompt(),
        # Sonnet45_Round181_Function2_StaticPrompt(),
        # Sonnet45_Round467_Function2_StaticPrompt(),
        # Opus45_Round476_Function2_StaticPrompt(),
        # Opus45_Round453_Function1_StaticPrompt(),
        # Opus45_Round441_Function1_StaticPrompt(),
        # Opus45_Round351_Function2_DynamicPrompt(),
        # Opus45_Round272_Function1_DynamicPrompt(),
        # Opus45_Round385_Function2_DynamicPrompt(),
        # Haiku45_Round36_Function1_StaticPrompt(),
        # Haiku45_Round30_Function1_StaticPrompt(),
        # Haiku45_Round348_Function1_StaticPrompt(),
        # Haiku45_Round490_Function1_DynamicPrompt(),
        # Haiku45_Round486_Function2_DynamicPrompt(),
        # Haiku45_Round404_Function1_DynamicPrompt(),
        # Opus46_Round476_Function1_StaticPrompt(),
        # Opus46_Round484_Function1_StaticPrompt(),
        # Opus46_Round200_Function2_StaticPrompt(),
        # Sonnet45_Round133_Function1_StaticPrompt_3datasettrainmix(),
        # Sonnet45_Round238_Function1_StaticPrompt_3datasettrainmix(),
        # Sonnet45_Round476_Function2_StaticPrompt_3datasettrainmix(),
        # Opus46_Round495_Function1_StaticPrompt_3datasettrainmix(),
        # Opus46_Round461_Function1_StaticPrompt_3datasettrainmix(),
        # Opus46_Round480_Function2_StaticPrompt_3datasettrainmix(),
        # Opus45_Round454_Function2_StaticPrompt_3datasettrainmix(),
        # Opus45_Round456_Function2_StaticPrompt_3datasettrainmix(),
        # Opus45_Round325_Function1_StaticPrompt_3datasettrainmix(),
        # Sonnet40_Round340_Function2_StaticPrompt_3datasettrainmix(),
        # Sonnet40_Round196_Function1_StaticPrompt_3datasettrainmix(),
        # Sonnet40_Round315_Function2_StaticPrompt_3datasettrainmix(),
        # Haiku45_Round309_Function1_StaticPrompt_3datasettrainmix(),
        # Haiku45_Round427_Function1_StaticPrompt_3datasettrainmix(),
        # Haiku45_Round275_Function1_StaticPrompt_3datasettrainmix(),
        RAUQ(),
        AttentionScore(layer=16),
    ]

    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)

    estimators_to_metrics, estimators_to_bootstrap_aucs = eval_estimators_on_split(
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

    # Save bootstrapped AUC values to a separate file for pairwise comparisons
    bootstrap_output_path = output_path.with_stem(output_path.stem + "_bootstrap_aucs")
    save_json(bootstrap_output_path, estimators_to_bootstrap_aucs)
    logging.info(f"[UQ evaluate_func] Saved bootstrap AUCs to {bootstrap_output_path}")


if __name__ == "__main__":
    main()