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
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from src.packing.evaluate.uncertainty_quantification.features import make_all_stats
from src.packing.evaluate.uncertainty_quantification.estimator_base import Estimator
from src.packing.evaluate.uncertainty_quantification.utils import (
    normalize_dataframe_classification,
    apply_chat_template,
    pack_single_sample_raw_states,
    bootstrap_auc_ci,
    delong_auc_ci,
    generate_bootstrap_indices,
)

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
)


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
    rendered_texts = apply_chat_template(tokenizer, texts, chat_template)
    
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
    estimators_to_bootstrap_aucs: Dict[str, List[float]] = {}

    # ROC-AUC objective; higher scores should mean higher hallucination prob.
    logging.info(f"[UQ evaluate_func] Computing ROC-AUC...")

    # Pre-generate bootstrap indices with fixed seed for reproducibility
    # This ensures aligned samples across all methods for pairwise comparisons
    B = 1000
    BOOTSTRAP_SEED = 31
    n_samples = len(labels)
    bootstrap_indices = generate_bootstrap_indices(n_samples, n_bootstrap=B, seed=BOOTSTRAP_SEED)
    logging.info(f"[UQ evaluate_func] Generated {B} bootstrap samples with seed {BOOTSTRAP_SEED}")

    for key, scores in estimators_to_scores.items():
        auc = float(roc_auc_score(labels, scores))
        score = round(100.0 * auc, 2)
        estimators_to_metrics[key]["ROC-AUC"] = score
        logging.info(f"[UQ evaluate_func] {key}: ROC-AUC = {auc:.2f}")

        # Bootstrap confidence interval for AUC (using pre-generated indices)
        bootstrap_start = time.time()
        auc_ci, bootstrap_aucs = bootstrap_auc_ci(labels, scores, B, bootstrap_indices)
        delong_ci = delong_auc_ci(labels, scores)
        bootstrap_elapsed = time.time() - bootstrap_start
        logging.info(f"[UQ evaluate_func] Bootstrap CI took {bootstrap_elapsed:.2f}s. Bootstrap CI: {auc_ci}, DeLong CI: {delong_ci}")
        estimators_to_metrics[key]["Bootstrap CI"] = auc_ci
        estimators_to_metrics[key]["DeLong CI"] = delong_ci
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

    df = normalize_dataframe_classification(args.data, text_col=args.text_col, label_col=args.label_col)
    if args.max_samples is not None:
        # Take first `max_samples` samples
        df = df.head(args.max_samples)

    model, tokenizer, model_metadata = load_model_and_tokenizer(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float32,
        device="auto",
    )

    context = {
        "supervised": False,
        "meta": model_metadata,
    }

    estimators = [
        SequenceLogProb(context=context),
        SequenceLogProbBottom(context=context),
        MinimalLogProb(context=context),
        MeanTokenEntropy(context=context),
        Perplexity(context=context),
        StdTokenEntropy(context=context),
        MeanStdTokenEntropy(context=context),
        QuartileEntropyDiff(context=context),
        QuartileEntropyDiffNormalized(context=context),
        PositionWeightedGap(context=context),
        PositionWeightedGapNormalized(context=context),
        Gap(context=context),
        EntropySlope(context=context),
        QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay(context=context),
        QuartileEntropyDiff_PositionWeightedGap(context=context),
        PositionWeightedGap_EntropySlope_TopKMassInv(context=context),
        LogProbSlope(context=context),
        ExpWeightedNegLogProb(context=context),
        ExpWeightedNegLogProbWithTop1(context=context),
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