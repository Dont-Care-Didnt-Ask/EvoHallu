"""
PRR evaluation script using lm-polygraph library.

This script computes Prediction-Rejection Ratio (PRR) metrics for pre-existing model
generations using uncertainty estimators from lm-polygraph.

Key design principles:

1. Log-likelihood statistics (greedy_log_likelihoods) from scoring match generation
   within floating point tolerance (~1e-5). This ensures MaximumSequenceProbability
   and Perplexity estimators produce consistent results.

2. Attention statistics (attention_all) will DIFFER between scoring (full forward pass)
   and generation (with KV cache). This is because:
   - Generation: at step j, query attends to [prompt, gen_0..gen_{j-1}], NOT including self
   - Scoring: each position attends to all previous INCLUDING self
   The softmax denominators differ, causing different attention values.

   IMPLICATION: RAUQ estimator (which uses attention) will produce different values
   when scoring vs when generating. The relative ordering should still be meaningful
   for uncertainty estimation, but exact values won't match.

3. Each sample is processed individually (not batched) to avoid left-padding artifacts
   that cause different results between batched generation and batched scoring.

Usage:
    python -m src.packing.evaluate.uncertainty_quantification.eval_prr_polygraph \\
        --data path/to/data.csv \\
        --model-id meta-llama/Llama-2-7b-chat-hf \\
        --output-path results.json \\
        --instruct
"""

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# lm-polygraph imports
from lm_polygraph.estimators import MaximumSequenceProbability, Perplexity
from lm_polygraph.estimators.rauq import RAUQ
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

# Local imports for data utilities
from src.packing.evaluate.uncertainty_quantification.utils import (
    normalize_dataframe_generation,
    detect_dataset_type_from_filename,
    get_metric_for_dataset_type,
    select_best_available_metric,
)


def compute_stats_for_existing_generation(
    model: WhiteboxModel,
    input_texts: List[str],
    generation_texts: List[str],
    output_attentions: bool = True,
) -> Dict[str, Any]:
    """
    Compute statistics (log-likelihoods, attention) for pre-existing generations.

    This function performs a forward pass on [input | generation] and extracts
    statistics only for the generation tokens, matching exactly what would be
    obtained during generation.

    IMPORTANT: To ensure exact matching with generation, each sample is processed
    individually (no batched padding). This avoids subtle differences that arise
    from left-padding interactions with attention masking during batched forward
    passes.

    Parameters
    ----------
    model : WhiteboxModel
        lm-polygraph WhiteboxModel wrapper.
    input_texts : List[str]
        Input prompts (will be formatted with chat template if model.instruct=True).
    generation_texts : List[str]
        Pre-existing model generations to score.
    output_attentions : bool
        Whether to compute attention (required for RAUQ).

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'greedy_log_likelihoods': List[List[float]] - log probs for each gen token
        - 'greedy_tokens': List[List[int]] - tokenized generations
        - 'greedy_texts': List[str] - generation texts
        - 'greedy_log_probs': List[np.ndarray] - full log prob distributions
        - 'attention_all': List[np.ndarray] - attention within generation tokens (if requested)
        - 'model': WhiteboxModel - reference to model (needed by some estimators)
    """
    tokenizer = model.tokenizer
    device = model.device()

    greedy_log_likelihoods = []
    greedy_log_probs = []
    greedy_tokens = []
    attention_all = []

    num_layers = model.model.config.num_hidden_layers
    num_heads = model.model.config.num_attention_heads

    # Process each sample individually to avoid batched padding issues
    # This ensures exact matching with generation statistics
    for i in range(len(input_texts)):
        inp = input_texts[i]
        gen_text = generation_texts[i]

        # Format input with chat template if instruct model
        if model.instruct:
            chat = [{"role": "user", "content": inp}]
            formatted_input = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted_input = inp

        # Tokenize input to get prompt length
        input_enc = tokenizer(
            formatted_input,
            return_tensors="pt",
            add_special_tokens=not model.instruct,
        )
        prompt_len = input_enc["input_ids"].shape[1]

        # Tokenize full sequence: input + generation
        full_text = formatted_input + gen_text
        full_enc = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=not model.instruct,
        ).to(device)

        full_len = full_enc["input_ids"].shape[1]
        gen_len = full_len - prompt_len

        # Get generation tokens
        gen_tokens_list = full_enc["input_ids"][0, prompt_len:].tolist()
        greedy_tokens.append(gen_tokens_list)

        # Forward pass (single sample, no padding)
        with torch.no_grad():
            outputs = model.model(
                **full_enc,
                output_attentions=output_attentions,
                output_hidden_states=False,
            )

        logits = outputs.logits  # (1, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probs for generation tokens
        # Logit at position (prompt_len - 1 + j) predicts gen token j
        gen_log_probs = log_probs[0, prompt_len - 1 : prompt_len - 1 + gen_len].cpu().numpy()
        greedy_log_probs.append(gen_log_probs)

        # Extract log likelihood for each generated token
        ll = []
        for j, token_id in enumerate(gen_tokens_list):
            ll.append(gen_log_probs[j, token_id].item())
        greedy_log_likelihoods.append(ll)

        # Extract attention within generation tokens
        if output_attentions:
            attentions = outputs.attentions
            # Shape: (num_layers * num_heads, gen_len, gen_len)
            attn_mask = np.zeros((num_layers * num_heads, gen_len, gen_len), dtype=np.float32)

            for j in range(1, gen_len):
                # Attention from gen position j to gen positions 0..j-1
                # In sequence: from position (prompt_len + j) to positions (prompt_len : prompt_len + j)
                attn_row = []
                for layer_idx in range(num_layers):
                    layer_attn = attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
                    for head_idx in range(num_heads):
                        attn_values = layer_attn[head_idx, prompt_len + j, prompt_len : prompt_len + j]
                        attn_row.append(attn_values.cpu().numpy())

                stacked = np.stack(attn_row, axis=0)  # (num_layers * num_heads, j)
                attn_mask[:, j, :j] = stacked

            attention_all.append(attn_mask)

    result = {
        "greedy_log_likelihoods": greedy_log_likelihoods,
        "greedy_tokens": greedy_tokens,
        "greedy_texts": generation_texts,
        "greedy_log_probs": greedy_log_probs,
        "model": model,
    }

    if output_attentions:
        result["attention_all"] = attention_all

    return result


def compute_stats_via_generation(
    model: WhiteboxModel,
    input_texts: List[str],
    max_new_tokens: int = 100,
    output_attentions: bool = True,
) -> Dict[str, Any]:
    """
    Generate text and compute statistics (for testing/comparison).

    This wraps lm-polygraph's generation to get stats in the same format
    as compute_stats_for_existing_generation.

    Parameters
    ----------
    model : WhiteboxModel
        lm-polygraph WhiteboxModel wrapper.
    input_texts : List[str]
        Input prompts.
    max_new_tokens : int
        Maximum tokens to generate.
    output_attentions : bool
        Whether to compute attention.

    Returns
    -------
    Dict[str, Any]
        Same format as compute_stats_for_existing_generation.
    """
    from lm_polygraph.stat_calculators import GreedyProbsCalculator

    calculator = GreedyProbsCalculator(
        output_attentions=output_attentions,
        output_hidden_states=False,
    )

    stats = calculator(
        dependencies={},
        texts=input_texts,
        model=model,
        max_new_tokens=max_new_tokens,
    )

    stats["model"] = model
    return stats


def evaluate_prr(
    stats: Dict[str, Any],
    quality_scores: List[float],
    estimators: List,
    max_rejection: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate PRR for given statistics and quality scores.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics from compute_stats_for_existing_generation.
    quality_scores : List[float]
        Quality scores for each sample (higher = better).
    estimators : List
        List of lm-polygraph estimators.
    max_rejection : float
        Maximum rejection fraction for PRR.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Results for each estimator.
    """
    prr_metric = PredictionRejectionArea(max_rejection=max_rejection)
    results = {}

    for estimator in estimators:
        estimator_name = str(estimator)
        try:
            # Compute uncertainty scores
            uncertainty_scores = estimator(stats)
            if not isinstance(uncertainty_scores, np.ndarray):
                uncertainty_scores = np.array(uncertainty_scores)

            # Compute PRR
            # Note: PRR expects higher uncertainty = more uncertain
            # Quality scores: higher = better
            prr = prr_metric(uncertainty_scores, quality_scores)
            prr_pct = round(100.0 * prr, 2)

            results[estimator_name] = {
                "PRR": prr_pct,
                "uncertainty_scores": uncertainty_scores.tolist(),
            }
            logging.info(f"  {estimator_name}: PRR = {prr_pct}%")

        except Exception as e:
            logging.warning(f"  {estimator_name}: FAILED - {e}")
            results[estimator_name] = {"PRR": None, "error": str(e)}

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="PRR evaluation using lm-polygraph estimators"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV with 'input', 'model_generation', and quality metric columns"
    )
    parser.add_argument(
        "--input-col",
        type=str,
        default="input",
        help="Name of input/prompt column"
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="model_generation",
        help="Name of generation text column"
    )
    parser.add_argument(
        "--metric-col",
        type=str,
        default=None,
        help="Name of quality metric column (auto-detected if not provided)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        help="Dataset type for metric auto-detection"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="VityaVitalich/Llama3.1-8b-instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model weights"
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Use instruction-tuned model with chat template"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--disable-rauq",
        action="store_true",
        help="Disable RAUQ estimator (saves memory by not computing attention)"
    )

    args = parser.parse_args()

    # Auto-detect dataset type
    if args.dataset_type is None:
        args.dataset_type = detect_dataset_type_from_filename(Path(args.data).name)
        if args.dataset_type:
            logging.info(f"Auto-detected dataset type: {args.dataset_type}")

    # Load data
    df = normalize_dataframe_generation(
        args.data,
        text_col=args.text_col,
        validate_metrics=False,
    )

    # Validate input column exists
    if args.input_col not in df.columns:
        raise ValueError(
            f"Input column '{args.input_col}' not found. Available: {list(df.columns)}"
        )

    # Auto-detect metric column
    if args.metric_col is None:
        if args.dataset_type:
            args.metric_col = get_metric_for_dataset_type(args.dataset_type)
        if args.metric_col is None or args.metric_col not in df.columns:
            common_metrics = ["bleu", "rougeL", "rouge1", "f1", "exact_match", "best_rougeL"]
            args.metric_col = select_best_available_metric(df, common_metrics)

    if args.metric_col is None or args.metric_col not in df.columns:
        raise ValueError(
            f"Could not determine metric column. Please specify --metric-col. "
            f"Available: {list(df.columns)}"
        )

    logging.info(f"Using metric column: {args.metric_col}")

    # Convert metric to numeric
    df[args.metric_col] = pd.to_numeric(df[args.metric_col], errors="coerce")

    if args.max_samples:
        df = df.head(args.max_samples)

    logging.info(f"Loaded {len(df)} samples")
    logging.info(f"Metric stats - mean: {df[args.metric_col].mean():.3f}, "
                 f"std: {df[args.metric_col].std():.3f}")

    # Load model
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    logging.info(f"Loading model {args.model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model.eval()

    # Create lm-polygraph WhiteboxModel
    gen_params = GenerationParameters(do_sample=False, max_new_tokens=100)
    model = WhiteboxModel(
        base_model,
        tokenizer,
        model_path=args.model_id,
        generation_parameters=gen_params,
        instruct=args.instruct,
    )

    # Setup estimators
    estimators = [
        MaximumSequenceProbability(),
        Perplexity(),
    ]

    output_attentions = not args.disable_rauq
    if not args.disable_rauq:
        estimators.append(RAUQ(instruct=args.instruct))

    logging.info(f"Evaluating {len(estimators)} estimators: {[str(e) for e in estimators]}")

    # Process in batches
    all_stats = {
        "greedy_log_likelihoods": [],
        "greedy_tokens": [],
        "greedy_texts": [],
        "greedy_log_probs": [],
        "model": model,
    }
    if output_attentions:
        all_stats["attention_all"] = []

    input_texts = df[args.input_col].astype(str).tolist()
    generation_texts = df[args.text_col].astype(str).tolist()
    quality_scores = df[args.metric_col].tolist()

    for start in tqdm(range(0, len(df), args.batch_size), desc="Computing stats"):
        end = min(start + args.batch_size, len(df))
        batch_inputs = input_texts[start:end]
        batch_generations = generation_texts[start:end]

        batch_stats = compute_stats_for_existing_generation(
            model=model,
            input_texts=batch_inputs,
            generation_texts=batch_generations,
            output_attentions=output_attentions,
        )

        # Accumulate stats
        all_stats["greedy_log_likelihoods"].extend(batch_stats["greedy_log_likelihoods"])
        all_stats["greedy_tokens"].extend(batch_stats["greedy_tokens"])
        all_stats["greedy_texts"].extend(batch_stats["greedy_texts"])
        all_stats["greedy_log_probs"].extend(batch_stats["greedy_log_probs"])
        if output_attentions:
            all_stats["attention_all"].extend(batch_stats["attention_all"])

        # Free memory
        del batch_stats
        torch.cuda.empty_cache()
        gc.collect()

    # Evaluate PRR
    logging.info("Computing PRR metrics...")
    estimator_results = evaluate_prr(
        stats=all_stats,
        quality_scores=quality_scores,
        estimators=estimators,
    )

    # Save results
    results = {
        "dataset": args.data,
        "dataset_type": args.dataset_type,
        "metric_col": args.metric_col,
        "model_id": args.model_id,
        "instruct": args.instruct,
        "num_samples": len(df),
        "estimators": {
            name: {"PRR": data["PRR"]}
            for name, data in estimator_results.items()
        },
    }

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {args.output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, data in estimator_results.items():
        prr = data.get("PRR", "FAILED")
        print(f"{name:40s} | PRR: {prr}")
    print("=" * 80)


if __name__ == "__main__":
    main()
