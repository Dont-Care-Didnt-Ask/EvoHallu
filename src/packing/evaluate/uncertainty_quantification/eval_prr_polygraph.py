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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# lm-polygraph imports
from lm_polygraph.estimators import (
    MaximumSequenceProbability, Perplexity,
    # Ported estimators from found_formulas.py
    SequenceLogProb, SequenceLogProbBottom, MinimalLogProb,
    StdTokenEntropy, MeanStdTokenEntropy, MeanTokenEntropy,
    QuartileEntropyDiffNormalized, QuartileEntropyDiff,
    PositionWeightedGap, PositionWeightedGapNormalized, Gap,
    QuartileEntropyDiff_PositionWeightedGap,
    QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay,
    EntropySlope, LogProbSlope,
    ExpWeightedNegLogProbWithTop1, ExpWeightedNegLogProb,
    PositionWeightedGap_EntropySlope_TopKMassInv,
    # Multi-criteria-optimized methods
    SequenceLogProb_MeanEnt_MeanGap,
    SequenceLogProb_MeanEnt_RatioGap,
)
# Evolved estimators from EvoTune rounds 289, 144, 177
from lm_polygraph.estimators.outlier_score import (
    LogProbSlope_OutlierScore,
    SequenceLogProb_OutlierScore,
)
from lm_polygraph.estimators.top_bottom_ratio import (
    LogProbSlope_TopBottomRatio,
    SequenceLogProb_TopBottomRatio,
)
from lm_polygraph.estimators.bottom_mass_avg import (
    LogProbSlope_BottomMassAvg,
    SequenceLogProb_BottomMassAvg,
)
# From 3000-round run
from lm_polygraph.estimators.entropy_logprobvolatility import (
    Entropy_LogProbVolatility,
)
from lm_polygraph.estimators.logvar_negtop1 import (
    LogVar_NegMeanTop1,
)
from lm_polygraph.estimators.seqlogprob_fraclow import (
    MeanSeqLogProb_FracLow,
)
# From another 3000-round run
from lm_polygraph.estimators.sequencelogprob_times_expectedrank import (
    SequenceLogProbTimesExpectedRank,
)
from lm_polygraph.estimators.sequencelogprob_times_meanentropy import (
    SequenceLogProbTimesMeanEntropy,
)
from lm_polygraph.estimators.sequencelogprob_meanentropy import (
    SequenceLogProb_MeanEntropy,
)
# Attention-based estimators
from lm_polygraph.estimators import (
    RAUQ,
    AttentionScore
)
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

MAX_REJECTION_RATIO = 0.5

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
    forwardpass_attention_weights = []

    # New stats for ported estimators
    sequence_logprobs_list = []
    greedy_logprobs_list = []
    token_entropies_list = []
    top_k_logits_list = []

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

        # NOTE: lm-polygraph's generation includes EOS token in greedy_tokens but not
        # in decoded text. When scoring pre-existing text, we cannot know if generation
        # ended with EOS (natural termination) or without (hit max_tokens). This may cause
        # length mismatches. Uncertainty estimators handle variable-length sequences.

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
        gen_log_probs = log_probs[0, prompt_len - 1 : prompt_len - 1 + gen_len].float().cpu().numpy()
        greedy_log_probs.append(gen_log_probs)

        # Extract log likelihood for each generated token
        ll = []
        for j, token_id in enumerate(gen_tokens_list):
            ll.append(gen_log_probs[j, token_id].item())
        greedy_log_likelihoods.append(ll)

        # ============================================================================
        # Compute additional stats for ported estimators
        # ============================================================================

        # 1. sequence_logprobs - log probs of ACTUALLY generated tokens
        # Note: Has shape [gen_len] (includes all generated tokens) to match original convention
        # Original design excludes only the first token of the FULL sequence (prompt+gen), not first gen token
        sequence_logprobs_item = np.array(ll, dtype=np.float32)
        sequence_logprobs_list.append(sequence_logprobs_item)

        # 2. greedy_logprobs - log probs of TOP-1 (most likely) token at each position
        # Has shape [gen_len] (includes all positions)
        greedy_logprobs_item = np.max(gen_log_probs, axis=1).astype(np.float32)
        greedy_logprobs_list.append(greedy_logprobs_item)

        # 3. token_entropies - Shannon entropy over full vocabulary: -sum(p * log(p))
        probs = np.exp(gen_log_probs)  # shape [gen_len, vocab_size]
        # Shannon entropy: -sum(p * log(p)) over vocab dimension
        entropies = -np.sum(probs * gen_log_probs, axis=1).astype(np.float32)  # shape [gen_len]
        token_entropies_list.append(entropies)

        # 4. top_k_logits - top-512 logits per position (captures almost all probability mass)
        # Use logits BEFORE softmax from the same positions as gen_log_probs
        # logits shape: (1, full_seq_len, vocab_size)
        # We need logits at positions [prompt_len-1 : prompt_len-1+gen_len]
        gen_logits = logits[0, prompt_len - 1 : prompt_len - 1 + gen_len].float().cpu()
        top_k_logits = torch.topk(gen_logits, k=min(512, gen_logits.shape[1]), dim=1).values.numpy().astype(np.float32)
        top_k_logits_list.append(top_k_logits)

        # Extract attention within generation tokens
        # Key insight: shift query positions by -1 to align with generation format
        # During generation at step j: query attends to positions [0..j-1] (no self-attention)
        # During scoring at position j: query attends to positions [0..j] (includes self-attention)
        # Solution: use scoring attention from position j-1 to mimic generation at step j
        #
        # Implementation follows HalluDetect/run_uq_methods.py:get_attentions (lines 62-82)
        if output_attentions:
            attentions = outputs.attentions
            # Concatenate attentions: tuple of (batch, heads, seq, seq) -> (layers, heads, seq, seq)
            attn_mask = tuple(attention.cpu() for attention in attentions)
            attn_mask = torch.cat(attn_mask).float().numpy()  # (layers, heads, seq, seq)

            # Reshape to (layers * heads, seq, seq)
            attn_mask_reshape = attn_mask.reshape(num_layers * num_heads, full_len, full_len)

            # Shift query positions by -1 and extract generation token attention
            # Query positions: [prompt_len-1, ..., prompt_len+gen_len-2] (shifted by -1)
            # Key positions: [prompt_len, ..., prompt_len+gen_len-1] (generation tokens)
            # This gives us shape (num_layers * num_heads, gen_len, gen_len)
            if prompt_len > 0:
                attn_mask_shifted = attn_mask_reshape[:, prompt_len - 1 : prompt_len + gen_len - 1, prompt_len : prompt_len + gen_len]
            else:
                # Edge case: if there's no prompt, we can't shift by -1
                # Use the first gen_len-1 queries attending to first gen_len keys
                attn_mask_shifted = attn_mask_reshape[:, : gen_len - 1, :gen_len]

            # Apply causal masking (zero out attention to future tokens)
            # After shifting: query at index j (absolute position prompt_len+j-1)
            # attends to keys at indices [0..gen_len-1] (absolute positions [prompt_len..prompt_len+gen_len-1])
            # Due to causal mask, query at prompt_len+j-1 can only attend to positions up to prompt_len+j-1
            # In key indices, this means up to index j-1 (since key index k = absolute position prompt_len+k)
            # So we need to zero out key indices j and onwards for query j
            for j in range(attn_mask_shifted.shape[1]):
                # Zero out attention to keys at indices j and onwards
                attn_mask_shifted[:, j, j:] = 0.0

            attention_all.append(attn_mask_shifted)
            forwardpass_attention_weights.append(attn_mask)

    result = {
        "greedy_log_likelihoods": greedy_log_likelihoods,
        "greedy_tokens": greedy_tokens,
        "greedy_texts": generation_texts,
        "greedy_log_probs": greedy_log_probs,
        "model": model,
        # New stats for ported estimators
        "sequence_logprobs": sequence_logprobs_list,
        "greedy_logprobs": greedy_logprobs_list,
        "token_entropies": token_entropies_list,
        "top_k_logits": top_k_logits_list,
    }

    if output_attentions:
        result["attention_all"] = attention_all
        result["forwardpass_attention_weights"] = forwardpass_attention_weights

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


def _bootstrap_prr_ci(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    max_rejection: float,
    B: int = 1000,
    bootstrap_indices: Optional[np.ndarray] = None,
) -> Tuple[str, List[float]]:
    """
    Compute 95% bootstrap confidence interval for PRR.

    Parameters
    ----------
    uncertainty_scores : np.ndarray
        Uncertainty scores for each sample.
    quality_scores : np.ndarray
        Quality scores for each sample.
    max_rejection : float
        Maximum rejection fraction for PRR.
    B : int
        Number of bootstrap iterations (used only if bootstrap_indices is None).
    bootstrap_indices : np.ndarray, optional
        Pre-generated bootstrap indices of shape (B, n). If provided, these are
        used instead of generating new random indices. This allows aligned
        bootstrap samples across multiple methods for pairwise comparisons.

    Returns
    -------
    ci_str : str
        Formatted confidence interval string "(lower, upper)".
    bootstrapped_scores : List[float]
        List of bootstrapped PRR values for each iteration.
    """
    n = len(uncertainty_scores)
    prr_metric = PredictionRejectionArea(max_rejection=max_rejection)

    if bootstrap_indices is None:
        # Fallback: generate indices on the fly (non-reproducible)
        bootstrap_indices = np.array([
            np.random.choice(np.arange(n), size=n, replace=True)
            for _ in range(B)
        ])

    bootstrapped_scores = []
    for idxs in bootstrap_indices:
        try:
            # Resample both uncertainty and quality scores together
            prr_bootstrap = prr_metric(uncertainty_scores[idxs], quality_scores[idxs])
            bootstrapped_scores.append(float(prr_bootstrap))
        except Exception:
            # Keep NaN to preserve alignment across methods
            bootstrapped_scores.append(float('nan'))

    valid_scores = [s for s in bootstrapped_scores if not np.isnan(s)]
    if valid_scores:
        lower = float(np.percentile(valid_scores, 2.5))
        upper = float(np.percentile(valid_scores, 97.5))
        ci_str = f"({100*lower:.2f}, {100*upper:.2f})"
    else:
        ci_str = "(nan, nan)"

    return ci_str, bootstrapped_scores


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
        help="Batch size for inference (within each chunk)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of samples per chunk for memory-efficient processing"
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

    # ============================================================================
    # CRITICAL WARNING: batch_size > 1 causes divergence from lm-polygraph
    # ============================================================================
    if args.batch_size > 1:
        warning_msg = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                          ⚠️  CRITICAL WARNING  ⚠️                          ║
║                                                                           ║
║  batch_size > 1 WILL CAUSE DIVERGENCE FROM LM-POLYGRAPH GENERATION!     ║
║                                                                           ║
║  Batched processing introduces left-padding artifacts that cause         ║
║  attention values to differ significantly from autoregressive            ║
║  generation (verified: 0.9+ difference with batching vs 1e-6 without).  ║
║                                                                           ║
║  For accurate PRR evaluation matching lm-polygraph:                      ║
║    --> USE --batch-size 1 <--                                            ║
║                                                                           ║
║  Current setting: batch_size = {batch_size}                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
        """.format(batch_size=args.batch_size)

        logging.warning(warning_msg)

        # Create warning file in output directory
        output_dir = Path(args.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        warning_file = output_dir / "BATCH_SIZE_GREATER_THAN_1_RESULTS_MAY_DIVERGE_WITH_LM_POLYGRAPH"
        warning_file.touch()
        logging.warning(f"Created warning file: {warning_file}")

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
        # Existing LM-Polygraph estimators
        MaximumSequenceProbability(),
        Perplexity(),
        # Ported estimators from found_formulas.py
        SequenceLogProb(),
        SequenceLogProbBottom(),
        MinimalLogProb(),
        StdTokenEntropy(),
        MeanStdTokenEntropy(),
        MeanTokenEntropy(),
        QuartileEntropyDiffNormalized(),
        QuartileEntropyDiff(),
        PositionWeightedGap(),
        PositionWeightedGapNormalized(),
        Gap(),
        QuartileEntropyDiff_PositionWeightedGap(),
        QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay(),
        EntropySlope(),
        LogProbSlope(),
        ExpWeightedNegLogProbWithTop1(),
        ExpWeightedNegLogProb(),
        PositionWeightedGap_EntropySlope_TopKMassInv(),
        # Evolved estimators from EvoTune (Rounds 289, 144, 177)
        # Round 289: Best performer (Test AUC: 74.35%)
        LogProbSlope_OutlierScore(),
        SequenceLogProb_OutlierScore(),
        # Round 144: Top-bottom ratio (Test AUC: 73.20%)
        LogProbSlope_TopBottomRatio(),
        SequenceLogProb_TopBottomRatio(),
        # Round 177: Bottom-mass average (Test AUC: 73.19%)
        LogProbSlope_BottomMassAvg(),
        SequenceLogProb_BottomMassAvg(),
        # From 3000-round run
        Entropy_LogProbVolatility(),
        LogVar_NegMeanTop1(),
        MeanSeqLogProb_FracLow(),
        # Multi-criteria optimized estimators
        SequenceLogProb_MeanEnt_MeanGap(),
        SequenceLogProb_MeanEnt_RatioGap(),
        # From another 3000-round run
        SequenceLogProbTimesExpectedRank(),
        SequenceLogProbTimesMeanEntropy(),
        SequenceLogProb_MeanEntropy(),
    ]

    output_attentions = not args.disable_rauq
    if not args.disable_rauq:
        estimators.append(RAUQ(instruct=args.instruct))
        estimators.append(AttentionScore(layer=16))

    logging.info(f"Evaluating {len(estimators)} estimators: {[str(e) for e in estimators]}")

    input_texts = df[args.input_col].astype(str).tolist()
    generation_texts = df[args.text_col].astype(str).tolist()
    quality_scores = np.array(df[args.metric_col].tolist())

    # Process in chunks to avoid OOM
    # Accumulate only scores (small), not stats (huge)
    chunk_size = args.chunk_size
    num_samples = len(df)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    logging.info(f"Processing {num_samples} samples in {num_chunks} chunks of size {chunk_size}")

    # Accumulate uncertainty scores for each estimator
    all_uncertainty_scores = {str(estimator): [] for estimator in estimators}

    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, num_samples)
        chunk_size_actual = end - start

        logging.info(f"Chunk {chunk_idx + 1}/{num_chunks}: samples {start}-{end}")

        # Compute stats for this chunk only
        chunk_stats = {
            "greedy_log_likelihoods": [],
            "greedy_tokens": [],
            "greedy_texts": [],
            "greedy_log_probs": [],
            "model": model,
            # New stats for ported estimators
            "sequence_logprobs": [],
            "greedy_logprobs": [],
            "token_entropies": [],
            "entropy": [],          # an alias for token_entropies for MeanTokenEntropy estimator
            "top_k_logits": [],
        }
        if output_attentions:
            chunk_stats["attention_all"] = []
            chunk_stats["forwardpass_attention_weights"] = []

        # Process samples in this chunk (with mini-batches)
        for mini_start in range(start, end, args.batch_size):
            mini_end = min(mini_start + args.batch_size, end)
            batch_inputs = input_texts[mini_start:mini_end]
            batch_generations = generation_texts[mini_start:mini_end]

            batch_stats = compute_stats_for_existing_generation(
                model=model,
                input_texts=batch_inputs,
                generation_texts=batch_generations,
                output_attentions=output_attentions,
            )

            # Accumulate in chunk_stats
            chunk_stats["greedy_log_likelihoods"].extend(batch_stats["greedy_log_likelihoods"])
            chunk_stats["greedy_tokens"].extend(batch_stats["greedy_tokens"])
            chunk_stats["greedy_texts"].extend(batch_stats["greedy_texts"])
            chunk_stats["greedy_log_probs"].extend(batch_stats["greedy_log_probs"])
            chunk_stats["sequence_logprobs"].extend(batch_stats["sequence_logprobs"])
            chunk_stats["greedy_logprobs"].extend(batch_stats["greedy_logprobs"])
            chunk_stats["token_entropies"].extend(batch_stats["token_entropies"])
            chunk_stats["entropy"].extend(batch_stats["token_entropies"]) # an alias for MeanTokenEntropy estimator
            chunk_stats["top_k_logits"].extend(batch_stats["top_k_logits"])
            if output_attentions:
                chunk_stats["attention_all"].extend(batch_stats["attention_all"])
                chunk_stats["forwardpass_attention_weights"].extend(batch_stats["forwardpass_attention_weights"])

            del batch_stats

        # Extract uncertainty scores from this chunk
        for estimator in estimators:
            estimator_name = str(estimator)
            try:
                chunk_scores = estimator(chunk_stats)
                if not isinstance(chunk_scores, np.ndarray):
                    chunk_scores = np.array(chunk_scores)
                all_uncertainty_scores[estimator_name].extend(chunk_scores.tolist())
            except Exception as e:
                logging.warning(f"Chunk {chunk_idx + 1}: {estimator_name} failed - {e}")
                # Append NaN for failed samples
                all_uncertainty_scores[estimator_name].extend([np.nan] * chunk_size_actual)

        # Discard chunk stats to free memory
        del chunk_stats
        torch.cuda.empty_cache()
        gc.collect()

    # Compute PRR with accumulated scores
    logging.info("Computing PRR metrics...")
    prr_metric = PredictionRejectionArea(max_rejection=MAX_REJECTION_RATIO)
    estimator_results = {}
    
    # Pre-generate bootstrap indices with fixed seed for reproducibility
    # This ensures aligned samples across all methods for pairwise comparisons
    B = 1000
    BOOTSTRAP_SEED = 31
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_samples = len(quality_scores)
    bootstrap_indices = rng.choice(n_samples, size=(B, n_samples), replace=True)
    logging.info(f"Generated {B} bootstrap samples with seed {BOOTSTRAP_SEED}")
    
    # Store bootstrap PRR values for each estimator
    estimator_bootstrap_prrs = {}

    for estimator in estimators:
        estimator_name = str(estimator)
        try:
            uncertainty_scores = np.array(all_uncertainty_scores[estimator_name])

            # Check for NaN values
            total_samples = len(uncertainty_scores)
            valid_mask = ~np.isnan(uncertainty_scores)
            num_valid = valid_mask.sum()
            num_failed = total_samples - num_valid

            if not valid_mask.all():
                # Skip estimator entirely to maintain aligned bootstrap indices
                logging.warning(
                    f"{estimator_name}: {num_failed}/{total_samples} samples failed. "
                    f"Skipping estimator to maintain aligned bootstrap indices for pairwise tests."
                )
                estimator_results[estimator_name] = {
                    "PRR": None,
                    "Bootstrap CI": None,
                    "num_samples": int(num_valid),
                    "num_failed": int(num_failed),
                    "total_samples": int(total_samples),
                    "error": f"Skipped due to {num_failed} NaN values (alignment required for pairwise tests)",
                }
                continue

            # All samples are valid, proceed with PRR computation
            prr = prr_metric(uncertainty_scores, quality_scores)
            prr_pct = round(100.0 * prr, 2)
            
            # Bootstrap confidence interval for PRR
            bootstrap_start = time.perf_counter()
            prr_ci, bootstrap_prrs = _bootstrap_prr_ci(
                uncertainty_scores,
                quality_scores,
                max_rejection=MAX_REJECTION_RATIO,
                B=B,
                bootstrap_indices=bootstrap_indices,
            )
            bootstrap_elapsed = time.perf_counter() - bootstrap_start
            logging.info(f"Bootstrap CI for {estimator_name} took {bootstrap_elapsed:.2f}s. CI: {prr_ci}")

            estimator_results[estimator_name] = {
                "PRR": prr_pct,
                "Bootstrap CI": prr_ci,
                "num_samples": int(num_valid),
                "num_failed": int(num_failed),
                "total_samples": int(total_samples),
            }
            estimator_bootstrap_prrs[estimator_name] = bootstrap_prrs
            logging.info(f"  {estimator_name}: PRR = {prr_pct}% ({num_valid}/{total_samples} samples)")

        except Exception as e:
            logging.warning(f"  {estimator_name}: FAILED - {e}")
            estimator_results[estimator_name] = {"PRR": None, "error": str(e)}

    # Save results
    results = {
        "dataset": args.data,
        "dataset_type": args.dataset_type,
        "metric_col": args.metric_col,
        "model_id": args.model_id,
        "instruct": args.instruct,
        "num_samples": len(df),
        "estimators": {
            name: {
                "PRR": data["PRR"],
                "Bootstrap CI": data.get("Bootstrap CI", None),
            }
            for name, data in estimator_results.items()
        },
    }

    output_path = Path(args.output_path)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {output_path}")
    
    # Save bootstrapped PRR values to a separate file for pairwise comparisons
    bootstrap_output_path = output_path.with_stem(output_path.stem + "_bootstrap_prrs")
    with open(bootstrap_output_path, "w") as f:
        json.dump(estimator_bootstrap_prrs, f, indent=2)
    logging.info(f"Saved bootstrap PRRs to {bootstrap_output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, data in estimator_results.items():
        prr = data.get("PRR", "FAILED")
        bootstrap_ci = data.get("Bootstrap CI", "N/A")
        print(f"{name:40s} | PRR: {prr} | Bootstrap CI: {bootstrap_ci}")
    print("=" * 80)


if __name__ == "__main__":
    main()
