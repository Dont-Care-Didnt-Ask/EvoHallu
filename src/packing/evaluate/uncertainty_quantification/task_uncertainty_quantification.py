import os
import logging
import time
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from packing.evaluate.registry import TASK_REGISTRY
from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.logging.function_class import FunctionClass


def _delong_roc_variance(y_true, y_score):
    """
    Computes the variance of the ROC AUC using DeLong's method.
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        True binary labels (0 or 1).
    y_score : array-like, shape = [n_samples]
        Prediction scores (e.g., probabilities).

    Returns
    -------
    auc_var : float
        Variance of the AUC estimate.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Ensure binary classification
    assert set(np.unique(y_true)) <= {0, 1}, "y_true must be binary (0/1)."
    
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_score_sorted = y_score[desc_score_indices]

    # Find thresholds where score changes
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[0, distinct_value_indices + 1, y_true.size]

    # Number of positive and negative samples
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples.")

    # Compute tie-adjusted AUC components
    cum_pos = np.cumsum(y_true_sorted)
    cum_neg = np.arange(1, len(y_true_sorted) + 1) - cum_pos

    # Sum of positive ranks
    sum_pos_ranks = np.sum(cum_neg[y_true_sorted == 1])
    auc = sum_pos_ranks / (n_pos * n_neg)

    # Compute V10 and V01 (covariance terms)
    # V10: variance component for positive class
    # V01: variance component for negative class

    # Compute placements for positives and negatives
    placements_pos = cum_neg[y_true_sorted == 1]
    placements_neg = cum_pos[y_true_sorted == 0][::-1]  # reverse to align

    # Build the "theta" matrix implicitly via placements
    # Using the formulation from Sun & Xu (2014): 
    # var(AUC) = (V10 / n_pos + V01 / n_neg)

    # Compute V10 = Var(E[phi_1(x_1, x_0) | x_1])
    # phi_1 = P(X_0 < X_1) + 0.5 * P(X_0 = X_1)
    # We approximate using placements
    V10 = np.var(placements_pos / n_neg, ddof=0)  # population variance

    # Compute V01 = Var(E[phi_1(x_1, x_0) | x_0])
    V01 = np.var(placements_neg / n_pos, ddof=0)

    auc_var = V10 / n_pos + V01 / n_neg
    return auc_var

def _delong_auc_ci(y_true: np.ndarray, y_score: np.ndarray) -> str:
    """
    Compute 95% confidence interval for ROC-AUC using DeLong's method.
    
    Parameters
    ----------
    y_true : np.ndarray, shape = [n_samples]
        True binary labels (0 or 1).
    y_score : np.ndarray, shape = [n_samples]
        Prediction scores (e.g., probabilities).
    alpha : float, default=0.05
        Significance level (0.05 → 95% CI).

    Returns
    -------
    ci_str : str
        Formatted confidence interval as "(lower%, upper%)" with 2 decimals,
        e.g., "(82.45, 91.30)". Returns "(nan, nan)" on failure.
    """
    try:
        auc = roc_auc_score(y_true, y_score)
        auc_var = _delong_roc_variance(y_true, y_score)
        auc_std = np.sqrt(auc_var)
        
        z = 1.96 # 95% asymtotic normal confidence interval
        #stats.norm.ppf(1 - alpha / 2)
        lower = auc - z * auc_std
        upper = auc + z * auc_std
        
        # Clamp to [0, 1]
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)
        
        return f"({100*lower:.2f}, {100*upper:.2f})"
    
    except (ValueError, ZeroDivisionError, AssertionError):
        return "(nan, nan)"

def _bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, B: int = 1000) -> str:
    """Compute 95% bootstrap confidence interval for ROC-AUC."""
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(B):
        idxs = np.random.choice(np.arange(n), size=n, replace=True)
        if (y_true[idxs] == 1).any() and (y_true[idxs] == 0).any():
            bootstrapped_scores.append(roc_auc_score(y_true[idxs], y_score[idxs]))
    if bootstrapped_scores:
        lower = float(np.percentile(bootstrapped_scores, 2.5))
        upper = float(np.percentile(bootstrapped_scores, 97.5))
        return f"({100*lower:.2f}, {100*upper:.2f})"
    return "(nan, nan)"


def _load_raw(path: str) -> Dict[str, Any]:

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw stats file not found at {path}")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def _prepare_train_val_indices(
    train_raw: Dict[str, Any],
    sample_size: int,
    val_size: float,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministically split the raw train set into train/val subsets.

    The splitting logic mirrors the hallucination task: first optionally
    subsample to `sample_size`, then perform a stratified train/val split.
    """
    samples = train_raw["samples"]
    n = len(samples)
    labels = np.asarray([float(s["label"]) for s in samples], dtype=float)

    indices = np.arange(n)

    # Optional stratified subsample
    if sample_size > 0 and sample_size < n:
        _, sample_idx = train_test_split(
            indices,
            test_size=sample_size,
            stratify=labels,
            random_state=split_seed,
        )
    else:
        sample_idx = indices

    sample_labels = labels[sample_idx]

    train_idx, val_idx = train_test_split(
        sample_idx,
        test_size=val_size,
        stratify=sample_labels,
        random_state=split_seed,
    )
    return train_idx, val_idx


def _subset_raw(raw_all: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    """
    Create a shallow subset of a raw-pt structure, selecting only the
    samples at the specified indices.
    """
    return {
        "samples": [raw_all["samples"][int(i)] for i in indices],
        "meta": raw_all.get("meta", {}),
    }


def generate_input(cfg_task: DictConfig, split: str):
    """
    Load raw statistics for the requested split and construct an input
    structure for EvoTune. This function is called once per run and its
    output is reused across candidate evaluations.
    """
    name = (split or "").lower()

    train_raw_path = getattr(cfg_task, "train_raw_path", None)
    test_raw_path = getattr(cfg_task, "test_raw_path", None)
    background_raw_path = getattr(cfg_task, "background_raw_path", None)

    if not train_raw_path:
        raise ValueError("cfg.task.train_raw_path must be provided for UQ task")

    # Load full train/test/raw structures
    train_raw_all = _load_raw(train_raw_path)
    test_raw_all = _load_raw(test_raw_path) if test_raw_path else None
    background_raw = _load_raw(background_raw_path) if background_raw_path else None

    # Deterministic train/val split on the train_raw_all set
    sample_size = int(cfg_task.sample_size)
    val_size = float(cfg_task.val_size)
    split_seed = int(cfg_task.split_seed)

    train_idx, val_idx = _prepare_train_val_indices(
        train_raw_all, sample_size=sample_size, val_size=val_size, split_seed=split_seed
    )

    supervised = bool(getattr(cfg_task, "supervised", False))
    train_raw_for_stats = _subset_raw(train_raw_all, train_idx) if supervised else None

    if name == "train":
        eval_raw = _subset_raw(train_raw_all, train_idx)
    elif name == "val":
        eval_raw = _subset_raw(train_raw_all, val_idx)
    elif name == "testset":
        if test_raw_all is None:
            raise ValueError(
                "cfg.task.test_raw_path must be provided for split='testset'"
            )
        eval_raw = test_raw_all
    else:
        # Default to validation semantics
        eval_raw = _subset_raw(train_raw_all, val_idx)

    return {
        "eval_raw": eval_raw,
        "train_raw": train_raw_for_stats,
        "background_raw": background_raw,
        "supervised": supervised,
    }


def get_initial_func(cfg: DictConfig):
    """
    Provide an initial baseline estimator implementation.

    The optimizer will see the source of this class and try to improve upon it.
    """

    class HallucinationEstimator(Estimator):
        def __init__(self, context: Dict[str, Any]) -> None:
            super().__init__(
                context,
                stats_dependencies=[
                    "sequence_logprobs",
                ],
                level="sequence",
            )

        def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
            # "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1.
            total_neg_logprob = np.array([-np.sum(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]], dtype=np.float32)
            return total_neg_logprob

    initial_class = HallucinationEstimator
    class_str_to_extract = "HallucinationEstimator"
    code_type = "class"
    return initial_class, class_str_to_extract, code_type


def evaluate_func(
    cfg: DictConfig,
    data: Dict[str, Any],
    function_class: FunctionClass,
) -> FunctionClass:
    """
    Evaluate a candidate estimator class on the provided statistics.

    The candidate must define an Estimator subclass with the name given by
    cfg.function_str_to_extract (default: \"BaselineEntropyEstimator\") that
    accepts a context dict in __init__ and produces scores via __call__.
    
    IMPORTANT: This function is READ-ONLY with respect to `data` (input_struct).
    It does not modify any tensors or data structures within `data`. All operations
    on tensors (hidden_states, attentions, logits) create NEW arrays via:
    - tensor.to(dtype).cpu().numpy() chains
    - torch.log_softmax() and similar functional ops
    - numpy slicing that creates copies
    
    This invariant allows the caller to skip deep-copying `data`, which would be
    prohibitively slow for large tensor datasets. If you add any in-place tensor
    operations (+=, .add_(), direct assignment), you MUST update the caller to
    deep-copy or handle data isolation differently.
    """
    eval_start_time = time.time()
    
    logging.info(f"[UQ evaluate_func] START - function_num={getattr(function_class, 'function_num', 'N/A')}")
    
    eval_raw = data["eval_raw"]
    train_raw = data.get("train_raw")
    background_raw = data.get("background_raw")
    supervised = bool(data.get("supervised", False))
    
    num_eval_samples = len(eval_raw.get("samples", []))
    logging.info(f"[UQ evaluate_func] Data loaded: {num_eval_samples} eval samples, supervised={supervised}")

    function_str = function_class.function_str
    imports = function_class.imports_str
    
    logging.info(f"[UQ evaluate_func] Function string length: {len(function_str) if function_str else 0}")
    logging.info(f"[UQ evaluate_func] Imports string length: {len(imports) if imports else 0}")

    globals_dict: Dict[str, Any] = {
        "Estimator": Estimator,
        "np": np,
        "Dict": Dict,
        "Any": Any,
        "Tuple": Tuple,
    }
    
    logging.info(f"[UQ evaluate_func] Executing imports...")
    if imports:
        exec(imports, globals_dict)
    logging.info(f"[UQ evaluate_func] Imports executed successfully")
    
    local_dict: Dict[str, Any] = {}
    logging.info(f"[UQ evaluate_func] Executing class definition...")
    exec(function_str, globals_dict, local_dict)
    logging.info(f"[UQ evaluate_func] Class definition executed. Keys in local_dict: {list(local_dict.keys())}")

    class_name = cfg.function_str_to_extract
    if class_name not in local_dict:
        raise ValueError(f"Class {class_name} not found in candidate program")

    EstimatorClass = local_dict[class_name]
    context = {
        "supervised": supervised,
        "meta": {
            "eval_meta": eval_raw.get("meta", {}),
        },
    }
    
    logging.info(f"[UQ evaluate_func] Instantiating class '{class_name}' with context...")
    estimator: Estimator = EstimatorClass(context)
    logging.info(f"[UQ evaluate_func] Created estimator: {type(estimator).__name__}")

    if not isinstance(estimator, Estimator):
        raise TypeError(
            f"{class_name} must be an Estimator subclass, "
            f"got {type(estimator)}"
        )

    eval_samples = eval_raw["samples"]
    chunk_size = max(
        1, int(getattr(cfg.task, "chunk_size", len(eval_samples)) or len(eval_samples))
    )
    
    num_chunks = (len(eval_samples) + chunk_size - 1) // chunk_size
    logging.info(f"[UQ evaluate_func] Processing {len(eval_samples)} samples in {num_chunks} chunks (chunk_size={chunk_size})")

    scores_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    train_stats_cache: Dict[str, Any] | None = None
    background_stats_cache: Dict[str, Any] | None = None

    for chunk_idx, start in enumerate(range(0, len(eval_samples), chunk_size)):
        chunk_start_time = time.time()
        end = start + chunk_size
        chunk_raw = {
            "samples": eval_samples[start:end],
            "meta": eval_raw.get("meta", {}),
        }
        logging.info(f"[UQ evaluate_func] Chunk {chunk_idx+1}/{num_chunks}: samples [{start}:{end}], calling make_all_stats...")
        
        stats_start_time = time.time()
        (
            all_stats_chunk,
            labels_chunk,
            train_stats_cache,
            background_stats_cache,
        ) = make_all_stats(
            eval_raw=chunk_raw,
            train_raw=train_raw if train_stats_cache is None else None,
            background_raw=background_raw if background_stats_cache is None else None,
            supervised=supervised,
            train_stats_cache=train_stats_cache,
            background_stats_cache=background_stats_cache,
            input_regime=cfg.task.input_regime,
        )
        stats_elapsed = time.time() - stats_start_time
        logging.info(f"[UQ evaluate_func] Chunk {chunk_idx+1}/{num_chunks}: make_all_stats took {stats_elapsed:.2f}s, got {len(all_stats_chunk)} stat keys")

        logging.info(f"[UQ evaluate_func] Chunk {chunk_idx+1}/{num_chunks}: calling estimator...")
        estimator_start_time = time.time()
        chunk_scores = estimator(all_stats_chunk)
        estimator_elapsed = time.time() - estimator_start_time
        logging.info(f"[UQ evaluate_func] Chunk {chunk_idx+1}/{num_chunks}: estimator took {estimator_elapsed:.2f}s")
        
        chunk_scores = np.asarray(chunk_scores, dtype=np.float32)
        labels_chunk = np.asarray(labels_chunk, dtype=np.float32)

        if chunk_scores.ndim != 1 or chunk_scores.shape[0] != labels_chunk.shape[0]:
            logging.warning(
                f"[UQ evaluate_func] Estimator returned scores of shape {chunk_scores.shape}, expected ({labels_chunk.shape[0]},). Marking as failed."
            )
            function_class.score = 0.0
            function_class.true_score = 0.0
            function_class.fail_flag = 1
            function_class.correct_flag = 0
            function_class.extra_metrics = {
                "error": f"Shape mismatch: got {chunk_scores.shape}, expected ({labels_chunk.shape[0]},)",
            }
            return function_class

        scores_list.append(chunk_scores)
        labels_list.append(labels_chunk)
        
        chunk_elapsed = time.time() - chunk_start_time
        logging.info(f"[UQ evaluate_func] Chunk {chunk_idx+1}/{num_chunks}: completed in {chunk_elapsed:.2f}s total")

    if not scores_list:
        raise ValueError("No evaluation samples available for UQ task")

    logging.info(f"[UQ evaluate_func] All chunks processed. Concatenating results...")
    scores = np.concatenate(scores_list)
    labels = np.concatenate(labels_list)
    logging.info(f"[UQ evaluate_func] Concatenated: {len(scores)} scores, {len(labels)} labels")

    # ROC-AUC objective; higher scores should mean higher hallucination prob.
    logging.info(f"[UQ evaluate_func] Computing ROC-AUC...")
    auc = float(roc_auc_score(labels, scores))
    score = round(100.0 * auc, 4)
    logging.info(f"[UQ evaluate_func] ROC-AUC = {auc:.4f}, score = {score}")

    # Bootstrap confidence interval for AUC
    B = int(getattr(cfg.task, "bootstrap_samples", 1000))
    logging.info(f"[UQ evaluate_func] Computing bootstrap CI with B={B}...")
    bootstrap_start = time.time()
    auc_ci = _bootstrap_auc_ci(labels, scores, B)
    delong_auc_ci = _delong_auc_ci(labels, scores)
    bootstrap_elapsed = time.time() - bootstrap_start
    logging.info(f"[UQ evaluate_func] Bootstrap CI took {bootstrap_elapsed:.2f}s. Bootstrap CI: {auc_ci}, DeLong CI: {delong_auc_ci}")

    # Precision/Recall at median score threshold
    median_score = np.median(scores)
    y_pred = (scores >= median_score).astype(int)
    prec = float(precision_score(labels, y_pred, zero_division=0))
    rec = float(recall_score(labels, y_pred, zero_division=0))
    logging.info(f"[UQ evaluate_func] Precision={prec:.4f}, Recall={rec:.4f}")

    function_class.score = score
    function_class.true_score = score
    function_class.fail_flag = 0
    function_class.correct_flag = 1
    function_class.extra_metrics = {
        "AUC": round(100.0 * auc, 2),
        "AUC 95% confidence interval (bootstrap)": auc_ci,
        "AUC 95% confidence interval (DeLong)": delong_auc_ci,
        "Precision at median score threshold": round(100.0 * prec, 2),
        "Recall at median score threshold": round(100.0 * rec, 2),
        "Number of samples": int(labels.shape[0]),
    }
    
    total_elapsed = time.time() - eval_start_time
    logging.info(f"[UQ evaluate_func] COMPLETE - score={score}, total time={total_elapsed:.2f}s")
    return function_class


system_prompt = """Take the role of a world-class NLP researcher with specialization in uncertainty quantification. 
Design an unsupervised Estimator class that detects LLM hallucinations using internal states of an LLM.

## Task
Write a class named `HallucinationEstimator` inheriting from `Estimator` (already provided).
Your `__call__` method receives `stats` dict and MUST return np.ndarray of shape [N] (one score per sample).

```python
class HallucinationEstimator(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(context, stats_dependencies=[...], level="sequence")
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # MUST return shape [N] where N = number of samples
        ...  # Return uncertainty scores (higher = more likely hallucination)
```

## Available Statistics in `stats` (N = number of samples)

**Token-level statistics:**
- "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1. Contains logits of tokens which were actually generated by the model, excluding the first generated token.
- "greedy_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i. Contains logits of the most likely token at each position.
- "token_entropies": List[np.ndarray] of length N, inner 1d array has length n_i. Contains entropy values of the probability distribution at each position.
- "top_k_logits": List[np.ndarray] of length N - each array shape [n_i, k]
- "top_k_indices": List[np.ndarray] of length N - each array shape [n_i, k]
- "greedy_texts": List[str] of length N - claim texts
- "greedy_tokens": List[List[int]] of length N - token IDs per claim

## Guidelines
- Use at most 3 features. You MUST explore estimators based on a single feature and two features as well, don't get fixed on 3-feature estimators.
- Start from a concrete hypothesis on why this design might work.
- You MUST go beyond generic features like "entropy over top-k logits", "gap between greedy and sequence logprobs". Make steps further to design deeper, more complex, semantically informed features.

## Constraints
- Class MUST be named `HallucinationEstimator`
- Output MUST be np.ndarray of shape [N] (one scalar per sample)
- Pure Python + numpy only
- Deterministic
- Do NOT redefine Estimator base class

## Response Format
First, write 3-4 sentences reflecting on estimators given below.
Second, formulate a concrete hypothesis on how hallucination detection can be improved leveraging given statistics.
Finally, provide a definition of a new estimator class in a single ```python code block.
"""

# **Embedding statistics:**
# - "token_embeddings_decoder": np.ndarray[float32, [sum(n_i), D]] - concatenated token embeddings
# - "token_embeddings_decoder_{layer}": np.ndarray[float32, [sum(n_i), D]] per layer (if stored)
# - "embeddings_decoder": np.ndarray[float32, [N, D]] - mean token embedding per sequence
# - "embeddings_decoder_{layer}": np.ndarray[float32, [N, D]] per layer (if stored)

# **Attention statistics:**
# - "recent_lookback_attention_weights_from_claim_tokens": List[Any] of length N. For each sample: List[List[List[float]]] with shape [n_claim_tokens, n_recent_lookback_tokens, num_layers * num_heads]
# - "claim_only_attention_maps": List[np.ndarray] of length N - each array shape [L*H, n_i, n_i]
# - "full_attention_maps": List[np.ndarray] of length N - each array shape [L, H, n_i, n_i]

# **Supervised train statistics (if supervised=True):**
# - Cached train stats (from train_stats_cache) prefixed with "train_"
# - "train_sequence_logprob": np.ndarray[float32, [N_train]] (computed if missing in cache)
# - "train_mean_token_entropy": np.ndarray[float32, [N_train]] (computed if missing in cache)

# **Background statistics (if background_raw provided):**
# - Cached background stats (from background_stats_cache) e.g.:
#     "background_train_token_embeddings_decoder": np.ndarray[float32, [sum(n_i_bg), D]]

append_prompt = ""


TASK_REGISTRY.register(
    "uncertainty_quantification",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)




