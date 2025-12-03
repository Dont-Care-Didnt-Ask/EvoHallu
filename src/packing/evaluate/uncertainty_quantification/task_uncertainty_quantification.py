import os
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from packing.evaluate.registry import TASK_REGISTRY
from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.logging.function_class import FunctionClass


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
    import torch

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw stats file not found at {path}")
    return torch.load(path, map_location="cpu")


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
                    "sequence_logprob",
                ],
                level="sequence",
            )

        def __str__(self) -> str:
            return "HallucinationEstimator"

        def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
            # sequence_logprob: shape [N], already summed per sequence
            seq_logprob = stats["sequence_logprob"]  # [N]
            seq_logprob = np.asarray(seq_logprob, dtype=np.float32)  # [N]
            # Lower log-prob means higher uncertainty (more likely hallucination)
            scores = -seq_logprob  # [N]
            return scores

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
    import logging
    import time
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
    bootstrap_elapsed = time.time() - bootstrap_start
    logging.info(f"[UQ evaluate_func] Bootstrap CI took {bootstrap_elapsed:.2f}s: {auc_ci}")

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
        "AUC 95% confidence interval": auc_ci,
        "Precision at median score threshold": round(100.0 * prec, 2),
        "Recall at median score threshold": round(100.0 * rec, 2),
        "Number of samples": int(labels.shape[0]),
    }
    
    total_elapsed = time.time() - eval_start_time
    logging.info(f"[UQ evaluate_func] COMPLETE - score={score}, total time={total_elapsed:.2f}s")
    return function_class


system_prompt = """
You are an Uncertainty Quantification Scientist. Design an unsupervised Estimator class that detects LLM hallucinations from precomputed statistics.

## Task
Write a class named `HallucinationEstimator` inheriting from `Estimator` (already provided).
Your `__call__` method receives `stats` dict and MUST return np.ndarray of shape [N] (one score per sample).

```python
class HallucinationEstimator(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(context, stats_dependencies=[...], level="sequence")
    def __str__(self) -> str:
        return "HallucinationEstimator"
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # MUST return shape [N] where N = number of samples
        ...  # Return uncertainty scores (higher = more likely hallucination)
```

## Available Statistics in `stats` (N = number of samples)

**Sequence-level (already aggregated, shape [N] - use directly):**
- `sequence_logprob`: np.ndarray[float32, [N]] - sum of token log-probs per sequence (more negative = less confident)
- `mean_token_entropy`: np.ndarray[float32, [N]] - mean entropy per sequence (higher = more uncertain)

**Token-level (lists of length N, inner lengths vary per sample):**
- `greedy_log_likelihoods`: List[List[float]] - length N, inner list[i] has n_i floats (per-token log p(token|prefix))
- `greedy_log_probs`: List[np.ndarray[float32, [n_i, V]]] - length N, each array is [n_tokens_i, vocab_size]
- `greedy_tokens`: List[List[int]] - length N, inner list[i] has n_i token IDs

**Attention (lists of length N):**
- `attention_all`: List[np.ndarray[float32, [L*H, n_i, n_i]]] - L=num_layers, H=num_heads, n_i=tokens in sample i

**Embeddings (D = hidden_dim):**
- `embeddings_decoder`: np.ndarray[float32, [N, D]] - mean embedding per sequence (last layer)
- `embeddings_decoder_{layer}`: np.ndarray[float32, [N, D]] - mean embedding per sequence (specific layer)
- `token_embeddings_decoder`: np.ndarray[float32, [sum(n_i), D]] - all tokens concatenated

## Expected Estimator Structure
- Use different feature combinations to create a single score for each sample.
- Avoid introducing hyperparameters, use simple combinations.
- Prioritize simple, interpretable combinations based on concrete hypotheses on how hallucination detection can be improved leveraging given statistics.

## Constraints
- Class MUST be named `HallucinationEstimator`
- Output MUST be np.ndarray of shape [N] (one scalar per sample)
- Pure Python + numpy only
- Deterministic
- Do NOT redefine Estimator base class

## Response Format
First, write 3-4 sentences reflecting on estimators given below.
Then, formlate a concrete hypothesis on how hallucination detection can be improved leveraging given statistics.
Then provide a definition of a new estimator class in a single ```python code block.
Keep reasoning concise - focus on the code.
"""

# **If context[\"supervised\"]=True:**
# - `train_sequence_logprob`: np.ndarray[float32, [N_train]]
# - `train_mean_token_entropy`: np.ndarray[float32, [N_train]]
# - `train_greedy_log_likelihoods`: List[List[float]] - length N_train
# - `train_greedy_log_probs`: List[np.ndarray[float32, [n_i, V]]] - length N_train
# - `train_metrics`: np.ndarray[float64, [N_train]] - binary labels (1=hallucination, 0=correct)
# - `train_embeddings_decoder`: np.ndarray[float32, [N_train, D]]

append_prompt = ""


TASK_REGISTRY.register(
    "uncertainty_quantification",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)




