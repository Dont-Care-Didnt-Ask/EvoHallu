import os
import logging
import time
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
import scipy
from omegaconf import DictConfig

from packing.evaluate.registry import TASK_REGISTRY
from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.pred_rej_area import PredictionRejectionArea
from packing.logging.function_class import FunctionClass


def _load_raw(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw stats file not found at {path}")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def _expected_suffix_for_input_regime(input_regime: str) -> str | None:
    suffix_by_regime = {
        "logits": "_logits.pt",
        "hidden_states": "_hidden_states.pt",
        "attentions": "_attentions.pt",
        "full": None,
    }
    if input_regime not in suffix_by_regime:
        raise ValueError(
            f"Unsupported input_regime='{input_regime}'. "
            "Expected one of: logits, hidden_states, attentions, full."
        )
    return suffix_by_regime[input_regime]


def _required_sample_keys_for_input_regime(input_regime: str) -> set[str]:
    required_by_regime = {
        "logits": {"sequence_logprobs", "top_k_logits", "top_k_indices"},
        "hidden_states": {"sequence_logprobs", "hidden_states"},
        "attentions": {"sequence_logprobs", "attentions"},
        "full": {"sequence_logprobs", "top_k_logits", "top_k_indices", "hidden_states", "attentions"},
    }
    return required_by_regime[input_regime]


def _validate_dataset_startup(
    dataset_name: str,
    dataset_path: str,
    raw_data: Dict[str, Any],
    input_regime: str,
    min_samples_per_dataset: int,
) -> None:
    samples = raw_data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"Dataset '{dataset_name}' has invalid 'samples' type: {type(samples)}")
    if not samples:
        raise ValueError(f"Dataset '{dataset_name}' has no samples")
    if len(samples) < min_samples_per_dataset:
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(samples)} samples, "
            f"which is below min_samples_per_dataset={min_samples_per_dataset}"
        )

    expected_suffix = _expected_suffix_for_input_regime(input_regime)
    if expected_suffix is not None and not dataset_path.endswith(expected_suffix):
        raise ValueError(
            f"Dataset '{dataset_name}' path mismatch for input_regime='{input_regime}'. "
            f"Expected suffix '{expected_suffix}', got path '{dataset_path}'."
        )

    missing_ground_truth = [idx for idx, sample in enumerate(samples) if "ground_truth_score" not in sample]
    if missing_ground_truth:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing 'ground_truth_score' for {len(missing_ground_truth)} samples. "
            f"First missing index: {missing_ground_truth[0]}."
        )

    required_sample_keys = _required_sample_keys_for_input_regime(input_regime)
    first_sample_keys = set(samples[0].keys())
    missing_keys = sorted(required_sample_keys - first_sample_keys)
    if missing_keys:
        raise ValueError(
            f"Dataset '{dataset_name}' is incompatible with input_regime='{input_regime}'. "
            f"First sample missing keys: {missing_keys}."
        )


def generate_input(cfg_task: DictConfig, split: str):
    """
    Load raw statistics for multiple datasets.

    Expects cfg_task.datasets to be a list of dataset configs, each with:
      - name: str (identifier for the dataset)
      - test_raw_path: str (path to the raw .pt file)
      - metric_name: str (label for the metric, e.g., "accuracy", "rouge_l")
    """
    datasets_cfg = getattr(cfg_task, "datasets", None)
    input_regime = str(getattr(cfg_task, "input_regime", "logits"))
    min_samples_per_dataset = int(getattr(cfg_task, "min_samples_per_dataset", 1))

    if datasets_cfg is None:
        raise ValueError(
            "cfg.task.datasets must be provided for multi-dataset UQ task. "
            "Expected a list of dataset configs with 'name', 'test_raw_path', and 'metric_name'."
        )
    if min_samples_per_dataset < 1:
        raise ValueError(
            f"cfg.task.min_samples_per_dataset must be >= 1, got {min_samples_per_dataset}"
        )

    supervised = bool(getattr(cfg_task, "supervised", False))
    if supervised:
        raise NotImplementedError(
            "Supervised mode is not implemented for multi-dataset UQ evaluation."
        )

    datasets = []
    for ds_cfg in datasets_cfg:
        name = ds_cfg.get("name")
        test_raw_path = ds_cfg.get("test_raw_path")
        metric_name = ds_cfg.get("metric_name", "unknown")

        if not name:
            raise ValueError("Each dataset config must have a 'name' field")
        if not test_raw_path:
            raise ValueError(f"Dataset '{name}' must have a 'test_raw_path' field")

        logging.info(f"[UQ Multi] Loading dataset '{name}' from {test_raw_path}")
        raw_data = _load_raw(test_raw_path)
        _validate_dataset_startup(
            dataset_name=name,
            dataset_path=test_raw_path,
            raw_data=raw_data,
            input_regime=input_regime,
            min_samples_per_dataset=min_samples_per_dataset,
        )

        datasets.append({
            "name": name,
            "raw_data": raw_data,
            "metric_name": metric_name,
        })

    return {
        "datasets": datasets,
        "supervised": False,
    }


def get_initial_func(cfg: DictConfig):
    """
    Provide an initial baseline estimator implementation.
    """

    class HallucinationDetector(Estimator):
        def __init__(self, context: Dict[str, Any]) -> None:
            super().__init__(
                context,
                stats_dependencies=[
                    "sequence_logprobs",
                ],
                level="sequence",
            )

        def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
            total_neg_logprob = np.array(
                [-np.sum(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]],
                dtype=np.float32
            )
            return total_neg_logprob

    initial_class = HallucinationDetector
    class_str_to_extract = "HallucinationDetector"
    code_type = "class"
    return initial_class, class_str_to_extract, code_type


def _evaluate_single_dataset(
    cfg: DictConfig,
    dataset_info: Dict[str, Any],
    EstimatorClass: type,
    input_regime: str,
) -> Dict[str, Any]:
    """
    Evaluate the estimator on a single dataset and return PRR metrics.
    """
    name = dataset_info["name"]
    raw_data = dataset_info["raw_data"]
    metric_name = dataset_info["metric_name"]

    eval_samples = raw_data["samples"]
    num_samples = len(eval_samples)

    logging.info(f"[UQ Multi] Evaluating dataset '{name}': {num_samples} samples, metric={metric_name}")

    # Extract ground truth scores
    ground_truth_scores = []
    for sample in eval_samples:
        if "ground_truth_score" not in sample:
            raise ValueError(
                f"Dataset '{name}': sample missing 'ground_truth_score' field. "
                "Ensure all samples have pre-computed ground truth scores."
            )
        ground_truth_scores.append(float(sample["ground_truth_score"]))

    ground_truth_scores = np.array(ground_truth_scores, dtype=np.float32)

    # Create estimator context
    context = {
        "supervised": False,
        "meta": {
            "eval_meta": raw_data.get("meta", {}),
        },
    }

    estimator: Estimator = EstimatorClass(context)

    if not isinstance(estimator, Estimator):
        raise TypeError(
            f"EstimatorClass must produce an Estimator subclass, got {type(estimator)}"
        )

    # Process in chunks
    chunk_size = max(
        1, int(getattr(cfg.task, "chunk_size", num_samples) or num_samples)
    )

    scores_list: List[np.ndarray] = []

    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk_raw = {
            "samples": eval_samples[start:end],
            "meta": raw_data.get("meta", {}),
        }

        # make_all_stats for the chunk (unsupervised mode)
        all_stats_chunk, _, _, _ = make_all_stats(
            eval_raw=chunk_raw,
            train_raw=None,
            background_raw=None,
            supervised=False,
            train_stats_cache=None,
            background_stats_cache=None,
            input_regime=input_regime,
        )

        chunk_scores = estimator(all_stats_chunk)
        chunk_scores = np.asarray(chunk_scores, dtype=np.float32)

        expected_len = end - start
        if chunk_scores.ndim != 1 or chunk_scores.shape[0] != expected_len:
            raise ValueError(
                f"Dataset '{name}': estimator returned shape {chunk_scores.shape}, "
                f"expected ({expected_len},)"
            )

        scores_list.append(chunk_scores)

    uncertainty_scores = np.concatenate(scores_list)

    # Compute PRR
    # PredictionRejectionArea expects:
    #   - estimator: higher = more uncertain
    #   - target: higher = better quality (less uncertainty)
    # ground_truth_score is already "higher = better", so use directly as target
    prr_metric = PredictionRejectionArea(max_rejection=1.0)
    prr_value = float(prr_metric(uncertainty_scores.tolist(), ground_truth_scores.tolist()))

    return {
        "PRR": round(prr_value, 4),
        "metric_name": metric_name,
        "n_samples": num_samples,
    }


def evaluate_func(
    cfg: DictConfig,
    data: Dict[str, Any],
    function_class: FunctionClass,
) -> FunctionClass:
    """
    Evaluate a candidate estimator class on multiple datasets.

    Returns a nested dictionary with PRR metrics for each dataset:
    {
        "dataset_name": {
            "PRR": float,
            "metric_name": str,
            "n_samples": int,
        },
        ...
    }
    """
    eval_start_time = time.time()

    logging.info(f"[UQ Multi evaluate_func] START - function_num={getattr(function_class, 'function_num', 'N/A')}")

    datasets = data["datasets"]
    supervised = data.get("supervised", False)

    if supervised:
        raise NotImplementedError(
            "Supervised mode is not implemented for multi-dataset UQ evaluation."
        )

    logging.info(f"[UQ Multi evaluate_func] Evaluating on {len(datasets)} datasets")

    # Parse and instantiate the estimator class
    function_str = function_class.function_str
    imports = function_class.imports_str

    globals_dict: Dict[str, Any] = {
        "Estimator": Estimator,
        "np": np,
        "scipy": scipy,
        "Dict": Dict,
        "Any": Any,
        "Tuple": Tuple,
    }

    if imports:
        exec(imports, globals_dict)

    local_dict: Dict[str, Any] = {}
    exec(function_str, globals_dict, local_dict)

    class_name = cfg.function_str_to_extract
    if class_name not in local_dict:
        raise ValueError(f"Class {class_name} not found in candidate program")

    EstimatorClass = local_dict[class_name]

    input_regime = cfg.task.input_regime

    # Evaluate on each dataset; fail the whole candidate on first dataset error.
    results: Dict[str, Dict[str, Any]] = {}
    all_prr_values: List[float] = []

    for dataset_info in datasets:
        name = dataset_info["name"]
        dataset_result = _evaluate_single_dataset(
            cfg=cfg,
            dataset_info=dataset_info,
            EstimatorClass=EstimatorClass,
            input_regime=input_regime,
        )
        results[name] = dataset_result
        all_prr_values.append(dataset_result["PRR"])

    mean_prr = float(np.mean(all_prr_values))
    score = round(100.0 * mean_prr, 4)

    function_class.score = score
    function_class.true_score = score
    function_class.fail_flag = 0
    function_class.correct_flag = 1
    function_class.extra_metrics = results

    total_elapsed = time.time() - eval_start_time
    logging.info(
        f"[UQ Multi evaluate_func] COMPLETE - mean_PRR={mean_prr:.4f}, "
        f"score={score}, datasets={len(datasets)}, time={total_elapsed:.2f}s"
    )

    return function_class


def system_prompt_fabric() -> str:
    """
    System prompt for multi-dataset UQ evaluation.
    Reuses the same prompt structure as single-dataset mode.
    """
    import random

    strong_baselines = [
        "LogProb: sum of negative log-probabilities over sequence",
        "LogProbSlope: negative pearson correlation between token position and log-probability",
        "PositionWeightedGap: for each token t compute (top1_logprob[t] - sequence_logprob[t-1]) and weight it by t / length",
    ]
    baseline = random.choices(population=strong_baselines, weights=[0.4, 0.5, 0.1], k=1)[0]

    ideas = [
        "anomaly detection",
        "position-based weighting (early vs late tokens)",
        "std of some metric across positions",
        "min/max statistics (most likely with low weight to avoid outlier dependence)",
        "scipy stats — skewness, kurtosis, or normality tests (shapiro, jarque_bera) on logprob distribution",
        "aggregation of top or bottom 20% of something instead of all values",
        "tail behavior (approximate using available 512 logits)",
        "consecutive runs — streaks of low-confidence tokens",
        "approximation of second derivative",
        "exponential weighted average",
        "sliding window average",
        "difference between some aggregate over first and last quarter of sequence",
    ]
    idea = random.choice(ideas)

    system_prompt = f"""You are designing general-purpose uncertainty quantification methods for LLM outputs, evaluated by Prediction-Rejection Area (PRR) across multiple tasks (QA, summarization, machine translation, math).

## Inputs (per sample)
- "sequence_logprobs": List[np.ndarray], shape (n_i-1,) — log probabilities of generated tokens
- "top_k_logits": List[np.ndarray], shape (n_i, k) — top-k logits per token (k=512)
- "top_k_indices": List[np.ndarray], shape (n_i, k) — corresponding token indices

## Critical Instructions

**EXPECTED STRUCTURE**
- Build off a strong baseline: {baseline}
- For additional signal, use {idea}

## Code Requirements
- Class named `HallucinationDetector`, subclass of `Estimator`
- Output shape [N] — higher values = more uncertain
- Pure numpy/scipy, deterministic
- Brief docstring describing approach

## Response Format
1. Analyze shown detectors
2. Ideate a novel detector based on EXPECTED STRUCTURE
3. Provide ```python code block with your detector
"""
    return system_prompt


system_prompt = system_prompt_fabric
append_prompt = ""


TASK_REGISTRY.register(
    "uncertainty_quantification_multi",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)
