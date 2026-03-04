import os
import logging
import time
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple, Generator, Optional

import torch
import numpy as np
import scipy
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from packing.evaluate.registry import TASK_REGISTRY
from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.pred_rej_area import PredictionRejectionArea
from packing.logging.function_class import FunctionClass


MAX_REJECTION_RATIO = 0.5

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
        "chunked": None,
    }
    if input_regime not in suffix_by_regime:
        raise ValueError(
            f"Unsupported input_regime='{input_regime}'. "
            "Expected one of: logits, hidden_states, attentions, full, chunked."
        )
    return suffix_by_regime[input_regime]


def _required_sample_keys_for_input_regime(input_regime: str) -> set[str]:
    required_by_regime = {
        "logits": {"sequence_logprobs", "top_k_logits", "top_k_indices"},
        "hidden_states": {"sequence_logprobs", "hidden_states"},
        "attentions": {"sequence_logprobs", "attentions"},
        "full": {"sequence_logprobs", "top_k_logits", "top_k_indices", "hidden_states", "attentions"},
        "chunked": {"sequence_logprobs", "top_k_logits", "top_k_indices", "hidden_states", "attentions"},
    }
    return required_by_regime[input_regime]


def _validate_sample_keys(sample: Dict[str, Any], input_regime: str) -> None:
    """Validate that a single sample has all required keys for the input regime."""
    required_keys = _required_sample_keys_for_input_regime(input_regime)
    sample_keys = set(sample.keys())
    missing_keys = sorted(required_keys - sample_keys)
    if missing_keys:
        raise ValueError(f"Sample missing required keys for input_regime='{input_regime}': {missing_keys}")


def _discover_chunk_files(base_path: str) -> List[str]:
    """
    Discover chunk files for a given base path.
    
    If base_path is 'train_raw.pt', looks for:
    - 'train_raw_chunk_0.pt', 'train_raw_chunk_1.pt', ... (if any exist)
    - Falls back to ['train_raw.pt'] if no chunks found
    
    Returns list of chunk file paths in order.
    """
    base = Path(base_path)
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    
    chunk_files = []
    for idx in itertools.count():
        chunk_path = parent / f"{stem}_chunk_{idx}{suffix}"
        if chunk_path.exists():
            chunk_files.append(str(chunk_path))
        else:
            break
    
    if not chunk_files:
        if base.exists():
            return [str(base)]
        else:
            raise FileNotFoundError(f"No chunk files or base file found for {base_path}")
    
    expected = list(range(len(chunk_files)))
    actual = []
    for f in chunk_files:
        idx = int(Path(f).stem.split("_chunk_")[-1])
        actual.append(idx)

    if actual != expected:
        raise ValueError("Chunk files must be contiguous starting at 0.")

    return chunk_files


def _stream_samples_from_chunks(
    chunk_files: List[str],
    indices: Optional[np.ndarray] = None,
    input_regime: str = "chunked",
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields samples from chunk files one by one.
    
    Args:
        chunk_files: List of chunk file paths in order
        indices: Optional array of indices to select (for train/val splits). 
                If None, yields all samples in order.
        input_regime: Input regime for validation
    
    Yields:
        Individual sample dictionaries
    """
    if indices is not None:
        indices_set = set(int(i) for i in indices)
    
    global_idx = 0
    for chunk_path in chunk_files:
        chunk_data = _load_raw(chunk_path)
        chunk_samples = chunk_data.get("samples", [])
        
        for sample in chunk_samples:
            if indices is None or global_idx in indices_set:
                _validate_sample_keys(sample, input_regime)
                yield sample
            global_idx += 1
        
        del chunk_data


def _count_samples_in_chunks(chunk_files: List[str]) -> int:
    """Count total number of samples across all chunk files."""
    total = 0
    for chunk_path in chunk_files:
        chunk_data = _load_raw(chunk_path)
        total += len(chunk_data.get("samples", []))
        del chunk_data
    return total


def _get_meta_from_chunks(chunk_files: List[str]) -> Dict[str, Any]:
    """Get metadata from the first chunk file."""
    if not chunk_files:
        return {}
    chunk_data = _load_raw(chunk_files[0])
    meta = chunk_data.get("meta", {})
    del chunk_data
    return meta


def _validate_dataset_startup(
    dataset_name: str,
    dataset_path: str,
    input_regime: str,
    min_samples_per_dataset: int,
    chunk_files: Optional[List[str]] = None,
) -> None:
    """Validate dataset startup parameters."""
    expected_suffix = _expected_suffix_for_input_regime(input_regime)
    
    if input_regime != "chunked" and expected_suffix is not None:
        if not dataset_path.endswith(expected_suffix):
            raise ValueError(
                f"Dataset '{dataset_name}' path mismatch for input_regime='{input_regime}'. "
                f"Expected suffix '{expected_suffix}', got path '{dataset_path}'."
            )
    
    files_to_check = chunk_files if chunk_files else [dataset_path]
    
    # Get first sample for validation
    if input_regime == "chunked":
        first_chunk = _load_raw(files_to_check[0])
        samples = first_chunk.get("samples", [])
        if not samples:
            raise ValueError(f"Dataset '{dataset_name}' first chunk has no samples")
        first_sample = samples[0]
        del first_chunk
    else:
        raw_data = _load_raw(dataset_path)
        samples = raw_data.get("samples", [])
        if not samples:
            raise ValueError(f"Dataset '{dataset_name}' has no samples")
        first_sample = samples[0]
        del raw_data
    
    if "ground_truth_score" not in first_sample:
        raise ValueError(
            f"Dataset '{dataset_name}' first sample is missing 'ground_truth_score' field."
        )
    
    _validate_sample_keys(first_sample, input_regime)
    
    # Count total samples for min check
    if input_regime == "chunked":
        total_samples = _count_samples_in_chunks(files_to_check)
    else:
        raw_data = _load_raw(dataset_path)
        total_samples = len(raw_data.get("samples", []))
        del raw_data
    
    if total_samples < min_samples_per_dataset:
        raise ValueError(
            f"Dataset '{dataset_name}' has {total_samples} samples, "
            f"which is below min_samples_per_dataset={min_samples_per_dataset}"
        )


def _prepare_train_val_indices_chunked(
    chunk_files: List[str],
    sample_size: int,
    val_size: float,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare train/val indices for chunked data."""
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    
    total_samples = _count_samples_in_chunks(chunk_files)
    indices = np.arange(total_samples)
    
    if sample_size > 0 and sample_size < total_samples:
        _, sampled_indices = train_test_split(
            indices,
            test_size=sample_size,
            random_state=split_seed,
            shuffle=True,
        )
    else:
        sampled_indices = indices

    if len(sampled_indices) < 2:
        raise ValueError(
            f"sample_size leaves too few rows for splitting: {len(sampled_indices)}"
        )

    train_idx, val_idx = train_test_split(
        sampled_indices,
        test_size=val_size,
        random_state=split_seed,
        shuffle=True,
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Empty split produced: train={len(train_idx)}, val={len(val_idx)}. "
            f"Adjust sample_size={sample_size} or val_size={val_size}."
        )
    return train_idx, val_idx


def _create_chunked_dataset_struct(
    chunk_files: List[str],
    indices: Optional[np.ndarray] = None,
    input_regime: str = "chunked",
) -> Dict[str, Any]:
    """Create a lightweight dataset structure for chunked data."""
    return {
        "chunk_files": chunk_files,
        "indices": indices,
        "input_regime": input_regime,
        "meta": _get_meta_from_chunks(chunk_files),
    }


def _load_samples_for_evaluation(dataset_struct: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Generator that yields samples for evaluation from a dataset struct."""
    if "samples" in dataset_struct and "chunk_files" in dataset_struct:
        raise ValueError("Dataset struct cannot contain both 'samples' and 'chunk_files'.")

    if "samples" not in dataset_struct and "chunk_files" not in dataset_struct:
        raise ValueError("Dataset struct must contain either 'samples' or 'chunk_files'.")

    if "samples" in dataset_struct:
        for sample in dataset_struct["samples"]:
            yield sample
        return
    
    if "chunk_files" in dataset_struct:
        yield from _stream_samples_from_chunks(
            chunk_files=dataset_struct["chunk_files"],
            indices=dataset_struct.get("indices"),
            input_regime=dataset_struct.get("input_regime", "chunked"),
        )
        return
    
    raise ValueError("Dataset struct missing required fields for sample loading")


def _subset_raw(raw_all: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    """Subset raw data by indices (for non-chunked regime)."""
    return {
        "samples": [raw_all["samples"][int(i)] for i in indices],
        "meta": raw_all.get("meta", {}),
    }


def generate_input(cfg_task: DictConfig, split: str):
    """Load raw statistics for multiple datasets with chunked input support."""
    split_name = (split or "").lower()
    datasets_cfg = getattr(cfg_task, "datasets", None)
    input_regime = str(getattr(cfg_task, "input_regime", "logits"))
    min_samples_per_dataset = int(getattr(cfg_task, "min_samples_per_dataset", 1))
    sample_size = int(getattr(cfg_task, "sample_size", 0))
    val_size = float(getattr(cfg_task, "val_size", 0.2))
    split_seed = int(getattr(cfg_task, "split_seed", 441))

    if datasets_cfg is None:
        raise ValueError(
            "cfg.task.datasets must be provided for multi-dataset UQ task. "
            "Expected a list of dataset configs with 'name', 'train_raw_path', "
            "'test_raw_path', and 'metric_name'."
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
        dataset_name = ds_cfg.get("name")
        train_raw_path = ds_cfg.get("train_raw_path")
        test_raw_path = ds_cfg.get("test_raw_path")
        metric_name = ds_cfg.get("metric_name", "unknown")

        if not dataset_name:
            raise ValueError("Each dataset config must have a 'name' field")
        
        if split_name == "testset":
            if not test_raw_path:
                raise ValueError(
                    f"Dataset '{dataset_name}' must have a 'test_raw_path' field for split='testset'"
                )
            logging.info(
                f"[UQ Multi] Loading dataset '{dataset_name}' test split from {test_raw_path}"
            )
            
            if input_regime == "chunked":
                chunk_files = _discover_chunk_files(test_raw_path)
                _validate_dataset_startup(
                    dataset_name=dataset_name,
                    dataset_path=test_raw_path,
                    input_regime=input_regime,
                    min_samples_per_dataset=min_samples_per_dataset,
                    chunk_files=chunk_files,
                )
                eval_struct = _create_chunked_dataset_struct(
                    chunk_files=chunk_files,
                    indices=None,
                    input_regime=input_regime,
                )
            else:
                eval_raw = _load_raw(test_raw_path)
                _validate_dataset_startup(
                    dataset_name=dataset_name,
                    dataset_path=test_raw_path,
                    raw_data=eval_raw,
                    input_regime=input_regime,
                    min_samples_per_dataset=min_samples_per_dataset,
                )
                eval_struct = eval_raw
                
        else:
            if not train_raw_path:
                raise ValueError(
                    f"Dataset '{dataset_name}' must have a 'train_raw_path' field "
                    f"for split='{split_name or 'val'}'"
                )
            logging.info(
                f"[UQ Multi] Loading dataset '{dataset_name}' train split source from {train_raw_path}"
            )
            
            if input_regime == "chunked":
                chunk_files = _discover_chunk_files(train_raw_path)
                _validate_dataset_startup(
                    dataset_name=dataset_name,
                    dataset_path=train_raw_path,
                    input_regime=input_regime,
                    min_samples_per_dataset=min_samples_per_dataset,
                    chunk_files=chunk_files,
                )
                train_idx, val_idx = _prepare_train_val_indices_chunked(
                    chunk_files=chunk_files,
                    sample_size=sample_size,
                    val_size=val_size,
                    split_seed=split_seed,
                )
                indices = train_idx if split_name == "train" else val_idx
                eval_struct = _create_chunked_dataset_struct(
                    chunk_files=chunk_files,
                    indices=indices,
                    input_regime=input_regime,
                )
            else:
                train_raw_all = _load_raw(train_raw_path)
                _validate_dataset_startup(
                    dataset_name=dataset_name,
                    dataset_path=train_raw_path,
                    raw_data=train_raw_all,
                    input_regime=input_regime,
                    min_samples_per_dataset=min_samples_per_dataset,
                )
                train_idx, val_idx = _prepare_train_val_indices(
                    raw_data=train_raw_all,
                    sample_size=sample_size,
                    val_size=val_size,
                    split_seed=split_seed,
                )
                subset_indices = train_idx if split_name == "train" else val_idx
                eval_struct = _subset_raw(train_raw_all, subset_indices)

        datasets.append({
            "name": dataset_name,
            "raw_data": eval_struct,
            "metric_name": metric_name,
        })

    return {
        "datasets": datasets,
        "supervised": False,
    }


def get_initial_func(cfg: DictConfig):
    """Provide an initial baseline estimator implementation."""

    class HallucinationDetector(Estimator):
        def __init__(self, context: Dict[str, Any]) -> None:
            super().__init__(
                context,
                stats_dependencies=["sequence_logprobs"],
                level="sequence",
            )

        def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
            total_neg_logprob = np.array(
                [-np.sum(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]],
                dtype=np.float32
            )
            return total_neg_logprob

    return HallucinationDetector, "HallucinationDetector", "class"


def _evaluate_single_dataset(
    cfg: DictConfig,
    dataset_info: Dict[str, Any],
    EstimatorClass: type,
    input_regime: str,
    use_gpu_hidden_states: bool,
    hidden_states_device: str,
    use_dependency_gated_stats: bool,
) -> Dict[str, Any]:
    """Evaluate the estimator on a single dataset and return PRR metrics."""
    name = dataset_info["name"]
    raw_struct = dataset_info["raw_data"]
    metric_name = dataset_info["metric_name"]

    logging.info(f"[UQ Multi] Evaluating dataset '{name}' (streaming mode)")

    # Create estimator context
    meta = raw_struct.get("meta", {}) if isinstance(raw_struct, dict) else {}
    context = {
        "supervised": False,
        "meta": {"eval_meta": meta},
    }

    estimator: Estimator = EstimatorClass(context)
    if not isinstance(estimator, Estimator):
        raise TypeError(
            f"EstimatorClass must produce an Estimator subclass, got {type(estimator)}"
        )
    estimator_stats_dependencies = (
        list(estimator.stats_dependencies) if use_dependency_gated_stats else None
    )

    # Determine chunk size
    chunk_size_cfg = getattr(cfg.task, "chunk_size", None)
    chunk_size = int(chunk_size_cfg) if chunk_size_cfg else None

    scores_list: List[np.ndarray] = []
    ground_truth_scores: List[float] = []
    sample_buffer: List[Dict[str, Any]] = []

    total_samples = 0

    for sample in _load_samples_for_evaluation(raw_struct):
        if "ground_truth_score" not in sample:
            raise ValueError(
                f"Dataset '{name}': sample missing 'ground_truth_score' field."
            )

        ground_truth_scores.append(float(sample["ground_truth_score"]))
        sample_buffer.append(sample)
        total_samples += 1

        # Process chunk
        if chunk_size is not None and len(sample_buffer) >= chunk_size:
            chunk_raw = {"samples": sample_buffer, "meta": meta}
            all_stats_chunk, _, _, _ = make_all_stats(
                eval_raw=chunk_raw,
                train_raw=None,
                background_raw=None,
                supervised=False,
                train_stats_cache=None,
                background_stats_cache=None,
                input_regime=input_regime,
                use_gpu_hidden_states=use_gpu_hidden_states,
                hidden_states_device=hidden_states_device,
                required_stats=estimator_stats_dependencies,
            )

            chunk_scores = np.asarray(estimator(all_stats_chunk), dtype=np.float32)

            if chunk_scores.ndim != 1 or chunk_scores.shape[0] != len(sample_buffer):
                raise ValueError(
                    f"Dataset '{name}': estimator returned shape {chunk_scores.shape}, "
                    f"expected ({len(sample_buffer)},)"
                )

            scores_list.append(chunk_scores)
            sample_buffer = []

    # Process remaining samples
    if sample_buffer:
        chunk_raw = {"samples": sample_buffer, "meta": meta}
        all_stats_chunk, _, _, _ = make_all_stats(
            eval_raw=chunk_raw,
            train_raw=None,
            background_raw=None,
            supervised=False,
            train_stats_cache=None,
            background_stats_cache=None,
            input_regime=input_regime,
            use_gpu_hidden_states=use_gpu_hidden_states,
            hidden_states_device=hidden_states_device,
            required_stats=estimator_stats_dependencies,
        )

        chunk_scores = np.asarray(estimator(all_stats_chunk), dtype=np.float32)

        if chunk_scores.ndim != 1 or chunk_scores.shape[0] != len(sample_buffer):
            raise ValueError(
                f"Dataset '{name}': estimator returned shape {chunk_scores.shape}, "
                f"expected ({len(sample_buffer)},)"
            )

        scores_list.append(chunk_scores)

    if total_samples == 0:
        raise ValueError(f"Dataset '{name}' has no samples to evaluate")

    ground_truth_scores_np = np.asarray(ground_truth_scores, dtype=np.float32)
    uncertainty_scores = np.concatenate(scores_list)

    # Safety check (prevents silent misalignment bugs)
    if len(uncertainty_scores) != len(ground_truth_scores_np):
        raise ValueError(
            f"Dataset '{name}': mismatch between scores ({len(uncertainty_scores)}) "
            f"and ground truth ({len(ground_truth_scores_np)})."
        )

    logging.info(
        f"[UQ Multi] Dataset '{name}': {len(ground_truth_scores_np)} samples, "
        f"metric={metric_name}"
    )

    prr_metric = PredictionRejectionArea(max_rejection=MAX_REJECTION_RATIO)
    prr_value = float(
        prr_metric(
            uncertainty_scores.tolist(),
            ground_truth_scores_np.tolist(),
        )
    )

    return {
        "prediction_rejection_ratio": round(prr_value, 4),
        "original_dataset_metric_name": metric_name,
        "n_samples": len(ground_truth_scores_np),
    }

def evaluate_func(
    cfg: DictConfig,
    data: Dict[str, Any],
    function_class: FunctionClass,
) -> FunctionClass:
    """Evaluate a candidate estimator class on multiple datasets."""
    eval_start_time = time.time()
    logging.info(f"[UQ Multi evaluate_func] START - function_num={getattr(function_class, 'function_num', 'N/A')}")

    datasets = data["datasets"]
    supervised = data.get("supervised", False)
    if supervised:
        raise NotImplementedError("Supervised mode is not implemented for multi-dataset UQ evaluation.")

    logging.info(f"[UQ Multi evaluate_func] Evaluating on {len(datasets)} datasets")

    # Parse and instantiate the estimator class
    function_str = function_class.function_str
    imports = function_class.imports_str
    globals_dict: Dict[str, Any] = {
        "Estimator": Estimator, "np": np, "scipy": scipy,
        "Dict": Dict, "Any": Any, "Tuple": Tuple,
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
    use_gpu_hidden_states = bool(getattr(cfg.task, "use_gpu_hidden_states", False))
    hidden_states_device = str(getattr(cfg.task, "hidden_states_device", "cuda"))
    use_dependency_gated_stats = bool(getattr(cfg.task, "use_dependency_gated_stats", True))

    # Evaluate on each dataset
    results: Dict[str, Dict[str, Any] | str] = {"task_variant": "multi-criteria optimization"}
    all_prr_values: List[float] = []

    for dataset_info in datasets:
        dataset_result = _evaluate_single_dataset(
            cfg=cfg, dataset_info=dataset_info,
            EstimatorClass=EstimatorClass,
            input_regime=input_regime,
            use_gpu_hidden_states=use_gpu_hidden_states,
            hidden_states_device=hidden_states_device,
            use_dependency_gated_stats=use_dependency_gated_stats,
        )
        results[dataset_info["name"]] = dataset_result
        all_prr_values.append(dataset_result["prediction_rejection_ratio"])

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


system_prompt = """Take the role of a world-class NLP researcher with specialization in Uncertainty Quantification. The overall goal is to design and implement an unsupervised hallucination detector. 
You are doing an evolutionary search for the best detector, so it is the following loop:
- You review the results of existing best performing detectors
- You design a new candidate detector
- It is evaluated on test sets for 4 datasets. If it performs well, it will be shown to you in the next iteration.

## Task description
The goal is to design and implement HallucinationDetector class that detects LLM hallucinations in atomic claims (10-25 tokens long). 

Here are examples of atomic claims (might be either true or false): "Made in Chelsea is a reality TV", "The father of Sybilla of Normandy is Rollo of Normandy", "Virginia Woolf is the author of The Unicorn in the Garden". We don't know claim's origin -- they might be generated by the same model, by different models, or by a human. For each claim, we use the following chat template: [{"role": "user", "content": "Generate true statement"}, {"role": "assistant", "content": claim_text}], where `claim_text` is the text of the claim, e.g. "Made in Chelsea is a reality TV".

Inputs for the detector are extracted from model forward outputs and include logits, hidden states, and attention maps.

## Performance metric
We use Prediction-Rejection Ratio (PRR) as the performance metric. It measures how well the uncertainty estimator ranks incorrect predictions higher than correct ones. Higher PRR values indicate better uncertainty estimation.
**Important**: we evaluate PRR on the test sets for 4 datasets. Since our goal is to develop a general-purpose uncertainty estimator, we want to avoid situations where the mean PRR is high, but on some datasets PRR is low. Ideally we want consistent improvements across all datasets, or at least no significant drops in PRR.

## HallucinationDetector inputs

(N = number of samples, n_i = claim-token length for sample i, L = number of layers, H = number of attention heads, D = hidden size, k = top-k width)

- "sequence_logprobs": List[np.ndarray] of length N, each array shape [n_i - 1]. Log probabilities (after log_softmax) of generated claim tokens, excluding the first generated token.
- "greedy_logprobs": List[np.ndarray] of length N, each array shape [n_i]. Log probability of the top-1 token at each position.
- "token_entropies": List[np.ndarray] of length N, each array shape [n_i]. Token-level entropy values.
- "greedy_texts": List[str] of length N. Claim texts.
- "greedy_tokens": List[List[int]] of length N. Token IDs for each claim.
- "top_k_logits": List[np.ndarray] of length N, each array shape [n_i, k]. Top-k logits (before log_softmax) for each token; [:, 0] is the top-1 logit.
- "top_k_indices": List[np.ndarray] of length N, each array shape [n_i, k]. Token IDs corresponding to top_k_logits.
- "recent_lookback_attention_weights_from_claim_tokens": List of length N. For sample i: nested list with shape [n_i, min(position, max_lookback), L*H], containing recent lookback attention weights from claim tokens.
- "claim_only_attention_maps": List[np.ndarray] of length N, each array shape [L*H, n_i, n_i]. Attention maps restricted to claim-token region.
- "full_attention_maps": List[np.ndarray] of length N, each array shape [L, H, n_i_full, n_i_full]. Full per-layer, per-head attention maps from forward pass.
- "token_embeddings_decoder": np.ndarray with shape [sum_i n_i, D]. Concatenated last-layer token embeddings over all samples.
- "token_embeddings_decoder_{layer}": np.ndarray with shape [sum_i n_i, D]. Concatenated token embeddings for a specific layer.
- "embeddings_decoder": np.ndarray with shape [N, D]. Mean-pooled last-layer token embedding per sample.
- "embeddings_decoder_{layer}": np.ndarray with shape [N, D]. Mean-pooled token embedding per sample for a specific layer.

You do not have access to logits beyond top-k, so you cannot estimate the missing tail probability mass.

## Design constraints
- Use up to 3 features. You MUST explore one-feature and two-feature detectors as well, don't get fixated on three-feature detectors.
- Use class doc string to 1) concisely describe the detector, 2) leave notes for future iterations.
- In notes, NEVER make up facts. Either summarize some observations about previous detectors as a preliminary conclusion, or formulate a concrete actionable hypothesis for the next iteration.
- In notes, NEVER make statements about performance of proposed candidate detector -- you don't know it yet, it will be evaluated on a test set in the next iteration.
- In notes, NEVER assume new types of inputs will be available in the next iteration -- it won't happen, inputs are restricted by design, the goal is to find the best detector given the inputs.
- In notes, avoid generic hypotheses. When uncertain, err towards not making a note at all.

## Code constraints
- Class MUST be named `HallucinationDetector`
- Output MUST be np.ndarray of shape [N] (one scalar per sample)
- Pure Python + numpy + scipy only
- Deterministic
- Do NOT redefine Estimator base class

## Response Format
First, think step by step about the detectors given below and their results.
Second, formulate a concrete hypothesis on how hallucination detection can be improved given the analysis.
Finally, provide a definition of a new candidate detector class in a single ```python code block.
Stop right after the code block, do not write anything else.
"""

append_prompt = ""


TASK_REGISTRY.register(
    "uncertainty_quantification_multi",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)