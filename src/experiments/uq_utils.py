import copy
import os
import csv
import numpy as np

def parse_auc_ci(ci_str: str) -> tuple[float, float]:
    """Parse AUC confidence interval string into numeric bounds.
    
    Args:
        ci_str: String in format "({lower:.4f}, {upper:.4f})"
        
    Returns:
        Tuple of (lower_bound, upper_bound) as floats, or (nan, nan) if parsing fails
    """
    parts = ci_str.strip("()").split(",", 1)
    try:
        lower = float(parts[0].strip())
        upper = float(parts[1].strip()) if len(parts) > 1 else float("nan")
    except Exception:
        lower, upper = float("nan"), float("nan")
    return lower, upper


def evaluate_uq(cfg, generate_input, evaluate_func, function_class, split):
    """Evaluate a function class on the specified split for uncertainty quantification task.
    
    Args:
        cfg: Configuration object
        generate_input: Function to generate the input data
        evaluate_func: Function to evaluate the function class
        function_class: FunctionClass instance to evaluate
        split: Split to evaluate on ("train", "val" or "testset")
    Returns:
        Tuple of (result, num_samples)
    """
    data = generate_input(cfg.task, split)
    num_samples = len(data.get("eval_raw", {}).get("samples", [])) if isinstance(data, dict) else -1
    fc = copy.deepcopy(function_class)
    result = evaluate_func(cfg, copy.deepcopy(data), fc)
    return result, num_samples


def evaluate_uq_multi(cfg, generate_input, evaluate_func, function_class, split):
    """Evaluate a function class on the specified split for uncertainty quantification multi-criteria optimization task.
    
    Args:
        cfg: Configuration object
        generate_input: Function to generate the input data
        evaluate_func: Function to evaluate the function class
        function_class: FunctionClass instance to evaluate
        split: Split to evaluate on ("train", "val" or "testset")
    Returns:
        Result object from evaluate_func with extra_metrics
    """
    data = generate_input(cfg.task, split)
    fc = copy.deepcopy(function_class)
    result = evaluate_func(cfg, copy.deepcopy(data), fc)
    return result


def write_uq_metrics_row(cfg, result, split, num_samples, iteration):
    """Write a UQ metrics row to uq_split_metrics.csv.
    
    Args:
        cfg: Configuration object
        result: Result object from evaluate_func with extra_metrics
        split: split used for evaluation ("train", "val" or "testset")
        num_samples: Number of samples
        iteration: Iteration number (use -1 for initial)
    """
    em = getattr(result, "extra_metrics", {}) or {}
    lower, upper = parse_auc_ci(em.get("AUC 95% confidence interval (bootstrap)", "(nan, nan)"))
    
    csv_path = os.path.join(cfg.logs_dir, "uq_split_metrics.csv")
    new_file = not os.path.exists(csv_path)
    
    with open(csv_path, "w" if new_file else "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "iteration",
                "function_num",
                "prompt_num",
                "split",
                "num_samples",
                "auc",
                "precision_at_median",
                "recall_at_median",
                "auc_lower_bound",
                "auc_upper_bound",
            ])
        writer.writerow([
            iteration,
            getattr(result, "function_num", ""),
            getattr(result, "prompt_num", ""),
            split,
            num_samples,
            em.get("AUC", ""),
            em.get("Precision at median score threshold", ""),
            em.get("Recall at median score threshold", ""),
            lower,
            upper,
        ])

def write_uq_multi_metrics_row(cfg, result, split, iteration):
    """Write a UQ multi metrics row to uq_multi_split_metrics.csv.
    
    Args:
        cfg: Configuration object
        result: Result object from evaluate_func with extra_metrics
        split: split used for evaluation ("train", "val" or "testset")
        num_samples: Number of samples
        iteration: Iteration number (use -1 for initial)
    """
    em = getattr(result, "extra_metrics", {}) or {}
    csv_path = os.path.join(cfg.logs_dir, "uq_multi_split_metrics.csv")
    new_file = not os.path.exists(csv_path)

    datasets = em.get("datasets", {})
    dataset_names = sorted(list(datasets.keys()))
    all_prr_values = em.get("all_prr_values", [])
    
    with open(csv_path, "w" if new_file else "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "iteration",
                "function_num",
                "prompt_num",
                "split",
                *[f"PRR_{dataset_name}" for dataset_name in dataset_names],
                "mean_prr",
                "min_prr",
                "score",
            ])
        
        writer.writerow([
            iteration,
            getattr(result, "function_num", ""),
            getattr(result, "prompt_num", ""),
            split,
            *[datasets[dataset_name].get("prediction_rejection_ratio", "") for dataset_name in dataset_names],
            float(np.mean(all_prr_values)),
            float(np.min(all_prr_values)),
            float(result.score),
        ])