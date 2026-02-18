import copy
import os
import csv
from experiments.uq_utils import parse_auc_ci

def evaluate_hallu(cfg, generate_input, evaluate_func, function_class, split):
    """Evaluate a function class on the test set.
    
    Args:
        cfg: Configuration object
        generate_input: Function to generate the input data
        evaluate_func: Function to evaluate the function class
        function_class: FunctionClass instance to evaluate
        split: Split to evaluate on ("train", "val" or "testset". Test set might be from different source compared to train/val)
    Returns:
        Tuple of (test_result, num_test_samples)
    """
    test_data = generate_input(cfg.task, split)
    num_test_samples = len(test_data[0]) if isinstance(test_data, tuple) else -1
    test_fc = copy.deepcopy(function_class)
    test_result = evaluate_func(cfg, copy.deepcopy(test_data), test_fc)
    return test_result, num_test_samples


def write_test_metrics_row(cfg, test_result, split, num_test_samples, iteration):
    """Write a test metrics row to test_metrics.csv.
    
    Args:
        cfg: Configuration object
        test_result: Result object from evaluate_func with extra_metrics
        split: split used for evaluation ("train", "val" or "testset". Test set might be from different source compared to train/val)
        num_test_samples: Number of test samples
        iteration: Iteration number (use -1 for initial)
    """
    em = getattr(test_result, "extra_metrics", {}) or {}
    lower, upper = parse_auc_ci(em.get("auc_ci", "(nan, nan)"))
    
    test_csv_path = os.path.join(cfg.logs_dir, "test_metrics.csv")
    new_file = not os.path.exists(test_csv_path)
    
    with open(test_csv_path, "w" if new_file else "a", newline="") as tf:
        writer = csv.writer(tf)
        if new_file:
            writer.writerow([
                "iteration",
                "function_num",
                "prompt_num",
                "split",
                "num_test_samples",
                "auc",
                "precision",
                "recall",
                "invalid_responses_percentage",
                "auc_lower_bound",
                "auc_upper_bound",
                "prompt_text",
            ])
        writer.writerow([
            iteration,
            test_result.function_num,
            test_result.prompt_num,
            split,
            num_test_samples,
            em.get("auc", ""),
            em.get("precision_at_0_5", ""),
            em.get("recall_at_0_5", ""),
            em.get("invalid_responses_percentage", ""),
            lower,
            upper,
            em.get("prompt_text", ""),
        ])
