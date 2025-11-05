import pandas as pd
import logging
import csv, os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from packing.evaluate.registry import TASK_REGISTRY


def _bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, B: int = 1000) -> str:
    """Compute bootstrap confidence interval for ROC-AUC."""
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(B):
        idxs = np.random.choice(np.arange(n), size=n, replace=True)
        if (y_true[idxs] == 1).any() and (y_true[idxs] == 0).any():
            try:
                bs = roc_auc_score(y_true[idxs], y_score[idxs])
                bootstrapped_scores.append(bs)
            except Exception:
                pass
    if bootstrapped_scores:
        lower = float(np.percentile(bootstrapped_scores, 2.5))
        upper = float(np.percentile(bootstrapped_scores, 97.5))
        return f"({lower:.4f}, {upper:.4f})"
    return "(nan, nan)"


def get_initial_func(cfg: DictConfig):
    def make_prompt() -> str:
        return (
            "You verify atomic claims. Output FALSE if the claim is hallucinated (false), "
            "otherwise output exactly TRUE."
        )

    initial_function = make_prompt
    function_str_to_extract = "make_prompt"
    return initial_function, function_str_to_extract


def _load_claims_and_labels(csv_path: str, do_filter: bool) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(csv_path)

    if do_filter:
        assert "is_good" in df.columns, "CSV must contain 'is_good' column"
        if pd.api.types.is_bool_dtype(df["is_good"]):
            df = df[df["is_good"] == True]
        else:
            df = df[(df["is_good"] == "GOOD") | (df["is_good"] == "True")]

    assert "claim" in df.columns and "is_hallu" in df.columns, (
        f"CSV must contain 'claim' and 'is_hallu' columns. Found {df.columns.tolist()}"
    )
    # Drop rows where claim is empty or NaN after stripping
    df["claim"] = df["claim"].astype(str).str.strip()
    df = df[df["claim"] != ""]

    # Convert is_hallu to int, remove rows where conversion fails
    df = df[pd.to_numeric(df["is_hallu"], errors="coerce").notnull()]
    df["is_hallu"] = df["is_hallu"].astype(int)

    claims = df["claim"].tolist()
    labels = df["is_hallu"].tolist()

    assert len(claims) == len(labels) and len(claims) > 0, f"No data loaded from CSV, or {len(claims)=} != {len(labels)=}"
    return claims, labels


def _prepare_splits(cfg_task: DictConfig) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    claims, labels = _load_claims_and_labels(cfg_task.dataset_path, cfg_task.filter_datasets)
    seed = int(cfg_task.split_seed)
    sample_size = int(cfg_task.sample_size)

    indices = np.arange(len(claims))
    # Stratified sampling of a fixed-size subset
    _, sample_idx = train_test_split(
        indices,
        test_size=sample_size,
        stratify=labels,
        random_state=seed,
    )

    # Build the sampled arrays
    sample_claims = [claims[i] for i in sample_idx]
    sample_labels = [labels[i] for i in sample_idx]

    # Even 50/50 stratified split into train/val
    claims_train, claims_val, labels_train, labels_val = train_test_split(
        sample_claims,
        sample_labels,
        test_size=0.5,
        stratify=sample_labels,
        random_state=seed,
    )
    return (claims_train, labels_train), (claims_val, labels_val)


def generate_input(cfg_task: DictConfig, set: str):
    name = (set or "").lower()
    if name == "testset":
        test_path = getattr(cfg_task, "test_dataset_path", None)
        assert isinstance(test_path, str) and len(test_path) > 0, "cfg.task.test_dataset_path must be provided"
        assert os.path.isfile(test_path), f"Test dataset not found at {test_path}"
        return _load_claims_and_labels(test_path, False)
    train_split, val_split = _prepare_splits(cfg_task)
    if name == "train":
        return train_split
    return val_split


def _call_vllm(detector_url: str, model_id: str, system_prompt: str, claim_text: str, max_tokens: int, temperature: float = 0.0) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Claim: {claim_text}",
            },
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 1.0,
    }
    resp = requests.post(detector_url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_label_to_score(text: str) -> float:
    # Normalize common noise
    s = (text or "").strip().upper()
    if "FALSE" in s:
        return 1.0
    if "TRUE" in s:
        return 0.0
    # Conservative approach: consider unclear responses as hallucination
    return 1.0
    # Approach below introduces invalid scores, which complicate the evaluation and misalign with the approach used in ICLR submission
    
    # # Avoid objective hacking via including both responses in the generation
    # if "TRUE" in s and "FALSE" in s:
    #     return -1.0
    # if "TRUE" in s:
    #     return 0.0  # TRUE => not hallucination => score for positive class = 0
    # if  "FALSE" in s:
    #     return 1.0  # FALSE => hallucination => score for positive class = 1
    # # Fallback: mark invalid
    # return -1.0


def evaluate_func(cfg: DictConfig, data: Tuple[List[str], List[int]], function_class):
    function_str = function_class.function_str
    imports = function_class.imports_str

    # Execute the candidate function to obtain a prompt string
    try:
        globals_dict = {}
        if imports:
            exec(imports, globals_dict)
        local_dict = {}
        exec(function_str, globals_dict, local_dict)
        make_prompt = local_dict.get(cfg.function_str_to_extract)
        assert make_prompt is not None, "make_prompt() not found"
        prompt_text = make_prompt()
        assert isinstance(prompt_text, str) and len(prompt_text) > 0, "make_prompt() must return non-empty str"
    except Exception as e:
        function_class.fail_flag = 1
        function_class.score = cfg.task.failed_score
        function_class.true_score = cfg.task.failed_score
        function_class.fail_exception = str(e)
        return function_class

    claims, labels = data
    assert isinstance(claims, list) and isinstance(labels, list)
    assert len(claims) == len(labels) and len(claims) > 0

    detector_url = f"http://localhost:{cfg.task.detector_vllm_port}/v1/chat/completions"
    model_id = cfg.task.detector_model
    max_tokens = getattr(cfg.task, "detector_max_tokens", 4)
    temperature = float(getattr(cfg.task, "detector_temperature", 0.0))
    max_workers = int(getattr(cfg.task, "detector_max_concurrent_requests", 8))
    # Per-sample threaded chat requests
    scores_list: List[float] = [0.0] * len(claims)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_call_vllm, detector_url, model_id, prompt_text, claim, max_tokens, temperature): idx
            for idx, claim in enumerate(claims)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                text = fut.result()
            except Exception as req_err:
                logging.warning(f"Detector request failed at idx={idx}: {req_err}")
                text = ""  # invalid
            scores_list[idx] = _parse_label_to_score(text)

    scores = np.asarray(scores_list, dtype=float)

    valid_mask = scores >= 0.0
    num_valid = int(valid_mask.sum())
    invalid_responses_percentage = 100.0 * (1.0 - (num_valid / max(1, len(scores))))

    # Always record the exact prompt used
    function_class.extra_metrics["prompt_text"] = prompt_text

    if num_valid >= 2:
        y_true = np.asarray(labels, dtype=int)[valid_mask]
        y_score = scores[valid_mask]
        # AUC requires at least one positive and one negative
        if (y_true == 1).any() and (y_true == 0).any():
            auc = float(roc_auc_score(y_true, y_score))
            score = round(100.0 * auc, 4)
            B = getattr(cfg.task, "bootstrap_samples", 1000)
            auc_ci = _bootstrap_auc_ci(y_true, y_score, B)
            # Precision/Recall at 0.5 threshold on valid predictions
            y_pred = (y_score >= 0.5).astype(int)
            try:
                prec = float(precision_score(y_true, y_pred, zero_division=0))
                rec = float(recall_score(y_true, y_pred, zero_division=0))
            except Exception:
                prec, rec = "unavailable", "unavailable"
            # Save detailed metrics
            function_class.extra_metrics = {
                "auc": auc,
                "auc_ci": auc_ci,
                "precision_at_0_5": prec,
                "recall_at_0_5": rec,
                "invalid_responses_percentage": float(invalid_responses_percentage),
                "num_valid_responses": int(num_valid),
                "num_total": int(len(scores)),
                "prompt_text": prompt_text,
            }
        else:
            score = float(cfg.task.failed_score)
    else:
        score = float(cfg.task.failed_score)

    function_class.score = score
    function_class.true_score = score
    function_class.fail_flag = 0
    function_class.correct_flag = 1
    return function_class


# Encourage formatting that keeps reasoning outside and code inside a fenced block
append_prompt = (
    "After any reasoning, provide the function inside a fenced python code block.\n"
    "Use exactly this format (no extra text inside the fence):\n\n"
    "```python\n"
    "def make_prompt() -> str:\n"
    "    \"\"\"Short description of the hypothesis being tested here ...\"\"\"\n"
    "    return (\n"
    "        \"... your system prompt string here ...\"\n"
    "    )\n"
    "```\n\n"
    "Keep auxiliary reasoning and explanations outside the code block."
)

system_prompt = (
    "You are an expert in evolutionary optimization for prompt engineering. You are designing a SYSTEM prompt for an LLM-based binary factuality detector.\n"
    "Your objective: maximize ROC-AUC when classifying atomic factual claims as hallucinated (FALSE) vs truthful (TRUE).\n"
    "Output format: a Python function make_prompt(), which takes no arguments and returns a system prompt string. Include a short description of the hypothesis behind the prompt design being tested in the documentation string.\n"
    "Input format: one or more examples of make_prompt() function, which were tested already and have known ROC-AUC/precision/recall scores."
    "Hard constraint: LLM-based binary factuality detector must eventually output exactly TRUE or FALSE.\n"
    "Take a deep breath and reason carefully step-by-step before returning the function make_prompt()."
)


TASK_REGISTRY.register(
    "hallu",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)


