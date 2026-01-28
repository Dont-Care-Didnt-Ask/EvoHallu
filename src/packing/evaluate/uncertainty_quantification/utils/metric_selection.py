"""Utilities for dataset type detection and metric selection."""

import re
from typing import Optional, List
import pandas as pd


# Dataset name -> dataset type mapping
DATASET_CONFIGS = {
    "wmt14_deen": "translation",
    "wmt14_fren": "translation",
    "wmt19_deen_simple_instruct": "translation",
    "wmt19_ruen_simple_instruct": "translation",
    "samsum_simple_instruct": "summarization",
    "xsum_simple_instruct": "summarization",
    # "aeslc_continuation": skipped - config mismatch
    "coqa_simple_instruct": "qa",
    "babi_qa_continuation": "qa_simple",
    "gsm8k_simple_instruct": "gsm8k",
    "triviaqa_simple_instruct": "multi_answer_qa",
    "truthfulqa_simple_instruct": "truthfulqa",
    "mmlu_simple_instruct": "mmlu",
}

# Dataset type -> primary metric mapping
DATASET_TYPE_TO_METRIC = {
    "translation": "bleu",
    "summarization": "rougeL",
    "qa": "f1",
    "qa_simple": "f1",
    "gsm8k": "exact_match",
    "multi_answer_qa": "f1",
    "truthfulqa": "best_rougeL",
    "mmlu": "exact_match",
}


def detect_dataset_type_from_filename(filename: str) -> Optional[str]:
    """
    Detect dataset type from filename by matching against known dataset prefixes.

    Handles files with patterns:
    - {dataset_name}.csv
    - {dataset_name}_limit_{n}.csv
    - PARTIAL_{dataset_name}.csv
    - PARTIAL_{dataset_name}_limit_{n}.csv

    Parameters
    ----------
    filename : str
        Filename to parse (e.g., "wmt14_deen_limit_100.csv").

    Returns
    -------
    dataset_type : str or None
        Dataset type if recognized (e.g., "translation"), None otherwise.
    """
    # Remove PARTIAL_ prefix if present
    base_name = filename
    if base_name.startswith("PARTIAL_"):
        base_name = base_name[8:]  # len("PARTIAL_") = 8

    # Remove .csv extension
    if base_name.endswith(".csv"):
        base_name = base_name[:-4]

    # Remove _limit_{number} suffix if present
    base_name = re.sub(r"_limit_\d+$", "", base_name)

    # Look up in DATASET_CONFIGS
    return DATASET_CONFIGS.get(base_name)


def get_metric_for_dataset_type(dataset_type: str) -> Optional[str]:
    """
    Get the primary metric for a given dataset type.

    Parameters
    ----------
    dataset_type : str
        Dataset type (e.g., "translation", "summarization").

    Returns
    -------
    metric : str or None
        Primary metric name (e.g., "bleu", "rougeL"), or None if unknown.
    """
    return DATASET_TYPE_TO_METRIC.get(dataset_type)


def select_best_available_metric(
    df: pd.DataFrame,
    preferred_metrics: List[str],
) -> Optional[str]:
    """
    Select the first available metric from a list of preferred metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metric columns.
    preferred_metrics : List[str]
        List of metric names in order of preference (e.g., ["bleu", "rouge1", "rougeL"]).

    Returns
    -------
    metric : str or None
        First metric from preferred_metrics that exists in df.columns, or None if none found.
    """
    for metric in preferred_metrics:
        if metric in df.columns:
            return metric
    return None
