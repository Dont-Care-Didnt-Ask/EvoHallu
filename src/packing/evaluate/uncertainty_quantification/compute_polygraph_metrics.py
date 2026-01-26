"""Compute quality metrics for LM-Polygraph benchmark generations.

For each dataset, computes sample-wise metrics and adds them as new columns to the CSV.
Metrics are computed based on dataset type (translation, summarization, QA, etc.).
"""

import argparse
import ast
import logging
import re
import string
import sys
from pathlib import Path

# Add the current directory to sys.path for alignscore imports
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from alignscore import AlignScore
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Dataset configurations: (dataset_file_prefix, metric_type)
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


def detect_dataset_type(filename: str) -> str | None:
    """
    Detect dataset type from filename by matching against known dataset prefixes.

    Handles files with patterns:
    - {dataset_name}.csv
    - {dataset_name}_limit_{n}.csv
    - PARTIAL_{dataset_name}.csv
    - PARTIAL_{dataset_name}_limit_{n}.csv

    Returns:
        metric_type if dataset is recognized, None otherwise
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


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, remove articles/punctuation)."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(c for c in s if c not in string.punctuation)
    # Normalize whitespace
    s = " ".join(s.split())
    return s

# =============================================================================
# AlignScore (general)
# =============================================================================

_align_scorer = AlignScore()

def compute_alignscore(generations: List[str], references: List[str]) -> np.ndarray:
    """Compute AlignScore metric (https://aclanthology.org/2023.acl-long.634/)."""
    scores = _align_scorer(stats={"greedy_texts": generations}, target_texts=references)
    return scores


# =============================================================================
# BLEU Metrics (Translation)
# =============================================================================

_bleu_scorer = BLEU(effective_order=True)


def compute_bleu(generation: str, reference: str) -> float:
    """Compute sentence-level BLEU score."""
    if not generation or not reference:
        return 0.0
    try:
        score = _bleu_scorer.sentence_score(generation.strip(), [reference.strip()])
        return score.score / 100  # Normalize to [0, 1]
    except Exception:
        return 0.0


# =============================================================================
# ROUGE Metrics (Summarization)
# =============================================================================

_rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(generation: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not generation or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        scores = _rouge_scorer.score(reference.strip(), generation.strip())
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# =============================================================================
# QA Metrics (Exact Match + F1)
# =============================================================================

def compute_exact_match(generation: str, reference: str) -> float:
    """Compute normalized exact match."""
    return float(normalize_answer(generation) == normalize_answer(reference))


def compute_f1(generation: str, reference: str) -> float:
    """Compute token-level F1 score."""
    gen_tokens = normalize_answer(generation).split()
    ref_tokens = normalize_answer(reference).split()

    if not gen_tokens and not ref_tokens:
        return 1.0
    if not gen_tokens or not ref_tokens:
        return 0.0

    common = set(gen_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# GSM8K Metrics (Numeric Extraction)
# =============================================================================

def extract_gsm8k_reference_answer(reference: str) -> str | None:
    """Extract numeric answer after #### from GSM8K reference."""
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", reference)
    if match:
        # Remove commas from numbers like "70,000"
        return match.group(1).replace(",", "")
    return None


def extract_gsm8k_generation_answer(generation: str) -> str | None:
    """Extract numeric answer from GSM8K generation."""
    # Try "The answer is X" pattern first
    match = re.search(r"[Tt]he answer is[:\s]*(-?\d+(?:,\d{3})*(?:\.\d+)?)", generation)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: find last number in text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", generation)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def compute_gsm8k_exact_match(generation: str, reference: str) -> float:
    """Compute exact match on extracted numeric answers."""
    ref_answer = extract_gsm8k_reference_answer(reference)
    gen_answer = extract_gsm8k_generation_answer(generation)

    if ref_answer is None or gen_answer is None:
        return 0.0

    try:
        return float(abs(float(ref_answer) - float(gen_answer)) < 1e-6)
    except ValueError:
        return float(ref_answer == gen_answer)


# =============================================================================
# Multi-Answer QA Metrics (TriviaQA)
# =============================================================================

def parse_answer_list(reference: str) -> list[str]:
    """Parse reference that may be a string representation of a list."""
    try:
        answers = ast.literal_eval(reference)
        if isinstance(answers, list):
            return [str(a) for a in answers]
    except (ValueError, SyntaxError):
        pass
    return [reference]


def compute_multi_answer_exact_match(generation: str, reference: str) -> float:
    """Check if generation matches any of the acceptable answers."""
    acceptable_answers = parse_answer_list(reference)
    gen_normalized = normalize_answer(generation)

    for answer in acceptable_answers:
        if normalize_answer(answer) == gen_normalized:
            return 1.0
    return 0.0


def compute_multi_answer_f1(generation: str, reference: str) -> float:
    """Compute best F1 score across all acceptable answers."""
    acceptable_answers = parse_answer_list(reference)
    best_f1 = 0.0

    for answer in acceptable_answers:
        f1 = compute_f1(generation, answer)
        best_f1 = max(best_f1, f1)

    return best_f1


# =============================================================================
# TruthfulQA Metrics (Contains Match + Best ROUGE)
# =============================================================================

def compute_truthfulqa_contains(generation: str, reference: str) -> float:
    """Check if any reference answer is contained in generation."""
    acceptable_answers = parse_answer_list(reference)
    gen_normalized = normalize_answer(generation)

    for answer in acceptable_answers:
        if normalize_answer(answer) in gen_normalized:
            return 1.0
    return 0.0


def compute_truthfulqa_best_rouge(generation: str, reference: str) -> dict[str, float]:
    """Compute best ROUGE scores across all acceptable answers."""
    acceptable_answers = parse_answer_list(reference)

    best_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for answer in acceptable_answers:
        scores = compute_rouge(generation, answer)
        for key in best_scores:
            best_scores[key] = max(best_scores[key], scores[key])

    return best_scores


# =============================================================================
# MMLU Metrics (Letter Extraction)
# =============================================================================

def extract_mmlu_answer(text: str) -> str:
    """Extract first A/B/C/D letter from text."""
    match = re.search(r"[ABCDabcd]", text)
    return match.group(0).upper() if match else ""


def compute_mmlu_exact_match(generation: str, reference: str) -> float:
    """Compute exact match on extracted letter answers."""
    gen_letter = extract_mmlu_answer(generation)
    ref_letter = reference.strip().upper()
    return float(gen_letter == ref_letter)


# =============================================================================
# Dataset Processing
# =============================================================================

def compute_metrics_for_dataset(
    df: pd.DataFrame,
    metric_type: str,
) -> pd.DataFrame:
    """Compute metrics for a dataset and add as new columns."""
    generations = df["model_generation"].fillna("").astype(str)
    references = df["output"].fillna("").astype(str)

    df["alignscore"] = compute_alignscore(generations.tolist(), references.tolist())

    if metric_type == "translation":
        df["bleu"] = [
            compute_bleu(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="BLEU")
        ]

    elif metric_type == "summarization":
        rouge_scores = [
            compute_rouge(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="ROUGE")
        ]
        df["rouge1"] = [s["rouge1"] for s in rouge_scores]
        df["rouge2"] = [s["rouge2"] for s in rouge_scores]
        df["rougeL"] = [s["rougeL"] for s in rouge_scores]

    elif metric_type == "qa":
        df["exact_match"] = [
            compute_exact_match(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="EM")
        ]
        df["f1"] = [
            compute_f1(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="F1")
        ]

    elif metric_type == "qa_simple":
        # bAbI QA - just exact match (single word answers)
        df["exact_match"] = [
            compute_exact_match(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="EM")
        ]

    elif metric_type == "gsm8k":
        df["exact_match"] = [
            compute_gsm8k_exact_match(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="GSM8K EM")
        ]

    elif metric_type == "multi_answer_qa":
        df["exact_match"] = [
            compute_multi_answer_exact_match(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="Multi-EM")
        ]
        df["f1"] = [
            compute_multi_answer_f1(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="Multi-F1")
        ]

    elif metric_type == "truthfulqa":
        df["contains_match"] = [
            compute_truthfulqa_contains(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="Contains")
        ]
        rouge_scores = [
            compute_truthfulqa_best_rouge(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="Best ROUGE")
        ]
        df["best_rouge1"] = [s["rouge1"] for s in rouge_scores]
        df["best_rouge2"] = [s["rouge2"] for s in rouge_scores]
        df["best_rougeL"] = [s["rougeL"] for s in rouge_scores]

    elif metric_type == "mmlu":
        df["exact_match"] = [
            compute_mmlu_exact_match(gen, ref)
            for gen, ref in tqdm(zip(generations, references), total=len(df), desc="MMLU EM")
        ]

    return df


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    # Bootstrap resampling
    rng = np.random.default_rng(42)
    bootstrap_means = []

    for _ in range(n_resamples):
        resample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(np.mean(values)), float(ci_lower), float(ci_upper)


def print_metrics_summary(df: pd.DataFrame, dataset_name: str) -> None:
    """Print summary statistics with bootstrap CIs for all metric columns."""
    metric_columns = [
        col for col in df.columns
        if col in ["bleu", "rouge1", "rouge2", "rougeL", "exact_match", "f1",
                   "contains_match", "best_rouge1", "best_rouge2", "best_rougeL"]
    ]

    if not metric_columns:
        return

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} (n={len(df)})")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Mean':>10} {'95% CI':>20}")
    print("-" * 60)

    for col in metric_columns:
        values = df[col].values
        mean, ci_lower, ci_upper = bootstrap_ci(values)
        print(f"{col:<20} {mean:>10.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute quality metrics for LM-Polygraph generations."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/uncertainty_quantification/polygraph_generations",
        help="Directory containing generation CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save CSV files with metrics. If not specified, saves to input-dir with '_metrics' suffix.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to filter CSV files (e.g., '*_limit_5.csv', 'PARTIAL_*.csv'). Default: '*.csv'",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary statistics with bootstrap CIs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be processed without actually processing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Add _metrics suffix to preserve original files
        output_dir = input_dir.parent / f"{input_dir.name}_metrics"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files matching the pattern
    csv_files = sorted(input_dir.glob(args.pattern))

    if not csv_files:
        logger.warning(f"No CSV files found matching pattern '{args.pattern}' in {input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files matching pattern '{args.pattern}'")

    processed_count = 0
    skipped_count = 0

    for input_path in csv_files:
        filename = input_path.name

        # Detect dataset type from filename
        metric_type = detect_dataset_type(filename)

        if metric_type is None:
            logger.warning(f"Skipping {filename}: unknown dataset type")
            skipped_count += 1
            continue

        # Check if file has required columns
        if args.dry_run:
            logger.info(f"[DRY RUN] Would process {filename} with {metric_type} metrics")
            processed_count += 1
            continue

        try:
            df = pd.read_csv(input_path)

            # Verify required columns exist
            required_cols = ["model_generation", "output"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Skipping {filename}: missing required columns {missing_cols}")
                skipped_count += 1
                continue

            logger.info(f"Processing {filename} ({len(df)} rows) with {metric_type} metrics")

            df = compute_metrics_for_dataset(df, metric_type)

            # Preserve original filename in output
            output_path = output_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Saved metrics to {output_path}")

            if args.print_summary:
                print_metrics_summary(df, filename)

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            skipped_count += 1
            continue

    logger.info(f"Processing complete: {processed_count} files processed, {skipped_count} files skipped")


if __name__ == "__main__":
    main()
