"""Tests for metric_selection module."""

import pytest
import pandas as pd
from src.packing.evaluate.uncertainty_quantification.utils.metric_selection import (
    detect_dataset_type_from_filename,
    get_metric_for_dataset_type,
    select_best_available_metric,
)


class TestDetectDatasetType:
    """Tests for detect_dataset_type_from_filename function."""

    def test_translation_datasets(self):
        """Test detection of translation datasets."""
        assert detect_dataset_type_from_filename("wmt14_deen.csv") == "translation"
        assert detect_dataset_type_from_filename("wmt14_fren.csv") == "translation"
        assert detect_dataset_type_from_filename("wmt19_deen_simple_instruct.csv") == "translation"

    def test_summarization_datasets(self):
        """Test detection of summarization datasets."""
        assert detect_dataset_type_from_filename("xsum_simple_instruct.csv") == "summarization"
        assert detect_dataset_type_from_filename("samsum_simple_instruct.csv") == "summarization"

    def test_qa_datasets(self):
        """Test detection of QA datasets."""
        assert detect_dataset_type_from_filename("coqa_simple_instruct.csv") == "qa"
        assert detect_dataset_type_from_filename("babi_qa_continuation.csv") == "qa_simple"

    def test_gsm8k_dataset(self):
        """Test detection of GSM8K dataset."""
        assert detect_dataset_type_from_filename("gsm8k_simple_instruct.csv") == "gsm8k"

    def test_with_partial_prefix(self):
        """Test detection with PARTIAL_ prefix."""
        assert detect_dataset_type_from_filename("PARTIAL_wmt14_deen.csv") == "translation"
        assert detect_dataset_type_from_filename("PARTIAL_gsm8k_simple_instruct.csv") == "gsm8k"

    def test_with_limit_suffix(self):
        """Test detection with _limit_{n} suffix."""
        assert detect_dataset_type_from_filename("wmt14_deen_limit_100.csv") == "translation"
        assert detect_dataset_type_from_filename("gsm8k_simple_instruct_limit_50.csv") == "gsm8k"
        assert detect_dataset_type_from_filename("PARTIAL_xsum_simple_instruct_limit_200.csv") == "summarization"

    def test_unknown_dataset(self):
        """Test that unknown datasets return None."""
        assert detect_dataset_type_from_filename("unknown_dataset.csv") is None
        assert detect_dataset_type_from_filename("random_file.csv") is None


class TestGetMetricForDatasetType:
    """Tests for get_metric_for_dataset_type function."""

    def test_translation_metric(self):
        """Test metric for translation datasets."""
        assert get_metric_for_dataset_type("translation") == "bleu"

    def test_summarization_metric(self):
        """Test metric for summarization datasets."""
        assert get_metric_for_dataset_type("summarization") == "rougeL"

    def test_qa_metrics(self):
        """Test metrics for QA datasets."""
        assert get_metric_for_dataset_type("qa") == "f1"
        assert get_metric_for_dataset_type("qa_simple") == "f1"
        assert get_metric_for_dataset_type("multi_answer_qa") == "f1"

    def test_gsm8k_metric(self):
        """Test metric for GSM8K."""
        assert get_metric_for_dataset_type("gsm8k") == "exact_match"

    def test_mmlu_metric(self):
        """Test metric for MMLU."""
        assert get_metric_for_dataset_type("mmlu") == "exact_match"

    def test_truthfulqa_metric(self):
        """Test metric for TruthfulQA."""
        assert get_metric_for_dataset_type("truthfulqa") == "best_rougeL"

    def test_unknown_type(self):
        """Test that unknown types return None."""
        assert get_metric_for_dataset_type("unknown_type") is None


class TestSelectBestAvailableMetric:
    """Tests for select_best_available_metric function."""

    def test_first_available(self):
        """Test that first available metric is selected."""
        df = pd.DataFrame({
            "bleu": [0.5, 0.8],
            "rouge1": [0.6, 0.7],
            "rougeL": [0.55, 0.75]
        })
        metric = select_best_available_metric(df, ["bleu", "rouge1", "rougeL"])
        assert metric == "bleu"

    def test_fallback_to_second(self):
        """Test fallback when first metric not available."""
        df = pd.DataFrame({
            "rouge1": [0.6, 0.7],
            "rougeL": [0.55, 0.75]
        })
        metric = select_best_available_metric(df, ["bleu", "rouge1", "rougeL"])
        assert metric == "rouge1"

    def test_no_metrics_available(self):
        """Test when no metrics are available."""
        df = pd.DataFrame({
            "other_col": [0.5, 0.8]
        })
        metric = select_best_available_metric(df, ["bleu", "rouge1", "rougeL"])
        assert metric is None

    def test_empty_preference_list(self):
        """Test with empty preference list."""
        df = pd.DataFrame({
            "bleu": [0.5, 0.8]
        })
        metric = select_best_available_metric(df, [])
        assert metric is None
