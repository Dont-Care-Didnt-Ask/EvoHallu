"""Tests for data_utils module."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.packing.evaluate.uncertainty_quantification.utils.data_utils import (
    normalize_dataframe_classification,
    normalize_dataframe_generation,
    validate_csv_columns,
)


class TestValidateCsvColumns:
    """Tests for validate_csv_columns function."""

    def test_valid_columns(self):
        """Test that validation passes for valid columns."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        # Should not raise
        validate_csv_columns(df, ["col1", "col2"], "test.csv")

    def test_missing_columns(self):
        """Test that validation fails for missing columns."""
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv_columns(df, ["col1", "col2", "col3"], "test.csv")


class TestNormalizeDataframeClassification:
    """Tests for normalize_dataframe_classification function."""

    def test_basic_loading(self, sample_csv_classification):
        """Test basic CSV loading and normalization."""
        df = normalize_dataframe_classification(
            sample_csv_classification,
            text_col="claim",
            label_col="is_hallu"
        )
        assert len(df) == 10
        assert "claim" in df.columns
        assert "is_hallu" in df.columns

    def test_text_truncation(self):
        """Test that text is truncated to 300 characters."""
        long_text = "a" * 500
        data = pd.DataFrame({
            "claim": [long_text, "short"],
            "is_hallu": [0, 1]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            df = normalize_dataframe_classification(temp_path)
            assert len(df["claim"].iloc[0]) == 300
            assert len(df["claim"].iloc[1]) == 5
        finally:
            Path(temp_path).unlink()

    def test_empty_text_filtering(self):
        """Test that whitespace-only text rows are filtered out."""
        # Note: When pandas reads CSV, truly empty cells become NaN,
        # which converts to "nan" string with astype(str). This "nan"
        # is not empty and won't be filtered. Only whitespace-only
        # strings that strip to "" will be filtered.
        data = pd.DataFrame({
            "claim": ["valid", "   ", "also valid"],
            "is_hallu": [0, 0, 1]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            df = normalize_dataframe_classification(temp_path)
            assert len(df) == 2  # Only "valid" and "also valid" (whitespace-only filtered)
        finally:
            Path(temp_path).unlink()

    def test_non_numeric_labels(self):
        """Test that non-numeric labels raise ValueError."""
        data = pd.DataFrame({
            "claim": ["test1", "test2"],
            "is_hallu": [0, "invalid"]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="non-numeric values"):
                normalize_dataframe_classification(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_columns(self):
        """Test that missing columns raise ValueError."""
        data = pd.DataFrame({
            "wrong_col": ["test1", "test2"],
            "is_hallu": [0, 1]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must contain"):
                normalize_dataframe_classification(temp_path, text_col="claim")
        finally:
            Path(temp_path).unlink()


class TestNormalizeDataframeGeneration:
    """Tests for normalize_dataframe_generation function."""

    def test_basic_loading(self, sample_csv_generation):
        """Test basic CSV loading for generation data."""
        df = normalize_dataframe_generation(
            sample_csv_generation,
            text_col="model_generation",
            required_metric_cols=["bleu", "rougeL"]
        )
        assert len(df) == 10
        assert "model_generation" in df.columns
        assert "bleu" in df.columns
        assert "rougeL" in df.columns

    def test_no_text_truncation(self):
        """Test that generation text is NOT truncated (unlike classification)."""
        long_text = "a" * 500
        data = pd.DataFrame({
            "model_generation": [long_text, "short"],
            "bleu": [0.5, 0.8]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            df = normalize_dataframe_generation(temp_path)
            # Text should NOT be truncated
            assert len(df["model_generation"].iloc[0]) == 500
            assert len(df["model_generation"].iloc[1]) == 5
        finally:
            Path(temp_path).unlink()

    def test_nan_preservation(self, sample_generation_data_with_nans):
        """Test that NaN values in metrics are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_generation_data_with_nans.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            df = normalize_dataframe_generation(
                temp_path,
                required_metric_cols=["bleu", "rougeL"],
                validate_metrics=True
            )
            # NaN values should be preserved
            assert df["bleu"].isna().sum() == 2
            assert df["rougeL"].isna().sum() == 2
        finally:
            Path(temp_path).unlink()

    def test_missing_metric_columns(self):
        """Test that missing metric columns raise ValueError."""
        data = pd.DataFrame({
            "model_generation": ["test1", "test2"],
            "bleu": [0.5, 0.8]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="missing required columns"):
                normalize_dataframe_generation(
                    temp_path,
                    required_metric_cols=["bleu", "rouge1", "missing_col"]
                )
        finally:
            Path(temp_path).unlink()

    def test_non_numeric_metrics(self):
        """Test that non-numeric metrics raise ValueError when validation enabled."""
        data = pd.DataFrame({
            "model_generation": ["test1", "test2"],
            "bleu": [0.5, "invalid"]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="non-numeric values"):
                normalize_dataframe_generation(
                    temp_path,
                    required_metric_cols=["bleu"],
                    validate_metrics=True
                )
        finally:
            Path(temp_path).unlink()

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        data = pd.DataFrame({
            "model_generation": ["test1", "test2"],
            "bleu": [0.5, 0.8]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            # Should not raise even with missing columns when validation disabled
            df = normalize_dataframe_generation(
                temp_path,
                required_metric_cols=None,
                validate_metrics=False
            )
            assert len(df) == 2
        finally:
            Path(temp_path).unlink()
