"""Pytest fixtures for uncertainty quantification tests."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_binary_classification_data():
    """Sample DataFrame for binary classification (ROC-AUC) tests."""
    data = {
        "claim": [
            "The sky is blue",
            "Water is dry",
            "The earth is flat",
            "Birds can fly",
            "Fish live in trees",
            "The sun is hot",
            "Ice is warm",
            "Grass is green",
            "Rocks are soft",
            "Humans need air",
        ],
        "is_hallu": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_generation_data():
    """Sample DataFrame for generation tasks (PRR) tests."""
    data = {
        "model_generation": [
            "Hello world",
            "Bonjour monde",
            "Hola mundo",
            "Ciao mondo",
            "Hallo Welt",
            "Olá mundo",
            "Привет мир",
            "こんにちは世界",
            "你好世界",
            "안녕하세요 세계",
        ],
        "bleu": [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55],
        "rougeL": [1.0, 0.93, 0.88, 0.83, 0.78, 0.73, 0.68, 0.63, 0.58, 0.53],
        "f1": [1.0, 0.92, 0.86, 0.81, 0.76, 0.71, 0.66, 0.61, 0.56, 0.51],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_generation_data_with_nans():
    """Sample DataFrame with NaN values in metrics."""
    data = {
        "model_generation": [
            "Hello world",
            "Bonjour monde",
            "Hola mundo",
            "Invalid response",
            "Hallo Welt",
            "Olá mundo",
            "Error",
            "こんにちは世界",
            "你好世界",
            "安녕하세요 세계",
        ],
        "bleu": [1.0, 0.95, 0.90, np.nan, 0.80, 0.75, np.nan, 0.65, 0.60, 0.55],
        "rougeL": [1.0, 0.93, 0.88, 0.83, np.nan, 0.73, 0.68, np.nan, 0.58, 0.53],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_uncertainty_scores():
    """Mock uncertainty scores for 10 samples (higher = more uncertain)."""
    return np.array([0.1, 0.9, 0.85, 0.15, 0.8, 0.2, 0.75, 0.25, 0.7, 0.3])


@pytest.fixture
def mock_quality_scores():
    """Mock quality scores for 10 samples (higher = better quality)."""
    return np.array([1.0, 0.1, 0.2, 0.95, 0.25, 0.9, 0.3, 0.85, 0.35, 0.8])


@pytest.fixture
def mock_binary_labels():
    """Mock binary labels for 10 samples."""
    return np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])


@pytest.fixture
def temp_csv_path():
    """Create a temporary CSV file path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_classification(temp_csv_path, sample_binary_classification_data):
    """Create a temporary CSV file with binary classification data."""
    sample_binary_classification_data.to_csv(temp_csv_path, index=False)
    return temp_csv_path


@pytest.fixture
def sample_csv_generation(temp_csv_path, sample_generation_data):
    """Create a temporary CSV file with generation data."""
    sample_generation_data.to_csv(temp_csv_path, index=False)
    return temp_csv_path
