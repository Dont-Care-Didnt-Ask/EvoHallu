"""Tests for bootstrap_utils module."""

import pytest
import numpy as np
from src.packing.evaluate.uncertainty_quantification.utils.bootstrap_utils import (
    generate_bootstrap_indices,
    bootstrap_auc_ci,
    bootstrap_prr_ci,
    delong_auc_ci,
)


class TestGenerateBootstrapIndices:
    """Tests for generate_bootstrap_indices function."""

    def test_reproducibility(self):
        """Test that same seed produces same indices."""
        indices1 = generate_bootstrap_indices(100, n_bootstrap=10, seed=42)
        indices2 = generate_bootstrap_indices(100, n_bootstrap=10, seed=42)
        np.testing.assert_array_equal(indices1, indices2)

    def test_shape(self):
        """Test output shape is correct."""
        n_samples = 50
        n_bootstrap = 100
        indices = generate_bootstrap_indices(n_samples, n_bootstrap=n_bootstrap, seed=42)
        assert indices.shape == (n_bootstrap, n_samples)

    def test_values_in_range(self):
        """Test all indices are valid (in [0, n_samples))."""
        n_samples = 30
        indices = generate_bootstrap_indices(n_samples, n_bootstrap=50, seed=42)
        assert indices.min() >= 0
        assert indices.max() < n_samples


class TestBootstrapAucCi:
    """Tests for bootstrap_auc_ci function."""

    def test_perfect_classifier(self, mock_binary_labels):
        """Test AUC CI for perfect classifier."""
        # Perfect classifier: higher scores for positive class (label=1)
        scores = mock_binary_labels.astype(float)
        ci_str, bootstrap_aucs = bootstrap_auc_ci(
            mock_binary_labels,
            scores,
            B=100,
            bootstrap_indices=generate_bootstrap_indices(len(mock_binary_labels), 100, seed=42)
        )

        # CI should be close to (100, 100)
        assert ci_str != "(nan, nan)"
        # Extract numbers from "(XX.XX, YY.YY)" format
        lower, upper = [float(x.strip()) for x in ci_str.strip("()").split(",")]
        assert lower >= 90.0  # Allow some variance due to bootstrap with small sample
        assert upper == 100.0

    def test_random_classifier(self, mock_binary_labels):
        """Test AUC CI for random classifier."""
        np.random.seed(42)
        random_scores = np.random.rand(len(mock_binary_labels))

        ci_str, bootstrap_aucs = bootstrap_auc_ci(
            mock_binary_labels,
            random_scores,
            B=100,
            bootstrap_indices=generate_bootstrap_indices(len(mock_binary_labels), 100, seed=42)
        )

        # CI should be reasonably wide for random classifier with small sample
        assert ci_str != "(nan, nan)"
        lower, upper = [float(x.strip()) for x in ci_str.strip("()").split(",")]
        assert 0.0 <= lower <= 100.0
        assert 0.0 <= upper <= 100.0
        assert lower <= upper

    def test_aligned_bootstrap_indices(self, mock_binary_labels):
        """Test that using same bootstrap indices produces aligned samples."""
        bootstrap_indices = generate_bootstrap_indices(len(mock_binary_labels), 100, seed=42)

        scores1 = np.random.RandomState(1).rand(len(mock_binary_labels))
        scores2 = np.random.RandomState(2).rand(len(mock_binary_labels))

        _, aucs1 = bootstrap_auc_ci(mock_binary_labels, scores1, bootstrap_indices=bootstrap_indices)
        _, aucs2 = bootstrap_auc_ci(mock_binary_labels, scores2, bootstrap_indices=bootstrap_indices)

        # Both should have same number of bootstrap samples
        assert len(aucs1) == len(aucs2) == 100


class TestBootstrapPrrCi:
    """Tests for bootstrap_prr_ci function."""

    def test_perfect_uncertainty(self, mock_quality_scores):
        """Test PRR CI for perfect uncertainty estimator."""
        # Perfect estimator: high uncertainty for low quality
        uncertainty = 1.0 - mock_quality_scores

        ci_str, bootstrap_prrs = bootstrap_prr_ci(
            uncertainty,
            mock_quality_scores,
            n_bootstrap=100,
            bootstrap_indices=generate_bootstrap_indices(len(mock_quality_scores), 100, seed=42)
        )

        # CI should indicate good performance (higher than random)
        # Note: With small sample size (10), variance is high
        assert ci_str != "(nan, nan)"
        lower, upper = [float(x.strip()) for x in ci_str.strip("()").split(",")]
        assert lower >= 50.0  # Should be better than random (50%)
        assert upper <= 100.0

    def test_random_uncertainty(self, mock_quality_scores):
        """Test PRR CI for random uncertainty estimator."""
        np.random.seed(42)
        random_uncertainty = np.random.rand(len(mock_quality_scores))

        ci_str, bootstrap_prrs = bootstrap_prr_ci(
            random_uncertainty,
            mock_quality_scores,
            n_bootstrap=100,
            bootstrap_indices=generate_bootstrap_indices(len(mock_quality_scores), 100, seed=42)
        )

        # CI should be wide for random estimator with small sample
        assert ci_str != "(nan, nan)"
        lower, upper = [float(x.strip()) for x in ci_str.strip("()").split(",")]
        assert 0.0 <= lower <= 100.0
        assert 0.0 <= upper <= 100.0
        assert lower <= upper

    def test_nan_handling(self):
        """Test that NaN values in quality scores are handled correctly."""
        uncertainty = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        quality_with_nans = np.array([1.0, np.nan, 0.8, np.nan, 0.6])

        ci_str, bootstrap_prrs = bootstrap_prr_ci(
            uncertainty,
            quality_with_nans,
            n_bootstrap=50,
            bootstrap_indices=generate_bootstrap_indices(len(uncertainty), 50, seed=42)
        )

        # Should still return valid CI (some bootstrap samples will have enough valid data)
        # At minimum, should not crash
        assert isinstance(ci_str, str)
        assert isinstance(bootstrap_prrs, list)

    def test_aligned_bootstrap_with_auc(self, mock_quality_scores, mock_binary_labels):
        """Test that PRR and AUC use aligned bootstrap samples."""
        bootstrap_indices = generate_bootstrap_indices(len(mock_quality_scores), 100, seed=42)

        uncertainty = np.random.RandomState(42).rand(len(mock_quality_scores))

        _, prrs = bootstrap_prr_ci(uncertainty, mock_quality_scores, bootstrap_indices=bootstrap_indices)
        _, aucs = bootstrap_auc_ci(mock_binary_labels, uncertainty, bootstrap_indices=bootstrap_indices)

        # Both should have same number of bootstrap samples
        assert len(prrs) == len(aucs) == 100


class TestDelongAucCi:
    """Tests for delong_auc_ci function."""

    def test_perfect_classifier(self, mock_binary_labels):
        """Test DeLong CI for perfect classifier."""
        scores = mock_binary_labels.astype(float)
        ci_str = delong_auc_ci(mock_binary_labels, scores)

        # Should return valid CI close to 100
        assert ci_str != "(nan, nan)"
        lower, upper = [float(x.strip()) for x in ci_str.strip("()").split(",")]
        assert lower >= 90.0
        assert upper == 100.0

    def test_invalid_labels(self):
        """Test that DeLong handles invalid input gracefully."""
        # All same label
        labels = np.ones(10)
        scores = np.random.rand(10)

        ci_str = delong_auc_ci(labels, scores)
        assert ci_str == "(nan, nan)"
