"""Bootstrap and confidence interval utilities for uncertainty quantification."""

import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_auc_score

from ..pred_rej_area import PredictionRejectionArea, skip_target_nans


def delong_roc_variance(y_true, y_score):
    """
    Computes the variance of the ROC AUC using DeLong's method.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        True binary labels (0 or 1).
    y_score : array-like, shape = [n_samples]
        Prediction scores (e.g., probabilities).

    Returns
    -------
    auc_var : float
        Variance of the AUC estimate.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Ensure binary classification
    assert set(np.unique(y_true)) <= {0, 1}, "y_true must be binary (0/1)."

    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_score_sorted = y_score[desc_score_indices]

    # Find thresholds where score changes
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[0, distinct_value_indices + 1, y_true.size]

    # Number of positive and negative samples
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples.")

    # Compute tie-adjusted AUC components
    cum_pos = np.cumsum(y_true_sorted)
    cum_neg = np.arange(1, len(y_true_sorted) + 1) - cum_pos

    # Sum of positive ranks
    sum_pos_ranks = np.sum(cum_neg[y_true_sorted == 1])
    auc = sum_pos_ranks / (n_pos * n_neg)

    # Compute V10 and V01 (covariance terms)
    # V10: variance component for positive class
    # V01: variance component for negative class

    # Compute placements for positives and negatives
    placements_pos = cum_neg[y_true_sorted == 1]
    placements_neg = cum_pos[y_true_sorted == 0][::-1]  # reverse to align

    # Build the "theta" matrix implicitly via placements
    # Using the formulation from Sun & Xu (2014):
    # var(AUC) = (V10 / n_pos + V01 / n_neg)

    # Compute V10 = Var(E[phi_1(x_1, x_0) | x_1])
    # phi_1 = P(X_0 < X_1) + 0.5 * P(X_0 = X_1)
    # We approximate using placements
    V10 = np.var(placements_pos / n_neg, ddof=0)  # population variance

    # Compute V01 = Var(E[phi_1(x_1, x_0) | x_0])
    V01 = np.var(placements_neg / n_pos, ddof=0)

    auc_var = V10 / n_pos + V01 / n_neg
    return auc_var


def delong_auc_ci(y_true: np.ndarray, y_score: np.ndarray) -> str:
    """
    Compute 95% confidence interval for ROC-AUC using DeLong's method.

    Parameters
    ----------
    y_true : np.ndarray, shape = [n_samples]
        True binary labels (0 or 1).
    y_score : np.ndarray, shape = [n_samples]
        Prediction scores (e.g., probabilities).
    alpha : float, default=0.05
        Significance level (0.05 → 95% CI).

    Returns
    -------
    ci_str : str
        Formatted confidence interval as "(lower%, upper%)" with 2 decimals,
        e.g., "(82.45, 91.30)". Returns "(nan, nan)" on failure.
    """
    try:
        auc = roc_auc_score(y_true, y_score)
        auc_var = delong_roc_variance(y_true, y_score)
        auc_std = np.sqrt(auc_var)

        z = 1.96 # 95% asymtotic normal confidence interval
        #stats.norm.ppf(1 - alpha / 2)
        lower = auc - z * auc_std
        upper = auc + z * auc_std

        # Clamp to [0, 1]
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)

        return f"({100*lower:.2f}, {100*upper:.2f})"

    except (ValueError, ZeroDivisionError, AssertionError):
        return "(nan, nan)"


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    B: int = 1000,
    bootstrap_indices: np.ndarray = None,
) -> Tuple[str, List[float]]:
    """
    Compute 95% bootstrap confidence interval for ROC-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Prediction scores.
    B : int
        Number of bootstrap iterations (used only if bootstrap_indices is None).
    bootstrap_indices : np.ndarray, optional
        Pre-generated bootstrap indices of shape (B, n). If provided, these are
        used instead of generating new random indices. This allows aligned
        bootstrap samples across multiple methods for pairwise comparisons.

    Returns
    -------
    ci_str : str
        Formatted confidence interval string "(lower, upper)".
    bootstrapped_scores : List[float]
        List of bootstrapped AUC values for each iteration.
    """
    n = len(y_true)

    if bootstrap_indices is None:
        # Fallback: generate indices on the fly (non-reproducible)
        bootstrap_indices = np.array([
            np.random.choice(np.arange(n), size=n, replace=True)
            for _ in range(B)
        ])

    bootstrapped_scores = []
    for idxs in bootstrap_indices:
        if (y_true[idxs] == 1).any() and (y_true[idxs] == 0).any():
            bootstrapped_scores.append(float(roc_auc_score(y_true[idxs], y_score[idxs])))
        else:
            # Keep NaN to preserve alignment across methods
            bootstrapped_scores.append(float('nan'))

    valid_scores = [s for s in bootstrapped_scores if not np.isnan(s)]
    if valid_scores:
        lower = float(np.percentile(valid_scores, 2.5))
        upper = float(np.percentile(valid_scores, 97.5))
        ci_str = f"({100*lower:.2f}, {100*upper:.2f})"
    else:
        ci_str = "(nan, nan)"

    return ci_str, bootstrapped_scores


def generate_bootstrap_indices(n_samples: int, n_bootstrap: int = 1000, seed: int = 31) -> np.ndarray:
    """
    Generate bootstrap sample indices for aligned paired comparisons.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    n_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    bootstrap_indices : np.ndarray, shape = (n_bootstrap, n_samples)
        Bootstrap sample indices. Each row contains indices for one bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)
    return bootstrap_indices


def bootstrap_prr_ci(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    n_bootstrap: int = 1000,
    bootstrap_indices: np.ndarray = None,
    max_rejection: float = 1.0,
) -> Tuple[str, List[float]]:
    """
    Compute 95% bootstrap confidence interval for PRR (Prediction-Rejection Area).

    Parameters
    ----------
    uncertainty_scores : np.ndarray
        Uncertainty estimation scores. Higher values indicate more uncertainty.
    quality_scores : np.ndarray
        Continuous quality scores (e.g., BLEU, ROUGE, F1). Higher values indicate
        better quality. NaN values are filtered out per bootstrap sample.
    n_bootstrap : int
        Number of bootstrap iterations (used only if bootstrap_indices is None).
    bootstrap_indices : np.ndarray, optional
        Pre-generated bootstrap indices of shape (n_bootstrap, n). If provided,
        these are used instead of generating new random indices. This allows aligned
        bootstrap samples across multiple methods for pairwise comparisons.
    max_rejection : float
        Maximum proportion of instances to reject (default 1.0 = entire set).

    Returns
    -------
    ci_str : str
        Formatted confidence interval string "(lower, upper)" with percentages.
    bootstrapped_scores : List[float]
        List of bootstrapped PRR values for each iteration. NaN entries indicate
        bootstrap samples with insufficient valid data (<2 samples after NaN filtering).
    """
    n = len(uncertainty_scores)

    if bootstrap_indices is None:
        # Fallback: generate indices on the fly (non-reproducible)
        bootstrap_indices = np.array([
            np.random.choice(np.arange(n), size=n, replace=True)
            for _ in range(n_bootstrap)
        ])

    prr_metric = PredictionRejectionArea(max_rejection=max_rejection)
    bootstrapped_scores = []

    for idxs in bootstrap_indices:
        # Get bootstrap sample
        ue_boot = uncertainty_scores[idxs]
        quality_boot = quality_scores[idxs]

        # Filter NaN values in quality scores
        quality_filtered, ue_filtered = skip_target_nans(quality_boot, ue_boot)

        # Need at least 2 valid samples for PRR computation
        if len(quality_filtered) >= 2:
            prr_score = prr_metric(ue_filtered, quality_filtered)
            bootstrapped_scores.append(float(prr_score))
        else:
            # Keep NaN to preserve alignment across methods
            bootstrapped_scores.append(float('nan'))

    valid_scores = [s for s in bootstrapped_scores if not np.isnan(s)]
    if valid_scores:
        lower = float(np.percentile(valid_scores, 2.5))
        upper = float(np.percentile(valid_scores, 97.5))
        ci_str = f"({100*lower:.2f}, {100*upper:.2f})"
    else:
        ci_str = "(nan, nan)"

    return ci_str, bootstrapped_scores
