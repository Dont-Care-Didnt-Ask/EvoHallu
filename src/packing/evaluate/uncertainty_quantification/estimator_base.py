from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class Estimator(ABC):
    """
    Base class for Uncertainty Quantification (UQ) estimators.

    The interface mirrors the design used in HalluDetect, adapted for EvoTune.
    An estimator consumes a dictionary of precomputed statistics and returns
    a one-dimensional array of uncertainty scores, where higher values
    correspond to higher uncertainty (e.g., higher hallucination likelihood).
    
    Subclasses receive a `context` dict in __init__ containing:
    - "supervised": bool - whether labeled training data is available
    - "meta": dict - additional metadata including "eval_meta"

    The __call__ method receives `stats` dict with these keys (N = num samples):

    **Sequence-level (shape [N], use directly for scoring):**
    - "sequence_logprob": np.ndarray[float32, [N]] - sum of token log-probs
    - "mean_token_entropy": np.ndarray[float32, [N]] - mean entropy per sequence

    **Token-level (lists of length N, variable inner lengths n_i):**
    - "greedy_log_likelihoods": List[List[float]] - per-token log p(token|prefix)
    - "greedy_log_probs": List[np.ndarray[float32, [n_i, V]]] - full vocab log-probs

    **Embeddings (D = hidden_dim):**
    - "embeddings_decoder": np.ndarray[float32, [N, D]] - mean embedding per sequence

    **If context["supervised"]=True:**
    - "train_sequence_logprob": np.ndarray[float32, [N_train]]
    - "train_metrics": np.ndarray[float64, [N_train]] - binary labels (1=hallucination)

    The __call__ method MUST return np.ndarray of shape [N] for level="sequence".
    """

    #: Allowed levels at which the estimator can operate.
    _ALLOWED_LEVELS = ("sequence", "claim", "token")

    def __init__(
        self,
        context: Dict[str, Any],
        *,
        stats_dependencies: List[str],
        level: str,
    ) -> None:
        """
        Parameters
        ----------
        context:
            Runtime context containing:
            - "supervised": bool indicating if training data is available
            - "train_stats": Dict[str, np.ndarray] of training statistics (if supervised)
            - "train_labels": np.ndarray of binary labels (if supervised)
            - "meta": dict with additional metadata
            
        stats_dependencies:
            Names of statistics that the estimator expects to find in the
            `stats` mapping passed to :meth:`__call__`. These are keys into
            the `all_stats["stats"]` dict constructed by `make_all_stats`.

        level:
            Granularity of the uncertainty estimates. Must be one of:

            - ``\"sequence\"`` – one scalar per sequence (most common).
            - ``\"claim\"`` – reserved for future use if claims differ from sequences.
            - ``\"token\"`` – one scalar per token, concatenated across samples.
        """
        if level not in self._ALLOWED_LEVELS:
            raise ValueError(
                f"Invalid level={level!r}. Expected one of {self._ALLOWED_LEVELS}."
            )
        self.level = level
        self.context = context
        # Copy to avoid accidental external mutation
        self.stats_dependencies = list(stats_dependencies)

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a unique, human-readable name for the estimator.

        All hyperparameters that affect the resulting scores should be part of
        this string so that different configurations of the same class have
        distinguishable identifiers.
        """

    @abstractmethod
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute uncertainty scores given precomputed statistics.

        Parameters
        ----------
        stats:
            Mapping from statistic names (strings) to arrays. The concrete
            keys available are determined by `make_all_stats`. Estimators
            should restrict themselves to `self.stats_dependencies` when
            possible to make their requirements explicit.

        Returns
        -------
        np.ndarray
            One-dimensional array of float scores. For ``level == \"sequence\"``,
            the length must equal the number of sequences in the evaluation
            split. For ``level == \"token\"``, the scores are concatenated
            across all tokens from all sequences.
        """


