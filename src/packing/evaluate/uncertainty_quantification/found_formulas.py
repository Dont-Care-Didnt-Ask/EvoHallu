from typing import Any, Dict, List

import numpy as np

from src.packing.evaluate.uncertainty_quantification.estimator_base import Estimator


class HallucinationEstimator(Estimator):
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")

    def __str__(self) -> str:
        return "HallucinationEstimator"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # MUST return shape [N] where N = number of samples
        total_neg_logprob = np.array([-np.sum(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]], dtype=np.float32)
        return total_neg_logprob


class QuartileEntropyDiffNormalized(Estimator):
    """
    Entropy acceleration – mean(entropy_last_quarter) - mean(entropy_first_quarter).
    Positive → uncertainty increasing.
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "token_entropies",   # List[np.ndarray] – shape [n_i]
            ],
            level="sequence",
        )

    def __str__(self) -> str:
        return "QuartileEntropyDiffNormalized"

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        """Zero‑mean, unit‑variance normalisation (eps protects against div‑0)."""
        mu = np.mean(x)
        sigma = np.std(x) + 1e-12
        return (x - mu) / sigma

    @staticmethod
    def _entropy_acceleration(entropies: List[np.ndarray]) -> np.ndarray:
        """Mean(last quarter) – mean(first quarter) per sequence."""
        out = np.empty(len(entropies), dtype=np.float32)
        for i, e in enumerate(entropies):
            n = e.size
            if n < 4:
                out[i] = 0.0
                continue
            q = n // 4
            out[i] = float(np.mean(e[-q:]) - np.mean(e[:q]))
        return out

    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        ent_acc = self._entropy_acceleration(stats["token_entropies"])
        ent_acc = self._zscore(ent_acc)
        return ent_acc


class QuartileEntropyDiff(Estimator):
    """
    Entropy acceleration – mean(entropy_last_quarter) - mean(entropy_first_quarter).
    Positive → uncertainty increasing.
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "token_entropies",   # List[np.ndarray] – shape [n_i]
            ],
            level="sequence",
        )

    def __str__(self) -> str:
        return "QuartileEntropyDiff"

    @staticmethod
    def _entropy_acceleration(entropies: List[np.ndarray]) -> np.ndarray:
        """Mean(last quarter) – mean(first quarter) per sequence."""
        out = np.empty(len(entropies), dtype=np.float32)
        for i, e in enumerate(entropies):
            n = e.size
            if n < 4:
                out[i] = 0.0
                continue
            q = n // 4
            out[i] = float(np.mean(e[-q:]) - np.mean(e[:q]))
        return out

    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        ent_acc = self._entropy_acceleration(stats["token_entropies"])
        return ent_acc


class PositionWeightedGap(Estimator):
    """
    Position-weighted greedy-vs-sampled log-probability gap.
    For each token t compute (greedy_logprob[t] - sequence_logprob[t-1])
    and weight it by t / length (later tokens get larger weight).
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "greedy_logprobs",   # List[np.ndarray] – shape [n_i]
                "sequence_logprobs", # List[np.ndarray] – shape [n_i-1]
            ],
            level="sequence",
        )
    
    def __str__(self) -> str:
        return "PositionWeightedGap"

    @staticmethod
    def _pos_weighted_gap(
        greedy_lp: List[np.ndarray],
        seq_lp: List[np.ndarray],
    ) -> np.ndarray:
        """
        For each sequence compute a position-weighted mean of
        (greedy_logprob[t] - sequence_logprob[t-1]).
        Later positions receive higher weight (linear ramp).
        """
        out = np.empty(len(greedy_lp), dtype=np.float32)

        for i, (g, s) in enumerate(zip(greedy_lp, seq_lp)):
            # Align: drop the first greedy token (no corresponding seq logprob)
            if s.size == 0 or g.shape[0] <= 1:
                out[i] = 0.0
                continue

            aligned_g = g[1 : s.shape[0] + 1]          # shape [len(s)]
            gap = aligned_g - s                         # token‑wise gaps

            # Position weights: 1..L / L
            L = gap.shape[0]
            weights = np.arange(1, L + 1, dtype=np.float32) / L
            weighted_mean = np.sum(gap * weights) / L
            out[i] = float(weighted_mean)

        return out

    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        gap = self._pos_weighted_gap(stats["greedy_logprobs"], stats["sequence_logprobs"])

        return gap

class PositionWeightedGapNormalized(Estimator):
    """
    Position-weighted greedy-vs-sampled log-probability gap.
    For each token t compute (greedy_logprob[t] - sequence_logprob[t-1])
    and weight it by t / length (later tokens get larger weight).
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "greedy_logprobs",   # List[np.ndarray] – shape [n_i]
                "sequence_logprobs", # List[np.ndarray] – shape [n_i-1]
            ],
            level="sequence",
        )

    def __str__(self) -> str:
        return "PositionWeightedGapNormalized"

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        """Zero‑mean, unit‑variance normalisation (eps protects against div‑0)."""
        mu = np.mean(x)
        sigma = np.std(x) + 1e-12
        return (x - mu) / sigma

    @staticmethod
    def _pos_weighted_gap(
        greedy_lp: List[np.ndarray],
        seq_lp: List[np.ndarray],
    ) -> np.ndarray:
        """
        For each sequence compute a position-weighted mean of
        (greedy_logprob[t] - sequence_logprob[t-1]).
        Later positions receive higher weight (linear ramp).
        """
        out = np.empty(len(greedy_lp), dtype=np.float32)

        for i, (g, s) in enumerate(zip(greedy_lp, seq_lp)):
            # Align: drop the first greedy token (no corresponding seq logprob)
            if s.size == 0 or g.shape[0] <= 1:
                out[i] = 0.0
                continue

            aligned_g = g[1 : s.shape[0] + 1]          # shape [len(s)]
            gap = aligned_g - s                         # token‑wise gaps

            # Position weights: 1..L / L
            L = gap.shape[0]
            weights = np.arange(1, L + 1, dtype=np.float32) / L
            weighted_mean = np.sum(gap * weights) / L
            out[i] = float(weighted_mean)

        return out

    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        gap = self._pos_weighted_gap(stats["greedy_logprobs"], stats["sequence_logprobs"])
        gap = self._zscore(gap)
        return gap

class Gap(Estimator):
    """
    Greedy-vs-sampled log-probability gap.
    For each token t compute (greedy_logprob[t] - sequence_logprob[t-1]).
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "greedy_logprobs",   # List[np.ndarray] – shape [n_i]
                "sequence_logprobs", # List[np.ndarray] – shape [n_i-1]
            ],
            level="sequence",
        )
    
    def __str__(self) -> str:
        return "Gap"

    @staticmethod
    def _gap(
        greedy_lp: List[np.ndarray],
        seq_lp: List[np.ndarray],
    ) -> np.ndarray:
        """
        For each sequence compute a position-weighted mean of
        (greedy_logprob[t] - sequence_logprob[t-1]).
        Later positions receive higher weight (linear ramp).
        """
        out = np.empty(len(greedy_lp), dtype=np.float32)

        for i, (g, s) in enumerate(zip(greedy_lp, seq_lp)):
            # Align: drop the first greedy token (no corresponding seq logprob)
            if s.size == 0 or g.shape[0] <= 1:
                out[i] = 0.0
                continue

            aligned_g = g[1 : s.shape[0] + 1]          # shape [len(s)]
            gap = aligned_g - s                         # token‑wise gaps

            out[i] = float(np.mean(gap))

        return out

    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        gap = self._gap(stats["greedy_logprobs"], stats["sequence_logprobs"])

        return gap
