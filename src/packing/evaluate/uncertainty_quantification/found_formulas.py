from typing import Any, Dict, List, Optional

import numpy as np

from src.packing.evaluate.uncertainty_quantification.estimator_base import Estimator


class SequenceLogProb(Estimator):
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")

    def __str__(self) -> str:
        return "SequenceLogProb"

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        # "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # MUST return shape [N] where N = number of samples
        total_neg_logprob = np.array([-np.sum(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]], dtype=np.float32)
        return total_neg_logprob

class SequenceLogProbBottom(Estimator):
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")

    def __str__(self) -> str:
        return "SequenceLogProbBottom"

    @staticmethod
    def _sum_bottom(sequence_logprobs: np.ndarray) -> float:
        length = sequence_logprobs.shape[0]
        bottom_half = np.sort(sequence_logprobs)[:length // 2]
        return np.sum(bottom_half)

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        sequence_logprob_bottom = np.array([self._sum_bottom(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]], dtype=np.float32)
        return -sequence_logprob_bottom

class MinimalLogProb(Estimator):
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")

    def __str__(self) -> str:
        return "MinimalLogProb"

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        minimal_logprob = np.array([np.min(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]], dtype=np.float32)
        return -minimal_logprob

class MeanTokenEntropy(Estimator):
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")

    def __str__(self) -> str:
        return "MeanTokenEntropy"

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        # MUST return shape [N] where N = number of samples
        mean_token_entropy = np.array([np.mean(token_entropy) for token_entropy in stats["token_entropies"]], dtype=np.float32)
        return mean_token_entropy


class Perplexity(Estimator):
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")

    def __str__(self) -> str:
        return "Perplexity"

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        perplexity = np.array([-np.exp(np.mean(sequence_logprobs)) for sequence_logprobs in stats["sequence_logprobs"]], dtype=np.float32)
        return perplexity


class StdTokenEntropy(Estimator):
    """Larger standard deviation should indicate more uncertainty.
    """
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["token_entropies"], level="sequence")

    def __str__(self) -> str:
        return "StdTokenEntropy"

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        std_token_entropy = np.array([np.std(token_entropy) for token_entropy in stats["token_entropies"]], dtype=np.float32)
        return std_token_entropy


class MeanStdTokenEntropy(Estimator):
    """Mix of mean and standard deviation of token entropy.
    """
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context, stats_dependencies=["token_entropies"], level="sequence")

    def __str__(self) -> str:
        return "MeanStdTokenEntropy"

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        mean_std_token_entropy = np.array([np.mean(token_entropy) + np.std(token_entropy) for token_entropy in stats["token_entropies"]], dtype=np.float32)
        return mean_std_token_entropy


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


class QuartileEntropyDiff_PositionWeightedGap(Estimator):
    """
    Unsupervised hallucination detector that fuses two temporally-aware cues:

    1. Entropy acceleration: mean(entropy_last_quarter) - mean(entropy_first_quarter).
        Positive -> uncertainty increasing.

    2. Position-weighted greedy-vs-sampled log-probability gap:
        For each token t compute (greedy_logprob[t] - sequence_logprob[t-1])
        and weight it by t / length (later tokens get larger weight).

    Each cue is z-scored across the batch, linearly combined with weights
    (0.5, 0.5), and shifted to be non-negative.  Higher scores indicate
    a higher likelihood of hallucination.
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "token_entropies",   # List[np.ndarray] – shape [n_i]
                "greedy_logprobs",   # List[np.ndarray] – shape [n_i]
                "sequence_logprobs", # List[np.ndarray] – shape [n_i-1]
                "top_k_logits",      # List[np.ndarray] – shape [n_i, k] (k ≥ 2)
            ],
            level="sequence",
        )

    def __str__(self) -> str:
        return "QuartileEntropyDiff_PositionWeightedGap"

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

    @staticmethod
    def _pos_weighted_gap(
        greedy_lp: List[np.ndarray],
        seq_lp: List[np.ndarray],
    ) -> np.ndarray:
        """
        For each sequence compute a position‑weighted mean of
        (greedy_logprob[t] – sequence_logprob[t‑1]).
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

    # ------------------------------------------------------------------
    # Core estimator
    # ------------------------------------------------------------------
    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        # 1️⃣ Entropy acceleration
        ent_acc = self._entropy_acceleration(stats["token_entropies"])

        # 2️⃣ Position‑weighted greedy‑vs‑sampled log‑prob gap
        gap = self._pos_weighted_gap(stats["greedy_logprobs"], stats["sequence_logprobs"])

        # Normalise each cue (higher → higher hallucination risk)
        f1 = self._zscore(ent_acc)                     # already risk‑oriented
        f2 = self._zscore(gap)                         # larger gap = riskier

        # Weighted linear fusion (weights sum to 1)
        combined = 0.5 * f1 + 0.5 * f2

        # Shift to non‑negative domain (helps with thresholding)
        min_val = combined.min()
        if min_val < 0:
            combined = combined - min_val

        return combined
        # Return ndarray of shape [N]


class QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay(Estimator):
    """
    Unsupervised hallucination detector that fuses three temporally-aware cues:

    1. Entropy acceleration: mean(entropy_last_quarter) - mean(entropy_first_quarter).
        Positive -> uncertainty increasing.

    2. Position-weighted greedy-vs-sampled log-probability gap:
        For each token t compute (greedy_logprob[t] - sequence_logprob[t-1])
        and weight it by t / length (later tokens get larger weight).

    3. Top-k mass decay: mean(mass_last_quarter) - mean(mass_first_quarter),
        where mass is the cumulative softmax probability of the top-k logits.
        Negative values (mass loss) are turned into a risk signal.

    Each cue is z-scored across the batch, linearly combined with weights
    (0.35, 0.35, 0.30), and shifted to be non-negative.  Higher scores indicate
    a higher likelihood of hallucination.
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "token_entropies",   # List[np.ndarray] – shape [n_i]
                "greedy_logprobs",   # List[np.ndarray] – shape [n_i]
                "sequence_logprobs", # List[np.ndarray] – shape [n_i-1]
                "top_k_logits",      # List[np.ndarray] – shape [n_i, k] (k ≥ 2)
            ],
            level="sequence",
        )

    def __str__(self) -> str:
        return "QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay"

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

    @staticmethod
    def _pos_weighted_gap(
        greedy_lp: List[np.ndarray],
        seq_lp: List[np.ndarray],
    ) -> np.ndarray:
        """
        For each sequence compute a position‑weighted mean of
        (greedy_logprob[t] – sequence_logprob[t‑1]).
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

    @staticmethod
    def _topk_mass_decay(topk_logits: List[np.ndarray], small_k: int = 8) -> np.ndarray:
        """
        Compute (mass_last_quarter - mass_first_quarter) where
        mass = Σ softmax(top-k logits) per position.
        """
        out = np.empty(len(topk_logits), dtype=np.float32)

        for i, tk in enumerate(topk_logits):
            # tk shape: [n_i, k]
            n, k = tk.shape
            if n < 4 or k < 1:
                out[i] = 0.0
                continue

            # Stable softmax over top‑k logits -- renormalized probabilities, which sum to 1 over k tokens
            max_logit = np.max(tk, axis=1, keepdims=True)          # [n,1]
            shifted = tk - max_logit
            exp_shift = np.exp(shifted)
            probs = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)  # [n,k]

            # Total probability mass captured by the top‑small_k set (should be < 1)
            mass = np.sum(probs[:, :small_k], axis=1)   # shape [n]

            q = n // 4
            first_mass = np.mean(mass[:q])
            last_mass = np.mean(mass[-q:])
            out[i] = float(last_mass - first_mass)   # negative → mass loss
        return out

    # ------------------------------------------------------------------
    # Core estimator
    # ------------------------------------------------------------------
    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        # 1️⃣ Entropy acceleration
        ent_acc = self._entropy_acceleration(stats["token_entropies"])

        # 2️⃣ Position‑weighted greedy‑vs‑sampled log‑prob gap
        gap = self._pos_weighted_gap(stats["greedy_logprobs"], stats["sequence_logprobs"])

        # 3️⃣ Top‑k mass decay (we flip sign later so that larger → higher risk)
        mass_decay = self._topk_mass_decay(stats["top_k_logits"])

        # Normalise each cue (higher → higher hallucination risk)
        f1 = self._zscore(ent_acc)                     # already risk‑oriented
        f2 = self._zscore(gap)                         # larger gap = riskier
        f3 = -self._zscore(mass_decay)                # mass loss (negative) → riskier

        # Weighted linear fusion (weights sum to 1)
        combined = 0.35 * f1 + 0.35 * f2 + 0.30 * f3

        # Shift to non‑negative domain (helps with thresholding)
        min_val = combined.min()
        if min_val < 0:
            combined = combined - min_val

        return combined
        # Return ndarray of shape [N]


class EntropySlope(Estimator):
    """
    Unsupervised hallucination detector that computes the slope of token-wise entropy across the sequence.
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=["token_entropies"],
            level="sequence",
        )

    def __str__(self) -> str:
        return "EntropySlope"

    # Feature 2 – slope of token‑wise entropy across the sequence
    @staticmethod
    def _entropy_slope(entropy_lists: List[np.ndarray]) -> np.ndarray:
        """
        Fit a simple linear regression (entropy = a * pos + b) per sequence and
        return the slope a.  If the sequence length < 2 the slope is defined as 0.
        """
        slopes = []
        for ent in entropy_lists:
            if ent.size < 2:
                slopes.append(0.0)
                continue
            positions = np.arange(ent.size, dtype=np.float32)
            # Closed‑form slope = cov(x, y) / var(x)
            cov = np.mean((positions - positions.mean()) * (ent - ent.mean()))
            var = np.mean((positions - positions.mean()) ** 2)
            slope = cov / var if var != 0 else 0.0
            slopes.append(float(slope))
        return np.array(slopes, dtype=np.float32)

    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        return self._entropy_slope(stats["token_entropies"])

# seed - 1044
# Iteration: 291
# Test AUC: 78.8555, n_samples: 6974
# Test AUC 95% confidence interval (bootstrap): (77.49, 80.07)
# Val AUC: 78.7663, n_samples: 1402
# Val AUC 95% confidence interval (bootstrap): (76.18, 81.59)
# Train AUC: 79.8255, n_samples: 5605
# Train AUC 95% confidence interval (bootstrap): (78.54, 81.05)
class LogProbSlope(Estimator):
    """
    Detect hallucinations via the *trend of token‑wise confidence*.

    Hypothesis (future work):
        In hallucinated continuations the model’s confidence (log‑probability)
        tends to decrease as generation proceeds, yielding a negative
        Pearson correlation between token index and log‑probability.
        Larger (more negative) correlation magnitudes indicate a higher
        likelihood of hallucination.

    Dependencies
    -------------
    - "sequence_logprobs": List[np.ndarray] where each element has shape
      (seq_len_i‑1,) containing the natural‑log probability of each generated token.
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=["sequence_logprobs"],
            level="sequence",
        )
    
    def __str__(self) -> str:
        return "LogProbSlope"

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        """
        Parameters
        ----------
        stats : dict
            "sequence_logprobs" – list of length N; each entry is a
            1-D array of log-probabilities for the generated tokens of one sample.

        Returns
        -------
        np.ndarray, shape (N,)
            Negative Pearson correlation between token position and log‑probability.
            Higher values (i.e. more negative correlation) correspond to a higher
            propensity to hallucinate.
        """
        scores: List[float] = []

        for logp in stats["sequence_logprobs"]:
            # Guard against empty generations.
            if logp.size == 0:
                scores.append(0.0)
                continue

            # Token positions: 0, 1, 2, …
            positions = np.arange(logp.shape[0], dtype=np.float64)

            # If all log‑probs are identical the correlation is undefined;
            # treat it as zero (no trend).
            if np.allclose(logp, logp[0]):
                scores.append(0.0)
                continue

            # Compute Pearson correlation coefficient.
            # np.corrcoef returns a 2×2 matrix; the off‑diagonal entry is the correlation.
            corr = np.corrcoef(positions, logp)[0, 1]

            # We want a score that grows with *more negative* correlation.
            # Flip the sign so that larger scores indicate higher hallucination risk.
            scores.append(-corr)

        return np.array(scores, dtype=np.float32)


# =============================================================================
# Experiment 1041 - Best method (Val AUC: 80.53%, iteration 478)
# Original had 3 features but topk_mass_deficit is always 0 (softmax sums to 1)
# Effective formula: exp_weighted(neg_logprob) + 0.7 * exp_weighted(one_minus_top1)
# =============================================================================
class ExpWeightedNegLogProbWithTop1(Estimator):
    """
    Two-feature hallucination detector with exponential positional weighting.

    Features (computed per token, then aggregated with exponential positional weights):
    1. neg_logprob - negative token log-probability from sequence_logprobs
    2. one_minus_top1 - (1 - p_top1) where p_top1 is softmax probability of most likely token

    Positional weighting:
        w_t = (exp(alpha * t/(T-1)) - 1) / (exp(alpha) - 1)
        alpha defaults to 2.5 - heavily weights later tokens (~10-12x more than early tokens)

    Score = weighted_mean(neg_logprob) + lambda_top1 * weighted_mean(one_minus_top1)
    Higher scores indicate higher hallucination likelihood.

    Note: First token gets ~0 weight due to exp(0)-1=0 in the weighting formula.
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
            ],
            level="sequence",
        )
        self.lambda_top1 = 0.7
        self.pos_alpha = 2.5

    def __str__(self) -> str:
        return "ExpWeightedNegLogProbWithTop1"

    @staticmethod
    def _exp_weights(length: int, alpha: float) -> np.ndarray:
        """Normalized exponential positional weights (later tokens get more weight)."""
        if length <= 1:
            return np.ones(length, dtype=np.float32)

        pos = np.arange(length, dtype=np.float32)
        w = np.exp(alpha * pos / (length - 1)) - 1.0
        denom = np.exp(alpha) - 1.0
        if denom > 0:
            w /= denom
        # Normalize to sum to 1
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        return w.astype(np.float32)

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        seq_logprobs: List[np.ndarray] = stats["sequence_logprobs"]
        topk_logits: List[np.ndarray] = stats["top_k_logits"]

        N = len(seq_logprobs)
        scores = np.empty(N, dtype=np.float32)

        for i in range(N):
            logp_seq = seq_logprobs[i]
            logits = topk_logits[i]

            T = logp_seq.shape[0]
            if T == 0:
                scores[i] = 0.0
                continue

            # Align logits to match logprobs length
            logits = logits[:T, :]

            # Positional weights
            w = self._exp_weights(T, self.pos_alpha)

            # Feature 1: weighted mean of negative log-probability
            wm_neg_logprob = np.sum(w * (-logp_seq))

            # Feature 2: weighted mean of (1 - top1 probability)
            # Softmax over top-k logits
            max_per_row = np.max(logits, axis=1, keepdims=True)
            shifted = logits - max_per_row
            exp_shifted = np.exp(shifted)
            probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
            p_top1 = probs[:, 0]
            wm_one_minus_top1 = np.sum(w * (1.0 - p_top1))

            # Linear combination
            scores[i] = float(wm_neg_logprob) + self.lambda_top1 * float(wm_one_minus_top1)

        return scores


class ExpWeightedNegLogProb(Estimator):
    """
    Simple hallucination detector using exponential positional weighting on neg log-prob.

    This is a minimal baseline to test whether exponential positional weighting
    alone improves over uniform weighting (like in SequenceLogProb).

    Positional weighting:
        w_t = (exp(alpha * t/(T-1)) - 1) / (exp(alpha) - 1)
        alpha defaults to 2.5 - heavily weights later tokens

    Score = weighted_mean(-log_prob)
    Higher scores indicate higher hallucination likelihood.

    Hypothesis: Hallucinations tend to "snowball" - errors at the end of a sequence
    are more indicative of hallucination than errors at the start.
    """

    def __init__(self, context: Dict[str, Any], alpha: float = 2.5) -> None:
        super().__init__(
            context,
            stats_dependencies=["sequence_logprobs"],
            level="sequence",
        )
        self.pos_alpha = alpha

    def __str__(self) -> str:
        return "ExpWeightedNegLogProb"

    @staticmethod
    def _exp_weights(length: int, alpha: float) -> np.ndarray:
        """Normalized exponential positional weights (later tokens get more weight)."""
        if length <= 1:
            return np.ones(length, dtype=np.float32)

        pos = np.arange(length, dtype=np.float32)
        w = np.exp(alpha * pos / (length - 1)) - 1.0
        denom = np.exp(alpha) - 1.0
        if denom > 0:
            w /= denom
        # Normalize to sum to 1
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        return w.astype(np.float32)

    def __call__(self, stats: Dict[str, List[np.ndarray]]) -> np.ndarray:
        scores: List[float] = []

        for logp_seq in stats["sequence_logprobs"]:
            T = logp_seq.shape[0]
            if T == 0:
                scores.append(0.0)
                continue

            # Exponential positional weights
            w = self._exp_weights(T, self.pos_alpha)

            # Weighted sum of negative log-probabilities
            score = float(np.sum(w * (-logp_seq)))
            scores.append(score)

        return np.array(scores, dtype=np.float32)


class PositionWeightedGap_EntropySlope_TopKMassInv(Estimator):
    """
    Unsupervised hallucination detector that fuses three complementary,
    sequence-level uncertainty cues:

      1. avg_gap_weighted   – position-weighted mean gap
         (greedy_logprob – sampled_logprob).  Larger gaps suggest the model
         generated tokens far from its own most-likely choice.

      2. entropy_slope      – linear-regression slope of token-wise entropy
         across the sequence.  Positive slope = growing uncertainty.

      3. topk_mass_inv      – 1 - average probability mass captured by the
         top-k logits (flatter top-k -> higher risk).

    Each cue is robust-scaled (median / IQR) across the batch, linearly combined
    with configurable weights, and finally squashed through a sigmoid so that
    the output lies in (0, 1).  Higher scores indicate a higher likelihood of
    hallucination.
    """

    def __init__(
        self,
        context: Dict[str, Any],
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        context : dict
            Forwarded to the base ``Estimator``.
        feature_weights : dict, optional
            Mapping from feature name (``gap``, ``slope``, ``mass``) to a scalar
            weight.  Missing entries default to 1.0; a weight of 0 disables the
            corresponding cue.
        """
        default_weights = {"gap": 1.0, "slope": 1.0, "mass": 1.0}
        if feature_weights is not None:
            default_weights.update(feature_weights)
        self.weights = default_weights

        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",   # List[np.ndarray] – shape [seq_len-1]
                "greedy_logprobs",     # List[np.ndarray] – shape [seq_len]
                "token_entropies",     # List[np.ndarray] – shape [seq_len]
                "top_k_logits",        # List[np.ndarray] – shape [seq_len, k]
            ],
            level="sequence",
        )

    def __str__(self) -> str:
        return "PositionWeightedGap_EntropySlope_TopKMassInv"

    # ------------------------------------------------------------------ #
    # Feature 1 – position‑weighted average gap (greedy – sampled)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _avg_gap_weighted(
        seq_logprobs: List[np.ndarray],
        greedy_logprobs: List[np.ndarray],
    ) -> np.ndarray:
        """
        For each sample compute a position‑weighted mean of
        (greedy_logprob – sampled_logprob).  Tokens later in the generation
        receive larger linear weight: w_t = (t+1) / T, where T is the total
        number of tokens (including the first token that is not present in
        ``sequence_logprobs``).
        """
        out = []
        for sampled_lp, greedy_lp in zip(seq_logprobs, greedy_logprobs):
            # sampled_lp corresponds to tokens 2..T (len = T‑1)
            T = sampled_lp.shape[0] + 1
            if sampled_lp.size == 0:
                out.append(0.0)
                continue

            # Align greedy length with sampled length (greedy includes first token)
            if greedy_lp.shape[0] == T:
                # drop the first greedy token so that lengths match sampled_lp
                aligned_greedy = greedy_lp[1:]
            else:
                # fallback: truncate or pad
                min_len = min(greedy_lp.shape[0] - 1, sampled_lp.shape[0])
                aligned_greedy = greedy_lp[1:1 + min_len]
                if min_len < sampled_lp.shape[0]:
                    pad = np.full(sampled_lp.shape[0] - min_len, -1e9, dtype=greedy_lp.dtype)
                    aligned_greedy = np.concatenate([aligned_greedy, pad])

            gap = aligned_greedy - sampled_lp  # positive → sampled less likely
            weights = np.arange(2, T + 1) / T   # positions 2..T
            weighted_mean = np.sum(gap * weights) / np.sum(weights)
            out.append(float(weighted_mean))
        return np.asarray(out, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Feature 2 – slope of token‑wise entropy across the sequence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _entropy_slope(entropy_lists: List[np.ndarray]) -> np.ndarray:
        """
        Fit a simple linear regression (entropy = a * pos + b) per sequence and
        return the slope a.  If the sequence length < 2 the slope is defined as 0.
        """
        slopes = []
        for ent in entropy_lists:
            if ent.size < 2:
                slopes.append(0.0)
                continue
            positions = np.arange(ent.size, dtype=np.float32)
            # Closed‑form slope = cov(x, y) / var(x)
            cov = np.mean((positions - positions.mean()) * (ent - ent.mean()))
            var = np.mean((positions - positions.mean()) ** 2)
            slope = cov / var if var != 0 else 0.0
            slopes.append(float(slope))
        return np.array(slopes, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Feature 3 – inverse average top‑k probability mass
    # ------------------------------------------------------------------ #
    @staticmethod
    def _topk_mass_inv(topk_logits: List[np.ndarray], small_k: int = 8) -> np.ndarray:
        """
        For each position compute a softmax over the top-k logits, sum the resulting
        probabilities (the probability mass captured by the top-small_k), average this
        mass over the sequence and return 1 - average_mass.  Flatter distributions
        -> smaller mass -> larger inverted value.
        """
        inv = []
        for logits in topk_logits:
            if logits.size == 0:
                inv.append(0.0)
                continue
            # stable softmax
            max_logit = np.max(logits, axis=1, keepdims=True)
            exp_shift = np.exp(logits - max_logit)
            probs = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
            avg_mass = np.mean(np.sum(probs[:, :small_k], axis=1))   # average top‑k mass per token
            inv.append(float(1.0 - avg_mass))
        return np.asarray(inv, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Robust scaling (median / IQR)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _robust_scale(x: np.ndarray) -> np.ndarray:
        """(x - median) / IQR, with fallback IQR = 1 to avoid div‑by‑zero."""
        median = np.median(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        if iqr == 0.0:
            iqr = 1.0
        return (x - median) / iqr

    # ------------------------------------------------------------------ #
    # Sigmoid – squashes to (0, 1)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def __call__(self, stats: Dict[str, Any]) -> np.ndarray:
        # 1️⃣ Compute raw cues
        gap_feat   = self._avg_gap_weighted(
            stats["sequence_logprobs"],
            stats["greedy_logprobs"],
        )
        slope_feat = self._entropy_slope(stats["token_entropies"])
        mass_feat  = self._topk_mass_inv(stats["top_k_logits"])

        # 2️⃣ Robust‑scale each cue across the batch
        gap_sc   = self._robust_scale(gap_feat)   # larger → higher risk
        slope_sc = self._robust_scale(slope_feat) # larger → higher risk
        mass_sc  = self._robust_scale(mass_feat)  # larger → higher risk

        # 3️⃣ Weighted linear combination (all cues positively correlated with hallucination)
        w = self.weights
        combined = (
            w["gap"]   * gap_sc +
            w["slope"] * slope_sc +
            w["mass"]  * mass_sc
        )

        # 4️⃣ Map to (0, 1) via sigmoid and return
        scores = self._sigmoid(combined).astype(np.float32)
        return scores
