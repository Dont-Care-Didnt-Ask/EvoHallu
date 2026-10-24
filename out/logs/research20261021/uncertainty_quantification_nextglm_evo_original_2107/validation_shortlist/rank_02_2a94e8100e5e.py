import numpy as np
from scipy.stats import pearsonr
from typing import Dict, Any
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detects hallucinations using the negative Pearson correlation of 
        token position and log-probability, smoothed with a sliding window.

        Hypothesis: 
        Future work should investigate whether the magnitude of the slope 
        (i.e., the rate of probability decay) is a stronger signal than the 
        correlation coefficient itself for identifying specific hallucination types.
        """
        sequence_logprobs = stats["sequence_logprobs"]
        scores = []

        for log_probs in sequence_logprobs:
            n_tokens = len(log_probs)
            if n_tokens < 2:
                scores.append(0.0)
                continue

            # Create position indices [0, 1, ..., n-2]
            positions = np.arange(n_tokens)

            # Sliding window average (window size 3)
            # This smooths the probability curve to reduce noise from individual tokens
            window_size = 3
            padded = np.pad(log_probs, (window_size // 2, window_size // 2), mode='edge')
            smoothed_probs = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')

            # Calculate Pearson correlation
            # We use scipy's pearsonr which returns (r, p-value). We take r.
            # We expect a negative correlation (probabilities decrease over sequence).
            r, _ = pearsonr(positions, smoothed_probs)
            
            # Negative correlation score: -r
            # If r is negative (decreasing prob), score is positive.
            # If r is positive (increasing prob), score is negative.
            scores.append(-r)

        return np.array(scores, dtype=np.float32)
