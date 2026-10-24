import numpy as np
from typing import Dict, Any
from scipy.signal import savgol_filter
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
        Detects hallucinations by analyzing the variance of the second derivative
        (curvature) of the log-probability sequence.
        
        Approach:
        1. Baseline: Uses the sum of negative log-probabilities (LogProb).
        2. Signal: Computes the second derivative of the log-probability curve.
           This captures the "jaggedness" or instability of the confidence path.
        3. Aggregation: Calculates the mean of the absolute second derivative.
        
        Hypothesis:
        Future work should investigate whether the spectral density of the 
        log-probability sequence (specifically the frequency of high-amplitude 
        oscillations) is a more robust indicator of semantic coherence than 
        the raw variance of the second derivative.
        """
        sequence_logprobs = stats["sequence_logprobs"]
        scores = []

        for log_probs in sequence_logprobs:
            n_tokens = len(log_probs)
            if n_tokens < 3:
                # Need at least 3 points to compute a second derivative
                scores.append(0.0)
                continue

            # 1. Baseline: Sum of negative log-probabilities
            baseline = -np.sum(log_probs)

            # 2. Approximation of Second Derivative (Curvature)
            # Using a simple finite difference method: d2f/dx2 approx (f[i+1] - 2*f[i] + f[i-1])
            # We compute the second derivative for the interior points.
            second_derivative = np.zeros_like(log_probs)
            second_derivative[1:-1] = log_probs[2:] - 2 * log_probs[1:-1] + log_probs[:-2]

            # 3. Signal: Variance of the second derivative
            # We use the mean of absolute values to emphasize instability regardless of direction.
            signal = np.mean(np.abs(second_derivative))

            # 4. Aggregation: Weighted combination of Baseline and Signal
            # We weight the signal higher (0.5) relative to the baseline (0.5) 
            # because the second derivative specifically targets the shape of the confidence curve.
            score = 0.5 * baseline + 0.5 * signal
            scores.append(score)

        return np.array(scores, dtype=np.float32)
