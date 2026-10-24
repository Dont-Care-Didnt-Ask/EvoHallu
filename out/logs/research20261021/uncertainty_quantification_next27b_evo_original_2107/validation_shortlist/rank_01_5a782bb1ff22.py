import numpy as np
from typing import Dict, Any
from scipy.stats import pearsonr
class HallucinationDetector(Estimator):
    """
    Detects hallucinations using a combination of global confidence slope and local confidence anomalies.
    
    Approach:
    1. Compute the global LogProbSlope: Negative Pearson correlation between token position and log-probabilities.
       This captures the overall trend of confidence decay across the sequence.
    2. Compute a Local Anomaly Score: For each token, calculate its Z-score relative to a local sliding window 
       (window size 11, centered). Tokens with significantly lower log-probabilities than their neighbors 
       (negative Z-scores) are considered anomalies, indicating sudden drops in confidence.
    3. Sum the negative Z-scores (anomalies) to create a penalty term.
    4. Combine the signals: Score = 1.0 * LogProbSlope + 0.5 * AnomalyPenalty.
       Higher scores indicate higher likelihood of hallucination.
    
    Hypothesis for Future:
    Integrating the entropy of the top-k logits distribution at each step could further distinguish 
    between low-confidence correct tokens and high-entropy hallucinated tokens.
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_logprobs = stats["sequence_logprobs"]
        scores = []
        
        window_size = 11  # Odd number for centered window
        
        for i in range(len(sequence_logprobs)):
            seq_logprobs = np.array(sequence_logprobs[i], dtype=np.float64)
            n_tokens = len(seq_logprobs)
            
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            # 1. Global LogProbSlope
            positions = np.arange(n_tokens, dtype=np.float64)
            if np.std(seq_logprobs) < 1e-6:
                r = 0.0
            else:
                r, _ = pearsonr(positions, seq_logprobs)
                if np.isnan(r):
                    r = 0.0
            
            logprob_slope_signal = -r  # Negative correlation -> positive signal
            
            # 2. Local Anomaly Score
            # Calculate Z-scores relative to local window
            z_scores = np.zeros(n_tokens, dtype=np.float64)
            half_w = window_size // 2
            
            for j in range(n_tokens):
                start = max(0, j - half_w)
                end = min(n_tokens, j + half_w + 1)
                window = seq_logprobs[start:end]
                
                # Avoid division by zero if window is constant
                std_w = np.std(window)
                if std_w < 1e-6:
                    z_scores[j] = 0.0
                else:
                    mean_w = np.mean(window)
                    z_scores[j] = (seq_logprobs[j] - mean_w) / std_w
            
            # Anomaly penalty: Sum of negative Z-scores (tokens much lower than local average)
            # We only penalize if the token is significantly lower than its neighbors
            anomaly_penalty = np.sum(np.clip(-z_scores, 0, None))
            
            # Normalize anomaly penalty by number of tokens to make it comparable across lengths
            anomaly_penalty_normalized = anomaly_penalty / n_tokens
            
            # 3. Combine signals
            # Weight slope heavily as it's the primary baseline, add anomaly as secondary
            score = 1.0 * logprob_slope_signal + 0.5 * anomaly_penalty_normalized
            
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)
