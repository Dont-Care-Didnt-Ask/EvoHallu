import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining total sequence uncertainty with position-weighted
    uncertainty and the maximum single-token uncertainty.
    
    Hypothesis: Hallucinated claims often involve a specific token or phrase with very low
    probability (high uncertainty) that appears later in the sequence, where specific details
    are provided. Positional weighting emphasizes these later tokens, while the max term
    captures extreme outliers in confidence.
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
        # "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # MUST return shape [N] where N = number of samples
        
        scores = []
        for sequence_logprobs in stats["sequence_logprobs"]:
            neg_logprobs = -sequence_logprobs
            n = len(neg_logprobs)
            
            if n == 0:
                scores.append(0.0)
                continue
                
            # 1. Baseline: Sum of negative log-probs
            total_neg_logprob = np.sum(neg_logprobs)
            
            # 2. Positional Weighting: Weight later tokens more heavily
            # Weights are normalized to sum to 1, then scaled by n to keep magnitude comparable to sum
            if n > 1:
                weights = np.arange(1, n + 1, dtype=np.float32)
                weights = weights / np.sum(weights) * n
                weighted_neg_logprob = np.sum(neg_logprobs * weights)
            else:
                weighted_neg_logprob = neg_logprobs[0]
                
            # 3. Max Uncertainty: The most uncertain single token
            max_neg_logprob = np.max(neg_logprobs)
            
            # Combine signals
            # We use a weighted combination. 
            # Total sum is the primary signal.
            # Weighted sum adds emphasis to later tokens.
            # Max term adds a penalty for extreme outliers.
            score = total_neg_logprob + 0.5 * weighted_neg_logprob + 0.5 * max_neg_logprob
            
            scores.append(score)
            
        return np.array(scores, dtype=np.float32)
