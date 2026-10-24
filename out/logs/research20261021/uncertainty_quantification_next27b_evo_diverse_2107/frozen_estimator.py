import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations using length-unnormalized, quadratically position-weighted negative log-likelihood.
    Applies a quadratic ramp weight (0 to 1) across tokens to strongly emphasize later positions,
    then sums to capture total accumulated uncertainty.
    
    Future hypothesis: Incorporating the variance of top-k logit gaps might help distinguish 
    between high-uncertainty due to ambiguity vs. hallucination.
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
        scores = []
        for sequence_logprobs in stats["sequence_logprobs"]:
            n = len(sequence_logprobs)
            if n == 0:
                scores.append(0.0)
                continue
            
            # Negative log-probs (higher = less confident)
            neg_logprobs = -np.asarray(sequence_logprobs, dtype=np.float32)
            
            # Quadratic ramp weights from 0 to 1
            if n == 1:
                weights = np.array([1.0], dtype=np.float32)
            else:
                # i ranges from 0 to n-1
                # Normalize index to [0, 1]
                indices = np.arange(n, dtype=np.float32)
                weights = (indices / (n - 1)) ** 2
            
            # Weighted sum (no length normalization)
            weighted_sum = np.sum(neg_logprobs * weights)
            scores.append(float(weighted_sum))
        
        return np.array(scores, dtype=np.float32)
