import numpy as np
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
        Detects hallucinations by calculating a position-weighted sum of negative log-probabilities.
        
        Hypothesis: Hallucinated claims exhibit a progressive degradation of token probability 
        towards the end of the sequence (coherence drift). Weighting later tokens higher 
        amplifies this tail-end instability compared to a uniform sum.
        """
        scores = []
        for seq_logprobs in stats["sequence_logprobs"]:
            # seq_logprobs shape: (n_i - 1,)
            # Weights: linear increase from 1 to n, normalized by n
            n_tokens = len(seq_logprobs)
            if n_tokens == 0:
                scores.append(0.0)
                continue
            
            # Create weights: [1, 2, ..., n] / n
            weights = np.arange(1, n_tokens + 1, dtype=np.float32) / n_tokens
            
            # Calculate weighted sum of negative log-probs
            # Note: input is logprobs (usually negative), so -logprobs is surprisal (positive)
            weighted_surprisal = np.sum(-seq_logprobs * weights)
            scores.append(weighted_surprisal)
        
        return np.array(scores, dtype=np.float32)
