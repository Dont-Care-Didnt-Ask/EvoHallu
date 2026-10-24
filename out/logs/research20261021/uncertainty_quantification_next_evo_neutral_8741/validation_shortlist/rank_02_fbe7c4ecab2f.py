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
        Detects hallucinations by measuring the cumulative surprise (sum of negative log-probs).
        
        Approach: Sums the negative log-probability of all generated tokens for each sequence.
        Hypothesis: Hallucinated sequences incur a higher total cumulative penalty because 
        they traverse a significantly lower-probability path through the model's latent space 
        compared to factual continuations, and summation preserves this total deviation signal 
        better than averaging (which dilutes it with long factual prefixes).
        """
        scores = []
        for seq_logprobs in stats["sequence_logprobs"]:
            if len(seq_logprobs) == 0:
                scores.append(0.0)
            else:
                # Sum of negative log-probabilities (cumulative surprise)
                score = -np.sum(seq_logprobs)
                scores.append(score)
        
        return np.array(scores, dtype=np.float32)
