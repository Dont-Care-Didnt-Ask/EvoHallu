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
        Detects hallucinations based on cumulative token unlikelihood.
        
        Approach: Sums the negative log-probabilities of all generated tokens.
        Hypothesis: Hallucinated sequences exhibit higher cumulative surprise than 
        factual sequences; summing preserves sensitivity to short sequences with 
        severe local errors that averaging would dilute.
        """
        # sequence_logprobs: List[np.ndarray], inner array length n_i - 1
        # Compute sum of negative log-probs for each sequence
        scores = np.array([
            -np.sum(seq_logprobs) 
            for seq_logprobs in stats["sequence_logprobs"]
        ], dtype=np.float32)
        
        return scores
