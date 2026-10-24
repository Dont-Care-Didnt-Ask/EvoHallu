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
        Detects hallucinations by measuring the total cumulative surprise (sum of negative log-probs).
        
        Approach: Sums the negative log-probabilities of all generated tokens (excluding the first). 
        This aggregates the total uncertainty of the generation path into a single scalar per sequence.
        
        Hypothesis: Future work should investigate whether adding a penalty term for the entropy of the 
        top-k distribution improves detection for cases where the model is confident but still wrong.
        """
        scores = []
        
        for seq_logprobs in stats["sequence_logprobs"]:
            if len(seq_logprobs) == 0:
                scores.append(0.0)
                continue
            
            # Calculate cumulative surprise (sum of -log_probs)
            # seq_logprobs contains log probabilities (negative values)
            # We sum them to get the total log-prob of the sequence path
            # Then negate to get cumulative surprise (higher is worse)
            cumulative_surprise = -np.sum(seq_logprobs)
            scores.append(cumulative_surprise)
        
        return np.array(scores, dtype=np.float32)
