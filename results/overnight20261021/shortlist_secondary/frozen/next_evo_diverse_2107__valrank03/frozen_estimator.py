import numpy as np
from typing import Dict, Any, List
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
        Detects hallucinations by summing the negative log-probabilities of tokens 
        that fall below the sequence-specific 90th percentile threshold.
        
        Hypothesis: Hallucinated tokens are significantly more rare than the model's 
        typical uncertainty for that specific sequence context; using a relative 
        quantile (90th percentile) rather than a fixed absolute threshold adapts to 
        varying perplexity baselines and isolates true outliers within each sequence.
        """
        seq_logprobs = stats["sequence_logprobs"]
        scores = []
        
        for seq in seq_logprobs:
            if len(seq) == 0:
                scores.append(0.0)
                continue
            
            # Calculate the 90th percentile of log-probs for this specific sequence
            # This defines the "extreme tail" boundary relative to the sequence's own distribution
            threshold = np.percentile(seq, 90.0)
            
            # Select tokens more surprising than the 90th percentile
            extreme_mask = seq < threshold
            
            # Sum the negative log-probabilities of these extreme tokens
            # If no tokens are below the threshold, sum is 0
            score = -np.sum(seq[extreme_mask])
            scores.append(float(score))
            
        return np.array(scores, dtype=np.float32)
