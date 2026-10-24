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
        Detects hallucinations by penalizing sequences with low average token probability density.
        
        Approach: Calculates the mean negative log-probability (avg-NLL) across all generated tokens 
        in each sequence. This normalizes the score by sequence length, isolating the model's 
        confidence per token rather than the total cumulative penalty.
        
        Hypothesis: Hallucinated claims often correspond to tokens generated with significantly lower 
        probability mass than factual claims, even within short sequences; length normalization 
        prevents long sequences from being penalized solely due to length.
        """
        # Calculate sum of negative logprobs for each sequence
        neg_log_sums = np.array([-np.sum(seq) for seq in stats["sequence_logprobs"]], dtype=np.float32)
        
        # Calculate sequence lengths (original length = sum_len + 1 because first token excluded)
        lengths = np.array([len(seq) + 1 for seq in stats["sequence_logprobs"]], dtype=np.float32)
        
        # Avoid division by zero if a sequence is empty (though unlikely given input shape constraints)
        # If length is 1, avg_nll equals the single token's neg_logprob
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_neg_logprobs = np.divide(neg_log_sums, lengths, out=np.zeros_like(neg_log_sums), where=lengths!=0)
        
        return avg_neg_logprobs
