import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations based solely on the cumulative improbability of the generated sequence.
    
    Approach: 
    Calculates the Sum of Negative Log-Probabilities (Sum-NLL) for each sequence.
    We sum the log-probabilities of all generated tokens (excluding the first).
    Higher sums indicate sequences the model finds increasingly unlikely, correlating with hallucination.
    
    Hypothesis: 
    For atomic claims, the primary discriminator between grounded and hallucinated content is the 
    model's inherent perplexity (improbability) of the specific claim, rendering secondary signals 
    like confidence gaps redundant for this specific evaluation metric.
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
        # sequence_logprobs: List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # We need to return an array of shape [N] where N is the number of samples.
        
        seq_logprobs_list = stats["sequence_logprobs"]
        
        # Calculate Sum-NLL for each sequence
        # -np.sum(logprobs) converts Sum(Log-P) to Sum(-Log-P)
        scores = np.array([-np.sum(logprobs, dtype=np.float32) for logprobs in seq_logprobs_list], dtype=np.float32)
        
        return scores
