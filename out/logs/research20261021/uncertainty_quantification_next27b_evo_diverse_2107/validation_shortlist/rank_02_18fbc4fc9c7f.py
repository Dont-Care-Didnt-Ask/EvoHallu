import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations using the unnormalized, quadratically position-weighted mean surprisal.
    This refines the linear position-weighted sum approach by applying a quadratic ramp to the weights,
    hypothesizing that the risk of hallucination is disproportionately concentrated in the final 
    portion of the sequence where the model must resolve narrative consistency.
    
    Future hypothesis: Incorporating a penalty for high entropy in the top-k logits at the 
    peak-weighted positions might further distinguish between coherent but uncertain 
    generations and incoherent hallucinations.
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
            
            neg_logprobs = -np.asarray(sequence_logprobs, dtype=np.float32)
            
            if n == 1:
                # For a single token, the weight is effectively 1 (or 0 depending on definition, 
                # but usually we want to capture the signal). 
                # In the limit of i/(n-1), 0/0 is undefined. 
                # Consistent with the "end-weighted" philosophy, a single token is the "end".
                # However, to maintain continuity with the sum formula where weights start at 0,
                # let's look at the behavior. 
                # If we use (i/(n-1))^2, for n=1, it's 0/0. 
                # Let's treat n=1 as weight 1.0 to avoid zeroing out the only signal.
                scores.append(float(neg_logprobs[0]))
                continue
            
            # Quadratic ramp weights from 0 to 1
            # This emphasizes the later tokens more strongly than a linear ramp
            indices = np.arange(n, dtype=np.float32)
            weights = (indices / (n - 1)) ** 2
            
            # Weighted sum (no length normalization)
            # This preserves the magnitude of the total uncertainty, which was beneficial in the linear version
            weighted_sum = np.sum(neg_logprobs * weights)
            scores.append(float(weighted_sum))
        
        return np.array(scores, dtype=np.float32)
