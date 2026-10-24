import numpy as np
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
        # Extract sequence logprobs
        seq_logprobs = stats["sequence_logprobs"]
        
        scores = np.zeros(len(seq_logprobs), dtype=np.float32)
        
        for i, probs in enumerate(seq_logprobs):
            if len(probs) == 0:
                scores[i] = 0.0
                continue
            
            # Signal 1: Global Surprise (Sum of negative log-probs)
            # Hallucinated sequences are often globally less probable.
            global_surprise = -np.sum(probs)
            
            # Signal 2: Instability Penalty (Variance of log-probs)
            # Hallucinations often involve erratic jumps between plausible and implausible tokens.
            # Using population variance for stability.
            instability = np.var(probs)
            
            # Aggregation: Weighted sum. 
            # Global surprise is the primary driver (based on 0.74 score), 
            # but instability helps distinguish coherent low-prob sequences from hallucinations.
            # Coefficients tuned to balance the two signals.
            score = global_surprise + 0.5 * instability
            
            scores[i] = score
            
        return scores
