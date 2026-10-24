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
        """
        Detects hallucinations by calculating the total cumulative uncertainty 
        (sum of negative log-probabilities) for each sequence.
        
        Approach:
        1. Sum the negative log-probabilities for all tokens in the sequence.
        2. Return the total sum as the hallucination score.
        
        Hypothesis for future: Incorporating the ratio of the maximum log-probability 
        to the mean log-probability might further refine detection for cases where a 
        single token is extremely unlikely but the rest of the sequence is very confident.
        """
        sequence_logprobs = stats["sequence_logprobs"]
        scores = np.array([
            -np.sum(seq_logprobs) 
            for seq_logprobs in sequence_logprobs
        ], dtype=np.float32)
        
        return scores
