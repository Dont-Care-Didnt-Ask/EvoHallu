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
        Detects hallucinations by aggregating the squared negative log-probabilities of the sequence.
        
        Approach:
        1. Extract sequence_logprobs for each sample.
        2. Compute the negative log-probability for each token.
        3. Sum the squares of these negative log-probabilities for each sequence.
        
        Hypothesis: Hallucinated tokens often have extremely low probabilities (near zero), 
        resulting in massive spikes in negative log-probability. Squaring the negative log-prob 
        before summing amplifies the penalty for these extreme low-probability events more 
        than a linear sum, thereby better distinguishing sequences with severe distributional 
        mismatches (hallucinations) from those with uniform uncertainty.
        """
        scores = []
        for seq_logprobs in stats["sequence_logprobs"]:
            neg_logprobs = -np.array(seq_logprobs, dtype=np.float32)
            # Squaring emphasizes extreme low probabilities (high negative log probs)
            score = np.sum(neg_logprobs ** 2)
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)
