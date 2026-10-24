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
        Detects hallucinations using the cumulative Negative Log-Probability (NLL) of the sequence.
        
        Approach:
        1. Sums the negative log-probabilities of all generated tokens in the sequence.
        2. Returns the raw sum as the score.
        
        Hypothesis for future: Hallucinated atomic claims are generated via token sequences that 
        are statistically further from the training distribution manifold than factual claims, 
        resulting in a consistently higher cumulative negative log-probability (total path cost) 
        regardless of local entropy patterns or trajectory smoothness.
        """
        # Extract list of log-prob arrays
        logprobs_list = stats["sequence_logprobs"]
        
        # Calculate total negative log-prob for each sequence
        # shape: (N,) where N is number of samples
        scores = np.array([
            -np.sum(seq_logprobs) 
            for seq_logprobs in logprobs_list
        ], dtype=np.float32)
        
        return scores
