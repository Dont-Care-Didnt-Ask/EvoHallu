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
        Detects hallucinations by summing the negative log-probabilities of all tokens 
        in each sequence.
        
        Signal: Total Negative Log-Likelihood (sum of -log(p) for all tokens).
        Aggregation: Summation across the sequence dimension.
        
        Hypothesis: Hallucinated sequences exhibit a higher cumulative divergence from the 
        model's training distribution, manifesting as a larger total negative log-probability 
        compared to grounded sequences, making the aggregate score a robust discriminator.
        """
        seq_logprobs = stats["sequence_logprobs"]
        n_samples = len(seq_logprobs)
        scores = np.zeros(n_samples, dtype=np.float32)
        
        for i, lp in enumerate(seq_logprobs):
            if len(lp) == 0:
                scores[i] = 0.0
            else:
                # Convert log-probs to negative log-probs and sum
                # log-probs are typically negative, so -sum(lp) is positive
                scores[i] = float(-np.sum(lp))
                
        return scores
