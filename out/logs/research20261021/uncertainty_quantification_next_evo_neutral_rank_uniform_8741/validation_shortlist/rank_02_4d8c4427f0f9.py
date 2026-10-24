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
        Detects hallucinations by measuring the cumulative surprise (sum of negative log-probabilities)
        across the entire generated sequence.
        
        Signal: Sum of negative log-probabilities for all tokens in the sequence.
        Aggregation: Global Sum (no length normalization).
        
        Hypothesis: Hallucinated atomic claims accumulate higher total surprise scores because they 
        often require the model to traverse a path of lower-probability tokens to fabricate specific 
        details, leading to a larger cumulative penalty compared to truthful claims, regardless of 
        sequence length.
        """
        # Extract sequence logprobs
        seq_logprobs = stats["sequence_logprobs"]
        
        # Calculate negative log-prob for each sequence and sum them up
        # sequence_logprobs is a list of 1D arrays. 
        # We sum the negative log-probs for each sequence to get a single scalar per sample.
        total_neg_logprobs = np.array([
            -np.sum(sequence_logprobs) 
            for sequence_logprobs in seq_logprobs
        ], dtype=np.float32)
        
        return total_neg_logprobs
