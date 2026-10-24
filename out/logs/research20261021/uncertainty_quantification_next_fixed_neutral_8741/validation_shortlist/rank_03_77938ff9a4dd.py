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
        Detects hallucinations by penalizing low average confidence per token.
        
        Approach:
        1. Calculate total negative log-prob for each sequence.
        2. Divide by sequence length (n_i - 1) to get mean negative log-prob (average perplexity proxy).
        3. Add a small epsilon to avoid division by zero and numerical instability.
        
        Hypothesis: Hallucinated claims often correspond to localized drops in token probability; 
        normalizing by length ensures that short sequences with high uncertainty are flagged 
        as strongly as long sequences, unlike raw NLL which favors longer sequences.
        """
        sequence_logprobs = stats["sequence_logprobs"]
        
        # Calculate total negative log-prob (sum of -logp)
        total_neg_logprob = np.array([-np.sum(seq) for seq in sequence_logprobs], dtype=np.float32)
        
        # Calculate sequence lengths (number of generated tokens = len(logprobs) + 1, but logprobs excludes first)
        # The input logprobs array has length n_i - 1.
        lengths = np.array([len(seq) + 1 for seq in sequence_logprobs], dtype=np.float32)
        
        # Avoid division by zero for empty sequences (though unlikely in generation)
        lengths_safe = np.maximum(lengths, 1.0)
        
        # Mean negative log-probability per token
        mean_neg_logprob = total_neg_logprob / lengths_safe
        
        return mean_neg_logprob
