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
        Detects hallucinations by penalizing sequences with high average negative log-probability.
        Uses Mean Negative Log-Probability (MNLP) squared to emphasize localized improbability 
        over total sequence length.
        
        Hypothesis: Hallucinated atomic claims often consist of short bursts of highly unlikely tokens, 
        leading to high average perplexity, whereas long fluent sequences have lower average perplexity 
        despite containing some rare but valid tokens.
        """
        # Calculate total negative log-prob for each sequence
        total_neg_logprob = np.array([-np.sum(seq) for seq in stats["sequence_logprobs"]], dtype=np.float32)
        
        # Calculate sequence lengths (original length = total_neg_logprob count + 1)
        # The input logprobs are for tokens 1 to n-1, so length is len(logprobs) + 1
        lengths = np.array([len(seq) + 1 for seq in stats["sequence_logprobs"]], dtype=np.float32)
        
        # Avoid division by zero for empty sequences (though unlikely in this context)
        lengths = np.where(lengths == 0, 1.0, lengths)
        
        # Calculate Mean Negative Log-Probability (MNLP)
        mnlp = total_neg_logprob / lengths
        
        # Apply squaring to amplify the signal of high-improbability sequences
        # This helps distinguish short, dense hallucinations from long, fluent ones
        score = mnlp ** 2
        
        return score
