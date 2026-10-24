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
        Detects hallucinations based on the average negative log-probability per token.
        
        Hypothesis: Hallucinated tokens are, on average, assigned lower probabilities 
        by the model compared to grounded tokens, and normalizing by sequence length 
        prevents bias towards longer generations.
        """
        sequence_logprobs = stats["sequence_logprobs"]
        n_sequences = len(sequence_logprobs)
        scores = np.zeros(n_sequences, dtype=np.float32)
        
        for i, seq_logprobs in enumerate(sequence_logprobs):
            # sequence_logprobs excludes the first generated token, so length is n_i - 1
            # Sum negative log probs
            neg_log_sum = -np.sum(seq_logprobs)
            
            # Length of the generated sequence (excluding the first token which is excluded from logprobs)
            # If seq_logprobs is empty, we assume length 1 (the first token) or handle as 0 score
            seq_len = len(seq_logprobs) + 1
            
            if seq_len > 0:
                scores[i] = neg_log_sum / seq_len
            else:
                scores[i] = 0.0
                
        return scores
