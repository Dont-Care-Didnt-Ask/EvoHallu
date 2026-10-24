import numpy as np
from typing import Dict, Any, List
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
        Detects hallucinations by computing the mean negative log-probability per token.
        
        Approach:
        1. For each sequence, sum the negative log-probabilities of all tokens (excluding the first).
        2. Divide by the number of tokens (length - 1) to get the mean negative log-prob.
        3. Return the array of mean scores for all sequences.
        
        Hypothesis:
        Hallucinated sequences tend to have a higher average negative log-probability than truthful ones,
        independent of sequence length, because the model assigns lower probability to factually incorrect content.
        """
        n_sequences = len(stats["sequence_logprobs"])
        scores = np.zeros(n_sequences, dtype=np.float32)
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            # seq_logprobs has shape (n_i - 1,)
            n_tokens = len(seq_logprobs)
            if n_tokens == 0:
                scores[i] = 0.0
            else:
                neg_logprobs = -np.array(seq_logprobs, dtype=np.float32)
                scores[i] = np.mean(neg_logprobs)
        
        return scores
