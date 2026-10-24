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
        Detects hallucinations by summing the negative log-probabilities of the generated sequence.
        
        Approach: 
        1. Iterates through each sequence in the input.
        2. Sums the negative log-probabilities (nll = -log_prob) for all tokens except the first.
        3. Returns the total cumulative NLL as the score.
        
        Hypothesis: Hallucinated sequences incur a higher cumulative information-theoretic penalty 
        (total negative log-probability) because the model assigns lower probabilities to the 
        specific sequence of tokens it generates, reflecting a lack of confidence in the 
        entire claim compared to factual ground-truth generations.
        """
        sequences = stats["sequence_logprobs"]
        N = len(sequences)
        
        if N == 0:
            return np.array([], dtype=np.float32)

        scores = np.zeros(N, dtype=np.float32)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                scores[i] = 0.0
                continue
            
            # Calculate negative log-probability for each token
            # sequence_logprobs excludes the first token, so we sum all provided values
            neg_logprobs = -seq
            scores[i] = np.sum(neg_logprobs)

        return scores
