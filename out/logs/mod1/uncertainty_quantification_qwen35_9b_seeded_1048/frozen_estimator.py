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
        Detects hallucinations by computing a position-weighted sum of negative log-probabilities.
        Hypothesis: Hallucinations are more likely to manifest or be less constrained in the latter 
        half of a generated sequence; weighting tokens linearly by their position index amplifies 
        the signal from these critical regions compared to a uniform global sum.
        """
        scores = []
        for seq_logprobs in stats["sequence_logprobs"]:
            # seq_logprobs shape: (n_i - 1,)
            # Weights: linear increase from start to end of the sequence
            # t goes from 0 to len(seq_logprobs)-1
            n_tokens = len(seq_logprobs)
            if n_tokens == 0:
                scores.append(0.0)
                continue
            
            # Create position weights: w_t = t / (n_tokens - 1) if n_tokens > 1 else 1
            positions = np.arange(n_tokens, dtype=np.float32)
            if n_tokens > 1:
                max_pos = n_tokens - 1
                weights = positions / max_pos
            else:
                weights = np.array([1.0], dtype=np.float32)
            
            # Calculate weighted negative log-prob sum
            neg_logprobs = -seq_logprobs
            weighted_score = np.sum(neg_logprobs * weights)
            scores.append(weighted_score)
        
        return np.array(scores, dtype=np.float32)
