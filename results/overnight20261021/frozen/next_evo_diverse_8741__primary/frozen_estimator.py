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
        Hypothesis: Hallucinated segments often consist of extended sequences of low-probability tokens; 
        summing surprisal captures the cumulative 'cost' of these unlikely sequences better than averaging, 
        as it is less sensitive to short, isolated anomalies in longer sequences.
        """
        log_probs_list = stats["sequence_logprobs"]
        # Calculate total negative log-prob (sum of surprisal) per sequence
        scores = np.array([-np.sum(lp, dtype=np.float32) for lp in log_probs_list], dtype=np.float32)
        return scores
