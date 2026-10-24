
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=["sequence_logprobs"],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the Negative Log-Likelihood (NLL) of the generated sequences.
        
        Approach:
        1. Sum the log probabilities of the generated tokens (excluding the first).
        2. Negate the sum to obtain the NLL.
        
        Hypothesis:
        The detector's performance is driven by the model's overall confidence
        (lower NLL = higher confidence = lower hallucination probability).
        """
        sequence_logprobs = stats["sequence_logprobs"]
        
        # Sum log probabilities for each sequence, then negate to get NLL
        nll_scores = np.array([-np.sum(probs) for probs in sequence_logprobs], dtype=np.float32)
        
        return nll_scores
