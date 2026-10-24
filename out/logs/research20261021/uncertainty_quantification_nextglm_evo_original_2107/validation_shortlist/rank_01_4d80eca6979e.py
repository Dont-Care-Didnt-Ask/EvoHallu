
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=["sequence_logprobs"],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the negative Pearson correlation between token position and log-probability.

        Approach:
        Computes the slope of confidence over the sequence length. Factual claims 
        typically maintain stable confidence (low negative slope), whereas hallucinations 
        often exhibit rapid confidence degradation (high negative slope).

        Hypothesis: 
        Future work should explore whether this slope is driven by model overconfidence 
        at generation start or by specific token-level instability patterns.
        """
        sequence_logprobs = stats["sequence_logprobs"]
        scores = []

        for i, log_probs in enumerate(sequence_logprobs):
            # Ensure log_probs is a numpy array
            log_probs = np.array(log_probs, dtype=np.float32)
            
            # Token positions: 0 to len-1
            positions = np.arange(len(log_probs))
            
            # Calculate Pearson correlation between position and log-probability
            # np.corrcoef returns a matrix; we want the [0, 1] element
            corr_matrix = np.corrcoef(positions, log_probs)
            slope = corr_matrix[0, 1]
            
            # Return negative slope as score (Higher score = More negative correlation = More Hallucination)
            scores.append(-slope)

        return np.array(scores, dtype=np.float32)
