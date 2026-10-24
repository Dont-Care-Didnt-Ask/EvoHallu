
class HallucinationDetector(Estimator):
    """
    Computes Pearson correlation between token position and log-probability.
    Negative correlations indicate increasing uncertainty over sequence length,
    which may signal hallucination onset.
    
    Hypothesis: Hallucinations emerge when models lose coherence, causing
    confidence to drop as they continue generating inconsistent content.
    """
    
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        scores = []
        for sequence_logprobs in stats["sequence_logprobs"]:
            n_tokens = len(sequence_logprobs)
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            positions = np.arange(n_tokens)
            logprobs = np.array(sequence_logprobs)
            
            # Compute Pearson correlation coefficient
            mean_pos = np.mean(positions)
            mean_lp = np.mean(logprobs)
            
            numerator = np.sum((positions - mean_pos) * (logprobs - mean_lp))
            denominator = np.sqrt(np.sum((positions - mean_pos)**2) * np.sum((logprobs - mean_lp)**2))
            
            if denominator == 0:
                correlation = 0.0
            else:
                correlation = numerator / denominator
            
            # Return negative correlation as score (higher = more likely hallucination)
            scores.append(-correlation)
        
        return np.array(scores, dtype=np.float32)
