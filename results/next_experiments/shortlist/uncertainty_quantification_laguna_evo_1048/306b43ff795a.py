
class HallucinationDetector(Estimator):
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
        """
        LogProbSlope baseline with top-k confidence range anomaly.
        Uses negative Pearson correlation between position and log-prob as baseline,
        enhanced with normalized range of top-k probabilities to detect confidence
        inconsistencies.
        
        Hypothesis: Hallucinated sequences exhibit abnormal confidence dynamics
        where the model's top-k distribution range doesn't align with the
        expected correlation pattern between position and token certainty.
        """
        scores = []
        sequence_logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        
        for seq_logprobs, top_k_logits in zip(sequence_logprobs_list, top_k_logits_list):
            n_tokens = len(seq_logprobs)
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            # Baseline: negative Pearson correlation (LogProbSlope)
            positions = np.arange(n_tokens)
            logprobs = np.array(seq_logprobs)
            
            # Compute Pearson correlation
            pos_mean = np.mean(positions)
            lp_mean = np.mean(logprobs)
            
            numerator = np.sum((positions - pos_mean) * (logprobs - lp_mean))
            denominator = np.sqrt(np.sum((positions - pos_mean)**2) * np.sum((logprobs - lp_mean)**2))
            
            if denominator == 0:
                baseline = 0.0
            else:
                baseline = -numerator / denominator  # Negative correlation as baseline
            
            # Enhanced signal: top-k confidence range anomaly
            top_k_logits_trunc = top_k_logits[:n_tokens]
            
            # Compute normalized top-k probabilities
            max_logits = np.max(top_k_logits_trunc, axis=1, keepdims=True)
            shifted_logits = top_k_logits_trunc - max_logits
            log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
            top_k_log_probs = shifted_logits - log_sum_exp
            top_k_probs = np.exp(top_k_log_probs)
            
            # Compute range (max - min) of top-k probabilities at each position
            prob_range = np.max(top_k_probs, axis=1) - np.min(top_k_probs, axis=1)
            
            # Min and max statistics (low weight to avoid outlier dependence)
            min_range = np.min(prob_range)
            max_range = np.max(prob_range)
            mean_range = np.mean(prob_range)
            
            # Normalize range statistics
            range_anomaly = (max_range - min_range) / (mean_range + 1e-8)
            
            # Combine baseline with range anomaly (low weight for min/max)
            final_score = baseline + 0.1 * range_anomaly
            scores.append(final_score)
        
        return np.array(scores, dtype=np.float32)
