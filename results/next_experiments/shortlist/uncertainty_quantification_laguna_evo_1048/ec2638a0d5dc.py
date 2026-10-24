
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
        LogProbSlope baseline with top-k distribution stability metric.
        Combines negative Pearson correlation between position and log-probability
        with a stability measure based on top-k probability variance across positions.
        
        Hypothesis: Hallucinated sequences exhibit unstable top-k distributions
        where probability mass fluctuates abnormally across token positions,
        indicating distributional shifts during generation.
        """
        scores = []
        sequence_logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        
        for seq_logprobs, top_k_logits in zip(sequence_logprobs_list, top_k_logits_list):
            n_tokens = len(seq_logprobs)
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            # Baseline: LogProbSlope - negative Pearson correlation
            positions = np.arange(n_tokens)
            logprobs = np.array(seq_logprobs)
            
            pos_mean = np.mean(positions)
            lp_mean = np.mean(logprobs)
            
            numerator = np.sum((positions - pos_mean) * (logprobs - lp_mean))
            denominator = np.sqrt(np.sum((positions - pos_mean)**2) * np.sum((logprobs - lp_mean)**2))
            
            if denominator == 0:
                baseline = 0.0
            else:
                baseline = -numerator / denominator
            
            # Enhanced signal: top-k distribution stability
            top_k_logits_trunc = top_k_logits[:n_tokens]
            
            # Compute top-k probabilities
            max_logits = np.max(top_k_logits_trunc, axis=1, keepdims=True)
            shifted_logits = top_k_logits_trunc - max_logits
            log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
            top_k_log_probs = shifted_logits - log_sum_exp
            top_k_probs = np.exp(top_k_log_probs)
            
            # Compute stability: variance of top-1 probabilities across positions
            top1_probs = top_k_probs[:, 0]
            stability_variance = np.var(top1_probs)
            
            # Min/max statistics for robustness (low weight)
            min_stability = np.min(top1_probs)
            max_stability = np.max(top1_probs)
            stability_range = max_stability - min_stability
            
            # Normalize by mean for scale invariance
            mean_stability = np.mean(top1_probs)
            normalized_stability = stability_range / (mean_stability + 1e-8)
            
            # Combine signals with low weight for min/max to avoid outlier dependence
            final_score = baseline + 0.15 * normalized_stability
            scores.append(final_score)
        
        return np.array(scores, dtype=np.float32)
