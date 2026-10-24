
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
        LogProbSlope baseline with top-k distribution concentration anomaly.
        Measures negative correlation between token position and log-probability,
        plus anomaly score based on deviation from expected top-k distribution
        concentration patterns.
        
        Hypothesis: Hallucinated sequences exhibit abnormal concentration in
        top-k distributions at specific positions, indicating model uncertainty
        or distributional shift.
        """
        scores = []
        sequence_logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        
        for seq_logprobs, top_k_logits in zip(sequence_logprobs_list, top_k_logits_list):
            n_tokens = len(seq_logprobs)
            if n_tokens < 2:
                scores.append(0.0 if n_tokens == 0 else np.sum(-np.array(seq_logprobs)))
                continue
            
            seq_logprobs_arr = np.array(seq_logprobs)
            
            # Baseline: LogProbSlope - negative pearson correlation
            positions = np.arange(n_tokens)
            if np.std(seq_logprobs_arr) < 1e-8 or np.std(positions) < 1e-8:
                slope_baseline = np.sum(-seq_logprobs_arr)
            else:
                correlation = np.corrcoef(positions, seq_logprobs_arr)[0, 1]
                slope_baseline = -correlation if not np.isnan(correlation) else np.sum(-seq_logprobs_arr)
            
            # Truncate top-k to actual sequence length
            top_k_logits_trunc = top_k_logits[:n_tokens]
            
            # Compute top-k probabilities
            max_logits = np.max(top_k_logits_trunc, axis=1, keepdims=True)
            shifted_logits = top_k_logits_trunc - max_logits
            log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
            log_probs = shifted_logits - log_sum_exp
            probs = np.exp(log_probs)
            
            # Compute distribution concentration: how much probability mass is in top-1
            top1_probs = probs[:, 0]
            
            # Compute expected concentration based on entropy
            entropy = -np.sum(probs * log_probs, axis=1)
            expected_concentration = 1.0 - entropy / np.log(512)  # Normalized entropy inverse
            
            # Anomaly: deviation from expected concentration pattern
            concentration_anomaly = np.abs(top1_probs - expected_concentration)
            
            # Weight by position (later tokens more important)
            position_weights = np.linspace(0.5, 1.5, n_tokens)
            weighted_anomaly = np.sum(position_weights * concentration_anomaly) / np.sum(position_weights)
            
            # Min/max statistics for robustness
            min_concentration = np.min(top1_probs)
            max_concentration = np.max(top1_probs)
            concentration_range = max_concentration - min_concentration
            
            # Combine signals
            final_score = slope_baseline + 0.3 * weighted_anomaly + 0.2 * concentration_range
            scores.append(final_score)
        
        return np.array(scores, dtype=np.float32)
