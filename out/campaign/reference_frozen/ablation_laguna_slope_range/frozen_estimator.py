class HallucinationDetector(Estimator):

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(context, stats_dependencies=['sequence_logprobs', 'top_k_logits'], level='sequence')

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        LogProbSlope baseline with top-k entropy deviation and distribution stability.
        Measures negative correlation between token position and log-probability,
        enhanced with normalized entropy deviation and top-k distribution stability metrics.
        
        Hypothesis: Hallucinated sequences exhibit abnormal entropy dynamics where
        the model's top-k distribution entropy deviates significantly from expected
        patterns, indicating distributional uncertainty or shift.
        """
        scores = []
        sequence_logprobs_list = stats['sequence_logprobs']
        top_k_logits_list = stats['top_k_logits']
        for seq_logprobs, top_k_logits in zip(sequence_logprobs_list, top_k_logits_list):
            n_tokens = len(seq_logprobs)
            if n_tokens < 2:
                scores.append(0.0)
                continue
            positions = np.arange(n_tokens)
            logprobs = np.array(seq_logprobs)
            pos_centered = positions - np.mean(positions)
            lp_centered = logprobs - np.mean(logprobs)
            denominator = np.sqrt(np.sum(pos_centered ** 2) * np.sum(lp_centered ** 2))
            if denominator < 1e-08:
                baseline = np.sum(-logprobs)
            else:
                correlation = np.sum(pos_centered * lp_centered) / denominator
                baseline = -correlation if not np.isnan(correlation) else np.sum(-logprobs)
            top_k_logits_trunc = top_k_logits[:n_tokens]
            max_logits = np.max(top_k_logits_trunc, axis=1, keepdims=True)
            shifted_logits = top_k_logits_trunc - max_logits
            log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
            log_probs = shifted_logits - log_sum_exp
            probs = np.exp(log_probs)
            entropy = -np.sum(probs * log_probs, axis=1)
            expected_entropy = np.log(512)
            entropy_deviation = np.abs(entropy - expected_entropy) / expected_entropy
            min_entropy_dev = np.min(entropy_deviation)
            max_entropy_dev = np.max(entropy_deviation)
            entropy_range = max_entropy_dev - min_entropy_dev
            top1_probs = probs[:, 0]
            top1_variance = np.var(top1_probs)
            mean_entropy_dev = np.mean(entropy_deviation)
            normalized_entropy_range = entropy_range / (mean_entropy_dev + 1e-08)
            final_score = baseline + 0.25 * normalized_entropy_range
            scores.append(final_score)
        return np.array(scores, dtype=np.float32)
