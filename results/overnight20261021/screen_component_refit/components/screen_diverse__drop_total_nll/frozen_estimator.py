import numpy as np
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
                "top_k_indices",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detects hallucinations by fusing total negative log-probability with a 
        'Normalized Temporal Drift' and 'Tail Gap Variance' signal.
        
        Approach:
        1. Calculate total negative log-probability (NLL) as a baseline.
        2. Compute the 'Gap' (Top-1 LogProb - Generated LogProb) for each token.
        3. Calculate Normalized Drift: (Mean_Gap_Last_Q - Mean_Gap_First_Q) / Mean_Gap_First_Q.
           This normalizes the drift by the initial confidence, making it robust to globally 
           low-confidence sequences.
        4. Calculate Tail Variance: Variance of gaps in the last quarter.
        5. Fuse signals: Score = NLL + alpha * Normalized_Drift + beta * Tail_Variance.
           We increase weight on variance (beta) to capture sudden confidence collapses 
           and normalize drift to handle varying baseline confidences.
        
        Hypothesis for future: Normalizing the temporal drift by initial confidence improves 
        discrimination for atomic claims where absolute log-probabilities vary widely but 
           relative deviation from the top-k path is the true indicator of hallucination.
        
        Args:
            stats: Dictionary containing 'sequence_logprobs', 'top_k_logits', 'top_k_indices'.
        
        Returns:
            np.ndarray of shape [N] with fused hallucination scores.
        """
        log_probs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        
        if len(log_probs_list) == 0:
            return np.array([], dtype=np.float32)
        
        N = len(log_probs_list)
        scores = np.zeros(N, dtype=np.float32)
        
        # Adjusted coefficients: Reduced drift weight, increased variance weight
        ALPHA = 2.0  
        BETA = 2.5   
        K = 512
        
        for i in range(N):
            lp = log_probs_list[i]
            logits_list = top_k_logits_list[i]
            
            n_tokens = len(lp)
            if n_tokens == 0:
                scores[i] = 0.0
                continue
            
            # 1. Baseline: Total Negative Log Prob
            total_neg_logprob = -np.sum(lp)
            
            # 2. Compute Gaps
            gaps = np.zeros(n_tokens, dtype=np.float32)
            
            for t_idx in range(n_tokens):
                gen_logp = lp[t_idx]
                logits = logits_list[t_idx]
                
                # Stable log-sum-exp for top-k log probs
                max_logit = np.max(logits)
                if max_logit == -np.inf:
                    # Fallback if all logits are -inf (rare but possible)
                    gaps[t_idx] = 0.0
                    continue
                
                exp_shifted = np.exp(logits - max_logit)
                log_sum_exp = max_logit + np.log(np.sum(exp_shifted))
                log_probs_top_k = logits - log_sum_exp
                
                top_1_logp = np.max(log_probs_top_k)
                gaps[t_idx] = top_1_logp - gen_logp
            
            # 3. Split into Quarters
            quarter_size = n_tokens // 4
            
            # Ensure quarters are non-empty
            if quarter_size < 1:
                scores[i] = total_neg_logprob
                continue

            first_q_indices = np.arange(quarter_size)
            last_q_start = 3 * quarter_size
            last_q_indices = np.arange(last_q_start, n_tokens)
            
            if len(first_q_indices) == 0 or len(last_q_indices) == 0:
                scores[i] = total_neg_logprob
                continue
                
            first_q_gaps = gaps[first_q_indices]
            last_q_gaps = gaps[last_q_indices]
            
            mean_first_gap = np.mean(first_q_gaps)
            mean_last_gap = np.mean(last_q_gaps)
            
            # Normalized Drift
            # If mean_first_gap is close to 0, the signal is noisy; clamp to avoid division issues
            # However, for hallucination detection, we expect mean_first_gap > 0 usually.
            # If it's 0, the model wasn't confident anywhere, so drift is less meaningful.
            # We use a small epsilon to prevent division by zero, but logic suggests 
            # if mean_first_gap is tiny, the normalized term should be dampened or ignored.
            # Here we clamp the denominator to avoid explosion if gaps are near 0.
            denom = np.clip(mean_first_gap, 1e-6, None)
            normalized_drift = (mean_last_gap - mean_first_gap) / denom
            
            # Tail Variance
            last_q_variance = np.var(last_q_gaps)
            
            # 4. Fuse
            score = 0.0 + ALPHA * normalized_drift + BETA * last_q_variance
            scores[i] = score
            
        return scores
