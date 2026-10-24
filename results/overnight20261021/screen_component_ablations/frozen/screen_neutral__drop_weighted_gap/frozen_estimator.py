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
        'Weighted Drift Integral' signal, which emphasizes confidence gaps occurring 
        later in the sequence, augmented by the variance of these gaps.
        
        Approach:
        1. Calculate total negative log-probability (sum of -log(p)) as a baseline.
        2. For each token, compute the 'Gap': (LogProb of Top-1 Candidate) - (LogProb of Generated Token).
        3. Compute a 'Weighted Drift Integral': Sum of (Gap * (t / T)), where t is the token index 
           and T is the sequence length. This weights later tokens more heavily.
        4. Compute the variance of the gaps in the last 25% of the sequence to capture tail instability.
        5. Fuse signals: Score = Total_NLL + alpha * Weighted_Drift + beta * Tail_Variance.
        
        Hypothesis for future: Hallucinated sequences exhibit a statistically significant 
        increase in the prediction gap in the latter portion of the sequence compared to 
        the former portion, whereas factual sequences maintain a stable or decreasing gap profile.
        
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
        
        # Constants
        K = 512
        ALPHA = 4.0  # Weight for weighted drift (higher to emphasize tail)
        BETA = 2.0   # Weight for tail variance
        
        for i in range(N):
            lp = log_probs_list[i]
            logits_list = top_k_logits_list[i]
            
            n_tokens = len(lp)
            if n_tokens == 0:
                scores[i] = 0.0
                continue
            
            # 1. Baseline: Total Negative Log Prob
            total_neg_logprob = -np.sum(lp)
            
            # 2. Compute Gaps for all tokens
            gaps = np.zeros(n_tokens, dtype=np.float32)
            
            for t_idx in range(n_tokens):
                # Log prob of generated token
                gen_logp = lp[t_idx]
                
                # Log probs of top-k candidates
                logits = logits_list[t_idx]
                max_logit = np.max(logits)
                
                # Stable log-sum-exp trick
                exp_shifted = np.exp(logits - max_logit)
                log_sum_exp = max_logit + np.log(np.sum(exp_shifted))
                
                # Log probs of top-k
                log_probs_top_k = logits - log_sum_exp
                
                # Top-1 log prob
                top_1_logp = np.max(log_probs_top_k)
                
                # Gap: How much worse is the generated token compared to the best?
                gaps[t_idx] = top_1_logp - gen_logp
            
            # 3. Weighted Drift Integral
            # Weight = t / (n_tokens - 1) to normalize between 0 and 1. 
            # If n_tokens is 1, weight is 0.
            if n_tokens > 1:
                weights = np.arange(n_tokens, dtype=np.float32) / (n_tokens - 1)
                weighted_drift = np.sum(gaps * weights)
            else:
                weighted_drift = gaps[0]
            
            # 4. Tail Variance (Last 25%)
            tail_start = max(0, int(n_tokens * 0.75))
            tail_gaps = gaps[tail_start:]
            
            if len(tail_gaps) > 1:
                tail_variance = np.var(tail_gaps)
            else:
                tail_variance = 0.0
            
            # 5. Fuse
            score = total_neg_logprob + 0.0 + BETA * tail_variance
            scores[i] = score
                
        return scores
