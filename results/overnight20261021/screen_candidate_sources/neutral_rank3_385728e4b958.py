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
        Detects hallucinations by fusing total negative log-probability with 
        a 'Tail-Weighted Gap Instability' signal.
        
        Approach:
        1. Calculate total negative log-probability (sum of -log(p)) as a baseline.
        2. For each token, compute the 'Gap': (LogProb of Top-1 Candidate) - (LogProb of Generated Token).
        3. Apply a linear weight to the Gap sequence, increasing from 0 at the start to 1 at the end.
        4. Compute two signals from the weighted gaps:
           a. Weighted Mean Gap: Captures the increasing divergence towards the end.
           b. Weighted Variance: Captures the instability of the divergence in the tail.
        5. Fuse signals: Score = Total_NLL + alpha * Weighted_Mean_Gap + beta * Weighted_Variance.
        
        Hypothesis for future: Hallucinated sequences exhibit a systematic increase in the 
        average gap between generated tokens and top-1 candidates as the sequence progresses, 
        which is better captured by tail-weighted statistics than global averages.
        
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
        ALPHA = 2.5  # Weight for weighted mean gap
        BETA = 1.5   # Weight for weighted variance
        
        for i in range(N):
            lp = log_probs_list[i]
            logits_list = top_k_logits_list[i]
            
            n_tokens = len(lp)
            if n_tokens == 0:
                scores[i] = 0.0
                continue
            
            # 1. Baseline: Total Negative Log Prob
            total_neg_logprob = -np.sum(lp)
            
            # 2. Compute Gaps and Apply Weights
            # Weights: linear ramp from 0 to 1
            weights = np.linspace(0.0, 1.0, n_tokens)
            
            weighted_gap_sum = 0.0
            weighted_gap_sq_sum = 0.0
            weighted_gap_mean = 0.0
            
            # Pre-allocate for vectorization if possible, but loop is clearer for weighted stats
            # To optimize, we compute gaps and accumulate weighted sums in one pass
            
            current_weighted_gap_sum = 0.0
            current_weighted_gap_sq_sum = 0.0
            
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
                # log_probs_top_k = logits - log_sum_exp
                
                # Top-1 log prob
                top_1_logp = np.max(log_probs_top_k) if (log_probs_top_k := logits - log_sum_exp).size > 0 else 0.0
                
                # Gap: How much worse is the generated token compared to the best?
                gap = top_1_logp - gen_logp
                
                w = weights[t_idx]
                
                current_weighted_gap_sum += gap * w
                current_weighted_gap_sq_sum += (gap * w) ** 2
            
            # Calculate Weighted Mean and Weighted Variance
            w_sum = np.sum(weights)
            if w_sum > 0:
                weighted_mean_gap = current_weighted_gap_sum / w_sum
                # Weighted Variance formula: E[X^2] - (E[X])^2
                weighted_var_gap = (current_weighted_gap_sq_sum / w_sum) - (weighted_mean_gap ** 2)
            else:
                weighted_mean_gap = 0.0
                weighted_var_gap = 0.0
            
            # 3. Fuse
            score = total_neg_logprob + ALPHA * weighted_mean_gap + BETA * weighted_var_gap
            scores[i] = score
            
        return scores
