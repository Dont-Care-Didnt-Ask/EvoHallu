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
        'Temporal Drift' signal: the increase in the gap between generated tokens 
        and top-1 candidates from the first quarter to the last quarter of the sequence.
        
        Approach:
        1. Calculate total negative log-probability (sum of -log(p)) as a baseline.
        2. For each token, compute the 'Gap': (LogProb of Top-1 Candidate) - (LogProb of Generated Token).
        3. Split the sequence into the first 25% and last 25% of tokens.
        4. Calculate the difference in mean gap (Last_Quarter_Mean_Gap - First_Quarter_Mean_Gap).
           - A large positive value indicates the model drifts away from its most likely paths 
             as generation progresses, a hallmark of hallucination.
        5. Add the variance of gaps in the last quarter to penalize instability at the end.
        6. Fuse signals: Score = Total_NLL + alpha * Temporal_Drift + beta * Last_Quarter_Variance.
        
        Hypothesis for future: Hallucinated sequences exhibit a statistically significant 
        increase in the prediction gap (deviation from top-1) in the latter quarter compared 
        to the former quarter, whereas factual sequences maintain a stable or decreasing gap profile.
        
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
        ALPHA = 3.0  # Weight for temporal drift
        BETA = 1.5   # Weight for tail variance
        
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
            # We need to convert logits to log_probs for the top-k list
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
            
            # 3. Split into First Quarter and Last Quarter
            quarter_size = n_tokens // 4
            
            # First Quarter: indices 0 to quarter_size - 1
            first_q_indices = np.arange(quarter_size)
            
            # Last Quarter: indices (3*quarter_size) to (4*quarter_size - 1)
            # Handle edge case where n_tokens is not perfectly divisible by 4
            last_q_start = 3 * quarter_size
            last_q_indices = np.arange(last_q_start, n_tokens)
            
            # 4. Calculate Metrics
            if len(first_q_indices) > 0 and len(last_q_indices) > 0:
                first_q_gaps = gaps[first_q_indices]
                last_q_gaps = gaps[last_q_indices]
                
                mean_first_gap = np.mean(first_q_gaps)
                mean_last_gap = np.mean(last_q_gaps)
                
                # Temporal Drift: Increase in gap from start to end
                temporal_drift = mean_last_gap - mean_first_gap
                
                # Variance in the last quarter
                last_q_variance = np.var(last_q_gaps)
                
                # 5. Fuse
                score = total_neg_logprob + ALPHA * temporal_drift + BETA * last_q_variance
                scores[i] = score
            else:
                # Fallback for very short sequences
                scores[i] = total_neg_logprob
                
        return scores
