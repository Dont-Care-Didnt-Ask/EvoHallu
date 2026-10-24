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
        Detects hallucinations by fusing Total Negative Log-Probability with a composite 
        uncertainty signal comprising 'Global Mean Gap' and 'Temporal Drift'.
        
        Approach:
        1. Calculate Total Negative Log-Probability (NLL) as a baseline confidence score.
        2. For each token, compute the 'Gap': (LogProb of Top-1 Candidate) - (LogProb of Generated Token).
        3. Compute two derived signals from these gaps:
           a. Global Mean Gap: The average deviation across the entire sequence.
           b. Temporal Drift: The difference between the mean gap in the last 25% and the first 25% of the sequence.
        4. Fuse signals: Score = NLL + alpha * Global_Mean_Gap + beta * Temporal_Drift.
        
        Hypothesis for future: Hallucinated sequences exhibit both a higher average deviation from 
        the model's most likely path (Global Mean Gap) and a systematic increase in this deviation 
        towards the end of the sequence (Temporal Drift), whereas factual sequences maintain low 
        and stable gaps.
        
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
        ALPHA = 2.5   # Weight for Global Mean Gap
        BETA = 2.0    # Weight for Temporal Drift
        
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
            
            # 3. Compute Derived Signals
            
            # Signal A: Global Mean Gap
            mean_gap = np.mean(gaps)
            
            # Signal B: Temporal Drift (Last 25% Mean - First 25% Mean)
            quarter_size = n_tokens // 4
            
            first_q_indices = np.arange(quarter_size)
            last_q_start = 3 * quarter_size
            last_q_indices = np.arange(last_q_start, n_tokens)
            
            temporal_drift = 0.0
            if len(first_q_indices) > 0 and len(last_q_indices) > 0:
                mean_first_gap = np.mean(gaps[first_q_indices])
                mean_last_gap = np.mean(gaps[last_q_indices])
                temporal_drift = mean_last_gap - mean_first_gap
            
            # 4. Fuse
            score = total_neg_logprob + ALPHA * mean_gap + BETA * temporal_drift
            scores[i] = score
            
        return scores
