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
        Detects hallucinations by fusing Total NLL with a 'Drift Rate' signal 
        (slope of surprise gap over sequence length) and 'Global Instability' 
        (variance of the surprise gap).
        
        Approach:
        1. Calculate Total Negative Log Probability (NLL).
        2. Compute the 'Surprise Gap' for each token: (Top-1 LogProb - Generated LogProb).
        3. Fit a linear regression (slope) of Gap vs. Token Index to capture the rate 
           of divergence from the model's most likely path.
        4. Calculate the global variance of the Surprise Gap to penalize erratic confidence.
        5. Fuse signals: Score = NLL + alpha * Slope + beta * Variance.
        
        Hypothesis for future: Hallucinated sequences exhibit a statistically significant 
           positive slope in the surprise gap (continuous drift) and higher global variance 
           compared to factual sequences which maintain a flatter, more consistent gap profile.
        
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
        
        # Coefficients tuned for the slope/variance signals
        ALPHA = 4.0  # Weight for drift rate
        BETA = 2.0   # Weight for global instability
        
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
                gen_logp = lp[t_idx]
                logits = logits_list[t_idx]
                
                # Stable log-sum-exp trick
                max_logit = np.max(logits)
                exp_shifted = np.exp(logits - max_logit)
                log_sum_exp = max_logit + np.log(np.sum(exp_shifted))
                
                log_probs_top_k = logits - log_sum_exp
                top_1_logp = np.max(log_probs_top_k)
                
                gaps[t_idx] = top_1_logp - gen_logp
            
            # 3. Calculate Drift Rate (Slope) using Linear Regression
            # y = gaps, x = indices [0, 1, ..., n_tokens-1]
            x = np.arange(n_tokens, dtype=np.float32)
            # Slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
            n = n_tokens
            sum_x = np.sum(x)
            sum_y = np.sum(gaps)
            sum_xy = np.sum(x * gaps)
            sum_x2 = np.sum(x * x)
            
            denom = n * sum_x2 - sum_x * sum_x
            if denom != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denom
            else:
                slope = 0.0
            
            # 4. Calculate Global Variance of Gaps
            mean_gap = np.mean(gaps)
            variance_gap = np.mean((gaps - mean_gap) ** 2)
            
            # 5. Fuse Signals
            scores[i] = total_neg_logprob + ALPHA * slope + BETA * variance_gap
            
        return scores
