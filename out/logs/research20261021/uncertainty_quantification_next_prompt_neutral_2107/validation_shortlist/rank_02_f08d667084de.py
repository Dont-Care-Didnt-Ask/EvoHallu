import numpy as np
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detects hallucinations by fusing total negative log-probability with 
        a normalized tail-instability ratio.
        
        Approach:
        1. Calculate Total Negative Log Prob (NLL) as a baseline for overall confidence.
        2. Split the sequence at 60% length. Compute a linearly increasing weighted 
           variance for the tail (40% of sequence), emphasizing tokens further from the split.
        3. Compute the mean absolute log-probability of the head (60% of sequence) as a stability baseline.
        4. Calculate an Instability Ratio = (Tail Weighted Variance) / (Head Mean Abs Log-Prob + epsilon).
        5. Final Score = Total NLL + beta * Instability Ratio.
        
        This approach hypothesizes that hallucinations cause a sharp divergence in probability 
        distribution in the later stages (tail) relative to the stable initial context (head).
        
        Args:
            stats: Dictionary containing 'sequence_logprobs' (List[np.ndarray]).
        
        Returns:
            np.ndarray of shape [N] with fused hallucination scores.
        """
        log_probs_list = stats["sequence_logprobs"]
        
        if len(log_probs_list) == 0:
            return np.array([], dtype=np.float32)
        
        # 1. Total Negative Log Prob (Baseline)
        total_neg_logprob = np.array([-np.sum(lp) for lp in log_probs_list], dtype=np.float32)
        
        # 2. Tail Instability Signal (Ratio)
        instability_ratios = []
        eps = 1e-6
        
        for lp in log_probs_list:
            n = len(lp)
            if n == 0:
                instability_ratios.append(0.0)
                continue
            
            # Split at 60%
            split_idx = int(n * 0.6)
            head = lp[:split_idx]
            tail = lp[split_idx:]
            
            if len(tail) == 0 or len(head) == 0:
                instability_ratios.append(0.0)
                continue
            
            # Head Stability: Mean Absolute Log-Prob
            head_stability = np.mean(np.abs(head))
            
            # Tail Instability: Linearly Increasing Weighted Variance
            # Weights increase from 1.0 at split_idx to len(tail) at the end
            tail_len = len(tail)
            weights = np.arange(1, tail_len + 1)
            
            # Weighted Mean
            w_sum = np.sum(weights)
            weighted_mean = np.sum(weights * tail) / w_sum
            
            # Weighted Variance
            weighted_var = np.sum(weights * (tail - weighted_mean) ** 2) / w_sum
            
            # Ratio
            ratio = weighted_var / (head_stability + eps)
            instability_ratios.append(ratio)
        
        instability_ratios = np.array(instability_ratios, dtype=np.float32)
        
        # 3. Fusion
        # Weight the instability ratio heavily as it captures the specific tail divergence
        beta = 5.0 
        score = total_neg_logprob + beta * instability_ratios
        
        return score
