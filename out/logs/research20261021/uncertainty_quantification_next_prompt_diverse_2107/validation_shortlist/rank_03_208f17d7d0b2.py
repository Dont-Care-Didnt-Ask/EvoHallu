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
        Detects hallucinations by combining total negative log-probability with 
        a linearly position-weighted standard deviation of log-probabilities.
        
        Approach:
        1. Calculate total negative log-probability (sum of -log(p)) as a baseline.
        2. Calculate a position-weighted standard deviation where weights increase linearly 
           from the start to the end of the sequence. This smooths the emphasis on the 
           tail compared to a binary split, better capturing gradual confidence degradation.
        3. Fuse these signals: High total NLL indicates low overall probability, while 
           high weighted std indicates erratic confidence specifically in the critical 
           later stages of generation.
        
        Hypothesis for future: Hallucinated sequences exhibit higher variance in the 
        latter portion of the token sequence compared to factual sequences, and this 
        variance scales linearly with token position rather than appearing only after a 
        fixed midpoint threshold.
        
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
        
        # 2. Linearly Position-Weighted Variance Signal
        weighted_stds = []
        for lp in log_probs_list:
            n = len(lp)
            if n == 0:
                weighted_stds.append(0.0)
                continue
            
            # Create linear weights: 1.0 at start, increasing to 2.0 at end
            # Formula: w_i = 1.0 + (1.0 * (i / (n-1))) if n>1 else 1.0
            if n > 1:
                positions = np.arange(n, dtype=np.float32)
                weights = 1.0 + positions / (n - 1)
            else:
                weights = np.array([1.0], dtype=np.float32)
            
            # Calculate weighted mean and weighted std
            w_sum = np.sum(weights)
            weighted_mean = np.sum(weights * lp) / w_sum
            weighted_var = np.sum(weights * (lp - weighted_mean) ** 2) / w_sum
            weighted_std = np.sqrt(weighted_var)
            
            weighted_stds.append(weighted_std)
        
        weighted_stds = np.array(weighted_stds, dtype=np.float32)
        
        # 3. Fusion
        # Weight the position signal heavily as it addresses the specific hypothesis
        alpha = 3.0 
        score = total_neg_logprob + alpha * weighted_stds
        
        return score
