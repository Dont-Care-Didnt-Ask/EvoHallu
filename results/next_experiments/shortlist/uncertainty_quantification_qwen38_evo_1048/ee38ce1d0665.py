import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    LogProbSlope with Position-Weighted Variance Detector.
    
    This detector builds on the LogProbSlope baseline (negative Pearson correlation 
    between token position and log-probability) to capture the trend of confidence 
    degradation. It augments this with a position-weighted variance term, which 
    measures the volatility of log-probabilities, giving higher weight to later 
    tokens. This is based on the hypothesis that hallucinations often involve 
    not just a drop in confidence, but an increase in uncertainty volatility 
    towards the end of atomic claims.
    
    Future hypothesis: Incorporating the entropy of the top-k distribution 
    as a per-token uncertainty measure could provide a more granular signal 
    than just the log-probability of the generated token.
    """
    
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )
        self.slope_weight = 2.0
        self.variance_weight = 1.0
        self.nll_weight = 0.5

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_logprobs = stats["sequence_logprobs"]
        N = len(sequence_logprobs)
        scores = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            seq_lp = sequence_logprobs[i]  # shape (n_i - 1,)
            n_tokens = len(seq_lp)
            
            if n_tokens < 2:
                # Not enough tokens for slope or variance
                scores[i] = -np.sum(seq_lp) if n_tokens > 0 else 0.0
                continue
            
            # 1. LogProbSlope: Negative Pearson correlation between position and log-prob
            positions = np.arange(n_tokens, dtype=np.float32)
            # Normalize positions to [0, 1] for consistency
            pos_norm = positions / (n_tokens - 1)
            
            # Calculate Pearson correlation
            # If std is 0, correlation is 0
            lp_mean = np.mean(seq_lp)
            pos_mean = np.mean(pos_norm)
            
            lp_std = np.std(seq_lp)
            pos_std = np.std(pos_norm)
            
            if lp_std > 1e-8 and pos_std > 1e-8:
                cov = np.mean((seq_lp - lp_mean) * (pos_norm - pos_mean))
                pearson_r = cov / (lp_std * pos_std)
            else:
                pearson_r = 0.0
            
            # We want negative correlation to be high score (hallucination)
            # So we take -pearson_r
            slope_score = -pearson_r
            
            # 2. Position-Weighted Variance
            # Weights: linearly increasing from 0 to 1
            weights = pos_norm
            # Normalize weights so they sum to 1 for weighted variance
            weights_norm = weights / np.sum(weights)
            
            # Weighted mean
            w_mean = np.sum(weights_norm * seq_lp)
            # Weighted variance
            w_var = np.sum(weights_norm * (seq_lp - w_mean) ** 2)
            
            # 3. Total NLL
            total_nll = -np.sum(seq_lp)
            
            # Combine scores
            # We normalize the components to have similar scales
            # slope_score is in [-1, 1], w_var is in [0, inf), total_nll is in [0, inf)
            # We'll use the raw values but with tuned weights
            score = (self.slope_weight * slope_score + 
                     self.variance_weight * w_var + 
                     self.nll_weight * total_nll)
            
            scores[i] = score
        
        return scores
