import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining total negative log-probability with a 
    position-weighted negative log-probability that emphasizes later tokens.
    
    Approach: Uses the sum of negative log-probs as the primary signal, augmented 
    by a linearly position-weighted sum of negative log-probs. This weights later 
    tokens more heavily, under the hypothesis that hallucinations often manifest 
    as increased uncertainty or drift in the latter part of the generated sequence.
    """
    
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # MUST return shape [N] where N = number of samples
        
        n_samples = len(stats["sequence_logprobs"])
        scores = np.zeros(n_samples, dtype=np.float32)
        
        for i, sequence_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(sequence_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            neg_logprobs = -sequence_logprobs
            n_tokens = len(neg_logprobs)
            
            # Baseline: Total sum of negative log-probs
            total_neg_logprob = np.sum(neg_logprobs)
            
            # Position-weighted sum: weights increase linearly from 1 to n_tokens
            # This emphasizes later tokens
            if n_tokens > 1:
                weights = np.arange(1, n_tokens + 1, dtype=np.float32)
                # Normalize weights to sum to 1? Or just use raw weighted sum?
                # Using raw weighted sum might make longer sequences inherently higher score.
                # To be comparable to total_neg_logprob, let's use a weighted average 
                # multiplied by n_tokens, or just a weighted sum scaled appropriately.
                # Let's use a weighted sum where the average weight is 1.
                # weights_normalized = weights / np.mean(weights)
                # weighted_sum = np.sum(neg_logprobs * weights_normalized)
                
                # Simpler: Just add the weighted sum directly. 
                # Since total_neg_logprob is the sum, and weighted_sum is roughly 
                # (n+1)/2 * total_neg_logprob, this might dominate.
                # Let's use a weighted *average* instead to keep scale similar.
                weighted_avg = np.average(neg_logprobs, weights=weights)
                # Scale weighted_avg by n_tokens to be comparable to sum?
                # weighted_sum_scaled = weighted_avg * n_tokens
                # This is equivalent to sum(neg_logprobs * weights / mean(weights))
                
                # Let's try: Score = Total Neg LogProb + (Weighted Avg Neg LogProb - Mean Neg LogProb) * n_tokens
                # This adds a penalty if later tokens are more uncertain than earlier ones.
                mean_neg_logprob = np.mean(neg_logprobs)
                position_penalty = (weighted_avg - mean_neg_logprob) * n_tokens
            else:
                position_penalty = 0.0
            
            scores[i] = total_neg_logprob + position_penalty
        
        return scores
