import numpy as np
from typing import Dict, Any
from scipy.special import logsumexp
class HallucinationDetector(Estimator):
    """
    Detects hallucinations using Position-Weighted Gap between top-1 prediction and actual token.
    
    Approach:
    1. For each token position (excluding the first), compute the log-probability of the top-1 
       candidate from the top-k logits.
    2. Compute the gap between this top-1 log-prob and the actual generated token's log-prob.
    3. Weight each gap by its relative position in the sequence (later tokens get higher weight).
    4. Apply a sliding window average to smooth the weighted gaps.
    5. Sum the smoothed weighted gaps to produce the final hallucination score.
    
    Hypothesis: Hallucinations are characterized by a larger divergence between the model's 
    highest-confidence prediction and the actual token generated, particularly in later positions 
    of the sequence where contextual constraints are stronger.
    """
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
        sequence_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        
        N = len(sequence_logprobs)
        scores = np.zeros(N, dtype=np.float32)
        
        window_size = 5  # Sliding window size for smoothing
        
        for i in range(N):
            seq_lp = sequence_logprobs[i]
            tk_logits = top_k_logits[i]
            
            if len(seq_lp) == 0 or len(tk_logits) == 0:
                scores[i] = 0.0
                continue
                
            n_tokens = len(tk_logits)
            # seq_lp has length n_tokens - 1
            # tk_logits[t] predicts token t+1
            # seq_lp[t-1] is the log-prob of token t+1 for t >= 1
            
            weighted_gaps = np.zeros(n_tokens, dtype=np.float32)
            
            for t in range(1, n_tokens):
                # Get logits for predicting token t+1
                logits_t = tk_logits[t]
                
                # Calculate log-prob of top-1 candidate
                # log(p_top1) = max_logit - logsumexp(logits)
                max_logit = np.max(logits_t)
                lse = logsumexp(logits_t)
                log_p_top1 = max_logit - lse
                
                # Get log-prob of actual token
                # seq_lp index t-1 corresponds to token t+1
                log_p_actual = seq_lp[t-1]
                
                # Calculate gap
                # Gap is how much lower the actual token's prob is compared to the top-1
                gap = log_p_top1 - log_p_actual
                
                # Ensure non-negative (should be, but numerical errors can occur)
                if gap < 0:
                    gap = 0.0
                
                # Position weight: t / n_tokens
                # t ranges from 1 to n_tokens-1
                pos_weight = t / n_tokens
                
                weighted_gaps[t] = gap * pos_weight
            
            # Apply sliding window average to smooth the weighted gaps
            if n_tokens > window_size:
                # Use a simple moving average
                kernel = np.ones(window_size) / window_size
                # Pad to handle edges
                padded = np.pad(weighted_gaps, (window_size // 2, window_size // 2), mode='edge')
                smoothed = np.convolve(padded, kernel, mode='valid')
                # Trim back to original length
                start = window_size // 2
                end = start + n_tokens
                smoothed_gaps = smoothed[start:end]
            else:
                smoothed_gaps = weighted_gaps
            
            # Final score is the sum of smoothed weighted gaps
            scores[i] = np.sum(smoothed_gaps)
            
        return scores
