import numpy as np
from typing import Dict, Any
from scipy.special import log_softmax
class HallucinationDetector(Estimator):
    """
    Detects hallucination using PositionWeightedGap: the difference between the top-1 
    log-probability and the actual token log-probability, weighted by token position.
    A sliding window average of these gaps is added to smooth local fluctuations.
    
    Hypothesis for future: Incorporating the entropy of the top-k distribution 
    (not just the top-1 gap) could better distinguish between 'confidently wrong' 
    and 'uncertain' hallucinations.
    """
    
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
            ],
            level="sequence",
        )
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        N = len(sequence_logprobs)
        scores = np.zeros(N, dtype=np.float32)
        
        # Sliding window size for smoothing
        window_size = 5
        
        for i in range(N):
            logprobs = sequence_logprobs[i]
            logits = top_k_logits[i]
            
            n_tokens = len(logits)
            if n_tokens < 2:
                scores[i] = 0.0
                continue
            
            # sequence_logprobs has length n_tokens - 1
            # It corresponds to tokens at indices 1 to n_tokens-1
            # logits has length n_tokens, corresponding to indices 0 to n_tokens-1
            
            # We can only compute gaps for tokens where we have both:
            # actual_logprob (from sequence_logprobs) and top1_logprob (from logits)
            # For token index j (0-based in logits), actual_logprob is logprobs[j-1] if j >= 1
            
            gaps = np.zeros(n_tokens - 1, dtype=np.float32)
            
            for j in range(1, n_tokens):
                # Compute log-probs for this token position
                log_probs_j = log_softmax(logits[j])
                top1_logprob_j = log_probs_j[0]
                
                # Actual log-prob for this token is in sequence_logprobs at index j-1
                actual_logprob_j = logprobs[j-1]
                
                # Gap: How much more confident was the top-1 choice than the actual choice?
                # If actual is top-1, gap is 0. If actual is lower, gap is positive.
                gaps[j-1] = top1_logprob_j - actual_logprob_j
            
            # Position weights: later tokens get higher weight
            # Positions for gaps are 1 to n_tokens-1
            positions = np.arange(1, n_tokens, dtype=np.float32)
            weights = positions / n_tokens
            
            # Main score: Sum of position-weighted gaps
            weighted_gaps = weights * gaps
            score_main = np.sum(weighted_gaps)
            
            # Sliding window average of gaps
            if len(gaps) >= window_size:
                kernel = np.ones(window_size) / window_size
                # Use 'same' mode to keep length same as gaps
                sliding_avg = np.convolve(gaps, kernel, mode='same')
                # Weight the sliding average by position as well
                weighted_sliding = np.sum(weights * sliding_avg)
            else:
                weighted_sliding = 0.0
            
            # Combine scores
            scores[i] = score_main + 0.5 * weighted_sliding
        
        return scores
