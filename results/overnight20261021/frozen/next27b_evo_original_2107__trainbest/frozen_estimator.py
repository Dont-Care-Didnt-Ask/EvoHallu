import numpy as np
from typing import Dict, Any
from scipy.special import logsumexp
from scipy.stats import pearsonr
class HallucinationDetector(Estimator):
    """
    Detects hallucinations using a combination of LogProbSlope and Tail Uncertainty.
    
    Approach:
    1. LogProbSlope: Computes the negative Pearson correlation between token position 
       and actual log-probability. A positive score indicates a declining confidence 
       trend, which is a hallmark of hallucinated content where the model loses 
       consistency as it generates more tokens.
    2. Tail Uncertainty: For each token, approximates the mass of the probability 
       distribution outside the top-1 token using the available top-512 logits. 
       Specifically, it calculates the negative log-probability of the top-1 token 
       relative to the sum of the top-512 logits (approximating the full distribution). 
       Higher values indicate flatter distributions (more uncertainty).
    3. Final Score: A weighted sum of the slope score and the mean tail uncertainty.
    
    Hypothesis for Future: 
    Analyzing the variance of the tail uncertainty across the sequence could help 
    distinguish between consistent low-confidence hallucinations and sudden 
    confidence drops that might indicate the start of a fabricated segment.
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
        seq_logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        N = len(seq_logprobs_list)
        scores = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            seq_lp = seq_logprobs_list[i]  # shape (n_i - 1,)
            topk_logits = top_k_logits_list[i]  # shape (n_i, k)
            
            n_gen = len(seq_lp)
            if n_gen == 0:
                scores[i] = 0.0
                continue
            
            # Align top_k_logits with seq_lp
            # seq_lp[j] corresponds to token index j+1
            # topk_logits[j+1] corresponds to token index j+1
            aligned_topk = topk_logits[1:1+n_gen]  # shape (n_gen, k)
            
            if aligned_topk.shape[0] != n_gen:
                # Fallback if alignment fails
                scores[i] = 0.0
                continue

            # 1. LogProbSlope
            # Positions: 1 to n_gen
            positions = np.arange(1, n_gen + 1, dtype=np.float64)
            # Use actual log-probs for slope
            # If n_gen < 2, slope is undefined, use 0
            if n_gen >= 2:
                # Check for constant logprobs to avoid division by zero in pearsonr
                if np.std(seq_lp) > 1e-8:
                    corr, _ = pearsonr(positions, seq_lp)
                    # We want high score for negative correlation (declining confidence)
                    slope_score = -corr
                else:
                    slope_score = 0.0
            else:
                slope_score = 0.0

            # 2. Tail Uncertainty
            # For each token, compute the log-prob of the top-1 token relative to the top-512 sum.
            # This approximates -log(p_top1) if the top-512 covers most of the mass.
            # Actually, we want a measure of uncertainty. 
            # Let L_max be the max logit. Let L_rest_lse be logsumexp of the rest.
            # log_p_top1_approx = L_max - log(exp(L_max) + L_rest_lse) = L_max - logsumexp(all_topk)
            # Uncertainty = -log_p_top1_approx
            # We compute this for all tokens and take the mean.
            
            # Efficient computation:
            # max_logits = np.max(aligned_topk, axis=1)
            # log_denom = logsumexp(aligned_topk, axis=1)
            # top1_logprobs = max_logits - log_denom
            # tail_uncertainty = -top1_logprobs
            
            max_logits = np.max(aligned_topk, axis=1)
            log_denom = logsumexp(aligned_topk, axis=1)
            top1_logprobs = max_logits - log_denom
            tail_uncertainty = -top1_logprobs
            
            mean_tail_uncertainty = np.mean(tail_uncertainty)
            
            # Normalize slope_score to be in a similar range as tail_uncertainty?
            # Slope is in [-1, 1]. Tail uncertainty is typically > 0 (can be large).
            # We'll weight them. 
            # Typical tail uncertainty for confident tokens is ~0.1-1.0. For uncertain, > 2.0.
            # Slope is -1 to 1.
            # Let's scale slope by 2 to make it comparable.
            
            score = 2.0 * slope_score + 0.5 * mean_tail_uncertainty
            scores[i] = score
            
        return scores
