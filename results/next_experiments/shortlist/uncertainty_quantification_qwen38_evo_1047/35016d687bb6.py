import numpy as np
from typing import Dict, Any
from scipy.special import logsumexp
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining total sequence surprisal with an exponentially 
    weighted sum of top-1 regret.
    
    Approach:
    1. Computes the sum of negative log-probabilities (baseline surprisal).
    2. For each token position, calculates the 'regret': the difference between the 
       log-probability of the top-1 candidate and the actual generated token.
    3. Applies exponential weighting to the regret values, giving higher weight to 
       later tokens in the sequence (decay factor 0.98).
    4. Returns the sum of the baseline surprisal and the weighted regret signal.
    
    Hypothesis: Hallucinations are more likely to manifest as deviations from the 
    model's top-1 prediction in later sequence positions, where contextual constraints 
    are strongest, and this deviation is more indicative when the model was confident 
    in its top choice.
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
        
        # Exponential decay factor for weighting later tokens more heavily
        decay = 0.98
        
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
            
            # 1. Baseline: Sum of negative log-probabilities
            total_neg_logprob = -np.sum(seq_lp)
            
            # 2. Exponentially Weighted Regret
            weighted_regret_sum = 0.0
            
            # Pre-compute weights for positions 1 to n_tokens-1
            # Weights are applied to the regret at position t (predicting token t+1)
            # We want later tokens to have higher weight.
            # Let's define weight for position t (0-indexed in tk_logits, corresponding to seq_lp index t-1)
            # We will iterate t from 1 to n_tokens-1
            
            # To avoid recalculating powers, we can iterate backwards or use cumulative products
            # But n is small (10-25 tokens), so direct calculation is fine.
            
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
                
                # Calculate regret: how much lower the actual token's prob is compared to the top-1
                # Regret is non-negative if top-1 is indeed the top-1 (which it is by definition of max)
                # However, due to floating point, log_p_actual might be slightly higher if the actual token WAS the top-1
                # In that case, regret should be 0.
                regret = log_p_top1 - log_p_actual
                if regret < 0:
                    regret = 0.0
                
                # Exponential weight: higher weight for larger t (later in sequence)
                # We use t as the exponent base for decay.
                # Weight = decay^(n_tokens - t) ? No, we want later tokens (higher t) to have higher weight.
                # Let's use weight = decay^(n_tokens - 1 - t) is wrong, that would weight early tokens more if decay < 1.
                # We want weight to increase with t.
                # Let's use weight = decay^(t) is decreasing.
                # Let's use weight = (1/decay)^(t) ? That grows exponentially.
                # Better: Normalize weights so they sum to 1, or just use a simple scheme.
                # The prompt says "Exponential weighted average". Usually, this implies recent items matter more.
                # So weight for position t (where t is 1..n-1) should be proportional to decay^(n-1-t) is wrong.
                # Standard EWA: w_t = (1-alpha) * alpha^(T-t) where T is total time.
                # Here, let's just use w_t = decay^(n_tokens - t).
                # If t is close to n_tokens (end), exponent is small, weight is high (close to 1).
                # If t is small (beginning), exponent is large, weight is low.
                weight = decay ** (n_tokens - t)
                
                weighted_regret_sum += weight * regret
            
            # Combine signals
            # Scale the regret sum to be comparable to the surprisal sum?
            # Surprisal is sum of -log p. Regret is difference of log p.
            # They are in the same units (nats).
            # We add them directly.
            scores[i] = total_neg_logprob + weighted_regret_sum
            
        return scores
