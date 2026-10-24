import numpy as np
from scipy.special import log_softmax
class HallucinationDetector(Estimator):
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
        """
        Detects hallucinations by scoring the combination of low observed probability 
        and high local entropy in the top-512 candidate distribution.
        
        Hypothesis: Hallucinated tokens typically exhibit both a high negative log-probability 
        (the model was unsure about the specific token) and high entropy in the top-k logits 
        (the model was generally uncertain at that step), whereas coherent tokens often have 
        moderate entropy and higher observed probability.
        """
        # Convert inputs to float64 for numerical stability
        logprobs = [arr.astype(np.float64) for arr in stats["sequence_logprobs"]]
        logits = [arr.astype(np.float64) for arr in stats["top_k_logits"]]
        
        N = len(logprobs)
        scores = np.zeros(N, dtype=np.float64)
        
        for i in range(N):
            lp = logprobs[i]
            lg = logits[i]
            
            L = len(lp)
            if L == 0:
                scores[i] = 0.0
                continue
            
            # Align: lp has length L. lg has length L+1.
            # We use the first L rows corresponding to the observed tokens.
            aligned_logits = lg[:L]
            
            # Precompute max entropy for normalization: log(512)
            max_entropy = np.log(512.0)
            
            per_token_scores = np.zeros(L, dtype=np.float64)
            
            for t in range(L):
                # 1. Get observed log-prob (already computed by the model)
                # Note: lp[t] is the log-prob of the actual token.
                # Negative log-prob is the NLL contribution.
                nll = -lp[t]
                
                # 2. Compute Entropy of the top-512 distribution at this step
                # Use log_softmax for numerical stability: log(softmax(x)) = x - logsumexp(x)
                # Entropy = -sum(p * log(p)) = -sum(exp(x - lse) * (x - lse))
                # Or simply: -sum(softmax(x) * log_softmax(x))
                
                # Compute log_softmax directly
                log_probs_topk = log_softmax(aligned_logits[t])
                
                # Compute entropy: -sum(log_probs_topk * exp(log_probs_topk))
                # Since log_probs_topk = log(p), exp(log_probs_topk) = p
                # We can compute p * log(p) directly if we have log_probs_topk
                # p = exp(log_probs_topk)
                # However, exp(log_probs_topk) might underflow if log_probs_topk is very negative.
                # Better: Entropy = logsumexp(log_probs_topk) - dot(log_probs_topk, exp(log_probs_topk - logsumexp(...)))
                # Simpler stable way: 
                # p = softmax(logits)
                # entropy = -np.sum(p * np.log(p + eps))
                
                # Let's use the logsumexp trick for stability with log_probs_topk
                # H = - E[log p] = - sum(p * log p)
                # log p = log_probs_topk
                # p = exp(log_probs_topk)
                
                # Safe exponentiation
                p = np.exp(log_probs_topk)
                
                # Handle potential underflow in p * log(p) if p is extremely small
                # But log_probs_topk is already normalized, so p sums to 1.
                # The term p * log(p) is safe if p > 0.
                
                # Calculate p * log(p)
                # If log_probs_topk is -inf, p is 0, term is 0.
                # np.exp(-inf) is 0. 0 * -inf is NaN. Need masking.
                
                mask = log_probs_topk > -np.inf
                if np.any(mask):
                    term = p[mask] * log_probs_topk[mask]
                    ent = -np.sum(term)
                else:
                    ent = 0.0
                
                # 3. Compute Score
                # Score = NLL / (1 + Entropy / log(K))
                # This weights high NLL more heavily when entropy is low, 
                # and dampens NLL when entropy is high? 
                # Wait, the hypothesis is: High NLL AND High Entropy -> High Score.
                # If Entropy is high, the denominator is large, reducing the score.
                # We want High Entropy to INCREASE the score.
                
                # Revised Formula: Score = NLL * (1 + Entropy / log(K))
                # Or Score = NLL + Entropy.
                # Let's stick to a multiplicative boost or additive.
                # Given the previous detector used weighting: Score = NLL * Weight.
                # If Weight = (1 + Entropy/LogK), then High Entropy -> High Score.
                # If Weight = 1 / (1 + Entropy/LogK), then High Entropy -> Low Score.
                # We want High Entropy to indicate Hallucination (along with High NLL).
                # So we use: Score = NLL * (1 + Entropy / LogK)
                
                # However, if NLL is low (model confident), we don't want high score even if entropy is high.
                # If Entropy is high, it means the model is unsure. If it's unsure and picked a token, 
                # that token is likely a hallucination.
                # So High Entropy is a positive signal.
                
                # Let's try: Score = NLL * (1 + Entropy / LogK)
                # If NLL is small, score is small. If Entropy is large, score is boosted.
                
                score_val = nll * (1.0 + ent / max_entropy)
                per_token_scores[t] = score_val
            
            # Aggregate: Mean per-token score
            scores[i] = np.mean(per_token_scores)
            
        return scores
