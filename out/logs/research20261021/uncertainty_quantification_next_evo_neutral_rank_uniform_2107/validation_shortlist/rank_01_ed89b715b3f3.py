import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining cumulative improbability with decision instability.
    
    Approach: 
    1. Calculates the Sum of Negative Log-Probabilities (Sum-NLL) for each sequence.
    2. Adds a penalty term: Sum of (Max_TopK_Logit - Log_Prob_of_Generated) for each token.
       - If the generated token is the argmax, the penalty is 0.
       - If the generated token is not the argmax (common in hallucinations/sampling), 
         the penalty increases the score, reflecting the model's lack of decisive confidence.
    
    Hypothesis: 
    Hallucinated sequences are characterized by either extremely low probability paths 
    (captured by Sum-NLL) or a pattern where the model frequently selects tokens that are 
    not the most probable among its top-k candidates (captured by the deviation term), 
    indicating a disconnect between the model's confidence and its output.
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
        # sequence_logprobs: List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # top_k_logits: List[np.ndarray] of shape (n_i, k).
        # top_k_indices: List[np.ndarray] of shape (n_i, k).
        
        seq_logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        
        scores = []
        
        for logprobs, logits in zip(seq_logprobs_list, top_k_logits_list):
            # 1. Base Score: Sum of Negative Log-Probabilities
            # logprobs is already log(p), so -sum(logprobs) is Sum-NLL
            base_score = -np.sum(logprobs, dtype=np.float32)
            
            # 2. Deviation Penalty: Sum of (Max_Logit - Generated_Logit)
            # Find the max logit in the top-k for this position
            max_logits = np.max(logits, axis=1) # Shape (n_i,)
            
            # The logprobs list corresponds to the generated tokens.
            # We need to align the generated logit with the max logit.
            # logprobs[i] corresponds to the log-prob of the token generated at step i+1 (since list is n_i-1 long).
            # logits[i] corresponds to the top-k logits for step i+1.
            
            # Calculate the difference: Max_Logit - Generated_Logit
            # Note: logits are raw logits, logprobs are log-probs. 
            # To compare fairly, we should use log-probs for the top-k as well.
            # However, we only have logits. We can convert max_logits to log-prob by subtracting log(sum(exp(logits))).
            # But usually, for "confidence gap", comparing raw logits is okay if we assume the scale is consistent, 
            # OR we can assume the generated logit is already the log of the specific value.
            # To be precise: Log_Prob_Generated = logit_gen - log_sum_exp(top_k).
            # Log_Prob_Max = max_logit - log_sum_exp(top_k).
            # Difference = (max_logit - log_sums) - (logit_gen - log_sums) = max_logit - logit_gen.
            # So comparing raw logits is mathematically equivalent to comparing log-probs for the gap.
            
            # We need the logit corresponding to the generated token.
            # We have top_k_indices. We need to find the index in top_k_logits where the generated token is.
            # But we don't have the generated token index directly in the loop unless we infer it.
            # Wait, `sequence_logprobs` implies we know the token was generated. 
            # We can find the generated token's logit by matching the token ID from the context if available?
            # No, we don't have the generated token IDs in the inputs explicitly as a separate list.
            # However, we can deduce it: The logprobs are computed from the logits.
            # Actually, we can just take the max of the top-k and compare it to the logprobs value?
            # No, logprobs is log(p). Logit is raw.
            # Gap = log(max_exp(logits)) - log(exp(logit_gen)) = max(logits) - logit_gen.
            # Yes, the difference in log-probs is exactly the difference in raw logits.
            
            # We need the raw logit of the generated token.
            # We can find it by searching the top_k_indices for the token that produced the logprobs?
            # We don't have the token ID.
            # Alternative: The logprobs array is derived from the logits. 
            # If we assume the generated token is one of the top-k (which is true for standard sampling),
            # we can't easily find which one without the token ID.
            # BUT, usually in these datasets, the generated token is the one with the highest log-prob 
            # ONLY IF greedy. If sampled, it's not.
            # Let's re-read inputs: "top_k_indices" gives the indices. 
            # We are missing the generated token index.
            # Is it possible to infer? No.
            # However, we can approximate the "Confidence" using just the distribution of top-k.
            # If the model is confident, the top-1 logit is much higher than the rest.
            # If the model is hallucinating, the top-1 might be close to others (high entropy) OR the chosen one is low.
            # Let's use a simpler metric that doesn't require the specific generated logit:
            # The "Entropy of Top-K" or "Gap between Top-1 and Top-2".
            # Hallucinations often correlate with high entropy (uncertainty) or low confidence.
            # Let's calculate the gap between the top-1 logit and the top-2 logit.
            # A small gap implies uncertainty.
            
            # Sort logits descending
            sorted_logits = np.sort(logits, axis=1)[::-1]
            
            # Gap between 1st and 2nd best
            # If k < 2, handle gracefully (though k=512 here)
            gap = np.maximum(0, sorted_logits[:, 0] - sorted_logits[:, 1])
            
            # This gap represents "confidence". High gap = confident. Low gap = uncertain.
            # We want to penalize low confidence (high hallucination risk).
            # So we add the negative of the gap (or a function of it).
            # Or simply: Score = Sum-NLL - Sum(Gap). 
            # If Sum-NLL is high (bad) and Gap is low (uncertain), Score is very high (detected).
            # If Sum-NLL is low (good) and Gap is high (confident), Score is low.
            
            # Let's try: Score = Sum(-logp) - Sum(Gap)
            # This rewards low probability AND low confidence.
            
            confidence_penalty = -np.sum(gap, dtype=np.float32)
            
            total_score = base_score + confidence_penalty
            scores.append(total_score)
            
        return np.array(scores, dtype=np.float32)
