import numpy as np
class HallucinationDetector(Estimator):
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
        """
        Detects hallucinations by measuring the 'regret' or confidence gap 
        between the generated token and the model's most confident alternative.
        
        Approach:
        1. For each token in the sequence, identify the log-prob of the generated token.
        2. Identify the log-prob of the most likely token among the top-k candidates 
           that is NOT the generated token (if the generated token is top-1, take the top-2).
        3. Compute the gap: (Log_Prob_Generated) - (Log_Prob_Best_Alternative).
        4. Aggregate: Score = Mean(Gap) over the sequence. A positive gap indicates 
           the model chose a less probable option over a significantly more probable one.
        
        Hypothesis for future: The relationship between this 'regret' score and the 
        semantic distance (if available) or the length of the sequence might reveal 
        if hallucinations are more common in long chains of low-regret errors vs. 
        single high-regret jumps.
        """
        sequence_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        top_k_indices = stats["top_k_indices"]
        
        n_sequences = len(sequence_logprobs)
        scores = np.zeros(n_sequences, dtype=np.float32)
        
        for i in range(n_sequences):
            seq_logprobs = sequence_logprobs[i]
            seq_len = len(seq_logprobs)
            
            if seq_len == 0:
                scores[i] = 0.0
                continue
                
            # Get top-k logits and indices for this sequence
            k = top_k_logits[i].shape[0]
            logits = top_k_logits[i]
            indices = top_k_indices[i]
            
            # We need the log-prob of the generated token and the best alternative
            # Note: sequence_logprobs corresponds to the log-prob of the generated token.
            # However, top_k_logits are raw logits. We need to convert to log-probs 
            # assuming the model output is already normalized or we can compute log_softmax.
            # Since we only have logits, we compute log_softmax locally for stability.
            
            # Compute log_softmax for the top-k to get log-probs
            # Add small epsilon for numerical stability if needed, but max is usually safe
            logits_max = np.max(logits, axis=1, keepdims=True)
            logits_shifted = logits - logits_max
            exp_logits = np.exp(logits_shifted)
            sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
            log_probs_topk = logits_shifted - np.log(sum_exp)
            
            # Ensure shape matches: (seq_len, k)
            if log_probs_topk.ndim == 1:
                log_probs_topk = log_probs_topk.reshape(-1, k)
            
            generated_log_probs = np.array(seq_logprobs, dtype=np.float32)
            
            # For each token position j:
            # 1. Find which rank the generated token has in top-k.
            # 2. If it's rank 0 (index 0 in top_k_indices), the best alternative is rank 1.
            # 3. If it's rank > 0, the best alternative is rank 0.
            # 4. If it's not in top-k, we can't compute a reliable gap from top-k, 
            #    so we default to a high penalty or ignore. Let's assume it's in top-k 
            #    for atomic claims as per typical high-quality generation settings, 
            #    or fallback to generated_log_prob (which is negative) if not found.
            
            gaps = []
            
            for j in range(seq_len):
                gen_idx = indices[j, 0]
                gen_log_prob = generated_log_probs[j]
                
                # Find the log-prob of the best alternative
                # Create a mask for the current top-k candidates
                # We want the max log-prob among top-k where index != gen_idx
                
                # Check if gen_idx is in the top-k
                is_in_topk = False
                alt_log_prob = -np.inf
                
                for rank in range(k):
                    if indices[j, rank] == gen_idx:
                        is_in_topk = True
                        # If it's the top choice, the alternative is the next one
                        if rank + 1 < k:
                            alt_log_prob = log_probs_topk[j, rank + 1]
                        break
                    else:
                        # It's an alternative
                        if log_probs_topk[j, rank] > alt_log_prob:
                            alt_log_prob = log_probs_topk[j, rank]
                
                if is_in_topk:
                    # Regret: Chosen - Best_Alternative
                    # If Chosen is much lower than Best_Alternative, regret is negative (large magnitude)
                    # We want a high score for hallucination, so we invert or take negative.
                    # Let's define Score = Best_Alternative - Chosen.
                    # If Chosen is bad, Score is high.
                    gap = alt_log_prob - gen_log_prob
                else:
                    # If not in top-k, we don't have a direct comparison from top-k.
                    # However, if it's not in top-k, the model was very unsure.
                    # Let's assume a high penalty for being outside the confident set.
                    # Or simply use the generated log prob (very negative) as a proxy for low confidence.
                    # Let's use a heuristic: if not in top-k, gap is effectively infinite relative to top-k.
                    # But to keep it bounded, let's just use the fact that it's low confidence.
                    # A simple heuristic: gap = -gen_log_prob (since gen_log_prob is negative, this is positive).
                    # This aligns with the "Cumulative Uncertainty" detector but per-token.
                    gap = -gen_log_prob
                
                gaps.append(gap)
            
            if len(gaps) > 0:
                scores[i] = np.mean(gaps)
            else:
                scores[i] = 0.0
                
        return scores
