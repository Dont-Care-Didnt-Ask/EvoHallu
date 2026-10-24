import numpy as np
from scipy.special import logsumexp
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
        self.k = 512

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # Baseline: Total Negative Log Prob (Sum of -log_p)
        # High values indicate low overall confidence in the generated path.
        total_neg_logprob = np.array([-np.sum(seq) for seq in stats["sequence_logprobs"]], dtype=np.float32)

        n_samples = len(stats["sequence_logprobs"])
        scores = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            seq_logprobs = stats["sequence_logprobs"][i]
            top_k_logits = stats["top_k_logits"][i]
            top_k_indices = stats["top_k_indices"][i]
            
            n_tokens = len(seq_logprobs) + 1 # +1 because logprobs excludes first token
            
            if n_tokens < 3:
                scores[i] = total_neg_logprob[i]
                continue

            # Hypothesis: Hallucinations cause sharp discontinuities in the "predictive confidence" 
            # derived from the previous token's top-k distribution.
            # We compute a "Transition Confidence" sequence and its second derivative.
            
            # Pre-allocate arrays for efficiency
            transition_confidence = np.zeros(n_tokens, dtype=np.float32)
            
            # Compute transition confidence for each token starting from index 1
            # token_log_probs[t] comes from top_k_logits[t-1]
            # We need the log-prob of the actual token t under the distribution of t-1
            
            for t in range(1, n_tokens):
                # Get logits of the previous token (t-1)
                prev_logits = top_k_logits[t-1]
                
                # Convert to log-probs (stable)
                max_logit = np.max(prev_logits)
                if max_logit == -np.inf:
                    transition_confidence[t] = -np.inf
                    continue
                
                log_sum_exp = max_logit + np.log(np.sum(np.exp(prev_logits - max_logit)))
                log_probs_prev = prev_logits - log_sum_exp
                
                # The actual token index at position t
                # Note: top_k_indices[t-1] gives indices relative to the vocab.
                # We need to find the log-prob of this specific index in log_probs_prev.
                # However, top_k_indices might not contain the actual token if it wasn't in top-k.
                # If it's not in top-k, the log-prob is effectively -inf (or very low), indicating a huge jump.
                
                actual_idx = top_k_indices[t-1, :] # This is a list of k indices
                # We need to find the log-prob of the true token t.
                # But wait, we don't have the true token index explicitly in the loop variables directly 
                # unless we infer it. 
                # Actually, we can reconstruct it: 
                # The sequence_logprobs[t-1] corresponds to the log-prob of token at index t.
                # But we need the index to look it up in top_k_logits[t-1].
                # Let's assume the standard format: top_k_indices[t-1] contains the indices of the top k tokens.
                # We need to find which of these k indices matches the actual token.
                # If the actual token is NOT in top_k_indices[t-1], then the model's top-k prediction 
                # did not include the correct token, implying a massive drop in probability.
                
                # To do this efficiently without a full vocab lookup (which we don't have):
                # We can check if the actual token is in the top_k_indices.
                # But we don't have the actual token index directly. 
                # Wait, we can derive the actual token index from the fact that it generated the logprob in sequence_logprobs.
                # No, we don't have the index. 
                # Alternative: The problem statement says "top_k_indices" are provided. 
                # Usually, the actual token is one of the top-k if the model is confident.
                # If the actual token is NOT in top_k_indices, the log-prob is effectively 0 (or very low).
                
                # Let's assume the actual token index is available or we can infer it.
                # Since we don't have the actual token index in the input list directly, 
                # we must rely on the property: 
                # If the model picked a token, it was the argmax of the full softmax.
                # If that argmax is not in the top-k, the top-k log-prob for that token is -inf.
                # If it IS in the top-k, we find its log-prob.
                
                # Since we don't have the actual index, we can't perfectly compute this.
                # HOWEVER, we can approximate the "Transition Confidence" using the gap between 
                # the top-1 log-prob of the previous step and the average of the rest, 
                # and compare that to the observed log-prob trend.
                
                # Revised Strategy: Use the "Discrimination Gap" of the previous step.
                # A high discrimination gap (top-1 >> others) followed by a low observed log-prob 
                # suggests the model was confident but failed to pick the right thing (Hallucination).
                # But we need the observed log-prob of the *chosen* token.
                # We have `seq_logprobs[t-1]`. This is the log-prob of the token at `t`.
                # We need to know if this token was in the top-k of `t-1`.
                # If we assume the provided `top_k_indices` are correct, we can't verify without the index.
                
                # Let's pivot to a robust metric using only available data:
                # 1. Compute the "Entropy" of the top-k distribution at each step.
                # 2. Compute the "Sharpness" (Top-1 LogProb - Mean of Rest).
                # 3. Compute the Second Derivative of the "Sharpness" sequence.
                # Hypothesis: Hallucinations often occur where the model's internal confidence (Sharpness) 
                # fluctuates wildly or drops suddenly, indicating loss of coherence.
                
                pass 
            
            # Let's implement the "Sharpness Second Derivative" approach which is robust and uses all inputs.
            # Sharpness at step t (for predicting t+1):
            # We use top_k_logits[t] to predict t+1.
            # But we only have logprobs for t.
            # Let's align: 
            # We have logprobs for tokens 1..N-1 (indices in seq_logprobs).
            # We have top_k_logits for tokens 0..N-1.
            # For each token t (where we have logits), we can compute the sharpness of its prediction for t+1.
            # Then we look at the sequence of these sharpness values.
            
            # Let's define Sharpness[t] = log_prob(top_1 of t) - mean(log_prob(others in top_k of t))
            # This is computed for t = 0 to N-2 (since we need t+1).
            
            sharpness = np.zeros(n_tokens - 1, dtype=np.float32)
            
            for t in range(n_tokens - 1):
                logits = top_k_logits[t]
                max_logit = np.max(logits)
                if max_logit == -np.inf:
                    sharpness[t] = 0.0
                    continue
                
                log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
                log_probs = logits - log_sum_exp
                
                # Top 1 log prob
                top1_logp = max_logit - log_sum_exp
                
                # Mean of others
                others_logp = log_probs[1:]
                mean_others = np.mean(others_logp) if len(others_logp) > 0 else top1_logp
                
                sharpness[t] = top1_logp - mean_others
            
            # Now compute the second derivative of the sharpness sequence
            # d2/dt2 = s[t] - 2*s[t-1] + s[t-2]
            # We need at least 3 points.
            
            if len(sharpness) >= 3:
                # Compute first differences
                d1 = np.diff(sharpness)
                # Compute second differences
                d2 = np.diff(d1)
                
                # The magnitude of the second derivative indicates instability in confidence
                # High magnitude = sudden change in how "sure" the model is about the next token
                instability_score = np.max(np.abs(d2))
            else:
                instability_score = 0.0
            
            # Normalize instability by sequence length
            norm_instability = instability_score / (n_tokens + 1e-8)
            
            # Combine: High Total NLL OR High Instability in Confidence Dynamics
            # We scale instability to be comparable to NLL (NLL is usually ~0.5-2.0, instability ~0-5)
            # Factor 10 is chosen to give weight to dynamic changes.
            final_score = total_neg_logprob[i] + (norm_instability * 10.0)
            
            scores[i] = final_score

        return scores
