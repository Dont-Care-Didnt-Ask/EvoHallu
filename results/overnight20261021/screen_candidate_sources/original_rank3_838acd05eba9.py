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
        Detects hallucinations by combining total negative log-probability with 
        the average normalized rank of generated tokens within the bottom 20% of 
        the model's top-k candidates, augmented by the standard deviation of these 
        ranks to capture selection instability.
        
        Approach:
        1. Calculate total negative log-probability (sum of -log(p)) as a baseline.
        2. For each token in the sequence:
           a. Identify the generated token's index within the model's top-k list.
           b. Normalize this index by k (512) to get a rank in [0, 1].
           c. Collect these normalized ranks.
        3. Aggregate signals from the bottom 20% of these normalized ranks:
           a. Mean Rank (Bottom): High values indicate the model consistently selects 
              tokens far down its own probability list (a strong hallucination signal).
           b. Std Rank (Bottom): High values indicate erratic selection patterns.
        4. Fuse signals: Score = Total_NLL + alpha * Mean_Bottom_Rank + beta * Std_Bottom_Rank.
        
        Hypothesis for future: Hallucinated sequences exhibit a systematically higher 
        average normalized rank when restricted to the bottom 20% of the model's 
        top-k distribution compared to factual sequences, reflecting a tendency to 
        select implausible options that are not the model's highest probability choice.
        
        Args:
            stats: Dictionary containing 'sequence_logprobs', 'top_k_logits', 'top_k_indices'.
        
        Returns:
            np.ndarray of shape [N] with fused hallucination scores.
        """
        log_probs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        top_k_indices_list = stats["top_k_indices"]
        
        if len(log_probs_list) == 0:
            return np.array([], dtype=np.float32)
        
        N = len(log_probs_list)
        K = 512
        ALPHA = 2.5
        BETA = 1.5
        
        scores = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            lp = log_probs_list[i]
            k_indices = top_k_indices_list[i]
            
            n_tokens = len(lp)
            if n_tokens == 0:
                scores[i] = 0.0
                continue
            
            # 1. Baseline: Total Negative Log Prob
            total_neg_logprob = -np.sum(lp)
            
            # 2. Compute Normalized Ranks
            # k_indices shape: (n_tokens, K)
            # The model's generated token at step t corresponds to the token that produced lp[t].
            # We need to find the column index 'c' in k_indices[t] such that k_indices[t, c] 
            # matches the generated token.
            # CRITICAL: We do not have the explicit token ID of the generated sequence in 'lp'.
            # However, in standard evaluation pipelines for this task, 'sequence_logprobs' 
            # corresponds to the logprobs of the tokens *actually generated*.
            # If the model used greedy decoding (top-1), the generated token is k_indices[t, 0].
            # If the model used sampling, the generated token is the one the model picked.
            # Without the explicit token ID, we cannot directly look up the rank in k_indices 
            # unless we assume the logprob 'lp[t]' uniquely identifies the token among the top-k.
            # 
            # Alternative Interpretation for this specific benchmark context:
            # Often, 'sequence_logprobs' is computed such that the value corresponds to the 
            # log-probability of the token at a specific rank, OR the token ID is available 
            # implicitly. 
            # Given the previous successful detector (Detector 2) used the "Gap" between 
            # generated logp and top-1 logp, it implies we can compare magnitudes.
            # But to get the "Rank" specifically, we need the position.
            # 
            # Let's assume the standard constraint: We must use the provided arrays.
            # If we cannot map the generated token ID to the top_k_indices columns, we cannot 
            # compute the exact rank.
            # HOWEVER, there is a common pattern in these datasets: The 'sequence_logprobs' 
            # provided are often the logprobs of the tokens in the top-k list, sorted? 
            # No, "of tokens actually generated".
            # 
            # Let's reconsider the "Gap" approach from Detector 2. It worked well (76.33).
            # Can we improve it by looking at the bottom 20% of the *logprobs* in the top-k list?
            # Yes.
            # Hypothesis: Hallucinations have generated tokens whose logprobs fall in the 
            # bottom 20% of the top-k logprobs at that step.
            # We can compute the rank of the generated token's logprob relative to the top-k 
            # logprobs WITHOUT knowing the token ID.
            # How? We don't know the generated token's logprob value relative to the top-k 
            # list unless we know which one it is.
            # 
            # WAIT. We have 'sequence_logprobs'. We have 'top_k_logits'.
            # We can convert 'top_k_logits' to 'top_k_logprobs'.
            # We have the logprob of the generated token: `lp[t]`.
            # We have the list of logprobs of the top-k candidates: `top_k_logprobs[t]`.
            # We can find the rank of `lp[t]` within `top_k_logprobs[t]` by finding its index 
            # in the sorted list of `top_k_logprobs[t]`.
            # Since `top_k_logprobs` are derived from logits, and `lp[t]` is the logprob of 
            # the generated token, if the generated token was indeed in the top-k, its logprob 
            # `lp[t]` must match one of the values in `top_k_logprobs[t]` (within floating point tolerance).
            # 
            # So the algorithm is:
            # 1. Convert top_k_logits to top_k_logprobs (normalized).
            # 2. For each step t, find the index `r` such that `top_k_logprobs[t][r] == lp[t]`.
            # 3. This `r` is the rank (0 is best, K-1 is worst).
            # 4. Normalize: `norm_rank = r / (K-1)`.
            # 5. Filter: Keep only `norm_rank` values that are in the bottom 20% (>= 0.8).
            # 6. Aggregate: Mean and Std of these filtered values.
            
            generated_logps = lp
            # Convert top-k logits to logprobs
            # top_k_logits shape: (n_tokens, K)
            # We need to normalize them to get logprobs relative to the sum of exp(logits)
            # But since we only care about relative order within the top-k, we can just 
            # subtract the max logit to make them comparable, then take logsumexp?
            # Actually, to compare `lp[t]` with `top_k_logprobs[t]`, we need them on the same scale.
            # `lp[t]` is the absolute logprob. `top_k_logits` are absolute logits.
            # So we convert `top_k_logits` to logprobs using the same normalization as `lp[t]`.
            # However, `lp[t]` might have been computed with a different temperature or context.
            # Assuming consistency: `lp[t]` is the logprob of the token.
            # The top-k logits are the raw logits.
            # We need to find which column in top_k_logits corresponds to the token with logprob `lp[t]`.
            # We can do this by finding the column where `top_k_logits[t, c]` is closest to `log_softmax(lp[t])`?
            # No, that's circular.
            # 
            # Let's assume the simplest case: The dataset provides `sequence_logprobs` which are 
            # the logprobs of the generated tokens. The `top_k_logits` are the logits of the 
            # top-k candidates at that step.
            # The generated token MUST be one of the top-k candidates (otherwise the logprob 
            # wouldn't be reported in this context usually, or it would be extremely low).
            # We can find the column `c` by matching the token ID? We don't have the ID.
            # 
            # Is there a property of `lp[t]` that matches `top_k_logits`?
            # If we assume the model's internal state is consistent, the logprob `lp[t]` corresponds 
            # to `log_softmax(top_k_logits[t])` for the specific column.
            # But we don't know which column.
            # 
            # Let's try a different angle: 
            # Maybe the "Gap" approach (Detector 2) is the most robust given the inputs.
            # But the prompt asks for a novel detector based on "bottom 20%".
            # How can we define "bottom 20%" without the rank?
            # We can define it as: "The logprob of the generated token is in the bottom 20% of the 
            # top-k logprobs".
            # To do this, we need to convert `top_k_logits` to logprobs.
            # `logp = logits - logsumexp(logits)`.
            # We can compute `top_k_logprobs` for each step.
            # Then we have a list of `top_k_logprobs` (size K) and a scalar `lp[t]`.
            # We find the rank of `lp[t]` in `top_k_logprobs`.
            # This requires `lp[t]` to be exactly equal to one of the values in `top_k_logprobs`.
            # Due to floating point precision, we might need a tolerance.
            # 
            # Let's implement this matching logic.
            
            gaps = []
            valid_ranks = []
            
            for t_idx in range(n_tokens):
                gen_logp = lp[t_idx]
                
                # Get top-k logits
                logits = top_k_logits_list[i][t_idx]
                
                # Convert logits to logprobs
                # Shift to avoid overflow
                max_logit = np.max(logits)
                shifted_logits = logits - max_logit
                log_sum_exp = max_logit + np.log(np.sum(np.exp(shifted_logits)))
                log_probs_top_k = logits - log_sum_exp
                
                # Find the rank of the generated token in the top-k list
                # We look for the index where log_probs_top_k matches gen_logp
                # Use a small tolerance
                diff = np.abs(log_probs_top_k - gen_logp)
                # Find the index of the minimum difference
                # There should be exactly one match (or very close)
                min_diff_idx = np.argmin(diff)
                
                # Rank is the index in the sorted list of top-k logprobs?
                # No, `log_probs_top_k` is NOT sorted by value. It's sorted by the order of 
                # `top_k_logits` (which is sorted by logit value descending).
                # So `log_probs_top_k` is also sorted by logprob descending.
                # Therefore, the index `min_diff_idx` IS the rank (0 is best).
                
                rank = min_diff_idx
                normalized_rank = rank / (K - 1) if K > 1 else 0.0
                
                # Check if this rank is in the bottom 20%
                if normalized_rank >= 0.8:
                    valid_ranks.append(normalized_rank)
                
                # Also compute the gap for the baseline signal (optional, but let's stick to the plan)
                # The plan is: NLL + Mean(Bottom20%) + Std(Bottom20%)
                # We don't need the gap for the final score, just the rank.
            
            if len(valid_ranks) == 0:
                # If no tokens were in the bottom 20%, the detector shouldn't trigger heavily 
                # based on this signal, but we can add a small penalty or just use 0.
                # Let's use 0 for mean/std if no data.
                mean_bottom_rank = 0.0
                std_bottom_rank = 0.0
            else:
                mean_bottom_rank = np.mean(valid_ranks)
                std_bottom_rank = np.std(valid_ranks)
            
            # Fusion
            scores[i] = total_neg_logprob + ALPHA * mean_bottom_rank + BETA * std_bottom_rank
            
        return scores
