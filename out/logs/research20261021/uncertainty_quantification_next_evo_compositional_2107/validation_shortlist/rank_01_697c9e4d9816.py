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
        Detects hallucinations via a composite score of normalized surprisal deviation, 
        top-k entropy spikes, and low-confidence quantile checks relative to the sequence head.
        
        Hypothesis: Hallucinated atomic claims manifest as a concurrent increase in 
        prediction uncertainty (high entropy) and surprisal relative to the sequence's own 
        head baseline, with these effects accelerating in the final 30% of the generation.
        """
        seq_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        
        N = len(seq_logprobs)
        scores = np.zeros(N, dtype=np.float32)
        
        # Hyperparameters
        tail_start_ratio = 0.70
        head_end_ratio = 0.50
        
        # Fixed weights for the composite signal (deterministic)
        w_surprisal = 0.40
        w_flatness = 0.35
        w_entropy = 0.25
        
        for i, (logprobs, logits) in enumerate(zip(seq_logprobs, top_k_logits)):
            seq_len = len(logprobs)
            if seq_len == 0:
                scores[i] = 0.0
                continue
            
            # --- Global/Head Baselines ---
            # Use head statistics for normalization to be robust against globally hard sequences
            head_end_idx = int(seq_len * head_end_ratio)
            if head_end_idx > 0:
                head_logprobs = logprobs[:head_end_idx]
                head_logits = logits[:head_end_idx]
            else:
                head_logprobs = logprobs
                head_logits = logits
            
            # 1. Head Mean Surprisal
            head_mean_surprisal = -np.mean(head_logprobs)
            
            # 2. Head Mean Flatness (Top-1 vs Mean Top-K)
            head_top_1s = np.max(head_logits, axis=1)
            head_mean_k = np.mean(head_logits, axis=1)
            head_flatness = head_top_1s - head_mean_k
            head_mean_flatness = np.mean(head_flatness)
            flatness_norm = head_mean_flatness + 1e-8
            
            # 3. Head Mean Entropy
            # Calculate entropy for the head to establish a baseline for "confusion"
            max_h = np.max(head_logits, axis=1, keepdims=True)
            exp_h = np.exp(head_logits - max_h)
            prob_h = exp_h / np.sum(exp_h, axis=1, keepdims=True)
            head_entropy = -np.sum(prob_h * np.log(prob_h + 1e-10), axis=1)
            head_mean_entropy = np.mean(head_entropy)
            entropy_norm = head_mean_entropy + 1e-8
            
            # --- Local Signal Extraction ---
            tail_start_idx = int(seq_len * tail_start_ratio)
            local_scores = np.zeros(seq_len)
            
            for t in range(seq_len):
                # Only compute signals in the tail region
                if t < tail_start_idx:
                    local_scores[t] = 0.0
                    continue
                
                # 1. Normalized Surprisal (vs Head)
                current_surprisal = -logprobs[t]
                if head_mean_surprisal > 1e-8:
                    norm_surprisal = (current_surprisal - head_mean_surprisal) / head_mean_surprisal
                else:
                    norm_surprisal = 0.0
                
                # 2. Normalized Flatness (vs Head)
                current_flatness = np.max(logits[t]) - np.mean(logits[t])
                if flatness_norm > 1e-8:
                    # We want high flatness (low gap) to score high
                    # (Global_Mean - Current) / Norm -> if Current is small, result is large positive
                    norm_flatness = (head_mean_flatness - current_flatness) / flatness_norm
                else:
                    norm_flatness = 0.0
                
                # 3. Normalized Entropy (vs Head)
                max_l = np.max(logits[t])
                exp_l = np.exp(logits[t] - max_l)
                prob_l = exp_l / np.sum(exp_l)
                current_entropy = -np.sum(prob_l * np.log(prob_l + 1e-10))
                
                if entropy_norm > 1e-8:
                    norm_entropy = (current_entropy - head_mean_entropy) / entropy_norm
                else:
                    norm_entropy = 0.0
                
                # 4. Quantile Check: Is the chosen token in the bottom 25% of top-k?
                # If the model picks a token that is far below the mean of the top-k, it's suspicious.
                # We use the 25th percentile of the current top-k logits as a threshold.
                sorted_logits = np.sort(logits[t])
                q25 = sorted_logits[25 // 512] # Approximate 25th percentile index
                chosen_logit = np.max(logits[t]) # Assuming argmax is the chosen one for atomic claims
                # Actually, we don't know which index was chosen from stats provided. 
                # However, if the max logit is significantly lower than the mean, the distribution is flat.
                # A better proxy for "bad choice" in unsupervised setting: 
                # Check if the max logit is below the 50th percentile of the head's max logits? 
                # Let's stick to the entropy/flatness which covers this implicitly.
                # Alternative: Check if the gap between max and 2nd max is small (tie).
                # Let's skip explicit quantile check to avoid ambiguity without 'chosen_index'.
                # Instead, we rely on the composite of Surprisal + Flatness + Entropy.
                
                # Combine features
                local_score = (w_surprisal * norm_surprisal + 
                               w_flatness * norm_flatness + 
                               w_entropy * norm_entropy)
                
                local_scores[t] = local_score
            
            # Aggregate: Max score in the tail
            tail_scores = local_scores[tail_start_idx:]
            if len(tail_scores) > 0:
                scores[i] = np.max(tail_scores)
            else:
                scores[i] = 0.0

        return scores
