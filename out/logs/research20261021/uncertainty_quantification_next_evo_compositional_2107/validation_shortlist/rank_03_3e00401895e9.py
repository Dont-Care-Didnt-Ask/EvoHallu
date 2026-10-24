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
        top-k flatness, and top-k entropy spikes in the sequence tail.
        
        Hypothesis: Hallucinated atomic claims manifest as a concurrent increase in 
        prediction uncertainty (flatness/entropy) and surprisal relative to the 
        sequence's own head baseline, with these effects accelerating in the final 
        30% of the generation.
        """
        seq_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        
        N = len(seq_logprobs)
        scores = np.zeros(N, dtype=np.float32)
        
        # Hyperparameters
        tail_start_ratio = 0.70
        head_end_ratio = 0.50
        alpha = 0.45  # Weight for Surprisal
        beta = 0.45   # Weight for Flatness
        gamma = 0.10  # Weight for Entropy
        
        for i, (logprobs, logits) in enumerate(zip(seq_logprobs, top_k_logits)):
            seq_len = len(logprobs)
            if seq_len == 0:
                scores[i] = 0.0
                continue
            
            # --- Global Baselines ---
            global_mean_surprisal = -np.mean(logprobs)
            
            top_1s = np.max(logits, axis=1)
            mean_k = np.mean(logits, axis=1)
            global_flatness = top_1s - mean_k
            global_mean_flatness = np.mean(global_flatness)
            flatness_norm = global_mean_flatness + 1e-8
            
            # Calculate Top-K Entropy for global baseline
            # Entropy = -sum(p * log(p))
            # p = softmax(logits). Approximate with softmax of top-k logits.
            # To avoid exp overflow, subtract max before softmax
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            global_entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            global_mean_entropy = np.mean(global_entropy)
            entropy_norm = global_mean_entropy + 1e-8
            
            # --- Local Signal Extraction ---
            tail_start_idx = int(seq_len * tail_start_ratio)
            head_end_idx = int(seq_len * head_end_ratio)
            
            local_scores = np.zeros(seq_len)
            
            # Pre-calculate head averages for normalization
            if head_end_idx > 0:
                head_avg_surprisal = -np.mean(logprobs[:head_end_idx])
            else:
                head_avg_surprisal = global_mean_surprisal
            
            for t in range(seq_len):
                # Only compute signals in the tail region
                if t < tail_start_idx:
                    local_scores[t] = 0.0
                    continue
                
                # 1. Normalized Surprisal Escalation
                current_surprisal = -logprobs[t]
                if global_mean_surprisal > 1e-8:
                    norm_surprisal = (current_surprisal - head_avg_surprisal) / global_mean_surprisal
                else:
                    norm_surprisal = 0.0
                
                # 2. Normalized Top-K Flatness
                current_flatness = top_1s[t] - mean_k[t]
                if flatness_norm > 1e-8:
                    # Higher flatness (lower gap) -> higher score
                    norm_flatness = (global_mean_flatness - current_flatness) / flatness_norm
                else:
                    norm_flatness = 0.0
                
                # 3. Normalized Top-K Entropy Spike
                max_logit = np.max(logits[t])
                exp_l = np.exp(logits[t] - max_logit)
                p_l = exp_l / np.sum(exp_l)
                current_entropy = -np.sum(p_l * np.log(p_l + 1e-10))
                
                if entropy_norm > 1e-8:
                    norm_entropy = (current_entropy - global_mean_entropy) / entropy_norm
                else:
                    norm_entropy = 0.0
                
                # Combine features
                local_score = alpha * norm_surprisal + beta * norm_flatness + gamma * norm_entropy
                local_scores[t] = local_score
            
            # Aggregate: Max score in the tail
            tail_scores = local_scores[tail_start_idx:]
            if len(tail_scores) > 0:
                scores[i] = np.max(tail_scores)
            else:
                scores[i] = 0.0

        return scores
