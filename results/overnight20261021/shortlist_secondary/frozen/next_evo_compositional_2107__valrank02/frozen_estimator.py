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
        Detects hallucinations by combining normalized top-k flatness deviations 
        and local surprisal escalation in the sequence tail.
        
        Hypothesis: Hallucinated atomic claims are characterized by a simultaneous 
        drop in top-k confidence (flattening of the probability distribution) and 
        a significant increase in surprisal (negative log-probability) relative to 
        the sequence's own head baseline, with these effects amplified in the latter 
        30% of the generation.
        """
        seq_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        
        N = len(seq_logprobs)
        scores = np.zeros(N, dtype=np.float32)
        
        # Hyperparameters
        tail_start_ratio = 0.70  # Start measuring tail from 70% of length
        tail_end_ratio = 1.00    # End at 100%
        pos_weight_slope = 2.0   # Linear increase in weight from 0 to slope at end
        
        for i, (logprobs, logits) in enumerate(zip(seq_logprobs, top_k_logits)):
            seq_len = len(logprobs)
            if seq_len == 0:
                scores[i] = 0.0
                continue
            
            # --- Global Baselines (Normalization Factors) ---
            # 1. Global Mean Surprisal
            global_mean_surprisal = -np.mean(logprobs)
            
            # 2. Global Mean Top-K Flatness (Gap between top-1 and mean top-k)
            # Shape: (seq_len, k) -> (seq_len,)
            top_1s = np.max(logits, axis=1)
            mean_k = np.mean(logits, axis=1)
            global_flatness = top_1s - mean_k
            global_mean_flatness = np.mean(global_flatness)
            
            # Avoid division by zero if global flatness is 0 (perfect confidence everywhere)
            flatness_norm = global_mean_flatness + 1e-8
            
            # --- Local Signal Extraction ---
            local_scores = np.zeros(seq_len)
            
            # Define the tail window for detailed analysis
            tail_start_idx = int(seq_len * tail_start_ratio)
            tail_end_idx = int(seq_len * tail_end_ratio)
            
            for t in range(seq_len):
                # Calculate Positional Weight (Higher for tail)
                pos = t / seq_len
                # Weight starts at 0.0, ends at pos_weight_slope
                weight = np.clip(pos * pos_weight_slope, 0.0, 2.0)
                
                local_score = 0.0
                
                # Only compute signals in the tail region
                if t >= tail_start_idx:
                    # Signal A: Normalized Surprisal Escalation
                    # Compare current surprisal to the head average
                    head_end_idx = int(seq_len * 0.5)
                    if head_end_idx > 0:
                        head_avg_surprisal = -np.mean(logprobs[:head_end_idx])
                        current_surprisal = -logprobs[t]
                        
                        # Normalize by global mean surprisal to be scale-invariant
                        if global_mean_surprisal > 1e-8:
                            normalized_surprisal = (current_surprisal - head_avg_surprisal) / global_mean_surprisal
                        else:
                            normalized_surprisal = 0.0
                        
                        # Signal B: Normalized Top-K Flatness
                        # How much flatter is this token's distribution compared to global average?
                        # High flatness = low (top_1 - mean_k)
                        current_flatness = top_1s[t] - mean_k[t]
                        
                        # Normalize: (Global_Mean - Current) / Global_Mean
                        # If current is flatter (smaller gap), this value becomes positive
                        if flatness_norm > 1e-8:
                            normalized_flatness = (global_mean_flatness - current_flatness) / flatness_norm
                        else:
                            normalized_flatness = 0.0
                        
                        # Combine signals with equal weight
                        local_score = 0.5 * normalized_surprisal + 0.5 * normalized_flatness
                
                local_scores[t] = local_score * weight
            
            # Aggregate: Use max score in the tail region, as a single hallucinated claim 
            # should trigger a high score even if the rest is clean.
            tail_scores = local_scores[tail_start_idx:tail_end_idx]
            
            if len(tail_scores) > 0:
                # Take the maximum deviation in the tail
                scores[i] = np.max(tail_scores)
            else:
                scores[i] = 0.0

        return scores
