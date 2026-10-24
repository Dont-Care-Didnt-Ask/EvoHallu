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
        logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        
        n_samples = len(logprobs_list)
        results = np.zeros(n_samples, dtype=np.float32)
        
        # Deterministic weights for the composite score
        w_global = 0.25
        w_tail = 0.25
        w_gap = 0.25
        w_local = 0.25
        
        for i in range(n_samples):
            lp = logprobs_list[i]
            k_logits = top_k_logits_list[i]
            
            # seq_len excludes the first generated token per input spec
            seq_len = len(lp)
            
            if seq_len == 0:
                results[i] = 0.0
                continue
            
            # --- Feature 1: Global Surprise (Mean NLL) ---
            # Hallucinations increase overall uncertainty.
            global_score = -np.mean(lp)
            
            # --- Feature 2: Tail Instability (Variance of last 20%) ---
            # Coherence often degrades at the end of generated sequences.
            tail_start = max(0, int(seq_len * 0.8))
            tail_lp = lp[tail_start:]
            if len(tail_lp) > 0:
                tail_score = np.var(tail_lp)
            else:
                tail_score = 0.0
            
            # --- Feature 3: Top-K Ambiguity (Normalized Gap) ---
            # Small gaps between top-1 and top-2 logits indicate low confidence/ambiguity.
            if k_logits.shape[1] >= 2:
                # Sort descending: axis 1, then reverse
                sorted_logits = np.sort(k_logits, axis=1)[:, ::-1]
                gaps = sorted_logits[:, 0] - sorted_logits[:, 1]
                
                # Normalize by the magnitude of the top logit to handle scale
                top_vals = sorted_logits[:, 0]
                # Use absolute value to avoid division by zero if top logit is negative
                scale = np.abs(top_vals) + 1e-9
                normalized_gaps = np.clip(gaps / scale, 0.0, 1.0)
                
                # Invert: High gap_score (low confidence) -> High hallucination risk
                gap_score = 1.0 - np.mean(normalized_gaps)
            else:
                gap_score = 0.0
            
            # --- Feature 4: Local Consistency (Mean Jump in last 30%) ---
            # Hallucinations often cause erratic swings in confidence.
            local_start = max(0, int(seq_len * 0.7))
            local_lp = lp[local_start:]
            if len(local_lp) > 1:
                # Calculate absolute differences between adjacent tokens
                jumps = np.abs(np.diff(local_lp))
                local_score = np.mean(jumps)
            else:
                local_score = 0.0
            
            # --- Aggregation ---
            # Weighted sum of normalized components
            score = (w_global * global_score + 
                     w_tail * tail_score + 
                     w_gap * gap_score + 
                     w_local * local_score)
            
            results[i] = score

        return results
