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
        Detects hallucinations using LogProbSlope and Top-1/Top-K Divergence.
        
        Approach:
        1. Computes the LogProbSlope (negative Pearson correlation) to detect systematic confidence decay.
        2. Computes the "Confidence Divergence": the difference between the mean of the top-k logits 
           and the maximum (top-1) logit. A growing gap suggests the model is spreading probability 
           mass across unlikely options, indicating a loss of specific focus (hallucination).
        3. Aggregates signals by taking the minimum of the absolute slope and the maximum divergence 
           in the tail, normalized by sequence length.
        
        Specific Hypothesis for Future:
        Hallucinations are characterized by a decoupling of the model's peak confidence from its 
        average confidence over the top-k candidates, where the model increasingly relies on a 
        diffuse set of low-probability tokens rather than a single high-probability path.
        """
        scores = np.zeros(len(stats["sequence_logprobs"]), dtype=np.float32)
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            n_tokens = len(seq_logprobs)
            if n_tokens < 3:
                scores[i] = 0.0
                continue
            
            # --- Signal 1: LogProbSlope ---
            # Negative correlation between position and log-prob (higher score = steeper decline)
            positions = np.arange(n_tokens) / (n_tokens - 1)
            # Use raw logprobs; correlation handles direction
            corr = np.corrcoef(positions, seq_logprobs)[0, 1]
            slope_score = -corr if not np.isnan(corr) else 0.0
            
            # --- Signal 2: Top-1 / Top-K Divergence ---
            # Extract top-k logits
            top_k_logits = stats["top_k_logits"][i] # Shape (k,)
            k = top_k_logits.shape[0]
            
            # Top-1 is the max
            top1_val = np.max(top_k_logits)
            # Top-K mean
            topk_mean = np.mean(top_k_logits)
            
            # Divergence: How much the top-k average exceeds the top-1?
            # In a coherent sequence, top1 >> topk_mean (large negative gap or small positive gap if logits are negative)
            # Actually, logits can be negative. Let's look at the gap: top1 - topk_mean.
            # If top1 is 2.0 and topk_mean is 1.0, gap is 1.0 (confident).
            # If top1 is 0.0 and topk_mean is 0.0, gap is 0.0 (uncertain).
            # We want to penalize small gaps (high uncertainty spread).
            # However, the instruction says "min/max statistics (most likely with low weight to avoid outlier dependence)".
            # Let's define a "Divergence Score" as the ratio of the gap to the magnitude of the top-1.
            # Or simpler: The difference (top1 - topk_mean). If this is small, the model is unsure.
            # Let's invert it: Score = 1.0 / (1.0 + (top1 - topk_mean)) ? 
            # Better: Just use the raw difference, but normalized.
            
            gap = top1_val - topk_mean
            
            # Normalize gap by the absolute value of the top1 to make it scale-invariant
            # If top1 is very negative (low prob), the gap might be small naturally.
            # We want high score when gap is small (confusion).
            # Let's use: score = 1.0 / (1.0 + abs(gap)) 
            # But wait, we want to combine with slope.
            # Let's stick to the prompt's "min/max" hint for aggregation logic.
            
            # Let's compute the "Uncertainty Index": Small Gap = High Uncertainty.
            # Normalize gap by the typical logit range or just use raw if distributions align.
            # Let's use the raw gap, but clamp it.
            # Actually, let's look at the ratio of Top-K mean to Top-1.
            # If Top-1 is -10 and Top-K mean is -10, ratio is 1.0 (bad).
            # If Top-1 is -10 and Top-K mean is -12, ratio is 1.2 (ok).
            # Let's use: 1.0 / (1.0 + gap) where gap = top1 - topk_mean.
            # If gap is large (confident), score is low. If gap is 0, score is 1.0.
            uncertainty_idx = 1.0 / (1.0 + gap)
            
            # --- Aggregation ---
            # Combine Slope (trend) and Uncertainty (state).
            # We want high score if slope is very negative (steep drop) OR uncertainty is high.
            # Let's use a weighted sum of the normalized slope and the uncertainty index.
            
            # Normalize slope to [0, 1] roughly. Correlation is [-1, 1]. 
            # -1 (perfect drop) -> 1.0 score. 0 -> 0.5.
            norm_slope = (slope_score + 1.0) / 2.0
            
            # Combine
            # Weight slope 0.5, uncertainty 0.5
            final_score = 0.5 * norm_slope + 0.5 * uncertainty_idx
            
            scores[i] = final_score

        return scores
