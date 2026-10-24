import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining LogProbSlope (negative correlation between position 
    and log-prob) with the mean of the bottom 20% of top-k confidence margins.
    
    Hypothesis: Hallucinated sequences show a decline in token confidence over time (negative 
    slope) and contain a significant subset of tokens where the model is ambiguous among 
    top-k candidates (low margin between top-1 and other top-k logits).
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
                "top_k_indices"
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # "top_k_logits": List[np.ndarray] of shape (n_i, k)
        # "top_k_indices": List[np.ndarray] of shape (n_i, k)
        
        scores = []
        
        for i in range(len(stats["sequence_logprobs"])):
            logprobs = stats["sequence_logprobs"][i]
            top_k_logits = stats["top_k_logits"][i]
            
            n_tokens = len(logprobs)
            
            if n_tokens == 0:
                scores.append(0.0)
                continue
            
            # 1. LogProbSlope: Negative Pearson correlation between position and log-prob
            # Positions: 0 to n_tokens - 1
            positions = np.arange(n_tokens, dtype=np.float32)
            
            if n_tokens > 1 and np.std(logprobs) > 1e-8 and np.std(positions) > 1e-8:
                # Pearson correlation
                corr = np.corrcoef(positions, logprobs)[0, 1]
                # We want "negative slope" to be a high score. 
                # If logprobs decrease over time, corr is negative.
                # So we take -corr. If corr is -0.5, score contribution is 0.5.
                slope_score = -corr
            else:
                slope_score = 0.0
            
            # 2. Top-K Confidence Margin Aggregation
            # Calculate margin for each token: top1_logit - mean(other_top_k_logits)
            # Or simply top1 - top2. Let's use top1 - mean(top_k) as a measure of "sharpness".
            # Actually, a better measure of ambiguity is the entropy of the top-k distribution 
            # or the gap between top1 and top2.
            # Let's use: margin = top1_logit - np.mean(top_k_logits)
            # If the model is confident, top1 is much higher than the rest.
            # If hallucinating/uncertain, top1 is close to the rest.
            
            # top_k_logits shape: (n_i, k)
            # Note: logprobs are for the *generated* token. 
            # We need to check if the generated token is in the top-k.
            # The problem description says "top_k_logits" and "top_k_indices" are provided.
            # It doesn't explicitly say the generated token is always in top-k, but usually it is.
            # However, "sequence_logprobs" gives the log-prob of the *actual* token.
            # To be robust, we can compute the margin based on the top-k logits directly,
            # assuming the top-k contains the most likely candidates.
            
            # Let's compute the "confidence margin" for each token position in top_k_logits.
            # margin_t = top_k_logits[t, 0] - np.mean(top_k_logits[t, :])
            # This measures how much the best candidate stands out.
            
            if top_k_logits.shape[0] == n_tokens:
                top1 = top_k_logits[:, 0]
                mean_top_k = np.mean(top_k_logits, axis=1)
                margins = top1 - mean_top_k
                
                # Aggregate bottom 20% of margins (most ambiguous tokens)
                num_bottom = max(1, int(0.2 * n_tokens))
                # Sort margins in ascending order (most ambiguous first)
                sorted_margins = np.sort(margins)
                bottom_margins = sorted_margins[:num_bottom]
                margin_score = np.mean(bottom_margins)
                
                # We want high score for hallucination.
                # Low margin (ambiguous) -> low margin_score.
                # So we invert: -margin_score or (max_margin - margin_score).
                # Let's normalize or just use -margin_score.
                # Since margins can be negative, -margin_score will be positive for low margins.
                margin_component = -margin_score
            else:
                # Fallback if shapes don't match (shouldn't happen)
                margin_component = 0.0
            
            # Combine signals
            # Slope score is in [-1, 1] range roughly.
            # Margin component depends on logit scale.
            # Let's weight them. Slope is a strong structural signal.
            score = slope_score + 0.1 * margin_component
            
            scores.append(score)
            
        return np.array(scores, dtype=np.float32)
