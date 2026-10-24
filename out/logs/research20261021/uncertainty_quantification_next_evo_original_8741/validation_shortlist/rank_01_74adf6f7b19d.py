import numpy as np
from scipy.stats import pearsonr
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detects hallucinations by combining the negative correlation of log-probability 
        with position (slope) and the local variance of log-probabilities.
        
        Approach:
        1. For each sequence, compute the Pearson correlation between token index and log-prob.
        2. Identify the segment with the steepest negative slope (largest drop in log-prob).
        3. Compute the standard deviation of log-probs within that specific steep segment.
        4. The score is a composite: high negative correlation (strong trend) multiplied by high local std 
           (instability), normalized by sequence length.
        
        Hypothesis for future: Hallucinated atomic claims induce a localized increase in log-probability 
        variance coinciding with a sharp negative slope, distinguishing them from gradual confidence decay 
        in difficult but truthful sequences.
        """
        scores = []
        window_size = 5  # Size of the window to check for local variance
        
        for seq_logprobs in stats["sequence_logprobs"]:
            if len(seq_logprobs) == 0:
                scores.append(0.0)
                continue
            
            seq = np.array(seq_logprobs, dtype=np.float32)
            n_tokens = len(seq)
            
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            # 1. Global Slope Signal (Pearson Correlation)
            # Higher negative correlation implies a stronger trend of decreasing confidence
            corr, _ = pearsonr(np.arange(n_tokens), seq)
            slope_signal = -corr # We want positive scores for negative correlations
            
            # 2. Local Variance Signal (Find the steepest drop and check its variance)
            # Calculate differences between consecutive log-probs
            diffs = np.diff(seq)
            
            # Find the index of the most negative difference (steepest drop)
            if len(diffs) > 0:
                idx_steepest = np.argmin(diffs)
                # Define a local window around the steepest drop
                start = max(0, idx_steepest - window_size // 2)
                end = min(n_tokens, idx_steepest + window_size // 2 + 1)
                local_window = seq[start:end]
                
                # Calculate std of the log-probs in this critical region
                local_var = np.std(local_window)
            else:
                local_var = 0.0
            
            # 3. Composite Score
            # If there is a strong negative trend (high slope_signal) AND high local variance,
            # it suggests a sudden, unstable drop in confidence -> likely hallucination.
            # Normalize local_var by a small constant to prevent vanishing gradients if values are small
            normalized_var = local_var / (np.std(seq) + 1e-8)
            
            final_score = slope_signal * (1.0 + normalized_var)
            
            scores.append(final_score)
        
        return np.array(scores, dtype=np.float32)
