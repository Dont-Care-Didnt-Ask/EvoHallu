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
        Detects hallucinations using a composite of Local Variance, Tail Mean, and Global Slope.
        
        Approach:
        1. Computes the variance of negative log-probabilities in the final 10% of the sequence 
           to capture local instability/unpredictability.
        2. Computes the mean negative log-probability of the final 20% of the sequence to capture 
           tail confidence degradation.
        3. Computes the Pearson correlation between token position and log-probability to capture 
           global confidence decay trends.
        4. Aggregates these three signals using fixed weights and deterministic normalization factors.
        
        Specific Hypothesis for Future:
        Hallucinations are characterized by a triad of symptoms: high local instability (variance) 
        at the end of the generation, a significant drop in average confidence in the tail segment, 
        and a negative correlation between position and confidence; combining these orthogonal 
        signals yields a more robust detector than any single metric alone.
        """
        scores = np.zeros(len(stats["sequence_logprobs"]), dtype=np.float32)
        
        # Deterministic normalization constants (approximate scale factors for typical LLM outputs)
        # These are fixed to ensure determinism without global dataset statistics.
        VAR_SCALE = 1.0 
        MEAN_SCALE = 1.0
        SLOPE_SCALE = 10.0  # Correlation is [-1, 1], needs scaling to match magnitude of other terms
        
        # Weights for the composite score
        W_VAR = 0.4
        W_MEAN = 0.3
        W_SLOPE = 0.3
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            logprobs = np.array(seq_logprobs, dtype=np.float32)
            positions = np.arange(len(logprobs), dtype=np.float32)
            n = len(logprobs)
            
            # Convert to negative log-prob (higher = worse)
            neg_logprobs = -logprobs
            
            # 1. Local Variance (Last 10%)
            start_idx_var = max(0, int(n * 0.9))
            if start_idx_var < n:
                window_var = np.var(neg_logprobs[start_idx_var:])
                score_var = window_var / VAR_SCALE
            else:
                score_var = 0.0
            
            # 2. Tail Mean (Last 20%)
            start_idx_mean = max(0, int(n * 0.8))
            if start_idx_mean < n:
                tail_mean = np.mean(neg_logprobs[start_idx_mean:])
                score_mean = tail_mean / MEAN_SCALE
            else:
                score_mean = 0.0
            
            # 3. Global Slope (Pearson Correlation)
            # We want to penalize negative correlation (confidence dropping with position).
            # So we take -correlation.
            try:
                corr, _ = pearsonr(positions, logprobs)
                # If logprobs drop, corr is negative. -corr is positive (bad).
                score_slope = -corr * SLOPE_SCALE
            except:
                score_slope = 0.0
            
            # Aggregate
            final_score = (W_VAR * score_var) + (W_MEAN * score_mean) + (W_SLOPE * score_slope)
            scores[i] = final_score

        return scores
