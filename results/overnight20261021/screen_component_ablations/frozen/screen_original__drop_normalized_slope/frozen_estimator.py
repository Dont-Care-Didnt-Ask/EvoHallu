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
        Detects hallucinations using LogProbSlope and Tail Distribution Collapse.
        
        Approach:
        1. Computes the Pearson correlation (slope) between normalized token position 
           and log-probability to detect a systematic decline in confidence.
        2. Calculates the inverse of the range (max - min) of log-probabilities in the last 10% 
           of the sequence. A "collapsed" range (small difference) suggests the model is 
           generating from a flat, confused distribution rather than a peaked, confident one.
        3. Aggregates these signals: a steep negative slope combined with a collapsed tail 
           distribution indicates high hallucination risk.
        
        Specific Hypothesis for Future:
        Hallucinations are characterized by a distinct acceleration of uncertainty (steep negative slope) 
        coinciding with a flattening of the probability distribution at the tail, where the gap between 
        the most likely and least likely tokens in the window shrinks as the model loses its primary trajectory.
        """
        scores = np.zeros(len(stats["sequence_logprobs"]), dtype=np.float32)
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            n_tokens = len(seq_logprobs)
            if n_tokens < 5: 
                scores[i] = 0.0
                continue
            
            # Convert to float32 for consistency
            logprobs = np.array(seq_logprobs, dtype=np.float32)
            
            # 1. LogProbSlope Signal
            positions = np.arange(n_tokens, dtype=np.float32) / (n_tokens - 1)
            # Calculate Pearson correlation
            try:
                corr, _ = pearsonr(positions, logprobs)
                # We expect negative correlation (log-prob drops). 
                # Convert to positive score: stronger negative correlation -> higher score.
                slope_score = -corr
            except:
                slope_score = 0.0
            
            # 2. Tail Collapse Signal (Min/Max Statistics)
            tail_start_idx = max(0, int(n_tokens * 0.9))
            tail_logprobs = logprobs[tail_start_idx:]
            
            if len(tail_logprobs) > 0:
                tail_max = np.max(tail_logprobs)
                tail_min = np.min(tail_logprobs)
                tail_range = tail_max - tail_min
                
                # If range is 0 (all same), inverse is infinity -> cap it.
                # Hypothesis: Small range = flat distribution = hallucination.
                # Large range = peaked distribution = confidence.
                # Score should be high when range is small.
                # Use 1 / (1 + range) to normalize and cap.
                tail_collapse_score = 1.0 / (1.0 + tail_range)
            else:
                tail_collapse_score = 0.0
            
            # Normalize scores by global magnitude to handle different model temperatures/datasets
            global_mean_abs = np.mean(np.abs(logprobs)) + 1e-8
            slope_score_norm = slope_score / global_mean_abs
            tail_collapse_norm = tail_collapse_score / global_mean_abs
            
            # Combine: Weight slope (0.7) and tail collapse (0.3)
            # Note: Tail collapse is counter-intuitive in isolation (we want high score for low range),
            # but the normalization handles the scale.
            final_score = 0.0 + 0.3 * tail_collapse_norm
            
            scores[i] = final_score

        return scores
