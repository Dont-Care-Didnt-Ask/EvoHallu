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
        Detects hallucinations using LogProbSlope and Longest Low-Confidence Streak.
        
        Approach:
        1. Computes the negative Pearson correlation (LogProbSlope) between token position and log-probability.
        2. Identifies the longest consecutive run of tokens with log-probabilities below the sequence median.
        3. Combines the slope score with the normalized length of this longest streak.
        
        Specific Hypothesis for Future:
        Hallucinations are characterized by extended periods of low-confidence generation (streaks) 
        rather than isolated low-probability tokens, and these streaks are more predictive than 
        the global linear trend alone when combined with the slope.
        """
        scores = np.zeros(len(stats["sequence_logprobs"]), dtype=np.float32)
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            logprobs = np.array(seq_logprobs, dtype=np.float32)
            n = len(logprobs)
            
            # Signal 1: LogProbSlope (Negative Correlation)
            positions = np.arange(n, dtype=np.float32)
            try:
                corr, _ = pearsonr(positions, logprobs)
                slope_score = -corr  # Positive score if correlation is negative (confidence drops)
            except:
                slope_score = 0.0
            
            # Signal 2: Longest Consecutive Low-Confidence Run
            # Threshold: Median log-prob of the current sequence
            median_lp = np.median(logprobs)
            
            # Binary mask: 1 if token is below median (low confidence), 0 otherwise
            # Note: logprobs are usually negative, so "low" means more negative (smaller value)
            is_low_conf = logprobs < median_lp
            
            # Find lengths of consecutive runs of True values
            run_lengths = []
            current_run = 0
            for val in is_low_conf:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        run_lengths.append(current_run)
                    current_run = 0
            if current_run > 0:
                run_lengths.append(current_run)
            
            # Longest run length
            longest_run = max(run_lengths) if run_lengths else 0
            
            # Normalize by sequence length to handle varying lengths
            streak_score = longest_run / n
            
            # Combine signals
            # Weight slope heavily (0.7) and streak moderately (0.3)
            final_score = 0.7 * slope_score + 0.3 * streak_score
            
            scores[i] = final_score

        return scores
