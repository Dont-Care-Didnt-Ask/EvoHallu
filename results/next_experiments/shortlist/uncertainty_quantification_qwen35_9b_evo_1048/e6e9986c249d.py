import numpy as np
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
        Detects hallucinations using LogProbSlope and Tail Extremes.
        
        Approach:
        1. Computes the Pearson correlation (slope) between normalized token position 
           and log-probability to detect a systematic decline in confidence.
        2. Calculates the range (max - min) of log-probabilities in the last 10% of the sequence 
           to capture erratic behavior or sustained low confidence at the tail.
        3. Aggregates these signals: higher negative slope (more negative correlation) 
           and larger tail range indicate higher hallucination risk.
        
        Specific Hypothesis for Future:
        Hallucinations are characterized not just by high average uncertainty, but by a 
        distinct acceleration of uncertainty (steep negative slope) coinciding with 
        increased variance in the probability distribution of the final generated tokens.
        """
        scores = np.zeros(len(stats["sequence_logprobs"]), dtype=np.float32)
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            n_tokens = len(seq_logprobs)
            if n_tokens < 5: # Need enough tokens for meaningful slope and tail stats
                scores[i] = 0.0
                continue
            
            # Convert to negative log-prob for intuitive "uncertainty" scale (higher = worse)
            neg_logprobs = -np.array(seq_logprobs, dtype=np.float32)
            
            # 1. LogProbSlope Signal
            # Correlation between normalized position (0 to 1) and log-prob
            positions = np.arange(n_tokens) / (n_tokens - 1)
            # Pearson correlation
            corr = np.corrcoef(positions, seq_logprobs)[0, 1]
            # We expect a negative correlation (log-prob drops as position increases).
            # Convert to a positive "risk" score: stronger negative correlation -> higher score.
            slope_score = -corr
            
            # 2. Tail Extremes Signal
            # Focus on the last 10% of the sequence
            tail_start_idx = max(0, int(n_tokens * 0.9))
            tail_logprobs = seq_logprobs[tail_start_idx:]
            
            if len(tail_logprobs) > 0:
                tail_min = np.min(tail_logprobs)
                tail_max = np.max(tail_logprobs)
                # Range indicates instability or consistently low confidence
                tail_range = tail_max - tail_min
                # Normalize tail_range by the max possible range for this sequence length 
                # (approx log(Vocab) ~ 512 -> 6-7 bits, but we don't know vocab size exactly, 
                # so we rely on relative magnitude or just use raw if distributions are similar).
                # Alternatively, normalize by the mean absolute log-prob to make it scale-invariant.
                mean_abs_logprob = np.mean(np.abs(seq_logprobs)) + 1e-8
                normalized_tail_range = tail_range / mean_abs_logprob
            else:
                normalized_tail_range = 0.0
            
            # Combine signals
            # Weight slope (0.6) and tail range (0.4)
            final_score = 0.6 * slope_score + 0.4 * normalized_tail_range
            
            scores[i] = final_score

        return scores
