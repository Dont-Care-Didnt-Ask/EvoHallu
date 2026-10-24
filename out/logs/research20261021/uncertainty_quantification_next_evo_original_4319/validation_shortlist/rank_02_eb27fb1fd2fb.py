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
        Detects hallucinations by measuring the stability of token probability over time.
        
        Approach:
        1. Baseline: Computes the negative Pearson correlation between token position and 
           log-probability (LogProbSlope). A strong negative correlation indicates the model 
           is becoming increasingly uncertain as it generates, a hallmark of drifting into 
           hallucination.
        2. Novel Signal: Applies an Exponential Weighted Average (EWA) to the log-probability 
           sequence before calculating the slope. This smooths out local fluctuations and 
           emphasizes sustained trends in confidence decay.
        3. Combination: The final score is the negative correlation of the smoothed sequence. 
           Higher scores (more negative correlation) indicate higher hallucination likelihood.
        
        Hypothesis for future: Combining this smoothed slope with the variance of the 
        exponential weights themselves could distinguish between 'early confusion' (high variance 
        in weights) and 'late hallucination' (stable but low weights).
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)

        for i, logprobs in enumerate(stats["sequence_logprobs"]):
            if len(logprobs) == 0:
                scores[i] = 0.0
                continue
            
            seq_len = len(logprobs)
            positions = np.arange(seq_len, dtype=np.float32)
            
            # 1. Baseline: LogProbSlope (Negative Pearson Correlation)
            # We use negative correlation because we expect log-probs to drop (negative slope)
            # in hallucinations. Pearson(r) will be negative; we want the magnitude of this drop.
            # Note: pearsonr returns (correlation, p-value). We take the correlation coefficient.
            try:
                corr, _ = pearsonr(positions, logprobs)
                baseline_score = -corr # Convert to positive score for "high hallucination"
            except:
                baseline_score = 0.0
            
            # 2. Novel Signal: Smoothed LogProbSlope via EWA
            # Alpha determines the decay rate. 0.95 gives a 20-token lookback window roughly.
            alpha = 0.95
            ewa_logprobs = np.zeros_like(logprobs, dtype=np.float32)
            ewa_logprobs[0] = logprobs[0]
            
            for t in range(1, len(logprobs)):
                ewa_logprobs[t] = alpha * logprobs[t] + (1 - alpha) * ewa_logprobs[t-1]
            
            # Calculate slope on the smoothed sequence
            try:
                corr_smooth, _ = pearsonr(positions, ewa_logprobs)
                smoothed_score = -corr_smooth
            except:
                smoothed_score = 0.0
            
            # 3. Combination
            # Weight the smoothed slope higher as it is more robust to noise
            final_score = 0.6 * baseline_score + 0.4 * smoothed_score
            
            scores[i] = final_score

        return scores
