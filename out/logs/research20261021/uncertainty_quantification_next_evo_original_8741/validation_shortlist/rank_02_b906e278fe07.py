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
        Detects hallucinations by combining total perplexity with a weighted sum of 
        deviations from an exponential moving average trend.
        
        Approach:
        1. Computes the baseline total negative log-probability (perplexity).
        2. Calculates an Exponential Weighted Moving Average (EWMA) of the log-probabilities 
           to establish a smooth trend of confidence.
        3. Computes the absolute residual (deviation) between each token's log-prob and the 
           EWMA trend.
        4. Sums these residuals with exponential weighting (giving more importance to recent 
           deviations) to form a 'Confidence Shock' score.
        5. Final Score = Total_Neg_LogProb + Alpha * Confidence_Shock.
        
        Hypothesis for future: Hallucinated atomic claims induce a sudden, statistically 
        anomalous deviation from the model's own recent confidence trend (high residual), 
        whereas truthful continuations typically follow a smoother, more predictable confidence 
        trajectory relative to their immediate context.
        """
        scores = []
        alpha = 0.95  # Decay factor for EWMA
        beta = 1.0    # Scaling factor for the shock component
        
        for seq_logprobs in stats["sequence_logprobs"]:
            if len(seq_logprobs) == 0:
                scores.append(0.0)
                continue
            
            seq = np.array(seq_logprobs, dtype=np.float32)
            n = len(seq)
            
            if n < 5:
                # For very short sequences, rely primarily on total perplexity
                scores.append(-np.sum(seq))
                continue
            
            # 1. Baseline: Total Negative Log Prob
            total_neg_logprob = -np.sum(seq)
            
            # 2. Compute EWMA
            # ewma[i] = alpha * x[i] + (1 - alpha) * ewma[i-1]
            # We initialize ewma with the first value
            ewma = np.zeros(n)
            ewma[0] = seq[0]
            
            for i in range(1, n):
                ewma[i] = alpha * seq[i] + (1 - alpha) * ewma[i-1]
            
            # 3. Compute Residuals (Deviation from trend)
            residuals = seq - ewma
            
            # 4. Compute Weighted Confidence Shock
            # We use a decaying weight from the end backwards or simply increasing weights 
            # from the start. Given we want to catch recent hallucinations, let's weight 
            # recent tokens higher.
            # weights[i] = (i + 1) to linearly increase weight, or exp(i) for exponential.
            # Let's use a simple linear increase relative to position to avoid overflow, 
            # capped or normalized implicitly by the sum.
            # Better: Weight by the inverse of the decay used in EWMA to align scales? 
            # Simpler: Just weight by position index to emphasize the tail where hallucinations often occur.
            weights = np.arange(1, n + 1)
            
            weighted_shock = np.sum(np.abs(residuals) * weights)
            
            # Normalize shock by sequence length to prevent bias towards longer sequences
            normalized_shock = weighted_shock / n
            
            # 5. Final Score
            final_score = total_neg_logprob + beta * normalized_shock
            
            scores.append(final_score)
        
        return np.array(scores, dtype=np.float32)
