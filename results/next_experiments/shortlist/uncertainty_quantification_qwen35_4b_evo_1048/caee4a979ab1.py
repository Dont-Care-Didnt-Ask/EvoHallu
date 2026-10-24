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
        Detects hallucinations by analyzing the 'Confidence Crash' pattern.
        
        Approach:
        1. LogProbSlope (Weighted): Computes the negative Pearson correlation between 
           token position and log-probability. 
        2. Decay Rate: Computes the average slope of the log-probability sequence.
        3. Positional Weighting: Applies a weight that emphasizes rapid decay occurring 
           early in the sequence, as hallucinations often stem from an initial confident 
           but wrong prediction.
        
        Score Construction:
        - We invert the correlation magnitude (low correlation = erratic = high score).
        - We penalize high initial log-probs followed by steep negative slopes.
        - The final score is a weighted combination of these signals.
        
        Future Hypothesis:
        We hypothesize that hallucinated sequences will exhibit a distinct 'confidence crash' 
        pattern where the model starts with high log-probability (strong internal prediction) 
        followed by a statistically significant negative slope in subsequent tokens, indicating 
        an internal realization of incoherence.
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)
        
        for i, logprobs in enumerate(stats["sequence_logprobs"]):
            # logprobs shape: (n_i - 1,)
            # These correspond to the generated tokens.
            
            if len(logprobs) < 2:
                scores[i] = 0.0
                continue
                
            # 1. Positional Vector
            # Positions are 0 to n-1 corresponding to the generated tokens
            pos = np.arange(len(logprobs), dtype=np.float64)
            logp = logprobs.astype(np.float64)
            
            # 2. LogProbSlope (Pearson Correlation)
            # We expect coherent generation to have a negative correlation (decay).
            # Hallucinations might be flat or erratic.
            corr, _ = pearsonr(pos, logp)
            # Invert correlation: strong negative correlation (good) -> low score
            # Correlation near 0 or positive (bad) -> high score
            slope_score = 1.0 / (1.0 + np.abs(corr))
            
            # 3. Decay Rate (Average Slope)
            # Calculate the slope between the first and last token
            first_val = logp[0]
            last_val = logp[-1]
            total_slope = (last_val - first_val) / (len(logp) - 1)
            
            # We are interested in rapid decay (large negative slope) following high confidence.
            # If total_slope is very negative, it indicates a crash.
            # Normalize by length to make it comparable across sequences
            avg_decay = -total_slope / len(logp)
            
            # 4. Positional Weighting
            # We hypothesize that a crash happening early is more indicative of a hallucination 
            # (wrong start) than a gradual decay.
            # Weight is highest at the beginning (pos=0) and decreases linearly.
            weights = 1.0 - (pos / len(logp))
            weighted_decay = np.sum(avg_decay * weights)
            
            # If the decay is negative (actual drop), weighted_decay will be negative.
            # We want to penalize this.
            decay_score = 1.0 / (1.0 + np.abs(weighted_decay))
            
            # 5. Combine Scores
            # Emphasize the slope stability as the primary indicator of coherence
            combined_score = 0.6 * slope_score + 0.4 * decay_score
            
            scores[i] = combined_score
            
        return scores
