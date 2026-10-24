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
        Detects hallucinations by analyzing the stability of confidence over the sequence.
        
        Approach:
        1. LogProbSlope: Measures the negative Pearson correlation between token position and log-probability. 
           Coherent text typically shows a smooth decay (high negative correlation). Hallucinations often show 
           erratic patterns or flat confidence (low correlation).
        2. Curvature Spike: Computes the second derivative of the log-probability sequence. 
           Hallucinations often involve sudden drops in confidence (sharp inflection points) after an initial 
           confident guess, leading to high curvature.
        
        Score Construction:
        We combine the inverse of the correlation magnitude and the mean absolute curvature. 
        Higher scores indicate higher likelihood of hallucination.
        
        Future Hypothesis:
        We hypothesize that hallucinated sequences will exhibit a distinct "confidence collapse" pattern 
        where the second derivative of log-probability shows localized maxima that do not align with 
        semantic boundaries (which would require external knowledge), but rather reflect the model's 
        internal struggle to maintain coherence.
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)
        
        for i, logprobs in enumerate(stats["sequence_logprobs"]):
            # logprobs shape: (n_i - 1,)
            # We treat the first token as position 0, though the input excludes the first generated token's logprob 
            # relative to the prompt. Let's assume logprobs corresponds to tokens 1 to n_i-1.
            # To align with position, we prepend a dummy value or just index relative to the array start.
            # Let's assume position t corresponds to logprobs[t].
            
            pos = np.arange(len(logprobs), dtype=np.float64)
            logp = logprobs.astype(np.float64)
            
            # 1. LogProbSlope (Pearson Correlation)
            # We use the negative correlation because we expect log-prob to decrease (become less positive/more negative)
            # as we generate, but actually, for hallucinations, the relationship is less monotonic.
            # A strong negative correlation implies consistent decay (good). 
            # A correlation close to 0 or positive implies erratic behavior (bad).
            corr, _ = pearsonr(pos, logp)
            slope_score = 1.0 / (1.0 + np.abs(corr))
            
            # 2. Second Derivative (Curvature)
            # Compute first differences
            diff1 = np.diff(logp, prepend=logp[0]) # Shape (n)
            # Compute second differences
            diff2 = np.diff(diff1) # Shape (n-1)
            
            # Mean absolute curvature
            mean_curvature = np.mean(np.abs(diff2))
            
            # Normalize curvature roughly by sequence length to make it comparable
            # Longer sequences naturally have more variance in 2nd derivative
            length_factor = len(diff2) / 10.0 # Arbitrary normalization constant
            curvature_score = 1.0 / (1.0 + mean_curvature * length_factor)
            
            # Combine scores
            # Weighted sum: slightly more emphasis on slope stability as it's the primary signal
            combined_score = 0.6 * slope_score + 0.4 * curvature_score
            scores[i] = combined_score
            
        return scores
