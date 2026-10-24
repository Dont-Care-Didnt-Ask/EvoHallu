import numpy as np
from scipy.stats import pearsonr, skew
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
        Detects hallucinations by combining the slope of log-probability decay 
        and the skewness of the log-probability distribution.
        
        Approach:
        1. Compute the Pearson correlation coefficient between token position 
           and log-probability for each sequence (LogProbSlope).
        2. Compute the skewness of the log-probability distribution for each sequence.
        3. Combine these two signals: Hallucinations are hypothesized to have 
           a steeper negative slope (lower correlation) and potentially distinct 
           distributional skewness compared to factual text.
        
        Hypothesis for Future: Hallucinated sequences exhibit a significantly 
        steeper negative slope in the relationship between token position and 
        log-probability compared to factual sequences, indicating a faster decay 
        of generation confidence over time.
        
        Output shape: [N] where N is the number of samples.
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)

        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) < 2:
                # If sequence is too short to compute slope or skew reliably, 
                # default to a neutral score or handle based on specific logic.
                # Here we default to 0.0.
                scores[i] = 0.0
                continue

            # 1. LogProbSlope: Pearson correlation between position and log_prob
            positions = np.arange(len(seq_logprobs))
            try:
                corr, _ = pearsonr(positions, seq_logprobs)
            except Exception:
                corr = 0.0

            # 2. Skewness of log-prob distribution
            try:
                skew_val = skew(seq_logprobs)
            except Exception:
                skew_val = 0.0

            # Combine signals:
            # We expect hallucinations to have lower (more negative) correlation 
            # (steeper drop) and potentially higher absolute skewness.
            # Simple weighted sum: prioritize the slope as per the baseline instruction.
            # Normalize skewness roughly by dividing by a constant to balance scales,
            # or use it as a secondary term.
            # Let's define a composite score where lower is better (more hallucination).
            # High negative correlation -> High hallucination risk.
            # High positive skew (many low probs) -> High hallucination risk.
            
            # Invert correlation so that more negative values contribute positively to the "risk" score
            # Actually, let's keep it intuitive: 
            # Score = -corr (so strong negative correlation becomes a high positive score)
            #        + abs(skew) (normalized)
            
            # Normalize skewness to be comparable to correlation range [-1, 1]
            # Skewness can be large, so we dampen it or just add it. 
            # Let's try: score = -corr + 0.5 * skew_val (assuming right skew is bad)
            # Or simply sum them if we know the direction. 
            # Let's assume: Lower corr (more negative) is bad. Higher positive skew is bad.
            
            risk_score = -corr + 0.5 * skew_val
            
            scores[i] = risk_score

        return scores
