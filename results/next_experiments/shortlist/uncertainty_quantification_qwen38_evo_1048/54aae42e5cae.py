import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining LogProbSlope (global confidence decay) with 
    the slope of the Top-N vs Rest Contrast. The contrast measures how much the top 
    5 logits stand out from the remaining 507 logits, capturing distribution sharpness.
    
    Hypothesis: Hallucinated claims exhibit a decline in distribution sharpness (contrast) 
    over the sequence, where the model's top candidates become less distinct from the 
    rest, indicating increasing uncertainty about the specific wording.
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        scores = []
        
        for i in range(len(stats["sequence_logprobs"])):
            logprobs = stats["sequence_logprobs"][i]
            top_k_logits = stats["top_k_logits"][i]
            
            n_tokens = len(logprobs)
            if n_tokens == 0:
                scores.append(0.0)
                continue
            
            logprobs = np.array(logprobs, dtype=np.float64)
            positions = np.arange(n_tokens, dtype=np.float64)
            
            # 1. Baseline: LogProbSlope
            # Negative correlation between position and log-prob.
            # If log-probs decrease over time, corr is negative, so -corr is positive.
            if n_tokens > 1 and np.std(logprobs) > 1e-8:
                corr_lp = np.corrcoef(positions, logprobs)[0, 1]
                slope_lp = -corr_lp
            else:
                slope_lp = 0.0
            
            # 2. Tail Contrast Signal
            # Calculate contrast for each token: Mean(top 5) - Mean(rest 507)
            # This measures the "peakiness" of the distribution.
            if top_k_logits.shape[0] == n_tokens and top_k_logits.shape[1] >= 10:
                # Ensure we have at least 5 tokens in top and rest
                top_n = 5
                rest_start = top_n
                
                # Extract top and rest means
                top_means = np.mean(top_k_logits[:, :top_n], axis=1)
                rest_means = np.mean(top_k_logits[:, rest_start:], axis=1)
                
                contrasts = top_means - rest_means
                
                # Slope of Contrast: Does the contrast decrease over time?
                # If contrast decreases, corr(pos, contrast) is negative, so -corr is positive.
                if n_tokens > 1 and np.std(contrasts) > 1e-8:
                    corr_c = np.corrcoef(positions, contrasts)[0, 1]
                    slope_c = -corr_c
                else:
                    slope_c = 0.0
                
                # Global Mean Contrast (inverted)
                # Lower contrast means more uncertainty. We want higher score for lower contrast.
                # Normalize mean contrast to [0, 1] range roughly? 
                # Or just use negative mean contrast.
                # Let's use the negative of the mean contrast, scaled.
                mean_contrast = np.mean(contrasts)
                # To make it comparable to correlation scores (which are in [-1, 1]),
                # we might need to scale. However, contrast values can be large.
                # Let's use the rank or a normalized version.
                # Simpler: Just use the slope signals which are bounded.
                # But adding a global uncertainty signal helps.
                # Let's normalize the mean contrast by the std of contrasts in the batch? No, per sample.
                # Let's just use the slope signals as they are robust.
                
                # Combine
                # We weight the slope of contrast heavily as it captures the dynamic change.
                score = 1.0 * slope_lp + 1.0 * slope_c
                
            else:
                score = slope_lp
            
            scores.append(score)
            
        return np.array(scores, dtype=np.float32)
