import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining global confidence decay (LogProbSlope) with 
    the detection of local low-confidence streaks. It identifies the longest consecutive 
    run of tokens where the top-k confidence margin is below the sequence's median margin.
    
    Hypothesis: Hallucinated claims often appear as contiguous segments of low model 
    confidence, where the model is uncertain about multiple consecutive tokens, 
    rather than just a gradual global decline in confidence.
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
                "top_k_indices",
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
            
            # 1. LogProbSlope (Baseline from Detector 2)
            positions = np.arange(n_tokens, dtype=np.float32)
            if n_tokens > 1 and np.std(logprobs) > 1e-8:
                corr = np.corrcoef(positions, logprobs)[0, 1]
                slope_score = -corr
            else:
                slope_score = 0.0
            
            # 2. Low-Confidence Streak Detection
            # Calculate confidence margin for each token: Top-1 logit - Mean(top-k logits)
            # This measures how much the top candidate stands out from the rest.
            if top_k_logits.shape[0] == n_tokens:
                top1 = top_k_logits[:, 0]
                mean_top_k = np.mean(top_k_logits, axis=1)
                margins = top1 - mean_top_k
                
                # Determine a threshold for "low confidence". 
                # We use the median margin of the sequence as a dynamic threshold.
                # Tokens with margin < median are considered "uncertain".
                median_margin = np.median(margins)
                
                # Binary mask: 1 if margin is below median (uncertain), 0 otherwise
                uncertain_mask = (margins < median_margin).astype(np.int8)
                
                # Find the longest consecutive run of 1s in uncertain_mask
                max_streak = 0
                current_streak = 0
                for val in uncertain_mask:
                    if val == 1:
                        current_streak += 1
                        if current_streak > max_streak:
                            max_streak = current_streak
                    else:
                        current_streak = 0
                
                # Normalize streak length by sequence length to get a ratio [0, 1]
                streak_ratio = max_streak / n_tokens if n_tokens > 0 else 0.0
                
                # Calculate the "intensity" of the longest streak:
                # Average margin deficit within the longest streak.
                # We need to find the indices of the longest streak.
                # For efficiency, we can just use the streak ratio for now, 
                # but adding intensity helps distinguish "long but slightly uncertain" 
                # from "long and very uncertain".
                
                # Let's refine: Find the specific longest streak and compute mean margin deficit.
                # Re-scan to find the start and end of the max streak.
                start_idx = -1
                end_idx = -1
                current_start = -1
                current_streak = 0
                for idx, val in enumerate(uncertain_mask):
                    if val == 1:
                        if current_streak == 0:
                            current_start = idx
                        current_streak += 1
                        if current_streak > max_streak:
                            max_streak = current_streak
                            start_idx = current_start
                            end_idx = idx
                    else:
                        current_streak = 0
                
                if max_streak > 0 and start_idx != -1:
                    # Get the margins in the longest streak
                    streak_margins = margins[start_idx:end_idx+1]
                    # Deficit is how much below the median they are
                    # We want a positive score for low margins.
                    # mean_deficit = median_margin - mean(streak_margins)
                    # If streak_margins are very low, deficit is high.
                    mean_deficit = median_margin - np.mean(streak_margins)
                    # Normalize deficit by the range of margins to make it scale-invariant
                    margin_range = np.max(margins) - np.min(margins)
                    if margin_range > 1e-8:
                        normalized_deficit = mean_deficit / margin_range
                    else:
                        normalized_deficit = 0.0
                else:
                    normalized_deficit = 0.0
                
                # Combine streak ratio and intensity
                # Streak ratio captures "how much of the sequence is uncertain in a block"
                # Normalized deficit captures "how uncertain is that block"
                streak_score = 0.5 * streak_ratio + 0.5 * normalized_deficit
                
            else:
                streak_score = 0.0
            
            # Combine signals
            # Slope is a global trend, Streak is a local structure.
            # We weight them equally, but the streak signal is novel and potentially more discriminative.
            score = 1.0 * slope_score + 1.0 * streak_score
            
            scores.append(score)
            
        return np.array(scores, dtype=np.float32)
