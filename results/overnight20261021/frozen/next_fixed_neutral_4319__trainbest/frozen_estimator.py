import numpy as np
from typing import Dict, Any
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
        Detects hallucinations by measuring the density of local probability drops 
        relative to a running mean baseline.
        
        Hypothesis: Hallucinations often occur as sharp local deviations from the 
        model's established confidence trajectory, rather than just globally low 
        probability tokens.
        """
        scores = []
        
        for seq_logprobs in stats["sequence_logprobs"]:
            # Ensure we have data
            if len(seq_logprobs) == 0:
                scores.append(0.0)
                continue
            
            seq_logprobs = np.asarray(seq_logprobs, dtype=np.float32)
            
            # Calculate running mean with a small window to avoid edge effects
            # Using a simple cumulative approach for stability
            n = len(seq_logprobs)
            if n == 0:
                scores.append(0.0)
                continue
                
            # Pad to handle edges for rolling mean calculation
            pad_size = 5
            padded = np.pad(seq_logprobs, (pad_size, pad_size), mode='edge')
            
            # Calculate local deviations from a smoothed baseline
            # We use a simple moving average as the baseline
            window = min(10, n)
            running_mean = np.convolve(seq_logprobs, np.ones(window)/window, mode='same')
            
            # Calculate absolute deviation from the running mean
            # High deviation indicates a sudden shift (hallucination onset)
            deviations = np.abs(seq_logprobs - running_mean)
            
            # Normalize by sequence length to make it length-invariant
            # Higher score = higher density of local anomalies
            score = np.sum(deviations) / n
            
            scores.append(float(score))
            
        return np.array(scores, dtype=np.float32)
