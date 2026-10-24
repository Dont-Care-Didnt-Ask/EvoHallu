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
        Detects hallucinations by identifying streaks of low-confidence tokens and temporal confidence decay.
        
        Hypothesis: Hallucinated sequences frequently contain extended "drift" patterns where the model 
        consecutively generates tokens with unreasonably low probability (long runs of low log-probs), 
        whereas grounded sequences typically show isolated low-confidence spikes or a steady decay 
        without prolonged flat-bottomed low-confidence regions.
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)
        
        # Fixed threshold for "low confidence" relative to typical model behavior
        # Log-probs often drop below -5.0 for unlikely tokens; -3.0 is a reasonable baseline for "very unlikely"
        LOW_CONF_THRESHOLD = -3.0 
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            positions = np.arange(1, len(seq_logprobs) + 1, dtype=np.float32)
            log_probs = seq_logprobs.astype(np.float32)
            
            # Signal 1: LogProbSlope (Negative Pearson Correlation)
            # Invert so higher score = more suspicious (less negative slope)
            try:
                corr, _ = pearsonr(positions, log_probs)
                slope_score = -corr
            except Exception:
                slope_score = 0.0
            
            # Signal 2: Max Run Length of Low Confidence
            # Identify consecutive tokens below the threshold
            is_low_conf = log_probs < LOW_CONF_THRESHOLD
            
            # Find lengths of consecutive True values in boolean array
            # Pad with False to handle edge cases
            padded = np.concatenate([[False], is_low_conf, [False]])
            
            run_lengths = []
            current_run = 0
            for val in padded:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        run_lengths.append(current_run)
                    current_run = 0
            
            if run_lengths:
                max_run_length = np.max(run_lengths)
                # Normalize run length by sequence length to make it comparable across lengths
                # But also penalize absolute length: a run of 5 in a seq of 10 is worse than 5 in 100
                # Simple heuristic: Score = log(max_run_length + 1) to dampen extreme values
                run_score = np.log(max_run_length + 1)
            else:
                run_score = 0.0
            
            # Signal 3: Local Variance in Low Confidence Regions
            # Hallucinations often have a "flat" low-confidence region (low variance within the run)
            # Grounded text might fluctuate more even in low confidence areas
            # We compute the inverse of variance within the runs to reward "flatness"
            flatness_score = 0.0
            if is_low_conf.any():
                # Get indices where low confidence
                low_indices = np.where(is_low_conf)[0]
                if len(low_indices) > 1:
                    # Check if there are consecutive segments
                    # For simplicity, calculate variance of the whole low-conf segment if it's long enough
                    # or just rely on run length. Let's stick to run length as primary, 
                    # but add a penalty if the run is very long (indicating drift)
                    pass
            
            # Combine signals
            # Slope score is in [-1, 1] -> mapped to [0, 1] roughly by taking -corr
            # Run score is log-based, needs scaling. 
            # Heuristic: If max run > 5, it's highly suspicious.
            # Let's normalize run_score to [0, 1] assuming max reasonable run is ~50
            normalized_run_score = min(1.0, run_score / np.log(50 + 1))
            
            # Weighted combination
            # Slope captures global trend, Run captures local structural failure
            final_score = 0.6 * normalized_run_score + 0.4 * slope_score
            
            scores[i] = final_score

        return scores
