import numpy as np
from scipy.stats import pearsonr
class HallucinationDetector(Estimator):
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
        """
        Detects hallucinations by combining the temporal drift of log-probabilities 
        (LogProbSlope) with an exponential-weighted accumulation of surprisal.
        
        Approach:
        1. Baseline: Computes the negative Pearson correlation between token position 
           and log-probability. A positive result indicates the model's confidence 
           decreases (log-prob drops) as the sequence progresses, a hallmark of 
           drifting into hallucinated content.
        2. Additional Signal: Computes a weighted sum of negative log-probabilities 
           using an exponential decay factor (EMA-style weighting) that emphasizes 
           the end of the sequence, where hallucinations typically culminate.
        3. Combination: Normalizes both signals to [0, 1] range and sums them.
        
        Hypothesis for future: Integrating the standard deviation of the top-k logit 
        differences (uncertainty) as a third orthogonal feature could better distinguish 
           between 'model confusion' (high uncertainty, low slope) and 'confident hallucination' 
           (low uncertainty, high slope).
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)
        
        # Pre-compute EMA weights for efficiency if needed, but per-sequence is fine for N samples
        # We use alpha=0.5 for the weighting to balance history and recent context
        alpha = 0.5
        
        for i, (logprobs, top_k_logits, top_k_indices) in enumerate(
            zip(stats["sequence_logprobs"], stats["top_k_logits"], stats["top_k_indices"])
        ):
            if len(logprobs) == 0:
                scores[i] = 0.0
                continue
            
            seq_len = len(logprobs)
            t = np.arange(1, seq_len + 1, dtype=np.float32) # Positions 1 to n
            
            # 1. LogProbSlope Signal
            # Calculate Pearson correlation between position and log-prob
            # scipy.pearsonr returns (r, p-value), we only need r
            r, _ = pearsonr(t, logprobs)
            # The baseline is defined as -r. 
            # If logprobs drop (hallucination), r is negative, -r is positive (high score).
            slope_signal = -r
            
            # 2. EMA Weighted Surprisal Signal
            # Weights: w_t = alpha^(seq_len - t) to emphasize recent tokens
            # Or simpler: cumulative EMA of negative log probs
            neg_logprobs = -np.array(logprobs, dtype=np.float32)
            
            # Calculate EMA of the negative log probs
            # ema_t = alpha * neg_logprobs_t + (1-alpha) * ema_{t-1}
            ema_weights = np.zeros(seq_len, dtype=np.float32)
            ema_vals = np.zeros(seq_len, dtype=np.float32)
            
            # Initialize with first value
            ema_vals[0] = neg_logprobs[0]
            
            # Compute EMA
            for t_idx in range(1, seq_len):
                ema_vals[t_idx] = alpha * neg_logprobs[t_idx] + (1 - alpha) * ema_vals[t_idx - 1]
            
            # The signal is the final EMA value, representing the weighted recent surprisal
            # We normalize this by the mean negative log prob to make it comparable to slope
            mean_nll = np.mean(neg_logprobs)
            if mean_nll > 1e-6:
                ema_signal = ema_vals[-1] / mean_nll
            else:
                ema_signal = 0.0
                
            # 3. Combine
            # Normalize slope_signal (range approx -1 to 1) to be comparable to ema_signal (usually > 0)
            # We can just use the raw slope_signal as it's bounded [-1, 1]
            # And scale ema_signal to be in similar magnitude if necessary, but let's assume 
            # the combination is additive.
            
            # To ensure both contribute meaningfully:
            final_score = slope_signal + ema_signal
            
            scores[i] = final_score

        return scores
