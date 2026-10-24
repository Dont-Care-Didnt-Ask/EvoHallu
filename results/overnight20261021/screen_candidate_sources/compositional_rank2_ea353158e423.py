import numpy as np
from scipy import stats
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
        Detects hallucinations using a composite of tail risk, shock detection, 
        top-k distribution collapse, and positional decay.
        
        Approach:
        1. Tail Risk (EMA): Exponentially weighted average of negative log-probs (alpha=0.9).
        2. Shock Detection: Normalized sum of absolute first and second derivatives of neg-log-probs.
        3. Top-K Collapse: 1 / (max_top_k_logit / mean_top_k_logit). High values indicate flat distributions.
        4. Positional Decay: Difference between mean log-prob of last 10% and first 10%.
        
        Weights: [0.35, 0.30, 0.25, 0.10].
        
        Specific Hypothesis for Future:
        Hallucinated tokens often exhibit a combination of sustained low confidence at the sequence end, 
        abrupt non-linear drops in likelihood, flattened top-k distributions (indicating lack of a clear best choice), 
        and a systematic degradation of confidence in the final segment compared to the beginning.
        """
        scores = np.zeros(len(stats["sequence_logprobs"]), dtype=np.float32)
        
        # Hyperparameters
        alpha = 0.9
        k = 512
        tail_frac = 0.1
        head_frac = 0.1
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue
            
            n = len(seq_logprobs)
            neg_logprobs = np.array(seq_logprobs, dtype=np.float64)
            
            term_ema = 0.0
            term_shock = 0.0
            term_topk = 0.0
            term_pos = 0.0
            
            # 1. EMA (Tail Risk)
            ema_val = 0.0
            for val in neg_logprobs:
                ema_val = alpha * val + (1 - alpha) * ema_val
            term_ema = ema_val
            
            # 2. Shock Detection (Slope + Accel)
            if n >= 4:
                slope = np.zeros(n, dtype=np.float64)
                slope[1:-1] = neg_logprobs[2:] - neg_logprobs[:-2]
                slope[0] = neg_logprobs[1] - neg_logprobs[0]
                slope[-1] = neg_logprobs[-1] - neg_logprobs[-2]
                
                accel = np.zeros(n, dtype=np.float64)
                accel[1:-1] = slope[2:] - slope[:-2]
                accel[0] = slope[1] - slope[0]
                accel[-1] = slope[-1] - slope[-2]
                
                # Normalize by length and max possible deviation to keep scale consistent
                # Max diff is roughly bounded by log-prob range, but we normalize by n
                term_shock = np.sum(np.abs(slope) + np.abs(accel)) / n
            
            # 3. Top-K Collapse
            if n > 0 and i < len(stats["top_k_logits"]):
                top_k_logits = stats["top_k_logits"][i]
                # Handle potential NaNs or inf if logits are extreme
                valid_mask = ~np.isnan(top_k_logits)
                if np.any(valid_mask):
                    top_k_vals = top_k_logits[valid_mask]
                    if len(top_k_vals) > 0:
                        peak = np.max(top_k_vals)
                        mean_val = np.mean(top_k_vals)
                        if peak > 0 and mean_val > 0:
                            # Inverse ratio: if peak is close to mean, ratio is ~1, inverse is ~1.
                            # If peak is much higher, ratio is high, inverse is low (good confidence).
                            # If flat, ratio is low, inverse is high (bad confidence).
                            term_topk = 1.0 / (peak / mean_val)
                        else:
                            term_topk = 1.0
                    else:
                        term_topk = 0.0
                else:
                    term_topk = 0.0
            
            # 4. Positional Decay
            if n >= int(n / head_frac) + int(n / tail_frac):
                head_start = 0
                head_end = int(n * head_frac)
                tail_start = int(n * (1 - tail_frac))
                tail_end = n
                
                head_mean = np.mean(neg_logprobs[head_start:head_end])
                tail_mean = np.mean(neg_logprobs[tail_start:tail_end])
                
                # Positive score if tail is worse (higher neg-log-prob) than head
                term_pos = tail_mean - head_mean
            
            # Composite Score
            score = (
                0.35 * term_ema +
                0.30 * term_shock +
                0.25 * term_topk +
                0.10 * term_pos
            )
            scores[i] = score

        return scores
