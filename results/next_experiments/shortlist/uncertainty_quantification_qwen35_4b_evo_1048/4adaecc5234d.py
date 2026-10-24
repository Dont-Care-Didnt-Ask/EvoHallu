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
        Detects hallucinations by computing a Positional Decay Weighted Uncertainty score.
        
        Approach:
        1. Calculate the Sum of Negative Log-Probabilities (SNLP) for the sequence.
        2. Apply a positional decay weight to each token's log-probability. 
           Tokens earlier in the sequence are given lower weight (assuming they are more grounded),
           while tokens later in the sequence are given higher weight.
        3. Sum the weighted negative log-probabilities to create a "drift" metric.
        4. Normalize the result by the sequence length to handle varying sequence lengths.
        
        Hypothesis: Hallucinations tend to accumulate uncertainty later in the sequence 
        as the model drifts from the initial context, whereas grounded text maintains 
        consistent confidence from start to finish.
        """
        n_samples = len(stats["sequence_logprobs"])
        results = np.zeros(n_samples, dtype=np.float32)

        for i, logprobs in enumerate(stats["sequence_logprobs"]):
            # logprobs shape: (n_i - 1,)
            # These are the log-probs for the generated tokens (excluding the first generated token/prompt token)
            # We assume these correspond to indices 1 to n_i-1 of the full sequence if we had full context,
            # but here we treat the input list as the sequence of interest.
            
            n_tokens = len(logprobs)
            if n_tokens == 0:
                results[i] = 0.0
                continue

            # Calculate raw Sum of Negative Log-Probabilities (SNLP)
            raw_uncertainty = -np.sum(logprobs)

            # Calculate positional decay weights
            # We use a linear decay: weight decreases as position increases? 
            # Actually, for "drift", we want to emphasize the END of the sequence.
            # So weight should be HIGHER for later tokens.
            # Let's define weight = 1 / (1 + alpha * position) -> this decreases.
            # We want weight = 1 - (position / max_pos) -> linear increase.
            # Let's use an exponential decay of confidence, so uncertainty weight increases.
            # weight_t = 1.0 - (t / (n_tokens - 1)) for t in 0..n_tokens-1?
            # If n_tokens=1, weight=1. If n_tokens=10, t=0->0.9, t=9->0.0? No, we want t=9 to be high.
            # Let's use: weight = (t + 1) / (n_tokens + 1) -> linear ramp up.
            # Or exponential ramp up: weight = exp(t / n_tokens).
            
            # Let's try a linear ramp that emphasizes the tail.
            # Weights for indices 0 to n_tokens-1
            positions = np.arange(n_tokens)
            # Normalize positions to [0, 1]
            pos_norm = positions / (n_tokens - 1 + 1e-8) # Avoid div by zero if n_tokens=1
            
            # Apply a decay factor to the uncertainty contribution
            # We want later tokens to contribute MORE to the score if they are uncertain.
            # So we multiply the negative log prob by a weight that increases with position.
            # weight = 1 + pos_norm (simple linear ramp)
            # Or weight = exp(pos_norm) (exponential ramp)
            
            # Let's use a mild exponential ramp to emphasize the tail without being too aggressive
            decay_factor = np.exp(pos_norm * 0.5) 
            
            # Weighted uncertainty
            weighted_uncertainty = -np.sum(logprobs * decay_factor)
            
            # Normalize by the sum of weights to make it comparable across lengths
            # Effectively: weighted_uncertainty / sum(decay_factor)
            # But since we sum logprobs * decay, the scale is roughly sum(logp) * avg_decay
            # Let's just return the raw weighted sum and rely on the model's internal normalization if needed,
            # but for ROC-AUC, relative ordering matters.
            # A better normalization: divide by the expected SNLP (raw_uncertainty) if raw > 0?
            # No, let's just normalize by the sequence length to ensure scale consistency.
            
            avg_weighted_uncertainty = weighted_uncertainty / n_tokens
            
            results[i] = avg_weighted_uncertainty

        return results
