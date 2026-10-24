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
        Detects hallucinations by combining mean negative log-probability with 
        a position-weighted slope of cumulative uncertainty.
        
        Approach:
        1. Computes the mean negative log-probability (normalized NLL) per sequence 
           to establish a length-invariant baseline of overall predictability.
        2. Computes the cumulative sum of negative log-probabilities.
        3. Calculates the slope of this cumulative sum relative to token position, 
           applying a linear weight that increases towards the end of the sequence.
           This amplifies signals where uncertainty accelerates in the latter half 
           of the generation, indicative of coherence breakdown.
        4. Sums the normalized baseline and the weighted slope.
        
        Hypothesis for future: Hallucinated segments often cause a non-linear 
        acceleration in prediction difficulty (steepening NLL curve) specifically 
        after the model commits to an incorrect premise, whereas factual sequences 
        maintain a more linear or stable uncertainty trajectory throughout.
        """
        sequences = stats["sequence_logprobs"]
        N = len(sequences)
        scores = np.zeros(N, dtype=np.float32)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                scores[i] = 0.0
                continue
            
            log_probs = seq
            n_tokens = len(log_probs)
            
            # Baseline: Mean Negative Log-Probability (Length Normalized)
            # Higher mean NLL -> Less predictable -> Potential hallucination
            mean_nll = -np.mean(log_probs)
            
            # Novel Signal: Position-Weighted Slope of Cumulative NLL
            # 1. Compute cumulative NLL
            cum_nll = -np.cumsum(log_probs)
            
            # 2. Create position weights (0 to 1), emphasizing the end
            # Weight w_t = t / (n-1) for t in 0..n-1
            positions = np.arange(n_tokens, dtype=np.float32)
            if n_tokens > 1:
                weights = positions / (n_tokens - 1)
            else:
                weights = np.array([0.0])
            
            # 3. Compute weighted slope: Cov(cum_nll, weights) / Var(weights)
            # This measures how much cum_nll increases as we move to later positions
            weighted_sum = np.sum(cum_nll * weights)
            mean_cum_nll = np.mean(cum_nll)
            mean_weight = np.mean(weights)
            cov = weighted_sum / n_tokens - mean_cum_nll * mean_weight
            
            # Normalize by variance of weights to make slope comparable across lengths
            var_weights = np.var(weights)
            if var_weights > 1e-8:
                slope = cov / var_weights
            else:
                slope = 0.0
            
            # Combine signals
            # Slope is expected to be positive (uncertainty grows). 
            # We add it directly to mean_nll.
            scores[i] = mean_nll + slope

        return scores
