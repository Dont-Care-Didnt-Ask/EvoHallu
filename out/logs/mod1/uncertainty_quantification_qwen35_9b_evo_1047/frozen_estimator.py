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
        Detects hallucinations by combining total sequence unlikelihood with 
        the variance of the log-probability gradient (instability in confidence change).
        
        Approach: 
        1. Computes the total negative log-probability (NLL) as a baseline for global unlikelihood.
        2. Computes the gradient (first difference) of the negative log-probabilities across tokens.
        3. Calculates the variance of these gradients. High variance in the gradient suggests 
           abrupt shifts in confidence, indicative of the model struggling to maintain coherence 
           or forcing an incorrect entity into the context.
        
        Hypothesis for future: 
        Factual sequences exhibit smooth, gradual changes in token confidence, whereas 
        hallucinated sequences show high-frequency fluctuations or sudden spikes in the 
        rate of confidence change due to the model compensating for context mismatches.
        
        Returns:
            np.ndarray of shape [N] containing the composite score.
        """
        seq_logprobs = stats["sequence_logprobs"]
        scores = []
        
        for seq_lp in seq_logprobs:
            if len(seq_lp) == 0:
                scores.append(0.0)
                continue
            
            # Convert to float32 for stability
            neg_lp = -np.array(seq_lp, dtype=np.float32)
            
            # Baseline: Total unlikelihood
            total_nll = np.sum(neg_lp)
            
            # Novel Signal: Variance of the gradient (first difference)
            # Gradient g_i = neg_lp[i] - neg_lp[i-1]
            if len(neg_lp) > 1:
                gradient = np.diff(neg_lp)
                grad_variance = np.var(gradient)
            else:
                grad_variance = 0.0
            
            # Composite score: Weighted sum. 
            # We use a fixed weight here, but in practice, scaling might be needed.
            # The gradient variance is unitless relative to the sum, so a simple sum works 
            # provided the scales are comparable in the training distribution.
            score = total_nll + grad_variance
            
            scores.append(score)
            
        return np.array(scores, dtype=np.float32)
