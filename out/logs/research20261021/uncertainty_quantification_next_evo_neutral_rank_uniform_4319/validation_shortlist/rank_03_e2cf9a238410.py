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
        Detects hallucinations by combining global improbability (sum of negative log-probs)
        with local coherence instability (variance of log-probs).
        
        Approach:
        1. Extract log-probabilities for each generated token.
        2. Compute global surprise: sum of negative log-probabilities.
        3. Compute instability: variance of log-probabilities.
        4. Aggregate: score = global_surprise + 0.3 * instability.
        
        Hypothesis: Hallucinated atomic claims exhibit both higher total surprisal than grounded claims 
        and greater variance in token-level confidence, reflecting a deviation from the model's 
        expected probability distribution for factual content.
        """
        seq_logprobs = stats["sequence_logprobs"]
        scores = np.zeros(len(seq_logprobs), dtype=np.float32)
        
        for i, probs in enumerate(seq_logprobs):
            if len(probs) == 0:
                scores[i] = 0.0
                continue
            
            # Global Surprise: Sum of negative log-probabilities
            global_surprise = -np.sum(probs)
            
            # Instability: Variance of log-probabilities
            # Using population variance for stability
            instability = np.var(probs)
            
            # Aggregation: Weighted sum
            # Primary signal is total rarity (global_surprise), 
            # secondary signal is coherence (instability)
            score = global_surprise + 0.3 * instability
            
            scores[i] = score
            
        return scores
