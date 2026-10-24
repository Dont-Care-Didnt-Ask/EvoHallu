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
        Detects hallucinations by combining global surprise with distributional instability.
        
        Approach:
        1. Calculate Global Surprise: Sum of negative log-probabilities (total improbability).
        2. Calculate Instability: Population variance of the log-probabilities (distributional erraticness).
        3. Aggregate: A weighted sum of Global Surprise and Instability.
        
        Hypothesis: Hallucinated atomic claims exhibit higher variance in token likelihoods (instability) 
        compared to grounded claims, which tend to have more consistent probability profiles (either 
        high confidence on common terms or consistent low confidence on specific rare terms).
        """
        seq_logprobs = stats["sequence_logprobs"]
        scores = np.zeros(len(seq_logprobs), dtype=np.float32)
        
        for i, probs in enumerate(seq_logprobs):
            if len(probs) == 0:
                scores[i] = 0.0
                continue
            
            # Signal 1: Global Surprise (Sum of negative log-probs)
            global_surprise = -np.sum(probs)
            
            # Signal 2: Instability (Variance of log-probs)
            # Using population variance to measure distributional spread
            instability = np.var(probs)
            
            # Aggregation: Weighted sum
            # Global surprise captures the "unlikely" nature, instability captures the "inconsistent" nature
            # of hallucinations vs grounded facts.
            score = global_surprise + 0.5 * instability
            
            scores[i] = score
            
        return scores
