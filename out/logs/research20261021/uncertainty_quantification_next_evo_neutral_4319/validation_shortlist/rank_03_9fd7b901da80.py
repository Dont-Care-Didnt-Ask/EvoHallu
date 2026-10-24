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
        Detects hallucinations by aggregating cumulative surprise (sum of negative log-probs).
        Hypothesis: Hallucinated claims exhibit a systematically lower probability mass 
        across the generation trajectory compared to factual claims, making the sum 
        a more discriminative signal than the maximum single-token surprise.
        """
        seq_logprobs = stats["sequence_logprobs"]
        
        # Compute negative log-prob for each token in each sequence
        # Input shape: List of arrays with shape (n_i - 1,)
        neg_logprobs = [-lp for lp in seq_logprobs]
        
        # Aggregate: Sum of negative log-probs per sequence.
        # Higher scores indicate lower total probability mass, suggesting hallucination.
        scores = np.array([np.sum(nlps) for nlps in neg_logprobs], dtype=np.float32)
        
        return scores
