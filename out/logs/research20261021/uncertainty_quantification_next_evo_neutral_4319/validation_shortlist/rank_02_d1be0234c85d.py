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
        Detects hallucinations by measuring the total accumulated surprisal (sum of negative log-probs).
        
        Approach: Computes the sum of negative log-probabilities across all generated tokens 
        in a sequence. This aggregates the 'surprise' of the entire claim, assuming that 
        hallucinated content accumulates higher total information distance from the true context 
        than grounded content.
        
        Hypothesis: Hallucinated atomic claims exhibit a higher total cumulative surprisal 
        (sum of -log P) compared to grounded claims, as the model struggles to assign 
        high probability to the entire fabricated narrative simultaneously.
        """
        seq_logprobs = stats["sequence_logprobs"]
        
        # Compute negative log-prob for each token
        # Input: List of arrays with shape (n_i - 1,)
        neg_logprobs = [-lp for lp in seq_logprobs]
        
        # Aggregate: Sum the negative log-probs per sequence.
        # This captures the total "surprise" of the sequence.
        # Higher scores indicate higher likelihood of hallucination.
        scores = np.array([np.sum(nlps) for nlps in neg_logprobs], dtype=np.float32)
        
        return scores
