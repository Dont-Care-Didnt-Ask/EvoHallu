import numpy as np
from typing import Dict, Any, List
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
        Detects hallucinations by summing the negative log-probabilities of tokens 
        that fall below a specific surprise threshold.
        
        Hypothesis: Hallucinated tokens exhibit extreme rarity (very negative log-prob) 
        compared to grounded text, which maintains moderate probabilities. Summing only 
        the 'extreme' outliers isolates the hallucination signal from the background 
        perplexity of valid sequences.
        """
        seq_logprobs = stats["sequence_logprobs"]
        threshold = -5.0  # Tokens with log-prob < -5.0 are considered highly surprising
        
        scores = []
        for seq in seq_logprobs:
            # Apply thresholding: keep only highly surprising tokens
            masked_probs = seq[seq < threshold]
            # Sum the negative log-probs of these extreme tokens
            score = -np.sum(masked_probs)
            scores.append(score)
            
        return np.array(scores, dtype=np.float32)
