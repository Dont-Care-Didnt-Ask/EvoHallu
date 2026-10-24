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
        position-weighted gradient instability.
        
        Approach: 
        1. Computes the total negative log-probability (NLL) as a baseline for global unlikelihood.
        2. Computes the first difference (gradient) of the negative log-probabilities.
        3. Applies a linear ramp-up weight to the absolute gradient values, emphasizing 
           instability in later tokens where specific entities/facts are typically asserted 
           in atomic claims.
        4. Normalizes the weighted gradient sum by sequence length to ensure comparability.
        
        Hypothesis for future: 
        Hallucinated claims often manifest as a localized spike in token cost (high gradient) 
        at the position of the incorrect entity. By weighting later tokens more heavily, 
        this detector prioritizes the specific failure point over general sequence noise.
        
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
            
            # Novel Signal: Position-weighted gradient instability
            if len(neg_lp) > 1:
                # Gradient: change in cost between consecutive tokens
                gradient = np.diff(neg_lp)
                abs_gradient = np.abs(gradient)
                
                # Position weights: ramp up from 0 to 1 across the sequence
                # This emphasizes instability in later tokens (where entities usually appear)
                positions = np.arange(1, len(gradient) + 1)
                max_pos = len(gradient)
                weights = positions / max_pos
                
                # Weighted sum of absolute gradients
                weighted_grad_sum = np.sum(abs_gradient * weights)
                
                # Normalize by sequence length to prevent long sequences from dominating
                # This creates a "mean weighted gradient" signal
                normalized_grad_signal = weighted_grad_sum / max_pos
            else:
                normalized_grad_signal = 0.0
            
            # Composite score: Sum of global unlikelihood and local position-weighted instability
            # The gradient term is scaled to be comparable to NLL (typically NLL is larger, 
            # but for atomic claims (10-25 tokens), the weighted gradient can be significant)
            # A simple sum works as a first-order approximation for ranking.
            score = total_nll + normalized_grad_signal
            
            scores.append(score)
            
        return np.array(scores, dtype=np.float32)
