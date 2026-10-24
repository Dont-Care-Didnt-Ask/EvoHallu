
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
        # Extract inputs
        sequence_logprobs = stats["sequence_logprobs"]  # List[np.ndarray]
        top_k_logits = stats["top_k_logits"]           # List[np.ndarray]
        
        scores = []
        
        for i, probs in enumerate(sequence_logprobs):
            # 1. Calculate total negative log-likelihood (actual generation)
            # Shape: (n_i - 1,)
            actual_ll = -np.sum(probs)
            
            # 2. Calculate max log-likelihood from top-k logits at the final step
            # top_k_logits shape is (n_i, k). We take the last row (n_i-1).
            final_logits = top_k_logits[i][-1, :]
            max_top_k_ll = -np.max(final_logits)
            
            # 3. Compute Log-Likelihood Ratio
            # Avoid division by zero by adding a small epsilon if max_top_k_ll is 0
            if max_top_k_ll > 0:
                llr = actual_ll / max_top_k_ll
            else:
                llr = actual_ll  # Fallback if model is completely unconfident
                
            scores.append(llr)
            
        return np.array(scores, dtype=np.float32)
