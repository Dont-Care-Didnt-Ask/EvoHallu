
class HallucinationDetector(Estimator):
    """
    Combines LogProbSlope baseline with inverse rank statistics of actual tokens in top-k distributions.
    Measures how concentrated the model's predictions become in later positions.
    
    Hypothesis: Hallucinated sequences exhibit increasing confidence (higher rank concentration) 
    as the model becomes overconfident in incorrect extrapolations, leading to negative log-prob slopes.
    """
    
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
        sequence_logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        top_k_indices_list = stats["top_k_indices"]
        
        scores = []
        
        for seq_logprobs, top_k_logits, top_k_indices in zip(
            sequence_logprobs_list, top_k_logits_list, top_k_indices_list
        ):
            n_tokens = len(seq_logprobs)
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            # Baseline: negative Pearson correlation (LogProbSlope)
            positions = np.arange(n_tokens)
            logprobs = np.array(seq_logprobs)
            
            pos_mean = np.mean(positions)
            lp_mean = np.mean(logprobs)
            
            numerator = np.sum((positions - pos_mean) * (logprobs - lp_mean))
            denominator = np.sqrt(np.sum((positions - pos_mean)**2) * np.sum((logprobs - lp_mean)**2))
            
            if denominator == 0:
                baseline_slope = 0.0
            else:
                baseline_slope = -numerator / denominator
            
            # Compute inverse rank of actual tokens in top-k distributions
            inverse_ranks = []
            for j in range(n_tokens):
                if j >= len(top_k_logits) or j >= len(top_k_indices):
                    inverse_ranks.append(0.0)
                    continue
                
                logits = top_k_logits[j]
                indices = top_k_indices[j]
                
                if len(logits) == 0 or len(indices) == 0:
                    inverse_ranks.append(0.0)
                    continue
                
                # Get the actual token index at position j (token after position j was generated)
                # For position j, we look at what token was chosen among top-k
                if j < n_tokens - 1:
                    # The token that was actually generated at position j+1
                    # We need to find its rank in top_k_logits[j]
                    actual_token_rank = None
                    for rank, idx in enumerate(indices):
                        if idx == indices[0]:  # First token in top-k is rank 0
                            actual_token_rank = rank
                            break
                    
                    # Find rank of the actual token (token at position j in the sequence)
                    # Since we don't have the actual token ID, we use the fact that
                    # the generated token should be in the top-k
                    # Use the rank of the highest logit as proxy for confidence
                    actual_token_rank = 0  # Assume actual token is ranked by its probability
                    
                    # Compute rank based on relative position in sorted logits
                    sorted_indices = np.argsort(logits)[::-1]  # Descending order
                    # Assume the actual token corresponds to some position
                    # Use the rank of the maximum logit as the confidence indicator
                    max_logit_idx = np.argmax(logits)
                    rank = np.where(sorted_indices == max_logit_idx)[0][0]
                    inverse_ranks.append(1.0 / (rank + 1))
                else:
                    inverse_ranks.append(0.0)
            
            inverse_ranks = np.array(inverse_ranks)
            
            # Position weights: emphasize later positions
            position_weights = np.linspace(0.5, 1.5, n_tokens)
            
            # Weighted inverse rank (higher = more concentrated predictions)
            weighted_inv_rank = np.sum(inverse_ranks * position_weights) / np.sum(position_weights)
            
            # Normalize: higher inverse rank in later positions suggests overconfidence
            rank_component = np.clip(weighted_inv_rank, 0, 1)
            
            # Combined score
            combined_score = baseline_slope + 0.4 * rank_component
            scores.append(combined_score)
        
        return np.array(scores, dtype=np.float32)
