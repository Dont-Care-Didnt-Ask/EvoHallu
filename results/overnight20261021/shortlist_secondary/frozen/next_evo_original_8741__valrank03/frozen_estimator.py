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
        Detects hallucinations by focusing on late-stage confidence collapse and tail instability.
        
        Approach:
        1. Applies an exponential positional weight (increasing towards the end) to emphasize late tokens.
        2. Computes a 'Weighted Tail Drop': the difference between the weighted mean of the last 20% 
           of the sequence and the weighted mean of the first 80%, normalized by sequence length.
        3. Computes 'Weighted Tail Variance': the variance of log-probs in the last 20% weighted by position.
        4. Final Score = |Weighted Tail Drop| + Weighted Tail Variance.
        
        Hypothesis for future: Hallucinated atomic claims induce a disproportionately sharp and unstable 
        drop in confidence specifically in the final segment of the generation, whereas truthful claims 
        maintain a smoother, more predictable probability trajectory even under uncertainty.
        """
        scores = []
        tail_ratio = 0.2  # Focus on the last 20% of tokens
        
        for seq_logprobs in stats["sequence_logprobs"]:
            if len(seq_logprobs) == 0:
                scores.append(0.0)
                continue
            
            seq = np.array(seq_logprobs, dtype=np.float32)
            n = len(seq)
            
            if n < 2:
                scores.append(0.0)
                continue
            
            # 1. Create Positional Weights (Exponential decay from start to end)
            # Higher weights for later tokens
            positions = np.arange(n, dtype=np.float32)
            alpha = 1.5
            weights = np.power(alpha, positions)
            
            # 2. Define Split Point
            split_idx = int(n * (1 - tail_ratio))
            
            # Ensure we have enough tokens in both parts
            if split_idx < 2:
                scores.append(0.0)
                continue
                
            head = seq[:split_idx]
            tail = seq[split_idx:]
            
            # 3. Calculate Weighted Means
            head_w_sum = np.sum(head * weights[:split_idx])
            head_w_sum_weights = np.sum(weights[:split_idx])
            head_weighted_mean = head_w_sum / head_w_sum_weights
            
            tail_w_sum = np.sum(tail * weights[split_idx:])
            tail_w_sum_weights = np.sum(weights[split_idx:])
            tail_weighted_mean = tail_w_sum / tail_w_sum_weights
            
            # 4. Calculate Weighted Tail Drop (Magnitude of decline)
            # We expect log-probs to decrease. A large drop in the tail is suspicious.
            tail_drop = head_weighted_mean - tail_weighted_mean
            
            # 5. Calculate Weighted Tail Variance (Instability)
            # How much does the tail fluctuate around its own weighted mean?
            tail_centered = tail - tail_weighted_mean
            weighted_var = np.sum((tail_centered ** 2) * weights[split_idx:]) / tail_w_sum_weights
            
            # 6. Combine Signals
            # Normalize tail_drop by sequence length to make it comparable across different lengths
            normalized_drop = abs(tail_drop) / n
            
            # Final score: Strong drop + High instability in the tail
            score = normalized_drop + weighted_var
            
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)
