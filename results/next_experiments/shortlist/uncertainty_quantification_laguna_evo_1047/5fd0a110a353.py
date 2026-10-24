
class HallucinationDetector(Estimator):
    """
    Uses negative Pearson correlation between token position and log-probability 
    (LogProbSlope baseline) combined with position-weighted entropy of top-k distributions.
    Early tokens are weighted less, later tokens more heavily to capture compounding uncertainty.
    
    Hypothesis: Hallucinated sequences show decreasing entropy over time as models become 
    overconfident in extrapolated reasoning, creating a negative slope in log-probabilities
    with compressed top-k distributions.
    """
    
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_logprobs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        
        scores = []
        k = 512
        
        for seq_logprobs, top_k_logits in zip(sequence_logprobs_list, top_k_logits_list):
            n_tokens = len(seq_logprobs)
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            # Baseline: negative Pearson correlation (LogProbSlope)
            positions = np.arange(n_tokens)
            logprobs = np.array(seq_logprobs)
            
            # Compute Pearson correlation
            pos_mean = np.mean(positions)
            lp_mean = np.mean(logprobs)
            
            numerator = np.sum((positions - pos_mean) * (logprobs - lp_mean))
            denominator = np.sqrt(np.sum((positions - pos_mean)**2) * np.sum((logprobs - lp_mean)**2))
            
            if denominator == 0:
                baseline_slope = 0.0
            else:
                baseline_slope = -numerator / denominator  # Negative correlation = higher score
            
            # Compute position-weighted entropy of top-k distributions
            entropies = []
            for j in range(min(n_tokens, len(top_k_logits))):
                logits = top_k_logits[j]
                if len(logits) < 2:
                    entropies.append(0.0)
                    continue
                
                # Normalize logits to get probabilities
                max_logit = np.max(logits)
                shifted_logits = logits - max_logit
                probs = np.exp(shifted_logits) / np.sum(np.exp(shifted_logits))
                
                # Compute entropy (higher entropy = more uncertain)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            entropies = np.array(entropies)
            
            # Position weights: emphasize later positions (linear increase)
            if n_tokens > 1:
                position_weights = np.linspace(0.5, 1.5, n_tokens)
            else:
                position_weights = np.array([1.0])
            
            # Weighted entropy (lower entropy in later positions = more certain = potential hallucination)
            weighted_entropy = np.sum(entropies * position_weights) / np.sum(position_weights)
            
            # Normalize entropy component
            if n_tokens > 1:
                entropy_range = np.max(entropies) - np.min(entropies)
                if entropy_range > 0:
                    normalized_entropy = (weighted_entropy - np.min(entropies)) / entropy_range
                else:
                    normalized_entropy = 0.5
            else:
                normalized_entropy = 0.5
            
            # Combined score: baseline slope + entropy penalty (low entropy = high penalty)
            # Lower entropy suggests overconfidence in later tokens
            entropy_penalty = 0.5 - normalized_entropy  # Higher when entropy is low
            
            combined_score = baseline_slope + 0.3 * entropy_penalty
            scores.append(combined_score)
        
        return np.array(scores, dtype=np.float32)
