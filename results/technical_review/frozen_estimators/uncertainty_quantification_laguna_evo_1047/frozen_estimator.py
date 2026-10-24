
class HallucinationDetector(Estimator):
    """Combines log-prob slope with top-k concentration analysis.
    
    Uses negative Pearson correlation between token position and log-probability
    as baseline, plus a concentration metric comparing top-50 vs rest of top-512
    logits to approximate tail behavior. Hallucinations often show both increasing
    confidence degradation and poor token concentration.
    
    Hypothesis: Sequences with steeper log-prob decline and lower concentration
    in top-k logits are more likely to hallucinate due to coherence breakdown.
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
        
        for seq_logprobs, top_k_logits in zip(sequence_logprobs_list, top_k_logits_list):
            n_tokens = len(seq_logprobs)
            
            # Baseline: LogProbSlope (negative Pearson correlation)
            positions = np.arange(n_tokens)
            if n_tokens > 1:
                slope_corr = np.corrcoef(positions, seq_logprobs)[0, 1]
                logprob_slope = -slope_corr if not np.isnan(slope_corr) else 0.0
            else:
                logprob_slope = 0.0
            
            # Tail behavior approximation: concentration in top-50 vs rest
            if len(top_k_logits) > 0:
                # Take logits from first n_tokens positions
                relevant_logits = top_k_logits[:n_tokens]
                
                # Compute softmax probabilities for each position's top-512
                top50_probs = []
                rest_probs = []
                
                for logits in relevant_logits:
                    logits = np.array(logits, dtype=np.float64)
                    logits = logits - np.max(logits)  # Numerical stability
                    probs = np.exp(logits) / np.sum(np.exp(logits))
                    
                    # Top-50 probability mass
                    top50_mass = np.sum(probs[:50])
                    # Rest (51-512) probability mass
                    rest_mass = np.sum(probs[50:]) if len(probs) > 50 else 0.0
                    
                    top50_probs.append(top50_mass)
                    rest_probs.append(rest_mass)
                
                # Concentration metric: how much probability is in top-50
                # Lower concentration = more spread out = higher uncertainty
                if len(top50_probs) > 0:
                    avg_concentration = np.mean(top50_probs)
                    # Invert so higher concentration = lower score (less hallucination-prone)
                    concentration_score = 1.0 - avg_concentration
                else:
                    concentration_score = 0.0
            else:
                concentration_score = 0.0
            
            # Combine signals
            total_score = logprob_slope + concentration_score
            scores.append(total_score)
        
        return np.array(scores, dtype=np.float32)
