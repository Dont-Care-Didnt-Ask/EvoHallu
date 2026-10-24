
class HallucinationDetector(Estimator):
    """
    Uses Pearson correlation between token positions and log-probabilities as baseline,
    combined with min/max logit statistics to capture distribution spread.
    
    Hypothesis: Hallucinated sequences show decreasing log-probability trends (negative slope)
    and increased logit distribution spread in later positions as models become less certain.
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
            if n_tokens == 0:
                scores.append(0.0)
                continue
            
            # Baseline: LogProbSlope - negative Pearson correlation between position and log-prob
            positions = np.arange(n_tokens)
            logprobs_arr = np.array(seq_logprobs)
            
            if n_tokens > 1:
                correlation = np.corrcoef(positions, logprobs_arr)[0, 1]
                baseline_score = -correlation if not np.isnan(correlation) else 0.0
            else:
                baseline_score = 0.0
            
            # Additional signal: min/max logit statistics
            min_logits = []
            max_logits = []
            mean_logits = []
            
            for j in range(min(n_tokens, len(top_k_logits))):
                logits = top_k_logits[j]
                if len(logits) > 0:
                    min_logits.append(np.min(logits))
                    max_logits.append(np.max(logits))
                    mean_logits.append(np.mean(logits))
            
            if len(min_logits) > 1:
                # Compute correlation of min/max with positions
                min_corr = np.corrcoef(positions[:len(min_logits)], min_logits)[0, 1]
                max_corr = np.corrcoef(positions[:len(max_logits)], max_logits)[0, 1]
                
                # Spread metric: difference between max and min correlations
                spread_metric = abs(max_corr - min_corr) if not np.isnan(min_corr) else 0.0
                
                # Weighted combination with low weight on spread to avoid outlier dependence
                additional_score = 0.1 * spread_metric
            else:
                additional_score = 0.0
            
            # Combined score
            combined_score = baseline_score + additional_score
            scores.append(combined_score)
        
        return np.array(scores, dtype=np.float32)
