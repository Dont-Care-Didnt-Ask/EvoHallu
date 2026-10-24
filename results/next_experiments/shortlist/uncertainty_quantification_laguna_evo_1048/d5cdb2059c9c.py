
class HallucinationDetector(Estimator):
    """
    Combines LogProb slope baseline with tail behavior approximation using top-k logits.
    Hypothesis: Hallucinated sequences exhibit flatter log-probability slopes and reduced 
    confidence contrast between top predictions and alternatives in later positions.
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
        scores = []
        for sequence_logprobs, top_k_logits in zip(stats["sequence_logprobs"], stats["top_k_logits"]):
            if len(top_k_logits) == 0:
                scores.append(0.0)
                continue
            
            n = len(top_k_logits)
            
            # Baseline: negative Pearson correlation between position and log-probability
            positions = np.arange(len(sequence_logprobs))
            logprobs = np.array(sequence_logprobs)
            
            if len(positions) > 1:
                slope_corr = np.corrcoef(positions, logprobs)[0, 1]
                if np.isnan(slope_corr):
                    slope_corr = 0.0
                baseline_score = -slope_corr  # Negative correlation = positive score
            else:
                baseline_score = 0.0
            
            # Tail behavior approximation: confidence contrast in last positions
            # Compare top-1 logit vs average of rest of top-k for last 25% of sequence
            last_quarter_size = max(1, n // 4)
            last_logits = top_k_logits[-last_quarter_size:]
            
            if len(last_logits) > 0:
                # Top-1 logits
                top1 = np.max(last_logits, axis=1, keepdims=True)
                
                # Average of remaining top-k (excluding top-1)
                sorted_logits = np.sort(last_logits, axis=1)
                rest_logits = sorted_logits[:, :-1]  # Exclude the maximum
                rest_avg = np.mean(rest_logits, axis=1, keepdims=True)
                
                # Confidence contrast: how much top-1 stands out from alternatives
                contrast = np.mean(top1 - rest_avg)
                
                # Lower contrast in tail suggests uncertainty/hallucination
                tail_penalty = 1.0 / (contrast + 1.0)
            else:
                tail_penalty = 1.0
            
            # Combined score: baseline slope + tail uncertainty penalty
            score = baseline_score + tail_penalty
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)
