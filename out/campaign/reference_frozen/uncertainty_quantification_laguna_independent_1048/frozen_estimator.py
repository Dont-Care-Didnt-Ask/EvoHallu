
class HallucinationDetector(Estimator):
    """
    Combines LogProbSlope baseline with top-k margin analysis focusing on bottom 20% of tokens.
    
    Hypothesis: Hallucinations manifest as tokens with low probability relative to top-k alternatives,
    particularly in sequence positions where the model's confidence naturally decays.
    """
    
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs", "top_k_logits", "top_k_indices"], level="sequence")
    
    def __call__(self, stats):
        sequence_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        top_k_indices = stats["top_k_indices"]
        
        scores = []
        
        for i in range(len(sequence_logprobs)):
            logprobs = sequence_logprobs[i]
            logits = top_k_logits[i]
            indices = top_k_indices[i]
            
            n_tokens = len(logprobs)
            if n_tokens < 2:
                scores.append(0.0)
                continue
            
            # Compute LogProbSlope baseline
            positions = np.arange(n_tokens)
            if np.std(logprobs) < 1e-10:
                slope_score = 0.0
            else:
                slope_score = -np.corrcoef(positions, logprobs)[0, 1]
                if np.isnan(slope_score):
                    slope_score = 0.0
            
            # Compute top-k margins for each token
            margins = []
            for j in range(n_tokens):
                token_logit = logits[j]  # logit of the actual generated token
                top_logits = logits[j]  # top-k logits for this position
                
                # Find the margin: actual token logit vs max alternative logit
                actual_idx_in_topk = np.where(indices[j] == indices[j][0])[0]
                if len(actual_idx_in_topk) > 0:
                    actual_logit = top_logits[actual_idx_in_topk[0]]
                else:
                    actual_logit = token_logit
                
                # Get max logit among alternatives (excluding the actual token)
                mask = np.ones(len(top_logits), dtype=bool)
                mask[actual_idx_in_topk[0]] = False if len(actual_idx_in_topk) > 0 else True
                if np.any(mask):
                    max_alt_logit = np.max(top_logits[mask])
                else:
                    max_alt_logit = actual_logit
                
                margin = actual_logit - max_alt_logit
                margins.append(margin)
            
            margins = np.array(margins)
            
            # Focus on bottom 20% of tokens by margin (lowest confidence)
            n_bottom = max(1, int(0.2 * n_tokens))
            bottom_margins = np.partition(margins, n_bottom)[:n_bottom]
            avg_bottom_margin = np.mean(bottom_margins)
            
            # Combine: penalize low margins in bottom 20%, weighted by slope
            # More negative margin = worse, so we negate it
            margin_score = -avg_bottom_margin
            
            # Final score: weighted combination
            # High slope (confidence decay) + low margins = high hallucination risk
            final_score = slope_score + 0.1 * margin_score
            
            scores.append(final_score)
        
        return np.array(scores)
