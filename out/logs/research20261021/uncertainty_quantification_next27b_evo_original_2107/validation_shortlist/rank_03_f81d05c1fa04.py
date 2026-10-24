import numpy as np
from typing import Dict, Any
class HallucinationDetector(Estimator):
    """
    Detects hallucinations by combining the LogProbSlope (negative Pearson correlation 
    of generated token log-probs) with the Mean Top-1 Probability from the top-k logits.
    
    Approach:
    1. Calculate the LogProbSlope: Negative Pearson correlation between token position 
       and the log-probabilities of the generated tokens. This captures the trend in 
       confidence over the sequence.
    2. Calculate the Mean Top-1 Probability: For each token position, compute the 
       softmax probability of the top-1 token within the top-k logits. Average this 
       over the sequence. This captures the overall level of certainty.
    3. Combine these signals: Hallucinations are hypothesized to exhibit either a 
       decaying confidence trend (negative slope) or lower overall certainty (lower 
       top-1 probability). The score is a weighted sum of the negative slope and 
       (1 - mean top-1 probability).
    
    Hypothesis: Hallucinated sequences often show a combination of decreasing 
    confidence over time and lower peak probability per token compared to factual 
    sequences, which tend to be more consistently confident.
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
        self.w_slope = 0.6
        self.w_conf = 0.4

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            # 1. LogProbSlope
            lp = np.asarray(stats["sequence_logprobs"][i], dtype=np.float64)
            n = len(lp)
            
            slope_score = 0.0
            if n >= 2:
                positions = np.arange(n, dtype=np.float64)
                if np.std(lp) > 1e-10:
                    corr = np.corrcoef(positions, lp)[0, 1]
                    if not np.isnan(corr):
                        slope_score = -corr # Negative correlation -> positive score
            
            # 2. Mean Top-1 Probability
            top_k_logits = np.asarray(stats["top_k_logits"][i], dtype=np.float64)
            # top_k_logits shape: (n_i, k)
            # Note: sequence_logprobs length is n_i - 1. 
            # top_k_logits likely corresponds to the generated tokens.
            # If lengths differ, we align to the shorter one or assume top_k_logits 
            # covers the same tokens as sequence_logprobs (excluding the first one if necessary).
            # The prompt says: "sequence_logprobs": ... excluding the first generated token.
            # "top_k_logits": ... top-k logits per token.
            # Usually, top_k_logits[i] corresponds to the logits used to generate token i+1.
            # So if sequence_logprobs has length L, top_k_logits should have length L.
            
            if top_k_logits.size == 0:
                conf_score = 0.0
            else:
                # Compute softmax probabilities for each token position
                # Subtract max for numerical stability
                logits = top_k_logits
                max_logits = np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits - max_logits)
                sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
                probs = exp_logits / sum_exp
                
                # Top-1 probability is the probability of the first column (assuming sorted or just the max)
                # The prompt says "top_k_indices", implying the logits are for the top-k tokens.
                # It does not explicitly state they are sorted. 
                # However, "top_k_logits" usually implies the logits of the top-k tokens.
                # To be safe, we should find the max probability in each row, 
                # but typically top-k are sorted. Let's assume the max is the top-1.
                # If they are not sorted, we take the max.
                top1_probs = np.max(probs, axis=1)
                
                mean_top1_prob = np.mean(top1_probs)
                # We want higher score for lower confidence (higher hallucination likelihood)
                # So we use (1 - mean_top1_prob)
                conf_score = 1.0 - mean_top1_prob
            
            # Combine
            scores[i] = self.w_slope * slope_score + self.w_conf * conf_score
            
        return scores
