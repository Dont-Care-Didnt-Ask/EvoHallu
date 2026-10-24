import numpy as np
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
        """
        Detects hallucinations by fusing global NLL, local log-prob variance, 
        and Top-K distribution flatness.
        
        Approach:
        1. Compute Total Negative Log-Probability (NLL) as a baseline for sequence likelihood.
        2. Compute normalized variance of log-probs to capture local confidence instability.
        3. Compute Top-K Flatness: Average difference between Top-1 logit and the mean of Top-K logits.
           High flatness implies the model lacks a clear preference among plausible candidates.
        4. Fuse signals with fixed weights: Score = 0.5*NLL + 1.0*NormVariance + 2.0*TopKFlatness.
        
        Hypothesis for future: Hallucinated sequences exhibit simultaneously high global negative log-probability, 
        high local variance in token-level confidence, and a consistently "flat" Top-K distribution where the 
        selected token is rarely the unique maximum, compared to factual sequences which show sharper peaks 
        in the Top-K window.
        
        Args:
            stats: Dictionary containing 'sequence_logprobs', 'top_k_logits', 'top_k_indices'.
        
        Returns:
            np.ndarray of shape [N] with fused hallucination scores.
        """
        log_probs_list = stats["sequence_logprobs"]
        top_k_logits_list = stats["top_k_logits"]
        top_k_indices_list = stats["top_k_indices"]
        
        if len(log_probs_list) == 0:
            return np.array([], dtype=np.float32)
        
        N = len(log_probs_list)
        scores = np.zeros(N, dtype=np.float32)
        
        # Fixed weights for the composite signal
        w_nll = 0.5
        w_var = 1.0
        w_flat = 2.0
        
        for i in range(N):
            lp = log_probs_list[i]
            k_logits = top_k_logits_list[i]
            k_indices = top_k_indices_list[i]
            
            n_tokens = len(lp)
            if n_tokens == 0:
                scores[i] = 0.0
                continue
            
            # 1. Global Uncertainty: Total Negative Log-Probability
            total_nll = -np.sum(lp)
            
            # 2. Local Instability: Normalized Variance of Log-Pros
            # Normalize by sequence length to make it comparable across different lengths
            var_lp = np.var(lp)
            norm_var = var_lp / (n_tokens + 1e-8)
            
            # 3. Top-K Flatness: How "spread out" is the probability mass in the top-k?
            # We use logits directly. A flat distribution means the max logit is close to the mean logit.
            # Metric: Mean( max(logits) - mean(logits) ) over the sequence.
            # If the model is confident, max >> mean. If hallucinating/unsure, max ~ mean.
            flatness_scores = []
            for t_idx in range(n_tokens):
                # k_logits shape is (n_tokens, k). We are at step t_idx.
                logits = k_logits[t_idx]
                max_logit = np.max(logits)
                mean_logit = np.mean(logits)
                flatness_diff = max_logit - mean_logit
                flatness_scores.append(flatness_diff)
            
            avg_flatness = np.mean(flatness_scores)
            
            # Fuse signals
            # High NLL -> High Score
            # High Variance -> High Score
            # High Flatness (low confidence in top-k) -> High Score
            score = (w_nll * total_nll) + (w_var * norm_var) + (w_flat * avg_flatness)
            
            scores[i] = score
            
        return scores
