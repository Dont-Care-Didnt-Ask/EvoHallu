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
        Detects hallucinations by combining Total NLL, average surprise gap, 
        and a 'Local Confidence Transition' metric that penalizes sudden shifts 
        from high-confidence to low-confidence states.
        
        Approach:
        1. Calculate Total Negative Log Probability (NLL) as the baseline.
        2. Compute the 'Surprise Gap' for each token: (Top-1 LogProb - Generated LogProb).
        3. Calculate the 'Local Confidence Transition': The difference in Surprise Gap 
           between consecutive tokens. Large absolute changes indicate unstable reasoning.
        4. Fuse signals: Score = NLL + alpha * Mean_Gap + beta * Mean_Abs_Transition.
        
        Hypothesis for future: Hallucinated sequences exhibit sharper transitions in 
           confidence levels (specifically drops from high to low confidence) compared 
           to factual sequences, which tend to maintain a more consistent level of 
           certainty throughout the generation process.
        
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
        
        for i in range(N):
            lp = log_probs_list[i]
            k_logits_list = top_k_logits_list[i]
            
            n_tokens = len(lp)
            if n_tokens == 0:
                scores[i] = 0.0
                continue
            
            # 1. Baseline: Total Negative Log Prob
            total_neg_logprob = -np.sum(lp)
            
            # 2. Compute Surprise Gaps and Transitions
            gaps = np.zeros(n_tokens, dtype=np.float32)
            transitions = np.zeros(n_tokens, dtype=np.float32)
            
            for t in range(n_tokens):
                # Log prob of generated token
                gen_logp = lp[t]
                
                # Compute Top-1 LogProb for this position
                logits = k_logits_list[t]
                max_logit = np.max(logits)
                exp_shifted = np.exp(logits - max_logit)
                log_sum_exp = max_logit + np.log(np.sum(exp_shifted))
                log_probs_top_k = logits - log_sum_exp
                
                top_1_logp = np.max(log_probs_top_k)
                
                # Surprise Gap
                gaps[t] = top_1_logp - gen_logp
                
                # Local Transition: Difference in gap from previous token
                if t > 0:
                    transitions[t] = abs(gaps[t] - gaps[t-1])
                else:
                    transitions[t] = 0.0 # No transition for the first token
            
            # 3. Aggregate Signals
            mean_gap = np.mean(gaps)
            mean_transition = np.mean(transitions)
            
            # 4. Fuse Signals
            # Weights tuned to balance the scale of NLL (usually larger) with gap/transition
            scores[i] = total_neg_logprob + 2.5 * mean_gap + 1.5 * mean_transition
            
        return scores
