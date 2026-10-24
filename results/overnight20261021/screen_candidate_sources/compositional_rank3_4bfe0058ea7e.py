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
        Detects hallucinations using a composite of tail instability, low-confidence run lengths, 
        top-k entropy, and positional decay.
        
        Approach:
        1. Tail Instability: Std dev of log-probs in the last 20% of the sequence.
        2. Low-Confidence Runs: Max length of consecutive tokens where max-logit < -2.0.
        3. Top-K Entropy: Average entropy of the top-512 logits per token.
        4. Positional Decay: Difference in mean log-prob between first and second half.
        
        Signals are normalized by sequence length and fused with fixed weights.
        
        Hypothesis for future: Hallucinated sequences exhibit a specific pattern of 
        "high entropy but low confidence" in the tail, combined with sudden drops 
        in local confidence (low-confidence runs), which simple global NLL misses.
        
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
        
        # Hyperparameters for normalization and weighting
        tail_ratio = 0.2
        low_conf_threshold = -2.0
        top_k_size = 512
        
        for i in range(N):
            lp = log_probs_list[i]
            k_logits = top_k_logits_list[i]
            k_indices = top_k_indices_list[i]
            
            seq_len = len(lp)
            if seq_len == 0:
                scores[i] = 0.0
                continue
            
            # 1. Tail Instability (Std in last 20%)
            tail_start = max(0, int(seq_len * (1 - tail_ratio)))
            tail_lp = lp[tail_start:]
            if len(tail_lp) > 0:
                tail_std = np.std(tail_lp)
            else:
                tail_std = 0.0
            
            # 2. Low-Confidence Run Length
            # Calculate max logit for each token
            max_logits = np.max(k_logits, axis=1)
            is_low_conf = max_logits < low_conf_threshold
            
            # Find max consecutive run
            max_run = 0
            current_run = 0
            for val in is_low_conf:
                if val:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            low_conf_score = max_run / seq_len
            
            # 3. Top-K Entropy
            # Entropy = -sum(p * log(p))
            # p = softmax(logits) approximated by normalizing top-k
            # To avoid overflow/underflow, we can use log-sum-exp trick or direct softmax if logits are reasonable
            # Given k=512, direct softmax is usually safe enough for log-prob estimation, 
            # but let's use a stable log-sum-exp approach for p calculation.
            # Actually, simpler: p_i = exp(logit_i) / sum(exp(logits))
            
            entropies = []
            for logits in k_logits:
                # Shift logits to avoid overflow
                logits_shifted = logits - np.max(logits)
                exp_logits = np.exp(logits_shifted)
                sum_exp = np.sum(exp_logits)
                if sum_exp == 0:
                    entropies.append(0.0)
                    continue
                
                probs = exp_logits / sum_exp
                # Filter out zero probs to avoid log(0)
                valid_probs = probs[probs > 0]
                if len(valid_probs) == 0:
                    entropies.append(0.0)
                    continue
                
                entropy = -np.sum(valid_probs * np.log(valid_probs))
                entropies.append(entropy)
            
            avg_entropy = np.mean(entropies) if entropies else 0.0
            
            # 4. Positional Decay
            split_idx = seq_len // 2
            if split_idx > 0:
                first_half_mean = np.mean(lp[:split_idx])
                second_half_mean = np.mean(lp[split_idx:])
                decay = first_half_mean - second_half_mean # Positive if confidence drops
            else:
                decay = 0.0
            
            # Normalize features roughly to [0, 1] range for fusion
            # Heuristic normalization based on typical ranges:
            # Tail std: ~0-2, Low conf ratio: 0-1, Entropy: 0-6.4, Decay: -inf to 0
            
            # Feature scaling factors (empirical estimates)
            f1 = tail_std / 2.0 
            f2 = low_conf_score 
            f3 = avg_entropy / 6.4 
            f4 = max(0, decay) / 2.0 # Only penalize decay
            
            # Weighted Sum
            # Weights chosen to balance global vs local signals
            w1 = 1.5
            w2 = 2.0 # Run lengths are strong indicators
            w3 = 1.0
            w4 = 1.5
            
            score = w1*f1 + w2*f2 + w3*f3 + w4*f4
            scores[i] = score
            
        return scores
