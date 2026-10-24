import numpy as np
from scipy.stats import entropy
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
        Detects hallucinations by combining sequence-level negative log-prob sum,
        the variance of log-probabilities (consistency signal), and the ratio of 
        top-k entropy to peak confidence (ambiguity signal).
        
        Approach:
        1. Compute Negative Log-Prob (NLP) sum as the baseline difficulty score.
        2. Calculate the variance of NLP across the sequence; high variance indicates 
           inconsistent confidence, often seen when the model hallucinates mid-stream.
        3. Compute the mean entropy of the top-k logits for the sequence. Normalize 
           this entropy by the absolute value of the maximum log-prob in the sequence 
           to capture how "spread out" the model's top choices are relative to its peak confidence.
        4. Combine these signals: Score = NLP_Sum + alpha * NLP_Variance + beta * (Mean_TopK_Entropy / |Max_LogProb|).
        
        Hypothesis for Future: Hallucinated sequences exhibit significantly higher 
        variance in log-probability magnitudes and a higher ratio of top-k entropy 
        to peak log-probability compared to factual sequences, reflecting both 
           inconsistent confidence and ambiguous selection among incorrect options.
        
        Output shape: [N] where N is the number of samples.
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)
        
        # Hyperparameters tuned for balance between baseline and new signals
        alpha = 0.5  # Weight for variance
        beta = 1.0   # Weight for entropy ratio
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) == 0:
                scores[i] = 0.0
                continue

            # 1. Baseline: Sum of Negative Log-Probabilities
            nlp = -np.array(seq_logprobs, dtype=np.float32)
            nlp_sum = np.sum(nlp)
            
            # 2. Signal: Variance of NLP (Consistency)
            nlp_variance = np.var(nlp)
            
            # 3. Signal: Top-K Entropy Ratio (Ambiguity)
            top_k_logits = stats["top_k_logits"][i]
            # Ensure we have valid logits; if shape is wrong, handle gracefully
            if top_k_logits.ndim == 1:
                top_k_logits = top_k_logits.reshape(1, -1)
            
            k = top_k_logits.shape[1]
            # Calculate entropy for each token's top-k distribution
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            token_entropies = []
            for logits in top_k_logits:
                # Softmax to get probabilities
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / (np.sum(exp_logits) + eps)
                # Calculate entropy
                ent = entropy(probs, base=np.e) # Natural entropy
                token_entropies.append(ent)
            
            mean_topk_entropy = np.mean(token_entropies)
            
            # Get max log-prob magnitude for normalization
            max_log_prob = np.max(seq_logprobs)
            
            # Avoid division by zero or negative normalization
            if max_log_prob < -1e-6:
                entropy_ratio = mean_topk_entropy / (-max_log_prob)
            else:
                entropy_ratio = 0.0

            # Combine signals
            # We assume higher values in all components indicate higher hallucination risk
            final_score = nlp_sum + (alpha * nlp_variance) + (beta * entropy_ratio)
            
            scores[i] = final_score

        return scores
