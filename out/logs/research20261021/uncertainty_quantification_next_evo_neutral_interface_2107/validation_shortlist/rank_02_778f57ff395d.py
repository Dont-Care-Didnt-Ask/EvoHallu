import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import entropy
class HallucinationDetector(Estimator):
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
        # Convert inputs to float64 for numerical stability
        logprobs = [np.array(lp, dtype=np.float64) for lp in stats["sequence_logprobs"]]
        logits = [np.array(lg, dtype=np.float64) for lg in stats["top_k_logits"]]
        
        N = len(logprobs)
        scores = np.zeros(N, dtype=np.float64)
        
        for i in range(N):
            lp = logprobs[i]
            lg = logits[i]
            
            L = len(lp)
            if L == 0:
                scores[i] = 0.0
                continue
            
            # 1. Mean Negative Log-Probability (Magnitude of surprise)
            mean_nll = -np.mean(lp)
            
            # 2. Mean Top-K Entropy (Model uncertainty/confusion)
            # Compute logsumexp for normalization
            # Handle potential -inf if logits are all -inf (rare but possible)
            lse = np.array([logsumexp(lg_t) for lg_t in lg], dtype=np.float64)
            
            # Compute softmax probabilities
            # Avoid division by zero or overflow in softmax by subtracting max
            # scipy.special.softmax handles this, but let's be explicit if needed
            # Actually, scipy.special.softmax is robust.
            probs = softmax(lg, axis=1)
            
            # Compute entropy for each step
            # entropy function expects probabilities summing to 1
            # Handle rows where sum(probs) might be slightly off due to float precision or if all logits are -inf
            entropies = np.array([entropy(p) for p in probs], dtype=np.float64)
            
            mean_entropy = np.mean(entropies)
            
            # Combine signals: High surprise + High confusion = High Hallucination Risk
            # Weight entropy heavily as it captures the "instability" of the generation
            score = mean_nll + 2.0 * mean_entropy
            
            scores[i] = score
            
        return scores
