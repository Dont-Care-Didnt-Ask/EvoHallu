import numpy as np
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detects hallucinations using a robust composite uncertainty score.
        
        Approach: Computes four features (Global NLL, Local Variance, Tail Severity, 
        Max Local Deviation) and normalizes them using Median and MAD (Median Absolute Deviation)
        to ensure robustness against batch outliers. The final score is a weighted sum of these 
        normalized Z-scores.
        
        Hypothesis: Hallucinated sequences statistically deviate from the median behavior of 
        the batch in terms of cumulative surprise, local probability instability, and 
        extreme low-confidence events, which robust normalization effectively isolates.
        """
        sequences = stats["sequence_logprobs"]
        N = len(sequences)
        if N == 0:
            return np.array([], dtype=np.float32)

        # Helper for robust normalization: Z = (x - median) / (MAD * 1.4826)
        def robust_norm(arr):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            if mad < 1e-8:
                return np.zeros_like(arr, dtype=np.float32)
            return (arr - med) / (mad * 1.4826)

        # Feature 1: Global NLL (Sum of negative log-probs)
        # Higher = more surprising
        f1 = np.array([-np.sum(seq) if len(seq) > 0 else 0.0 for seq in sequences], dtype=np.float32)
        
        # Feature 2: Local Variance (Mean variance in windows of size 5)
        f2 = np.zeros(N, dtype=np.float32)
        w_size = 5
        for i, seq in enumerate(sequences):
            if len(seq) <= w_size:
                f2[i] = np.var(seq) if len(seq) > 0 else 0.0
            else:
                vars = []
                for start in range(0, len(seq) - w_size + 1, 2):
                    vars.append(np.var(seq[start:start+w_size]))
                f2[i] = np.mean(vars)

        # Feature 3: Tail Severity (Mean of bottom 10% log-probs)
        # Higher (less negative) = more severe low confidence
        f3 = np.array([
            np.mean(np.sort(seq)[-int(len(seq)*0.1):]) if len(seq) > 0 else 0.0
            for seq in sequences
        ], dtype=np.float32)

        # Feature 4: Max Local Deviation (Max absolute difference from local mean)
        # Captures sudden "jumps" in probability distribution
        f4 = np.zeros(N, dtype=np.float32)
        for i, seq in enumerate(sequences):
            if len(seq) <= 5:
                f4[i] = np.max(np.abs(seq)) if len(seq) > 0 else 0.0
            else:
                max_dev = 0.0
                for start in range(0, len(seq) - 5 + 1, 2):
                    sub = seq[start:start+5]
                    local_mean = np.mean(sub)
                    dev = np.max(np.abs(sub - local_mean))
                    if dev > max_dev:
                        max_dev = dev
                f4[i] = max_dev

        # Normalize all features robustly
        z1 = robust_norm(f1)
        z2 = robust_norm(f2)
        z3 = robust_norm(f3)
        z4 = robust_norm(f4)

        # Fixed weights: NLL is most important, followed by variance and tail.
        weights = np.array([0.40, 0.25, 0.20, 0.15], dtype=np.float32)
        
        final_scores = (
            weights[0] * z1 +
            weights[1] * z2 +
            weights[2] * z3 +
            weights[3] * z4
        )

        return final_scores
