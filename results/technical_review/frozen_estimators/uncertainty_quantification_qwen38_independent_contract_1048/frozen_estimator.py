import numpy as np
from scipy import stats as sp_stats
class HallucinationDetector(Estimator):
    """
    Combines LogProbSlope (negative Pearson correlation of position vs log-prob)
    with a position-weighted tail-mass entropy estimated from top-512 logits.
    Tail mass is approximated by contrasting top-16 logits against the remaining 496.
    Hypothesis for future: incorporating variance of top-k logit gaps could better
    separate confidently-wrong hallucinations from genuinely uncertain generations.
    """

    def __init__(self, context):
        super().__init__(
            context,
            stats_dependencies=["sequence_logprobs", "top_k_logits", "top_k_indices"],
            level="sequence",
        )
        self.top_n = 16
        self.alpha = 0.6
        self.beta = 0.4

    def __call__(self, stats):
        sequence_logprobs = stats["sequence_logprobs"]
        top_k_logits = stats["top_k_logits"]
        top_k_indices = stats["top_k_indices"]

        N = len(sequence_logprobs)
        scores = np.zeros(N, dtype=np.float64)

        for i in range(N):
            logprobs = np.asarray(sequence_logprobs[i], dtype=np.float64)
            logits = np.asarray(top_k_logits[i], dtype=np.float64)
            n_tokens = len(logprobs)

            if n_tokens < 2:
                scores[i] = 0.0
                continue

            # --- LogProbSlope: negative Pearson correlation between position and log-prob ---
            positions = np.arange(n_tokens, dtype=np.float64)
            if np.std(logprobs) < 1e-12:
                slope_score = 0.0
            else:
                corr, _ = sp_stats.pearsonr(positions, logprobs)
                if np.isnan(corr):
                    corr = 0.0
                slope_score = -corr  # negative correlation => positive score

            # --- Position-weighted tail mass ---
            # For each token position, compute tail mass: 1 - softmax(top-16 logits)
            # Use log-sum-exp for numerical stability
            tail_masses = np.zeros(n_tokens, dtype=np.float64)
            for t in range(n_tokens):
                tok_logits = logits[t]  # shape (k,)
                if tok_logits.size < self.top_n + 1:
                    tail_masses[t] = 0.0
                    continue
                top_part = tok_logits[: self.top_n]
                rest_part = tok_logits[self.top_n :]

                # log-softmax over all k logits
                max_logit = np.max(tok_logits)
                exp_logits = np.exp(tok_logits - max_logit)
                sum_exp = np.sum(exp_logits)
                if sum_exp < 1e-300:
                    tail_masses[t] = 0.0
                    continue
                top_prob_sum = np.sum(np.exp(top_part - max_logit)) / sum_exp
                tail_masses[t] = 1.0 - top_prob_sum

            # Position-weighted: later tokens weighted more heavily
            if n_tokens > 1:
                weights = positions / positions[-1]  # linearly increasing from ~0 to 1
            else:
                weights = np.ones(n_tokens)
            weighted_tail = np.sum(weights * tail_masses) / np.sum(weights) if np.sum(weights) > 0 else 0.0

            # Combine
            scores[i] = self.alpha * slope_score + self.beta * weighted_tail

        # Ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores
