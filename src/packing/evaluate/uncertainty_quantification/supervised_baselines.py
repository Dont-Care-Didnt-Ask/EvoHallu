"""Small supervised comparators with fixed, interpretable feature extraction."""
import numpy as np
from scipy.special import softmax


FEATURE_NAMES = [
    "length", "nll_sum", "nll_mean", "nll_std", "nll_max", "nll_q90", "nll_q10",
    "nll_position_slope", "nll_exp_weighted", "entropy_mean", "entropy_std",
    "entropy_last_minus_first", "top1_mass_top512_mean", "top1_top2_gap_mean",
    "top512_entropy_mean", "nll_last_quarter", "nll_first_quarter",
]
DOCUMENTED_FEATURE_INDICES = [i for i, name in enumerate(FEATURE_NAMES) if not name.startswith("entropy_")]
DOCUMENTED_FEATURE_NAMES = [FEATURE_NAMES[i] for i in DOCUMENTED_FEATURE_INDICES]


def compact_features(stats, feature_set="full"):
    if feature_set not in {"full", "documented"}:
        raise ValueError(f"Unknown feature set: {feature_set}")
    rows = []
    entropies = stats["token_entropies"] if feature_set == "full" else [None]*len(stats["sequence_logprobs"])
    for lp, entropy, top in zip(stats["sequence_logprobs"], entropies, stats["top_k_logits"]):
        nll = -np.asarray(lp, dtype=np.float64)
        e = np.asarray(entropy, dtype=np.float64) if feature_set == "full" else np.zeros(1)
        top = np.asarray(top, dtype=np.float64)
        probs = softmax(top, axis=1)
        length = len(nll)
        if not length or not len(e) or not len(top):
            raise ValueError("Empty token statistics")
        position = np.linspace(0, 1, length)
        centered = position - position.mean()
        slope = np.dot(nll - nll.mean(), centered) / max(np.dot(centered, centered), 1e-12)
        weights = np.expm1(2.5 * position)
        if weights.sum() == 0:
            weights = np.ones(length)
        q = max(1, length // 4)
        eq = max(1, len(e) // 4)
        row = [length, nll.sum(), nll.mean(), nll.std(), nll.max(), np.quantile(nll, .9),
               np.quantile(nll, .1), slope, np.average(nll, weights=weights), e.mean(), e.std(),
               e[-eq:].mean() - e[:eq].mean(), probs[:, 0].mean(),
               (top[:, 0] - top[:, 1]).mean(),
               -(probs * np.log(np.maximum(probs, 1e-300))).sum(axis=1).mean(),
               nll[-q:].mean(), nll[:q].mean()]
        rows.append(row)
    result = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite compact features")
    return result[:, DOCUMENTED_FEATURE_INDICES] if feature_set == "documented" else result


class FittedDetector:
    def __init__(self, model):
        self.model = model
        self.feature_set = getattr(model, "mod1_feature_set", "full")

    def __call__(self, stats):
        return self.score_features(compact_features(stats, self.feature_set))

    def score_features(self, features):
        if hasattr(self.model, "decision_function"):
            # AUC needs ordering. Sigmoid probabilities can round to 1 on long
            # out-of-domain claims even when decision scores remain distinct.
            return self.model.decision_function(features)
        return self.model.predict_proba(features)[:, 1]
