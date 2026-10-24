"""Predeclared controls for the supplied cache's template/claim boundary.

The legacy crop matches end2end_eval_original.py. Exact controls locate the
claim token subsequence, allowing for tokenizer-specific suffix lengths.
These controls do not alter any features supplied to evolved candidates.
"""
import numpy as np


def legacy_tail_crop_likelihoods(logprobs, claim_tokens):
    n = len(claim_tokens)
    return np.asarray(logprobs)[-n-1:-1] if len(logprobs) > n else np.asarray(logprobs)


def claim_span(input_tokens, claim_tokens):
    a, b = np.asarray(input_tokens).tolist(), np.asarray(claim_tokens).tolist()
    matches = [i for i in range(len(a)-len(b)+1) if a[i:i+len(b)] == b]
    if len(matches) != 1:
        raise ValueError(f"Claim alignment requires a unique token span; found {len(matches)}")
    return matches[0], matches[0] + len(b)


def claim_likelihoods(logprobs, claim_tokens, input_tokens):
    start, end = claim_span(input_tokens, claim_tokens)
    return np.asarray(logprobs)[max(0, start-1):end-1]


class ClaimOnlySequenceLogProb:
    def __call__(self, stats):
        return np.array([-claim_likelihoods(lp, tokens, full).astype(float).sum()
                         for lp, tokens, full in zip(stats["sequence_logprobs"], stats["greedy_tokens"], stats["input_tokens"])])


class ClaimOnlyMeanSurprisal:
    def __call__(self, stats):
        return np.array([-claim_likelihoods(lp, tokens, full).astype(float).mean()
                         for lp, tokens, full in zip(stats["sequence_logprobs"], stats["greedy_tokens"], stats["input_tokens"])])


class LastInputTokenSurprisal:
    def __call__(self, stats):
        return np.array([-np.asarray(lp)[-1] for lp in stats["sequence_logprobs"]])


class PostClaimSurprisal:
    def __call__(self, stats):
        result = []
        for lp, tokens, full in zip(stats["sequence_logprobs"], stats["greedy_tokens"], stats["input_tokens"]):
            _, end = claim_span(full, tokens)
            result.append(-np.asarray(lp)[end-1:].astype(float).sum())
        return np.asarray(result)


class LegacyTailCropSequenceLogProb:
    def __call__(self, stats):
        return np.array([-legacy_tail_crop_likelihoods(lp, tokens).astype(float).sum()
                         for lp, tokens in zip(stats["sequence_logprobs"], stats["greedy_tokens"])])


class ClaimLength:
    def __call__(self, stats):
        return np.array([len(tokens) for tokens in stats["greedy_tokens"]], dtype=float)
