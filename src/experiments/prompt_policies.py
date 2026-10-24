"""Opt-in prompt ablations; preserve the historical prompt by default."""
from __future__ import annotations

import copy
import re


POLICIES = ("original", "neutral", "diverse", "compositional")
FAMILIES = (
    "confidence gaps between the observed token and alternatives",
    "contiguous runs or local windows of token surprisal",
    "concentration or shape of the top-k distribution",
    "positional changes in confidence across the sequence",
    "robust quantile or trimmed aggregation of uncertainty",
)


def candidate_prompt(chat, policy="original", round_index=0, index=0):
    """Return a separate chat for one call; never mutate the parent context."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown prompt policy: {policy}")
    result = copy.deepcopy(chat)
    if policy == "original":
        return result
    system = next(m for m in result if m["role"] == "system")
    text, replacements = re.subn(
        r"\*\*EXPECTED STRUCTURE\*\*\n- Build off a strong baseline: [^\n]+\n"
        r"- For additional signal, use [^\n]+",
        "Choose a deterministic scoring formula using the available inputs.\n"
        "The signal family, aggregation principle, and formula structure are open.\n"
        "Use the shown training results as evidence when choosing your approach.",
        system["content"],
    )
    if replacements != 1:
        raise ValueError("Neutral-policy ablation expects exactly one historical formula-guidance block")
    text = text.replace(
        "Ideate a novel detector, based on EXPECTED STRUCTURE and insights from your analysis above.",
        "Propose a detector and briefly state its signal and aggregation principle.",
    )
    if "EXPECTED STRUCTURE" in text:
        raise ValueError("Unrecognized response-format reference to historical guidance")
    if policy == "compositional":
        text += (
            "\n\n## Composite-detector proposal\n"
            "Explore a weighted combination of several complementary uncertainty features. "
            "Consider quantiles, local windows, low-confidence runs, final-segment statistics, "
            "positional effects, and top-k distribution shape. Select useful components and "
            "define their scaling and weights explicitly. There is no fixed feature-count limit. "
            "Use fixed deterministic formulas and coefficients; labels and fitted model weights "
            "are unavailable to this unsupervised detector. Explain how the terms complement one another.\n"
        )
    elif policy == "diverse":
        if index % 2 == 0:
            instruction = (
                "Local-refinement role: improve one shown method with a focused change to its "
                "aggregation, weighting, or coefficients. Identify the specific change."
            )
        else:
            instruction = (
                "Exploration role: propose a different primary signal or aggregation principle "
                "from the shown methods. Explore " + FAMILIES[round_index % len(FAMILIES)] + ". "
                "If that family is already represented, choose a different aggregation principle. "
                "Briefly state the structural difference."
            )
        text += "\n\n## Candidate-specific role\n" + instruction + "\n"
    system["content"] = text
    return result
