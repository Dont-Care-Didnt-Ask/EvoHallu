from __future__ import annotations

import numpy as np


def render(tokenizer, claim: str, model_key: str):
    messages = [
        {"role": "user", "content": "Generate true statement"},
        {"role": "assistant", "content": claim},
    ]
    options = {"enable_thinking": False} if model_key == "gemma" else {}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **options
    )
    span = claim if claim in text else claim.strip()
    start = text.rfind(span)
    if start < 0 or not span:
        raise ValueError("The chat template did not preserve the claim")
    end = start + len(span)
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    targets = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and left >= start and right <= end
    ]
    overlaps_boundary = any(
        right > left and left < end and right > start
        and not (left >= start and right <= end)
        for left, right in offsets
    )
    if not targets or targets[0] == 0 or overlaps_boundary:
        raise ValueError("Claim boundaries do not align with token boundaries")
    return text, encoded["input_ids"], targets


def features(logits, token_ids, claim_indices):
    import torch

    logits = logits.float()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(-1)
    greedy = log_probs.max(-1).values
    sequence = log_probs[:-1].gather(-1, token_ids[1:, None]).squeeze(-1)
    top = logits.topk(512, dim=-1)
    # A claim token at position t is predicted by logits at position t - 1.
    predictors = torch.tensor(claim_indices, device=logits.device) - 1

    def array(value):
        return value.detach().cpu().numpy()

    result = {
        "sequence_logprobs": array(sequence),
        "greedy_logprobs": array(greedy),
        "token_entropies": array(entropy),
        "top_k_logits": array(top.values.half()),
        "input_ids": array(token_ids.int()),
        "top_k_indices": array(top.indices.int()),
        "claim_indices": np.asarray(claim_indices, dtype=np.int32),
        "claim_sequence_logprobs": array(
            log_probs[predictors].gather(-1, token_ids[predictors + 1, None]).squeeze(-1)
        ),
        # Existing greedy-based formulas discard g[0]; keep that interface unchanged.
        "claim_greedy_logprobs": np.concatenate(
            [np.zeros(1, dtype=np.float32), array(greedy[predictors])]
        ),
        "claim_token_entropies": array(entropy[predictors]),
        "claim_top_k_logits": array(top.values[predictors].half()),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Model features contain non-finite values")
    return result
