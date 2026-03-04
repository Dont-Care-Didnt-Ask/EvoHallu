from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import logging
import time


def _get_attention_features(attn_mask: np.ndarray, n_tokens: int, max_n_tokens: int = 10):
    """
    Port of HalluDetect.get_attention_features.

    Parameters
    ----------
    attn_mask:
        Attention mask of shape [num_layers * num_heads, n_tokens, n_tokens].
    n_tokens:
        Number of claim tokens (without chat template).
    """
    current_top_n = min(max_n_tokens, n_tokens)
    attn_features_seq = []
    for j in range(1, n_tokens):
        attn_features_tok = []
        for k in range(min(j, current_top_n)):
            attn_features_tok.append(attn_mask[:, j, j - k - 1].tolist())
        attn_features_seq.append(attn_features_tok)
    return attn_features_seq

def get_attentions(attn_mask: torch.Tensor, num_layers: int, num_heads: int, n_tokens: int, orig_n_tokens: int, max_n_tokens: int = 10):
    attn_mask = attn_mask.to(torch.float16).cpu().numpy()
    if n_tokens > (orig_n_tokens + 1):
        # attentions are cropped because of the eos token
        attn_mask = attn_mask[:, :, :-1, :-1]
        attn_mask_reshape = attn_mask.reshape(num_layers * num_heads, n_tokens - 1, n_tokens - 1)
    else:
        attn_mask_reshape = attn_mask.reshape(num_layers * num_heads, n_tokens, n_tokens)
        
    # use only claim tokens without chat template
    # shift attention mask by 1 token to align it with the generation format from lm-polygraph
    attn_mask_reshape = attn_mask_reshape[:, -orig_n_tokens-1:-1, -orig_n_tokens:]

    attn_features_seq = _get_attention_features(attn_mask_reshape, orig_n_tokens)
    return attn_mask, attn_mask_reshape, attn_features_seq


def _compute_attentions(
    attn: torch.Tensor,
    num_layers: int,
    num_heads: int,
    n_tokens: int,
    orig_n_tokens: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[List[float]]]]:
    """
    Reimplementation of HalluDetect.get_attentions but operating on stored
    attention tensors instead of live model outputs.

    Parameters
    ----------
    attn:
        Attention tensor for a single sample of shape
        [num_layers, num_heads, n_tokens, n_tokens].
    """
    print(f"DEBUG _compute_attentions: attn.shape = {attn.shape}, num_layers = {num_layers}, num_heads = {num_heads}, n_tokens = {n_tokens}, orig_n_tokens = {orig_n_tokens}")
    # Validate input tensor shape
    if attn.dim() != 4:
        logging.error(f"Expected 4D tensor, got shape {attn.shape}")
        raise ValueError(f"Expected 4D tensor, got shape {attn.shape}")
    
    if attn.shape[0] != num_layers:
        logging.error(f"First dimension should be {num_layers}, got {attn.shape[0]}")
        raise ValueError(f"First dimension should be {num_layers}, got {attn.shape[0]}")
    
    if attn.shape[1] != num_heads:
        logging.error(f"Second dimension should be {num_heads}, got {attn.shape[1]}")
        raise ValueError(f"Second dimension should be {num_heads}, got {attn.shape[1]}")
    
    if attn.shape[2] != n_tokens:
        logging.error(f"Third dimension should be {n_tokens}, got {attn.shape[2]}")
        raise ValueError(f"Third dimension should be {n_tokens}, got {attn.shape[2]}")
    
    if attn.shape[3] != n_tokens:
        logging.error(f"Fourth dimension should be {n_tokens}, got {attn.shape[3]}")
        raise ValueError(f"Fourth dimension should be {n_tokens}, got {attn.shape[3]}")
    
    # Convert to float32 numpy and match the concatenation over layers used in
    # the original implementation (batch size is 1 here).
    attn_mask = attn.to(torch.float32).cpu().numpy()  # [num_layers, num_heads, L, L]
    
    # Assert shape after conversion
    if attn_mask.shape != (num_layers, num_heads, n_tokens, n_tokens):
        logging.error(f"Shape mismatch after conversion: {attn_mask.shape}")
        raise ValueError(f"Shape mismatch after conversion: {attn_mask.shape}")

    if n_tokens > (orig_n_tokens + 1):
        # attentions are cropped because of the eos token
        # Add check that we have enough tokens to crop
        if n_tokens < 2:
            logging.error(f"Cannot crop: n_tokens must be at least 2, got {n_tokens}")
            raise ValueError(f"Cannot crop: n_tokens must be at least 2, got {n_tokens}")
        attn_mask = attn_mask[:, :, :-1, :-1]
        L_eff = n_tokens - 1
    else:
        L_eff = n_tokens
    
    # Validate L_eff calculation
    if L_eff <= 0:
        logging.error(f"L_eff must be positive, got {L_eff}")
        raise ValueError(f"L_eff must be positive, got {L_eff}")
    
    if L_eff > n_tokens:
        logging.error(f"L_eff ({L_eff}) cannot exceed n_tokens ({n_tokens})")
        raise ValueError(f"L_eff ({L_eff}) cannot exceed n_tokens ({n_tokens})")
    
    # Validate shape after potential cropping
    expected_shape = (num_layers, num_heads, L_eff, L_eff)
    if attn_mask.shape != expected_shape:
        logging.error(f"Expected shape {expected_shape} after cropping, got {attn_mask.shape}")
        raise ValueError(f"Expected shape {expected_shape} after cropping, got {attn_mask.shape}")
    
        f"Expected shape {expected_shape} after cropping, got {attn_mask.shape}"

    attn_mask_reshape = attn_mask.reshape(num_layers * num_heads, L_eff, L_eff)
    
    # Validate reshape
    if attn_mask_reshape.shape != (num_layers * num_heads, L_eff, L_eff):
        logging.error(f"Reshape failed: expected {(num_layers * num_heads, L_eff, L_eff)}, got {attn_mask_reshape.shape}")
        raise ValueError(f"Reshape failed: expected {(num_layers * num_heads, L_eff, L_eff)}, got {attn_mask_reshape.shape}")
        f"Reshape failed: expected {(num_layers * num_heads, L_eff, L_eff)}, got {attn_mask_reshape.shape}"

    # use only claim tokens without chat template
    # shift attention mask by 1 token to align it with the generation format
    
    # CRITICAL FIX: Validate slice indices to prevent empty or invalid slices
    # The original slicing could produce empty arrays or out-of-bounds errors
    
    start_idx_from = -orig_n_tokens - 1
    end_idx_from = -1
    start_idx_to = -orig_n_tokens
    
    # Check if indices are within bounds
    if abs(end_idx_from) > L_eff:
        logging.error(f"Slice end index {end_idx_from} out of bounds for dimension of size {L_eff}")
        raise ValueError(f"Slice end index {end_idx_from} out of bounds for dimension of size {L_eff}")
    
    if start_idx_to > 0:
        logging.error(f"start_idx_to must be non-positive, got {start_idx_to}")
        raise ValueError(f"start_idx_to must be non-positive, got {start_idx_to}")
    
    # Calculate actual slice sizes
    slice_from_len = end_idx_from - start_idx_from if end_idx_from != -1 else L_eff + start_idx_from + 1
    if slice_from_len <= 0:
        logging.error(f"Source slice would be empty: length {slice_from_len}")
        raise ValueError(f"Source slice would be empty: length {slice_from_len}")
    
    # Apply slice with bounds checking
    try:
        attn_mask_reshape = attn_mask_reshape[
            :, 
            max(start_idx_from, -L_eff):(end_idx_from if end_idx_from != 0 else None),
            start_idx_to:
        ]
    except IndexError as e:
        raise IndexError(f"Slice indices out of bounds: L_eff={L_eff}, "
                        f"orig_n_tokens={orig_n_tokens}, start_idx_from={start_idx_from}, "
                        f"end_idx_from={end_idx_from}, start_idx_to={start_idx_to}") from e
    
    # Validate that the slice produced non-empty results
    if attn_mask_reshape.shape[1] <= 0:
        logging.error(f"Resulting attention mask has empty sequence dimension (axis 1)")
        raise ValueError(f"Resulting attention mask has empty sequence dimension (axis 1)")
    
    if attn_mask_reshape.shape[2] <= 0:
        logging.error(f"Resulting attention mask has empty sequence dimension (axis 2)")
        raise ValueError(f"Resulting attention mask has empty sequence dimension (axis 2)")

    attn_features_seq = _get_attention_features(attn_mask_reshape, orig_n_tokens)
    
    # Validate output shape matches expectations
    if not isinstance(attn_features_seq, list):
        logging.error(f"Expected list, got {type(attn_features_seq)}")
        raise ValueError(f"Expected list, got {type(attn_features_seq)}")
    
    return attn_mask, attn_mask_reshape, attn_features_seq

def _get_embeddings(
    hidden_states: torch.Tensor,
    num_layers: int,
    n_tokens: int,
    orig_n_tokens: int,
    key_prefix: str = "",
) -> Dict[str, np.ndarray]:
    """
    Reimplementation of HalluDetect.get_embeddings using stored hidden states.

    Parameters
    ----------
    hidden_states:
        Tensor of shape [num_layers + 1, n_tokens, hidden_size].
    """
    if num_layers + 1 != len(hidden_states):
        logging.warning(f"Expected {num_layers + 1} hidden states, got {len(hidden_states)}. This is okay if stride > 1 is used.")
        num_layers = len(hidden_states) - 1

    token_embeddings: Dict[str, np.ndarray] = {}
    for layer in list(range(num_layers)) + [-1]:
        layer_name = "" if layer == -1 else f"_{layer}"
        hs_layer = hidden_states[layer]  # [n_tokens, hidden_size]

        if n_tokens > (orig_n_tokens + 1):
            # embeddings are cropped because of the eos token
            hs_np = hs_layer[:-1].to(torch.float32).cpu().numpy()
        else:
            hs_np = hs_layer.to(torch.float32).cpu().numpy()

        # use only claim tokens without the chat template and shift left
        token_embeddings[f"{key_prefix}token_embeddings_decoder{layer_name}"] = (
            hs_np[-orig_n_tokens - 1 : -1]
        )
    return token_embeddings


def _resolve_hidden_states_device(hidden_states_device: str) -> torch.device:
    assert isinstance(hidden_states_device, str), "hidden_states_device must be a string"
    assert hidden_states_device, "hidden_states_device must be non-empty"
    device = torch.device(hidden_states_device)
    if device.type == "cuda:0" and not torch.cuda.is_available():
        raise ValueError("use_gpu_hidden_states=True requires CUDA, but CUDA is not available")
    return device


def _get_embeddings_gpu(
    hidden_states: torch.Tensor,
    num_layers: int,
    n_tokens: int,
    orig_n_tokens: int,
    device: torch.device,
    key_prefix: str = "",
) -> Dict[str, np.ndarray]:
    assert hidden_states.dim() == 3, f"hidden_states must be 3D, got shape {tuple(hidden_states.shape)}"
    assert n_tokens > 0, f"n_tokens must be positive, got {n_tokens}"
    assert orig_n_tokens > 0, f"orig_n_tokens must be positive, got {orig_n_tokens}"
    if num_layers + 1 != len(hidden_states):
        logging.warning(
            f"Expected {num_layers + 1} hidden states, got {len(hidden_states)}. "
            "This is okay if stride > 1 is used."
        )
        num_layers = len(hidden_states) - 1

    # Move once to target device and crop there to minimize host<->device traffic.
    hs = hidden_states.to(device=device, dtype=torch.float32, non_blocking=True)
    if n_tokens > (orig_n_tokens + 1):
        hs = hs[:, :-1, :]

    hs_claim = hs[:, -orig_n_tokens - 1 : -1, :]
    assert hs_claim.shape[1] > 0, "Claim-token slice is empty; check n_tokens/orig_n_tokens alignment"

    token_embeddings: Dict[str, np.ndarray] = {}
    for layer in list(range(num_layers)) + [-1]:
        layer_name = "" if layer == -1 else f"_{layer}"
        token_embeddings[f"{key_prefix}token_embeddings_decoder{layer_name}"] = (
            hs_claim[layer].detach().cpu().numpy()
        )
    return token_embeddings


_LOGITS_STATS = {
    "top_k_logits",
    "top_k_indices",
    "greedy_texts",
    "greedy_tokens",
}
_ATTENTION_STATS = {
    "recent_lookback_attention_weights_from_claim_tokens",
    "claim_only_attention_maps",
    "full_attention_maps",
    "train_attention_features",
}
_ALWAYS_AVAILABLE_STATS = {
    "sequence_logprobs",
    "greedy_logprobs",
    "token_entropies",
    "eval_meta",
}


def _requires_hidden_state_features(stats_dependencies: set[str]) -> bool:
    prefixes = (
        "token_embeddings_decoder",
        "embeddings_decoder",
        "train_token_embeddings_decoder",
        "train_embeddings_decoder",
        "background_train_token_embeddings_decoder",
    )
    return any(dep.startswith(prefixes) for dep in stats_dependencies)


def _resolve_feature_gates(
    input_regime: str,
    required_stats: List[str] | None,
) -> Tuple[bool, bool, bool]:
    assert input_regime in ["full", "logits", "hidden_states", "attentions", "chunked"]
    if input_regime == "full":
        return True, True, True
    if input_regime == "logits":
        return True, False, False
    if input_regime == "hidden_states":
        return False, True, False
    if input_regime == "attentions":
        return False, False, True

    # input_regime == "chunked"
    if required_stats is None:
        return True, True, True
    assert isinstance(required_stats, list), "required_stats must be a list of stat names or None"
    if len(required_stats) == 0:
        logging.warning("[make_all_stats] Empty required_stats in chunked mode; falling back to full feature computation")
        return True, True, True
    if not all(isinstance(s, str) and s for s in required_stats):
        raise ValueError("required_stats must contain non-empty strings")

    required = set(required_stats)
    known = set(_ALWAYS_AVAILABLE_STATS) | set(_LOGITS_STATS) | set(_ATTENTION_STATS)
    unknown = [s for s in required if s not in known and not _requires_hidden_state_features({s})]
    if unknown:
        logging.warning(
            "[make_all_stats] Unknown required_stats in chunked mode: %s. Falling back to full feature computation.",
            sorted(unknown),
        )
        return True, True, True

    need_logits = any(s in _LOGITS_STATS for s in required)
    need_attentions = any(s in _ATTENTION_STATS for s in required)
    need_hidden_states = _requires_hidden_state_features(required)
    return need_logits, need_hidden_states, need_attentions


def _compute_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    orig_input_ids: torch.Tensor,
    n_tokens: int,
    orig_n_tokens: int,
    alternatives: bool = False,
    n_alternatives: int = 10,
) -> Tuple[List[float], np.ndarray, List[List[List[Tuple[int, float]]]]]:
    """
    Reimplementation of HalluDetect.get_all_probs operating on stored logits.

    Returns
    -------
    ll:
        List of greedy log-likelihoods per token for the claim.
    log_probs_orig:
        Array of shape [orig_n_tokens, vocab_size] with log-probs for claim tokens.
    cut_alternatives:
        Nested list of alternatives per token as in the original code.
    """
    # logits: [n_tokens, vocab_size]
    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1).cpu().numpy()

    if n_tokens > (orig_n_tokens + 1):
        # logits are cropped because of the eos token
        log_probs = log_probs[:-1]

    # remove the <bos> token in input_ids when computing greedy log-likelihoods
    input_ids_np = input_ids.cpu().numpy()
    ll = [float(l[tok]) for tok, l in zip(input_ids_np[1:], log_probs)]
    if n_tokens > (orig_n_tokens + 1):
        ll = ll[-orig_n_tokens - 1 : -1]

    # store only the claim tokens without the chat template and shift left
    log_probs3 = log_probs[None, :, :]  # [1, L_eff, vocab]
    log_probs_orig = log_probs3[:, -orig_n_tokens - 1 : -1, :]  # [1, orig_n_tokens, vocab]

    cut_alternatives: List[List[List[Tuple[int, float]]]] = []
    if alternatives:
        seq = orig_input_ids.cpu().numpy().tolist()
        length = len(seq)
        cut_alternatives.append([[] for _ in range(length)])
        for j in range(length):
            lt = log_probs_orig[0, j, :]
            best_tokens = np.argpartition(lt, -n_alternatives)
            ln = len(best_tokens)
            best_tokens = best_tokens[ln - n_alternatives : ln]
            if seq[j] not in best_tokens:
                cut_alternatives[0][j].append((int(seq[j]), float(lt[seq[j]])))
                for t in best_tokens[1:]:
                    cut_alternatives[0][j].append((int(t), float(lt[t])))
            else:
                for t in best_tokens:
                    cut_alternatives[0][j].append((int(t), float(lt[t])))
            cut_alternatives[0][j].sort(
                key=lambda x: x[0] == seq[j],
                reverse=True,
            )

    return ll, log_probs_orig[0, :], cut_alternatives


def _build_train_stats(
    train_raw: Dict[str, Any],
    num_layers: int,
    num_heads: int,
) -> Dict[str, Any]:
    train_attention_features = []
    train_greedy_log_likelihoods = []
    train_log_probs = []
    train_greedy_texts = []
    train_greedy_tokens = []
    train_metrics = []
    train_token_embeddings: Dict[str, List[np.ndarray]] = {}

    for sample in train_raw["samples"]:
        input_ids: torch.Tensor = sample["input_ids"]
        orig_input_ids: torch.Tensor = sample["orig_input_ids"]
        hidden_states: torch.Tensor = sample["hidden_states"]
        attentions: torch.Tensor = sample["attentions"]
        logits: torch.Tensor = sample["logits"]

        n_tokens = int(input_ids.shape[0])
        orig_n_tokens = int(orig_input_ids.shape[0])

        _, _, attn_features_seq = get_attentions(
            attentions, num_layers, num_heads, n_tokens, orig_n_tokens
        )
        train_attention_features.append(attn_features_seq)

        ll, log_probs, _ = _compute_probs(
            logits,
            input_ids,
            orig_input_ids,
            n_tokens,
            orig_n_tokens,
            alternatives=False,
        )
        train_greedy_log_likelihoods.append(ll)
        train_log_probs.append(log_probs)

        train_greedy_texts.append(sample["claim"])
        train_greedy_tokens.append(orig_input_ids.cpu().numpy().tolist())
        train_metrics.append(float(sample["label"]))

        token_embs = _get_embeddings(
            hidden_states, num_layers, n_tokens, orig_n_tokens, key_prefix="train_"
        )
        for key, arr in token_embs.items():
            train_token_embeddings.setdefault(key, []).append(arr)

    for key in list(train_token_embeddings.keys()):
        seq_key = key.replace("token_", "")
        per_seq_means = [t.mean(axis=0) for t in train_token_embeddings[key]]
        train_token_embeddings[seq_key] = np.stack(per_seq_means, axis=0)
        train_token_embeddings[key] = np.concatenate(train_token_embeddings[key], axis=0)

    train_stats = {
        "train_attention_features": train_attention_features,
        "train_greedy_log_likelihoods": train_greedy_log_likelihoods,
        "train_greedy_log_probs": train_log_probs,
        "train_greedy_texts": train_greedy_texts,
        "train_greedy_tokens": train_greedy_tokens,
        "train_metrics": np.asarray(train_metrics, dtype=float),
    }
    train_stats.update(train_token_embeddings)
    return train_stats


def _build_background_stats(
    background_raw: Dict[str, Any],
    num_layers: int,
    num_heads: int,
) -> Dict[str, Any]:
    background_train_token_embeddings: Dict[str, List[np.ndarray]] = {}
    for sample in background_raw["samples"]:
        input_ids: torch.Tensor = sample["input_ids"]
        hidden_states: torch.Tensor = sample["hidden_states"]
        n_tokens = int(input_ids.shape[0])
        token_embs = _get_embeddings(
            hidden_states,
            num_layers,
            n_tokens,
            n_tokens,
            key_prefix="background_train_",
        )
        for key, arr in token_embs.items():
            background_train_token_embeddings.setdefault(key, []).append(arr)

    for key in list(background_train_token_embeddings.keys()):
        background_train_token_embeddings[key] = np.concatenate(
            background_train_token_embeddings[key], axis=0
        )
    return background_train_token_embeddings


def make_all_stats(
    eval_raw: Dict[str, Any],
    train_raw: Dict[str, Any] | None = None,
    background_raw: Dict[str, Any] | None = None,
    supervised: bool = False,
    train_stats_cache: Dict[str, Any] | None = None,
    background_stats_cache: Dict[str, Any] | None = None,
    input_regime: str = "full",
    use_gpu_hidden_states: bool = False,
    hidden_states_device: str = "cuda:0",
    required_stats: List[str] | None = None,
) -> Tuple[
    Dict[str, Any],
    np.ndarray,
    Dict[str, Any] | None,
    Dict[str, Any] | None,
]:
    """
    Construct an `all_stats` dictionary using stored raw model outputs instead of live forward passes.

    NLI-based and entropy-calculator-based features are *not* supported.

    Parameters
    ----------
    eval_raw : Dict[str, Any]
        Raw evaluation data with "samples" list and "meta" dict.
    train_raw : Dict[str, Any] | None
        Raw training data (required if supervised=True and no cache).
    background_raw : Dict[str, Any] | None
        Raw background data for Mahalanobis-style methods.
    supervised : bool
        Whether to compute train statistics for supervised methods.
    train_stats_cache : Dict[str, Any] | None
        Cached train stats to avoid recomputation.
    background_stats_cache : Dict[str, Any] | None
        Cached background stats to avoid recomputation.
    input_regime : str
        Controls computed features: "full" (all), "logits" (token preds only),
        "hidden_states" (embeddings only), "attentions" (attention only).

    Returns
    -------
    all_stats : Dict[str, Any]
        Dictionary of statistics. Let N = number of eval samples, L = num_layers,
        H = num_heads, D = hidden_dim, k = top-k value. Per-sample token counts
        vary (denoted n_i for sample i).

        **Metadata:**
        - "eval_meta": dict with model metadata (e.g., "num_layers", "num_attention_heads")

        **Sequence-level statistics (always computed):**
        - "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1. Contains logits of tokens which were actually generated by the model, excluding the first generated token.
        - "greedy_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i. Contains logits of the most likely token at each position.
        - "token_entropies": List[np.ndarray] of length N, inner 1d array has length n_i. Contains entropy values of the probability distribution at each position.

        **Token-level statistics (computed if input_regime in ["logits", "full"]):**
        - "greedy_texts": List[str] of length N - claim texts
        - "greedy_tokens": List[List[int]] of length N - token IDs per claim
        - "top_k_logits": List[np.ndarray] of length N - each array shape [n_i, k]
        - "top_k_indices": List[np.ndarray] of length N - each array shape [n_i, k]

        **Attention statistics (computed if input_regime in ["attentions", "full"]):**
        - "recent_lookback_attention_weights_from_claim_tokens": List[Any] of length N. For each sample: List[List[List[float]]] with shape [n_claim_tokens, n_recent_lookback_tokens, num_layers * num_heads]
        - "claim_only_attention_maps": List[np.ndarray] of length N - each array shape [L*H, n_i, n_i]
        - "full_attention_maps": List[np.ndarray] of length N - each array shape [L, H, n_i, n_i]

        **Embedding statistics (computed if input_regime in ["hidden_states", "full"]):**
        - "token_embeddings_decoder": np.ndarray[float32, [sum(n_i), D]] - concatenated token embeddings
        - "token_embeddings_decoder_{layer}": np.ndarray[float32, [sum(n_i), D]] per layer (if stored)
        - "embeddings_decoder": np.ndarray[float32, [N, D]] - mean token embedding per sequence
        - "embeddings_decoder_{layer}": np.ndarray[float32, [N, D]] per layer (if stored)

        **Supervised train statistics (if supervised=True):**
        - Cached train stats (from train_stats_cache) prefixed with "train_"
        - "train_sequence_logprob": np.ndarray[float32, [N_train]] (computed if missing in cache)
        - "train_mean_token_entropy": np.ndarray[float32, [N_train]] (computed if missing in cache)

        **Background statistics (if background_raw provided):**
        - Cached background stats (from background_stats_cache) e.g.:
          "background_train_token_embeddings_decoder": np.ndarray[float32, [sum(n_i_bg), D]]

    eval_labels : np.ndarray[float32, [N]]
        Binary labels for eval samples (1=hallucination, 0=correct).
    train_stats_cache : Dict[str, Any] | None
        Cached train stats (pass back to future calls). None if supervised=False.
    background_stats_cache : Dict[str, Any] | None
        Cached background stats (pass back to future calls). None if no background provided.
    """
    assert input_regime in ["full", "logits", "hidden_states", "attentions", "chunked"]
    assert isinstance(use_gpu_hidden_states, bool), "use_gpu_hidden_states must be bool"
    need_logits, need_hidden_states, need_attentions = _resolve_feature_gates(
        input_regime=input_regime,
        required_stats=required_stats,
    )
    hs_device = None
    if use_gpu_hidden_states and need_hidden_states:
        hs_device = _resolve_hidden_states_device(hidden_states_device)

    start_time = time.time()
    
    meta = eval_raw.get("meta", {})
    num_layers = int(meta.get("num_layers", 0))
    num_heads = int(meta.get("num_attention_heads", 0))
    num_eval_samples = len(eval_raw.get("samples", []))
    logging.info(f"[make_all_stats] START: {num_eval_samples} samples, num_layers={num_layers}, num_heads={num_heads}")

    if supervised:
        if input_regime != "full":
            raise ValueError(f"Input regime {input_regime} not supported for supervised mode")

        if train_stats_cache is None:
            if train_raw is None:
                raise ValueError(
                    "train_raw must be provided when supervised=True and no cache exists"
                )
            logging.info(f"[make_all_stats] Building train stats (supervised mode)...")
            train_build_start = time.time()
            train_stats_cache = _build_train_stats(train_raw, num_layers, num_heads)
            logging.info(f"[make_all_stats] Train stats built in {time.time() - train_build_start:.2f}s")
    else:
        train_stats_cache = None

    if background_stats_cache is None and background_raw is not None:
        if input_regime != "full":
            raise ValueError(f"Input regime {input_regime} not supported for background mode")

        logging.info(f"[make_all_stats] Building background stats...")
        bg_build_start = time.time()
        background_stats_cache = _build_background_stats(
            background_raw, num_layers, num_heads
        )
        logging.info(f"[make_all_stats] Background stats built in {time.time() - bg_build_start:.2f}s")

    # ------------------------ Evaluation statistics ------------------------
    eval_samples: List[Dict[str, Any]] = eval_raw["samples"]
    logging.info(f"[make_all_stats] Processing {len(eval_samples)} eval samples...")

    attention_features: List[Any] = []
    attention_all: List[np.ndarray] = []
    forwardpass_attention_weights: List[np.ndarray] = []
    # greedy_log_likelihoods: List[List[float]] = []
    # greedy_tokens_alternatives: List[Any] = []
    # greedy_log_probs: List[np.ndarray] = []
    top_k_logits: List[np.ndarray] = []
    top_k_indices: List[np.ndarray] = []
    greedy_texts: List[str] = []
    greedy_tokens: List[List[int]] = []
    eval_labels: List[float] = []                   # binary labels for hallucination detection
    ground_truth_scores: List[float] = []           # continuous scores from 0 to 1, measuring output quality
    eval_token_embeddings: Dict[str, List[np.ndarray]] = {}
    sequence_logprobs: List[float] = []
    greedy_logprobs: List[float] = []
    token_entropies: List[float] = []

    eval_loop_start = time.time()
    for sample_idx, sample in enumerate(eval_samples):
        if sample_idx % 300 == 0:
            logging.info(f"[make_all_stats] Processing eval sample {sample_idx+1}/{len(eval_samples)}...")
        
        # NOTE: All tensor access below is READ-ONLY. We only read tensors and create
        # NEW arrays via .to().cpu().numpy() chains. No in-place modifications (+=, .add_(), etc.)
        # are performed. This allows callers to skip expensive deep copies of input data.
        input_ids = sample["input_ids"] # length
        orig_input_ids = sample["orig_input_ids"]
        sequence_logprobs.append(sample["sequence_logprobs"]) # length - 1
        greedy_logprobs.append(sample["greedy_logprobs"]) # length
        token_entropies.append(sample["token_entropies"]) # length
        if "label" in sample:
            eval_labels.append(float(sample["label"]))
        elif "ground_truth_score" in sample:
            ground_truth_scores.append(float(sample["ground_truth_score"]))
        else:
            raise ValueError(f"Sample {sample_idx} has no label or ground_truth_score")

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        if isinstance(orig_input_ids, list):
            orig_input_ids = torch.tensor(orig_input_ids)

        n_tokens = int(input_ids.shape[0])
        orig_n_tokens = int(orig_input_ids.shape[0])

        if need_attentions:
            attentions = sample["attentions"]
            attn_mask, attn_mask_reshape, attn_features_seq = get_attentions(
                attentions, num_layers, num_heads, n_tokens, orig_n_tokens
            )

            attention_features.append(attn_features_seq)
            attention_all.append(attn_mask_reshape)
            forwardpass_attention_weights.append(attn_mask)

        if need_logits:
            # logits = sample["logits"]
            # ll, log_probs, cut_alternatives = _compute_probs(
            #     logits,
            #     input_ids,
            #     orig_input_ids,
            #     n_tokens,
            #     orig_n_tokens,
            #     alternatives=True,
            # )
            # greedy_log_likelihoods.append(ll)
            # greedy_log_probs.append(log_probs)
            # greedy_tokens_alternatives.append(cut_alternatives)
            top_k_logits.append(sample["top_k_logits"].cpu().numpy() if isinstance(sample["top_k_logits"], torch.Tensor) else sample["top_k_logits"])
            top_k_indices.append(sample["top_k_indices"].cpu().numpy() if isinstance(sample["top_k_indices"], torch.Tensor) else sample["top_k_indices"])

            greedy_texts.append(sample["claim"])
            greedy_tokens.append(orig_input_ids.cpu().numpy().tolist())

        if need_hidden_states:
            hidden_states = sample["hidden_states"]

            # key -> [n_i, hidden_dim]
            if use_gpu_hidden_states:
                assert hs_device is not None, "Internal error: hs_device must be initialized"
                token_embs = _get_embeddings_gpu(
                    hidden_states,
                    num_layers,
                    n_tokens,
                    orig_n_tokens,
                    device=hs_device,
                    key_prefix="",
                )
            else:
                token_embs = _get_embeddings(
                    hidden_states, num_layers, n_tokens, orig_n_tokens, key_prefix=""
                )
            for key, arr in token_embs.items():
                eval_token_embeddings.setdefault(key, []).append(arr)
    
    eval_loop_elapsed = time.time() - eval_loop_start
    logging.info(f"[make_all_stats] Eval sample loop completed in {eval_loop_elapsed:.2f}s")

    # ------------------------ Assemble all_stats ------------------------
    all_stats: Dict[str, Any] = {
        "eval_meta": meta,
        "sequence_logprobs": sequence_logprobs,
        "greedy_logprobs": greedy_logprobs,
        "token_entropies": token_entropies,
    }
    if need_attentions:
        all_stats["recent_lookback_attention_weights_from_claim_tokens"] = attention_features
        all_stats["claim_only_attention_maps"] = attention_all
        all_stats["full_attention_maps"] = forwardpass_attention_weights
    if need_logits:
        all_stats["top_k_logits"] = top_k_logits
        all_stats["top_k_indices"] = top_k_indices
        all_stats["greedy_texts"] = greedy_texts
        all_stats["greedy_tokens"] = greedy_tokens

    if need_hidden_states:
        for key, seq_list in eval_token_embeddings.items():
            seq_key = key.replace("token_", "")
            per_seq_means = [t.mean(axis=0) for t in seq_list]
            all_stats[key] = np.concatenate(seq_list, axis=0)
            all_stats[seq_key] = np.stack(per_seq_means, axis=0)

    # ------------------------ Convenience aliases ------------------------
    # These do not exist in the original HalluDetect implementation but are
    # cheap to compute from existing statistics and make it easier to write
    # simple baselines.

    # Sequence log-probability for eval split: sum of token log-likelihoods
    # if greedy_log_likelihoods:
    #     all_stats["sequence_logprob"] = np.asarray(
    #         [float(sum(ll_i)) for ll_i in greedy_log_likelihoods],
    #         dtype=np.float32,
    #     )

    # # Mean token entropy for eval split, derived from greedy_log_probs
    # if greedy_log_probs:
    #     mean_ent_eval: List[float] = []
    #     for log_probs_i in greedy_log_probs:
    #         lp = log_probs_i.astype(np.float32)
    #         p = np.exp(lp)
    #         ent_tok = -(p * lp).sum(axis=-1)
    #         mean_ent_eval.append(float(ent_tok.mean()))
    #     all_stats["mean_token_entropy"] = np.asarray(mean_ent_eval, dtype=np.float32)

    if train_stats_cache is not None:
        all_stats.update(train_stats_cache)
        if "train_sequence_logprob" not in train_stats_cache:
            all_stats["train_sequence_logprob"] = np.asarray(
                [float(sum(ll_i)) for ll_i in train_stats_cache["train_greedy_log_likelihoods"]],
                dtype=np.float32,
            )
        if "train_mean_token_entropy" not in train_stats_cache:
            mean_ent_train: List[float] = []
            for log_probs_i in train_stats_cache["train_greedy_log_probs"]:
                lp = log_probs_i.astype(np.float32)
                p = np.exp(lp)
                ent_tok = -(p * lp).sum(axis=-1)
                mean_ent_train.append(float(ent_tok.mean()))
            all_stats["train_mean_token_entropy"] = np.asarray(
                mean_ent_train, dtype=np.float32
            )

    if background_stats_cache is not None:
        all_stats.update(background_stats_cache)

    total_elapsed = time.time() - start_time
    logging.info(f"[make_all_stats] COMPLETE: {len(all_stats)} stat keys, {len(eval_labels)} labels, total time={total_elapsed:.2f}s")
    
    assert len(eval_labels) == 0 or len(ground_truth_scores) == 0, "eval_labels and ground_truth_scores cannot both be present"
    eval_labels = eval_labels if len(eval_labels) > 0 else ground_truth_scores
    assert len(all_stats["sequence_logprobs"]) == len(eval_labels), f"{len(all_stats['sequence_logprobs'])=}, {len(eval_labels)=}"
    assert len(all_stats["greedy_logprobs"]) == len(eval_labels), f"{len(all_stats['greedy_logprobs'])=}, {len(eval_labels)=}"
    assert len(all_stats["token_entropies"]) == len(eval_labels), f"{len(all_stats['token_entropies'])=}, {len(eval_labels)=}"
    return (
        all_stats,
        np.asarray(eval_labels, dtype=np.float32),
        train_stats_cache,
        background_stats_cache,
    )

