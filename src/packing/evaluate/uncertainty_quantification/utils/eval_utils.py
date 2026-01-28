"""Model loading and inference utilities for uncertainty quantification."""

from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.nn.functional import log_softmax, softmax
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id: str,
    cache_dir: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Dict[str, Any]]:
    """
    Load a causal language model and tokenizer from HuggingFace Hub.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct").
    cache_dir : str, optional
        Cache directory for model weights.
    torch_dtype : torch.dtype
        Data type for model weights (default: torch.float32).
    device : str
        Device to load model on. "auto" uses CUDA if available, else CPU.

    Returns
    -------
    model : AutoModelForCausalLM
        Loaded model in eval mode.
    tokenizer : AutoTokenizer
        Loaded tokenizer with pad_token set.
    metadata : Dict[str, Any]
        Model metadata extracted from config (hidden_size, num_layers, etc.).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    ).eval().to(device)

    config = model.config
    model_metadata = {
        "model_id": getattr(config, "_name_or_path", None),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
    }

    return model, tokenizer, model_metadata


def apply_chat_template(tokenizer, texts: List[str], template: Optional[str]) -> List[str]:
    """
    Apply a chat template to a list of texts.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Tokenizer with chat template support.
    texts : List[str]
        List of text strings to wrap in chat template.
    template : str, optional
        Template name. Supported values:
        - None: No template applied, return texts as-is
        - "assistant-only": Wrap as assistant-only message
        - "generate_true_statement": User asks to generate true statement, assistant responds

    Returns
    -------
    rendered : List[str]
        List of texts with chat template applied.

    Raises
    ------
    ValueError
        If template name is not recognized.
    """
    if not template:
        return texts

    if template == "assistant-only":
        # Simple assistant-only wrapper by default
        build_prompt = lambda t: [
            {"role": "assistant", "content": t},
        ]
    elif "generate_true_statement" in template:
        build_prompt = lambda t: [
            {"role": "user", "content": "Generate true statement"},
            {"role": "assistant", "content": t},
        ]
    else:
        raise ValueError(f"Unknown value for --chat-template: {template}")

    rendered: List[str] = []
    for t in texts:
        rendered.append(
            tokenizer.apply_chat_template(
                build_prompt(t),
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return rendered


def pack_single_sample_raw_states(
    i: int,
    batch_raw: List[str],
    batch_labels: List[float],
    batch_ids: List[str],
    chat_inputs: Dict[str, Any],
    raw_inputs: Dict[str, Any],
    hidden_states: List[torch.Tensor],
    attentions: List[torch.Tensor],
    attn_mask: torch.Tensor,
    logits: torch.Tensor,
    layer_stride: int,
    top_k: int,
) -> Dict[str, Any]:
    """
    Pack model internal states for a single sample into a dictionary.

    This function extracts and processes hidden states, attention weights, logits,
    and other internal representations for a single sample from a batched forward pass.

    Parameters
    ----------
    i : int
        Index of the sample in the batch.
    batch_raw : List[str]
        Raw text strings for each sample in the batch.
    batch_labels : List[float]
        Labels for each sample in the batch.
    batch_ids : List[str]
        Unique IDs for each sample in the batch.
    chat_inputs : Dict[str, Any]
        Tokenized inputs with chat template applied.
    raw_inputs : Dict[str, Any]
        Tokenized inputs without chat template (original claim).
    hidden_states : List[torch.Tensor]
        Hidden states from all layers. Each tensor has shape [batch, seq_len, hidden_size].
    attentions : List[torch.Tensor]
        Attention weights from all layers. Each tensor has shape [batch, num_heads, seq_len, seq_len].
    attn_mask : torch.Tensor
        Attention mask with shape [batch, seq_len].
    logits : torch.Tensor
        Output logits with shape [batch, seq_len, vocab_size].
    layer_stride : int
        Stride for subsampling hidden state layers (e.g., stride=2 keeps every other layer).
    top_k : int
        Number of top-k logits to save per token.

    Returns
    -------
    raw_internal_states : Dict[str, Any]
        Dictionary containing:
        - claim (str): Raw text
        - label (float): Ground truth label
        - id (str): Sample ID
        - input_ids (torch.Tensor): Tokenized input IDs (chat form)
        - attention_mask (torch.Tensor): Attention mask
        - orig_input_ids (torch.Tensor): Tokenized input IDs (original, non-chat)
        - sequence_logprobs (np.ndarray): Log probabilities of each token in sequence
        - greedy_logprobs (np.ndarray): Top-1 log probability at each position
        - token_entropies (np.ndarray): Entropy of distribution at each position
        - hidden_states (torch.Tensor): Stacked hidden states [num_layers, length, hidden_size]
        - attentions (torch.Tensor): Stacked attention weights [num_layers, num_heads, length, length]
        - top_k_logits (torch.Tensor): Top-k logits per token
        - top_k_indices (torch.Tensor): Top-k token indices per token

    Raises
    ------
    ValueError
        If attention mask is empty (length <= 0).
    """
    length = int(attn_mask[i].sum().item())
    if length <= 0:
        raise ValueError("Empty attention mask")

    # Input IDs and attention mask (chat form)
    input_ids_i = chat_inputs["input_ids"][i, :length].to("cpu", dtype=torch.int32)
    attention_mask_i = attn_mask[i, :length].to("cpu", dtype=torch.int8)

    # Raw (non-chat) input IDs for the original claim
    raw_attn_mask_i = raw_inputs["attention_mask"][i]
    raw_length = int(raw_attn_mask_i.sum().item())
    orig_input_ids_i = raw_inputs["input_ids"][i, :raw_length].to("cpu", dtype=torch.int32)

    # Stack hidden states into [num_layers + 1, length, hidden_size]
    hs_layers: List[torch.Tensor] = []
    for j, h in enumerate(hidden_states):
        if j % layer_stride != 0:
            continue

        hs_layers.append(
            h[i, :length].to(dtype=torch.float16, device="cpu")
        )
    hidden_i = torch.stack(hs_layers, dim=0)

    # Stack attentions into [num_layers, num_heads, length, length]
    attn_layers: List[torch.Tensor] = []
    for a in attentions:
        attn_layers.append(
            a[i, :, :length, :length].to(dtype=torch.float16, device="cpu")
        )
    attn_i = torch.stack(attn_layers, dim=0)

    raw_logits_i = logits[i, :length]  # already on CPU, float16
    top_k_vals, top_k_inds = torch.topk(raw_logits_i, k=top_k, dim=-1)
    logits_i = log_softmax(raw_logits_i, dim=-1)
    probs_i = softmax(raw_logits_i, dim=-1)
    del raw_logits_i

    # Log probabilities of each token in the sequence (excluding the first token)
    sequence_logprobs_i = logits_i[:-1][torch.arange(length - 1), input_ids_i[1:]]
    # Top-1 logit for each token in the sequence
    greedy_logprobs_i = logits_i.max(dim=-1).values
    # Entropy of probability distribution at each token
    token_entropies_i = -probs_i.mul(logits_i).sum(dim=-1).to(dtype=torch.float16)

    # Common fields for all outputs
    raw_internal_states = {
        "claim": batch_raw[i],
        "label": float(batch_labels[i]),
        "id": batch_ids[i],
        "input_ids": input_ids_i,
        "attention_mask": attention_mask_i,
        "orig_input_ids": orig_input_ids_i,
        "sequence_logprobs": sequence_logprobs_i.numpy(), # length - 1
        "greedy_logprobs": greedy_logprobs_i.numpy(), # length
        "token_entropies": token_entropies_i.numpy(), # length
        "hidden_states": hidden_i,
        "attentions": attn_i,
        "top_k_logits": top_k_vals,
        "top_k_indices": top_k_inds,
    }
    return raw_internal_states
