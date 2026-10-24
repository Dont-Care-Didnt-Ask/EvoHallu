"""Exploratory neutral evolution with an explicit interface/metric contract.

The sole runtime intervention appends a truthful contract to the neutral
candidate prompt. It does not edit pinned files, repair programs, add model
calls, restrict formula complexity, or modify sampling/evaluation/selection.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydra import main as hydra_main
from omegaconf import OmegaConf


CONTRACT = """## Exact evaluator interface and metric contract
The following specifies the actual evaluator API and array indexing. Formula
structure, signal families, coefficients and the number of components remain open.

- Provide class HallucinationDetector(Estimator). Estimator is supplied by the
  evaluator; do not import or redefine it. Its constructor takes context and
  keyword-only stats_dependencies and level. In __init__(self, context), call:
  super().__init__(context, stats_dependencies=["sequence_logprobs", "top_k_logits", "top_k_indices"], level="sequence")
  The dependency list may instead name the available inputs your formula uses.
- Implement __call__(self, stats) directly: this is the method the evaluator
  invokes. stats is a DICTIONARY for a batch, not a list of claim dictionaries
  and not the scipy.stats module. N = len(stats["sequence_logprobs"]).
- The documented token inputs are lists of N NumPy arrays. For sample i, let
  L_i = len(stats["sequence_logprobs"][i]). That observed-token logprob array
  has shape (L_i,). The corresponding top_k_logits[i] and top_k_indices[i]
  arrays have shape (L_i + 1, 512). L_i varies across samples; the batch is
  ragged, so handle each sample's own arrays or explicitly pad/group them.
- Alignment is exact: observed logprob entry t was computed from logits row t
  for the next observed token. To compare rowwise with all L_i observed
  logprobs, use stats["top_k_logits"][i][:-1] (and the same slice of
  stats["top_k_indices"][i] when needed). The
  final top-k row has no corresponding observed-token logprob. There is no
  additional one-position shift between those aligned arrays. These arrays
  cover the cached token sequence, including its template tokens.
- top_k_logits contains raw logits. A softmax over its 512 columns defines a
  top-512-conditional distribution, not a full-vocabulary distribution. Its
  logprobs and full-vocabulary observed-token logprobs have different
  normalizers. top_k_indices contains vocabulary token IDs, not column
  positions within the 512-wide array; map IDs to columns when that is needed.
- Stored floating arrays can be float16. Promote arrays to float32 or float64
  before arithmetic/reductions where needed for finite numerical results.
  Handle short/constant arrays and zero denominators explicitly; correlations
  of constant vectors and divisions by zero can yield NaN.
- Import NumPy/SciPy APIs you use. entropy and linregress are in scipy.stats;
  softmax and logsumexp are in scipy.special. The parameter stats remains the
  input dictionary, so stats.linregress is not a SciPy function call.
- Return one finite, one-dimensional NumPy array of shape (N,), with one
  uncertainty score per sample. Higher score means greater hallucination
  risk (is_hallu=1); the training/validation fitness uses directed ROC-AUC.
  Labels and fitted model weights are not supplied to this unsupervised
  detector. Any deterministic NumPy/SciPy scoring formula is allowed; there
  is no added feature-count ceiling or required baseline-plus-extra structure.
"""
CONTRACT_SHA256 = hashlib.sha256(CONTRACT.encode()).hexdigest()


def chat_sha(chat):
    return hashlib.sha256(json.dumps(chat, sort_keys=True).encode()).hexdigest()


@contextlib.contextmanager
def clarified_candidate_prompts(prompt_policies, on_call=None):
    """Patch the import source used locally inside run_trial, then restore it."""
    original = prompt_policies.candidate_prompt
    events = []

    def dispatch(chat, policy="original", round_index=0, index=0):
        if policy != "neutral":
            raise ValueError("The interface-and-metric contract arm requires the neutral prompt")
        result = original(chat, policy, round_index, index)
        before = chat_sha(result)
        systems = [message for message in result if message["role"] == "system"]
        if len(systems) != 1 or CONTRACT in systems[0]["content"]:
            raise ValueError("Interface clarification expects one unmodified system message")
        systems[0]["content"] += "\n\n" + CONTRACT
        event = dict(round=round_index, index=index, neutral_prompt_sha256=before,
                     actual_prompt_sha256=chat_sha(result), contract_sha256=CONTRACT_SHA256)
        events.append(event)
        if on_call is not None:
            on_call(event)
        return result

    prompt_policies.candidate_prompt = dispatch
    try:
        yield events
    finally:
        prompt_policies.candidate_prompt = original


def verify_saved_prompts(output, expected_requests):
    """Verify the unchanged runner saved the actual injected requests/hashes."""
    records = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    checked = 0
    for row in records:
        chats = json.loads((output / f"candidate_prompts_{row['round']:04d}.json").read_text())
        chat = chats[row["index"]]
        system = next(message for message in chat if message["role"] == "system")
        if not system["content"].endswith(CONTRACT) or system["content"].count(CONTRACT) != 1:
            raise ValueError("Saved request does not contain exactly one complete interface contract")
        if chat_sha(chat) != row["prompt_sha256"]:
            raise ValueError("Saved candidate prompt hash differs from the issued ledger request")
        checked += 1
    if checked != expected_requests:
        raise ValueError("Saved contract request count differs from the actual run budget")
    return checked


@hydra_main(config_path="../configs", config_name="config_uq_trial", version_base="1.3")
def main(cfg):
    from experiments import prompt_policies
    from experiments.extension_trial import run_trial

    if not cfg.get("extension_trial") or cfg.search_mode != "evolution":
        raise ValueError("Interface clarification requires a normal evolutionary extension trial")
    if cfg.trial.get("prompt_policy") != "neutral" or cfg.trial.get("parent_sampling", "inherited") != "inherited":
        raise ValueError("Keep neutral guidance and inherited parent sampling fixed in this arm")
    if cfg.trial.get("interface_clarification") != "interface_and_metric_contract":
        raise ValueError("Explicit interface_and_metric_contract opt-in is required")
    if cfg.trial.get("interface_contract_sha256") != CONTRACT_SHA256:
        raise ValueError("The prepared interface contract hash does not match this wrapper")
    if cfg.trial.get("resume", False):
        raise ValueError("This exploratory interface arm requires a fresh run name")
    evidence_path = ROOT / cfg.trial.interface_evidence_path
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if evidence_hash != cfg.trial.interface_evidence_sha256:
        raise ValueError("The frozen training-failure design evidence changed")
    output = Path(cfg.logs_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "interface_contract_provenance.json"
    if provenance_path.exists() or (output / "candidates.jsonl").exists():
        raise FileExistsError("Refusing to overwrite existing interface-arm artifacts")
    provenance = dict(intervention="interface_and_metric_contract", prompt_policy="neutral",
        parent_sampling="inherited", contract_sha256=CONTRACT_SHA256,
        contract_text=CONTRACT, training_failure_evidence=str(evidence_path),
        training_failure_evidence_sha256=evidence_hash,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_hook="Scoped replacement of experiments.prompt_policies.candidate_prompt; restored in finally",
        reason_for_hook_location="run_trial imports candidate_prompt locally from prompt_policies",
        immutable_core_files_edited=False, repair_calls=0, extra_writer_calls=0,
        feature_count_ceiling=None, test_results_used_to_choose_contract=False,
        selection="Unchanged directed training population fitness and PopQA validation run-winner rule")
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    (output / "interface_contract.txt").write_text(CONTRACT)
    trace = output / "interface_prompt_injections.jsonl"

    def record(event):
        with trace.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    OmegaConf.set_struct(cfg, False)
    with clarified_candidate_prompts(prompt_policies, on_call=record) as events:
        run_trial(cfg)
    summary = json.loads((output / "summary.json").read_text())
    checked = verify_saved_prompts(output, summary["requests"])
    if len(events) != checked:
        raise ValueError("Prompt injection count does not match issued requests")
    (output / "interface_prompt_verification.json").write_text(json.dumps(dict(
        status="passed", actual_saved_prompts_verified=checked, contract_sha256=CONTRACT_SHA256,
        prompt_hashes_match_ledgers=True, runtime_hook_restored=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
