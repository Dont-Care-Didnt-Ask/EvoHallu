"""Optional frozen-formula transfer, run only after the primary benchmark.

The shared tokenizer keeps input token IDs and chat formatting fixed. Source
validation chooses formulas; target labels never enter selection or fitting.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def run(args):
    subprocess.run([sys.executable, *args], check=True)


def main():
    from transformers import AutoTokenizer
    model = "unsloth/Llama-3.2-1B-Instruct"
    revision = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
    destination = "/dev/shm/mod1-models/Llama3.2-1B"
    shared_tokenizer = "/workspace/mod1-models/llama31"
    core = Path("out/campaign/evaluation/all_datasets.json")
    if not core.exists() or len(json.loads(core.read_text())) != 9:
        raise ValueError("Complete the primary nine-dataset benchmark before optional transfer")
    staged = Path("/dev/shm/mod1-models/Qwen3.5-9B")
    if (staged / "download_manifest.json").exists():
        run(["scripts/rotate_checkpoint.py", str(staged)])
    additional_writer = Path("/dev/shm/mod1-models/Devstral-Small-2")
    if (additional_writer / "download_manifest.json").exists():
        run(["scripts/rotate_checkpoint.py", str(additional_writer)])
    run(["scripts/download_checkpoint.py", model, destination, "--revision", revision])
    reference = AutoTokenizer.from_pretrained(shared_tokenizer)
    target = AutoTokenizer.from_pretrained(destination)
    if reference.get_vocab() != target.get_vocab():
        raise ValueError("Shared-tokenizer transfer requires identical token-to-ID vocabularies")
    output = Path("out/campaign/transfer_llama32_1b")
    output.mkdir(parents=True, exist_ok=True)
    probe_path = Path("out/campaign/small_feature_probe.json")
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    gc_interval = 32 if probe.get("gc_schedule_features_identical", False) and probe["gc32_seconds"] < probe["seconds"] else 1
    extended_path = Path("out/campaign/small_feature_extended_probe.json")
    extended = json.loads(extended_path.read_text()) if extended_path.exists() else {}
    cuda_interval = 32 if extended.get("cuda32_features_identical", False) and extended["cuda32_seconds"] < extended["gc32_seconds"] else 1
    extra = []
    supplementary = Path("out/campaign/large_linear_budget_frozen")
    if len(list(supplementary.glob("*/frozen_estimator.py"))) == 2:
        extra = ["--extra-frozen-glob", str(supplementary / "*" / "frozen_estimator.py")]
    (output / "protocol.json").write_text(json.dumps({
        "target_model": model, "target_revision": revision, "tokenizer": shared_tokenizer,
        "token_to_id_vocabulary_identical": True, "dtype": "float32",
        "selection": "Best source-model validation seed per evolution writer; fixed-feature evolution winner",
        "supervised_models_refit": False, "target_labels_used_for_selection": False,
        "cpu_gc_interval_batches": gc_interval,
        "cuda_cache_cleanup_interval_batches": cuda_interval,
        "supplementary_fixed_search_methods_included": bool(extra),
    }, indent=2) + "\n")
    run(["scripts/evaluate_cached_claims.py", "--mode", "cache", "--model-id", destination,
         "--tokenizer-id", shared_tokenizer, "--cache-dir", "/dev/shm/mod1-transfer-cache",
         "--output", str(output), "--primary-only", "--batch-size", "16", "--cpu-threads", "12",
         "--gc-interval", str(gc_interval), "--cuda-cache-interval", str(cuda_interval), *extra])
    run(["scripts/evaluate_parallel.py", "--workers", "3", "--cache-dir", "/dev/shm/mod1-transfer-cache",
         "--output", str(output), "--model-id", destination, "--tokenizer-id", shared_tokenizer,
         "--primary-only", "--bootstrap", "500", *extra])


if __name__ == "__main__":
    main()
