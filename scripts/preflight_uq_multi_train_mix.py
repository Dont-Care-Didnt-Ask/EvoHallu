#!/usr/bin/env python3
"""Preflight checks for UQ multi-dataset train-mix logits config."""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def _expected_suffix(input_regime: str) -> str | None:
    mapping = {
        "logits": "_logits.pt",
        "hidden_states": "_hidden_states.pt",
        "attentions": "_attentions.pt",
        "chunked": None,
        "full": None,
    }
    assert input_regime in mapping, (
        f"Unsupported input_regime='{input_regime}'. "
        "Expected one of: logits, hidden_states, attentions, chunked, full."
    )
    return mapping[input_regime]


def _required_keys(input_regime: str) -> set[str]:
    mapping = {
        "logits": {"sequence_logprobs", "top_k_logits", "top_k_indices"},
        "hidden_states": {"sequence_logprobs", "hidden_states"},
        "attentions": {"sequence_logprobs", "attentions"},
        "full": {"sequence_logprobs", "top_k_logits", "top_k_indices", "hidden_states", "attentions"},
        "chunked": {"sequence_logprobs", "top_k_logits", "top_k_indices", "hidden_states", "attentions"},
    }
    return mapping[input_regime]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight check for UQ multi task config.")
    parser.add_argument(
        "--task-config",
        type=str,
        default="configs/task/uncertainty_quantification_multi_train_mix_generations.yaml",
        help="Path to task config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    task_cfg_path = Path(args.task_config)
    assert task_cfg_path.exists(), f"Task config not found: {task_cfg_path}"

    cfg = OmegaConf.load(task_cfg_path)
    assert cfg.task_name == "uncertainty_quantification_multi", (
        f"Unexpected task_name='{cfg.task_name}' in {task_cfg_path}"
    )

    input_regime = str(cfg.input_regime)
    min_samples_per_dataset = int(cfg.min_samples_per_dataset)
    datasets = list(cfg.datasets)

    assert min_samples_per_dataset >= 1, (
        f"min_samples_per_dataset must be >= 1, got {min_samples_per_dataset}"
    )
    assert len(datasets) == 4 or len(datasets) == 2, f"Expected exactly 4 or 2 datasets, got {len(datasets)}"

    suffix = _expected_suffix(input_regime)
    required_keys = _required_keys(input_regime)

    print(f"[preflight] task config: {task_cfg_path}")
    print(f"[preflight] input_regime={input_regime} min_samples_per_dataset={min_samples_per_dataset}")
    print(f"[preflight] required sample keys for regime: {sorted(required_keys)}")

    file_suffixes = set()
    for ds_cfg in datasets:
        name = str(ds_cfg.name)
        path = Path(str(ds_cfg.test_raw_path))
        metric_name = str(ds_cfg.metric_name)

        chunked_path = Path(ds_cfg.test_raw_path.replace(".pt", "_chunk_0.pt"))

        assert path.exists() or chunked_path.exists(), f"Dataset '{name}' path does not exist: both {path} and {chunked_path} are missing"
        file_suffixes.add(path.suffix)

        if suffix is not None:
            assert str(path).endswith(suffix), (
                f"Dataset '{name}' path must end with '{suffix}' for regime '{input_regime}', got '{path}'."
            )

        raw = torch.load(path if path.exists() else chunked_path, map_location="cpu", mmap=True, weights_only=False)
        samples = raw.get("samples", [])
        assert isinstance(samples, list), f"Dataset '{name}' has non-list samples: {type(samples)}"
        assert len(samples) >= min_samples_per_dataset, (
            f"Dataset '{name}' has {len(samples)} samples; "
            f"requires at least {min_samples_per_dataset}."
        )

        missing_gt = [i for i, sample in enumerate(samples) if "ground_truth_score" not in sample]
        assert not missing_gt, (
            f"Dataset '{name}' is missing ground_truth_score for {len(missing_gt)} samples. "
            f"First missing index: {missing_gt[0]}"
        )

        first_keys = set(samples[0].keys())
        missing_keys = sorted(required_keys - first_keys)
        assert not missing_keys, (
            f"Dataset '{name}' missing required keys for regime '{input_regime}': {missing_keys}"
        )

        gt = np.asarray([float(sample["ground_truth_score"]) for sample in samples], dtype=np.float32)
        assert np.isfinite(gt).all(), f"Dataset '{name}' has non-finite ground_truth_score values."

        print(
            f"[preflight] {name}: samples={len(samples)} metric={metric_name} "
            f"gt_min={float(np.min(gt)):.6f} gt_max={float(np.max(gt)):.6f} gt_mean={float(np.mean(gt)):.6f}"
        )

    # Strict no mixed regime artifacts: all files must be .pt and all must satisfy single suffix rule above.
    assert file_suffixes == {".pt"}, f"Unexpected mixed file extensions: {sorted(file_suffixes)}"

    print("[preflight] OK: all checks passed.")


if __name__ == "__main__":
    main()
