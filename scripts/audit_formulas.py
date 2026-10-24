"""Structural and invariance diagnostics; synthetic scores are not AUC evidence."""
import ast
import copy
import inspect
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import spearmanr

from packing.evaluate.uncertainty_quantification import found_formulas


def main():
    rng = np.random.default_rng(441)
    stats = {k: [] for k in ["sequence_logprobs", "greedy_logprobs", "token_entropies", "top_k_logits", "top_k_indices"]}
    for _ in range(128):
        length = int(rng.integers(3, 80))
        logits = rng.normal(0, rng.uniform(0.5, 3.0), (length, 1024))
        order = np.argsort(logits, axis=1)[:, ::-1][:, :512]
        top = np.take_along_axis(logits, order, axis=1)
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        token = rng.integers(0, 1024, length - 1)
        stats["sequence_logprobs"].append(lp[np.arange(length - 1), token].astype(np.float32))
        stats["greedy_logprobs"].append(lp.max(axis=1).astype(np.float32))
        stats["token_entropies"].append(-(np.exp(lp) * lp).sum(axis=1).astype(np.float32))
        stats["top_k_logits"].append(top.astype(np.float32))
        stats["top_k_indices"].append(order)
    rows, predictions = [], {}
    for name, cls in inspect.getmembers(found_formulas, inspect.isclass):
        if cls.__module__ != found_formulas.__name__:
            continue
        source = inspect.getsource(cls)
        tree = ast.parse(source)
        row = {"name": name, "lines": len(source.splitlines()), "ast_nodes": sum(1 for _ in ast.walk(tree))}
        try:
            started = time.perf_counter()
            estimator = cls({"supervised": False})
            full = np.asarray(estimator(copy.deepcopy(stats)), dtype=float)
            row["seconds_128_samples"] = time.perf_counter() - started
            row["finite"] = bool(full.shape == (128,) and np.isfinite(full).all())
            pieces = [np.asarray(estimator({k: v[i:i+16] for k, v in copy.deepcopy(stats).items()})) for i in range(0, 128, 16)]
            chunked = np.concatenate(pieces)
            row["chunk_invariant"] = bool(np.allclose(full, chunked, rtol=1e-5, atol=1e-6))
            shifted = copy.deepcopy(stats)
            shifted["top_k_logits"] = [a + rng.uniform(-10, 10, (len(a), 1)) for a in shifted["top_k_logits"]]
            changed = np.asarray(estimator(shifted))
            row["logit_shift_invariant"] = bool(np.allclose(full, changed, rtol=1e-4, atol=1e-5))
            row["logit_shift_rank_correlation"] = float(spearmanr(full, changed).statistic)
            predictions[name] = full
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    pairs = []
    names = sorted(predictions)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            rho = float(spearmanr(predictions[first], predictions[second]).statistic)
            if abs(rho) >= 0.995:
                pairs.append({"first": first, "second": second, "spearman": rho})
    result = {"data": "synthetic diagnostics only; no hallucination-quality conclusions",
              "seed": 441, "samples": 128, "methods": rows, "nearly_identical_rankings": pairs}
    output = Path("results/formula_diagnostics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"methods": len(rows), "failed": [r["name"] for r in rows if "error" in r],
                      "batch_dependent": [r["name"] for r in rows if r.get("chunk_invariant") is False],
                      "logit_shift_dependent": [r["name"] for r in rows if r.get("logit_shift_invariant") is False],
                      "nearly_identical_pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
