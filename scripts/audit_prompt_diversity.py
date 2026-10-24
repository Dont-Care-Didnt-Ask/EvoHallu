"""Inspect saved prompts and prediction ranks without running generated code.

Run from the repository root with .venv/bin/python. Validation comparisons are
retrospective diagnostics, never a proposed search-time novelty signal.
"""
from __future__ import annotations

import ast
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import rankdata


THRESHOLD = 0.995
BASELINES = {
    "LogProb": "SequenceLogProb",
    "LogProbSlope": "LogProbSlope",
    "PositionWeightedGap": "PositionWeightedGap",
}


def read(path):
    return json.loads(path.read_text())


def rank_vector(values):
    ranks = rankdata(values).astype(np.float64)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm else None


def fingerprint(values):
    inverse = np.unique(values, return_inverse=True)[1].astype("<i4")
    return hashlib.sha256(inverse.tobytes()).hexdigest()


def input_accesses(code):
    """Literal input accesses outside __init__; a static proxy, not execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            node.body = []
    found = set()
    keys = {"sequence_logprobs", "top_k_logits", "top_k_indices"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str) and node.slice.value in keys:
                found.add(node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in keys:
                found.add(node.args[0].value)
    return sorted(found)


def describe_pairs(pairs):
    if not pairs:
        return {"pairs": 0, "near_pairs": 0, "near_fraction": None, "median_spearman": None}
    return dict(pairs=len(pairs), near_pairs=sum(r["rho"] >= THRESHOLD for r in pairs),
                near_fraction=sum(r["rho"] >= THRESHOLD for r in pairs) / len(pairs),
                median_spearman=float(np.median([r["rho"] for r in pairs])),
                exact_rank_pairs=sum(r["exact"] for r in pairs))


def aggregate(runs):
    prompts = [p for r in runs for p in r["prompts"]]
    pairs = [p for r in runs for p in r["pairs"]]
    candidates = [c for r in runs for c in r["candidates"]]
    comparable = [c for c in candidates if c.get("rho_to_prompt_baseline") is not None]
    return dict(
        runs=len(runs), requests=sum(r["requests"] for r in runs), rounds=len(prompts),
        valid=sum(r["valid"] for r in runs),
        unique_valid_source_hashes=len({c["code_hash"] for c in candidates if c["valid"]}),
        unique_system_prompts=len({p["system_hash"] for p in prompts}),
        named_baseline_exposures=dict(Counter(p["baseline"] for p in prompts)),
        auxiliary_idea_exposures=dict(Counter(p["idea"] for p in prompts)),
        median_parent_to_system_character_ratio=float(np.median([
            p["user_characters"] / p["system_characters"] for p in prompts])),
        valid_candidates_near_prompt_baseline=sum(c["rho_to_prompt_baseline"] >= THRESHOLD for c in comparable),
        valid_nonconstant_candidates_compared_to_prompt_baseline=len(comparable),
        static_input_access_patterns=dict(Counter(
            "+".join(c["input_accesses"]) or "none_detected"
            for c in candidates if c["valid"] and c["input_accesses"] is not None)),
        pairs={
            "same_round_siblings": describe_pairs([p for p in pairs if p["round_distance"] == 0]),
            "other_round_distance_1_or_2": describe_pairs([p for p in pairs if 1 <= p["round_distance"] <= 2]),
            "all_other_rounds": describe_pairs([p for p in pairs if p["round_distance"] > 0]),
            "other_rounds_same_system_text": describe_pairs([p for p in pairs if p["round_distance"] > 0 and p["same_system"]]),
            "other_rounds_different_system_text": describe_pairs([p for p in pairs if p["round_distance"] > 0 and not p["same_system"]]),
        })


def main():
    root = Path(__file__).resolve().parents[1]
    references = np.load(root / "out/campaign/original_validation/stratified_val_scores.npz")
    reference_vectors = {key: rank_vector(references[name]) for key, name in BASELINES.items()}
    runs = []
    for config_path in sorted((root / "out/logs/mod1").glob("*/trial_config.json")):
        folder = config_path.parent
        if (folder / "exclude_from_evaluation.json").exists() or "pilot" in folder.name:
            continue
        cfg = read(config_path)
        if cfg.get("search_mode") not in {"evolution", "independent", "baseline_seeded"}:
            continue
        if not (folder / "candidates.jsonl").exists():
            continue
        rows = [json.loads(line) for line in (folder / "candidates.jsonl").read_text().splitlines()]
        prompts = {}
        for round_index in sorted({r["round"] for r in rows}):
            chat = read(folder / f"prompt_{round_index:04d}.json")
            system = chat[0]["content"]
            baseline = re.search(r"Build off a strong baseline: (\w+):", system).group(1)
            idea = re.search(r"For additional signal, use ([^\n]+)", system).group(1)
            prompts[round_index] = dict(round=round_index, baseline=baseline, idea=idea,
                system_hash=hashlib.sha256(system.encode()).hexdigest(),
                system_characters=len(system), user_characters=len(chat[1]["content"]))
        vectors, fingerprints, candidates = {}, {}, []
        previous = [rank_vector(np.load(folder / "baseline_val_scores.npy"))]
        for row in rows:
            valid = "score" in row.get("val", {})
            item = dict(round=row["round"], index=row["index"], code_hash=row.get("code_hash"),
                        valid=valid, input_accesses=input_accesses(row.get("code", "")),
                        prompt_baseline=prompts[row["round"]]["baseline"])
            if valid:
                values = np.load(folder / f"val_scores_{row['code_hash']}.npy")
                key = (row["round"], row["index"])
                vector = rank_vector(values)
                fingerprints[key] = fingerprint(values)
                item["constant_predictions"] = vector is None
                item["rho_to_prompt_baseline"] = None if vector is None else float(vector @ reference_vectors[item["prompt_baseline"]])
                item["near_previous_or_initial_nll"] = False if vector is None else any(
                    old is not None and float(vector @ old) >= THRESHOLD for old in previous)
                if vector is not None:
                    vectors[key] = vector
                previous.append(vector)
            candidates.append(item)
        pairs = []
        keys = list(vectors)
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                pairs.append(dict(a=list(a), b=list(b), round_distance=abs(a[0] - b[0]),
                    same_system=prompts[a[0]]["system_hash"] == prompts[b[0]]["system_hash"],
                    rho=float(vectors[a] @ vectors[b]), exact=fingerprints[a] == fingerprints[b]))
        run = dict(run=folder.name, writer=cfg["model"]["hf_model_id"], mode=cfg["search_mode"],
                   seed=cfg["seed"], requests=len(rows), valid=sum(c["valid"] for c in candidates),
                   prompts=list(prompts.values()), candidates=candidates, pairs=pairs,
                   generation={k:cfg["model"][k] for k in ["temperature", "topp", "topk", "max_tokens"]})
        run["summary"] = aggregate([run])
        runs.append(run)
    primary = [r for r in runs if r["mode"] == "evolution"]
    assert len(primary) == 7 and sum(r["requests"] for r in primary) == 402
    assert sum(r["valid"] for r in primary) == 283
    result = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        no_gpu_or_model_calls=True, generated_candidate_code_executed=False,
        near_repeat_positive_spearman_threshold=THRESHOLD,
        scope="Retrospective saved validation ranks; pair counts are dependent and validity-conditioned; no causal prompt ablation.",
        primary=aggregate(primary),
        by_writer={writer:aggregate([r for r in primary if r["writer"] == writer]) for writer in sorted({r["writer"] for r in primary})},
        controls=[dict(run=r["run"], writer=r["writer"], mode=r["mode"], **r["summary"]) for r in runs if r["mode"] != "evolution"],
        runs=[dict({k:v for k,v in r.items() if k != "pairs"},
                   sibling_pairs=[p for p in r["pairs"] if p["round_distance"] == 0]) for r in runs])
    output = root / "results/prompt_diversity_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:result[k] for k in ["primary", "by_writer", "controls"]}, indent=2))


if __name__ == "__main__":
    main()
