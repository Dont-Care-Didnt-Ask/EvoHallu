"""Prepare immutable, CPU-only component drops for all four screen winners.

Default execution exports programs, verifies them on synthetic inputs, and writes
commands; it never evaluates a held-out dataset. --check-caches is an optional
read-only preflight used by the generated evaluation commands.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import difflib
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/overnight20261021/screen_component_ablations"
POLICIES = {
    "original": {
        "sha256": "624e5d7457cff945b4a47b947309e7f84c499431ba134408595e265a435e69d0",
        "assignment": "final_score",
        "expression": "0.7 * slope_score_norm + 0.3 * tail_collapse_norm",
        "terms": [
            ("normalized_slope", "0.7 * slope_score_norm", "0.7 times negative Pearson(position, token log-probability), divided by mean absolute token log-probability plus 1e-8"),
            ("inverse_tail_range", "0.3 * tail_collapse_norm", "0.3 / (1 + range(last 10% token log-probabilities)), divided by the same mean absolute token log-probability plus 1e-8"),
        ],
    },
    "neutral": {
        "sha256": "f307243eb19d36da7dff35740ba7c2b87d86ba3fcd37fff762b46069241eb53e",
        "assignment": "score",
        "expression": "total_neg_logprob + ALPHA * weighted_drift + BETA * tail_variance",
        "terms": [
            ("total_nll", "total_neg_logprob", "-sum(token log-probabilities); coefficient 1"),
            ("weighted_gap", "ALPHA * weighted_drift", "4 times sum_t [(t/(n-1)) * gap_t], with the original n=1 special case retained"),
            ("tail_gap_variance", "BETA * tail_variance", "2 times population variance of gaps from floor(0.75*n) to n"),
        ],
    },
    "diverse": {
        "sha256": "d7d9834511f783cf6b429ddca86a309cc6376598ba2eea61f63f9bc184ecf8de",
        "assignment": "score",
        "expression": "total_neg_logprob + ALPHA * normalized_drift + BETA * last_q_variance",
        "terms": [
            ("total_nll", "total_neg_logprob", "-sum(token log-probabilities); coefficient 1, on the final-sum branch"),
            ("relative_gap_drift", "ALPHA * normalized_drift", "2 times (mean(last-quarter gaps)-mean(first-quarter gaps))/max(mean(first-quarter gaps),1e-6), q=floor(n/4), last quarter starts at 3q"),
            ("tail_gap_variance", "BETA * last_q_variance", "2.5 times population variance of gaps from 3*floor(n/4) to n"),
        ],
    },
    "compositional": {
        "sha256": "cfe0f01cc27a635a30d4c3152c1d6d6c6a68e3da5a7423e23c9d535719438f4f",
        "assignment": "final_score",
        "expression": "(W_VAR * score_var) + (W_MEAN * score_mean) + (W_SLOPE * score_slope)",
        "terms": [
            ("tail_surprisal_variance", "W_VAR * score_var", "0.4 times population variance of negative log-probabilities in the last 10%; VAR_SCALE=1"),
            ("tail_surprisal_mean", "W_MEAN * score_mean", "0.3 times mean negative log-probability in the last 20%; MEAN_SCALE=1"),
            ("global_correlation", "W_SLOPE * score_slope", "0.3 times [-10 * Pearson(position, token log-probability)], effective coefficient 3"),
        ],
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def additive_terms(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return additive_terms(node.left) + additive_terms(node.right)
    return [node]


def final_assignment(source, spec):
    expected = ast.dump(ast.parse(spec["expression"], mode="eval").body)
    matches = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == spec["assignment"] for t in node.targets)
               and ast.dump(node.value) == expected]
    if len(matches) != 1:
        raise ValueError("Expected exactly one pinned final-score assignment")
    return matches[0]


def replace_term(source, spec, index):
    """Use UTF-8 AST offsets; retain all bytes outside one additive term."""
    node = additive_terms(final_assignment(source, spec).value)[index]
    expected = spec["terms"][index][1]
    if ast.dump(node) != ast.dump(ast.parse(expected, mode="eval").body):
        raise ValueError("Pinned term ordering differs")
    lines = source.encode().splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
    original = source.encode()
    modified = original[:start] + b"0.0" + original[end:]
    # Independently verify the complete AST changes at this term only.
    expected_tree = ast.parse(source)
    assignment = next(n for n in ast.walk(expected_tree) if isinstance(n, ast.Assign)
                      and n.lineno == final_assignment(source, spec).lineno)
    target = additive_terms(assignment.value)[index]

    class Drop(ast.NodeTransformer):
        def visit(self, item):
            if item is target:
                return ast.Constant(value=0.0)
            return super().visit(item)

    expected_tree = Drop().visit(expected_tree)
    if ast.dump(ast.parse(modified)) != ast.dump(expected_tree):
        raise AssertionError("Change exceeds the selected additive term")
    return modified, dict(line_1based=node.lineno, column_0based=node.col_offset,
                         end_line_1based=node.end_lineno, end_column_0based=node.end_col_offset,
                         before=original[start:end].decode(), after="0.0")


def load_detector(source):
    # Use the evaluator's exact parser and estimator interface, without loading
    # any feature cache or language model.
    from experiments.extension_trial import parse_candidate
    from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
    import scipy
    code, imports = parse_candidate(source)
    namespace = dict(Estimator=Estimator, np=np, scipy=scipy, Any=Any, Dict=Dict,
                     List=List, Tuple=Tuple, Optional=Optional)
    exec(imports + "\n" + code, namespace)
    return namespace["HallucinationDetector"]({"supervised": False, "meta": {"eval_meta": {}}})


def fixture(lengths, dtype=np.float32):
    """Variable-length, observed-token probabilities from a 768-token vocabulary."""
    from scipy.special import logsumexp
    rng = np.random.default_rng(5817)
    stats = dict(sequence_logprobs=[], top_k_logits=[], top_k_indices=[])
    for n in lengths:
        logits = rng.normal(0, 1.4, size=(n+1, 768)).astype(np.float32)
        # Nonconstant observations, alternating positive/negative confidence trends.
        observed = rng.integers(0, 768, size=n)
        lp = logits[np.arange(n), observed] - logsumexp(logits[:n], axis=1)
        indices = np.argsort(logits, axis=1)[:, ::-1][:, :512]
        stats["sequence_logprobs"].append(lp.astype(dtype))
        stats["top_k_logits"].append(np.take_along_axis(logits, indices, axis=1).astype(dtype))
        stats["top_k_indices"].append(indices)
    return stats


def trace_components(source, spec, stats):
    """Observe actual final-assignment locals; do not reimplement any feature."""
    detector = load_detector(source)
    # parse_candidate extracts/reformats the class, so locate the compiled
    # method's line using inspectable code text from the same parser.
    from experiments.extension_trial import parse_candidate
    code, imports = parse_candidate(source)
    actual_source = imports + "\n" + code
    line = final_assignment(actual_source, spec).lineno
    terms = [compile(ast.Expression(ast.parse(expr, mode="eval").body), "<component>", "eval")
             for _, expr, _ in spec["terms"]]
    captured = []

    def trace(frame, event, arg):
        if frame.f_code is detector.__call__.__func__.__code__ and event == "line" and frame.f_lineno == line:
            captured.append([float(eval(expr, frame.f_globals, frame.f_locals)) for expr in terms])
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        full = detector(copy.deepcopy(stats))
    finally:
        sys.settrace(previous)
    if len(captured) != len(stats["sequence_logprobs"]):
        raise AssertionError("Synthetic fixture did not exercise every final sum")
    return np.asarray(full), np.asarray(captured)


def verify(output, manifest):
    checks = []
    for policy, spec in POLICIES.items():
        rows = [r for r in manifest["methods"] if r["policy"] == policy]
        source = (ROOT / rows[0]["original_source_file"]).read_text()
        full_source = (ROOT / rows[0]["file"]).read_text()
        if source.encode() != full_source.encode():
            raise AssertionError("Full control is not byte-identical")
        for dtype in [np.float32, np.float16]:
            stats = fixture([8, 13, 31, 48, 67], dtype)
            full, components = trace_components(source, spec, stats)
            actual = load_detector(full_source)(copy.deepcopy(stats))
            np.testing.assert_array_equal(full, actual)
            if not np.isfinite(full).all() or full.shape != (5,):
                raise AssertionError("Unexpected synthetic full prediction")
            np.testing.assert_allclose(full, components.sum(axis=1), rtol=3e-6, atol=2e-5)
            drops = []
            for row in rows[1:]:
                modified = (ROOT / row["file"]).read_text()
                values = load_detector(modified)(copy.deepcopy(stats))
                expected = components.sum(axis=1) - components[:, row["dropped_component_index"]]
                np.testing.assert_allclose(values, expected, rtol=3e-6, atol=8e-5)
                if np.array_equal(values, full):
                    raise AssertionError("Dropped component never changes this fixture")
                drops.append(values)
            np.testing.assert_allclose(np.sum(drops, axis=0), (len(spec["terms"])-1)*full,
                                       rtol=3e-6, atol=1e-4)
            # Each row must remain independent of its neighbors and chunk size.
            singles = np.concatenate([load_detector(full_source)({k: [v[i]] for k,v in stats.items()})
                                      for i in range(len(full))])
            np.testing.assert_array_equal(full, singles)
            checks.append(dict(policy=policy, stored_feature_dtype=np.dtype(dtype).name,
                lengths=[8,13,31,48,67], full_control_byte_parity=True,
                exact_full_prediction_parity=True, actual_traced_component_identity=True,
                all_drop_identities=True, additive_reconstruction_identity=True,
                prediction_independent_of_chunk_neighbors=True))
        # Preserve the diverse program's short-sequence NLL fallback explicitly.
        if policy == "diverse":
            short = fixture([0,1,2,3])
            expected = np.array([0 if len(lp)==0 else -np.sum(lp) for lp in short["sequence_logprobs"]], dtype=np.float32)
            for row in rows:
                np.testing.assert_array_equal(load_detector((ROOT/row["file"]).read_text())(copy.deepcopy(short)), expected)
    dump(output / "verification.json", dict(status="passed", gpu_or_model_calls=False,
        held_out_predictions_read=False, checks=checks, diverse_short_nll_fallback_preserved=True))


def cache_preflight(args, output):
    """Check pinned raw bytes and row identity before separately launched scoring."""
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((output / "manifest.json").read_text())
    actual_paths = {relative(p) for p in (output/"frozen").glob("*/frozen_estimator.py")}
    if actual_paths != {row["file"] for row in manifest["methods"]}:
        raise ValueError("Frozen folder contents differ from the exact 15-method manifest")
    for method in manifest["methods"]:
        if sha(ROOT / method["file"]) != method["sha256"]:
            raise ValueError("Exported estimator hash changed: " + method["method"])
    for file, fingerprint in manifest["data_sha256"].items():
        if sha(ROOT / file) != fingerprint:
            raise ValueError("Dataset bytes changed: " + file)
    files = [ROOT / row["csv"] for row in manifest["datasets"]]
    group = "mod1-eval-cache" if args.check_caches == "source-nine" else "mod1-transfer-cache"
    indices = range(9) if args.check_caches == "source-nine" else range(6)
    cache_dir = args.source_cache if group == "mod1-eval-cache" else args.target_cache
    rows = []
    for index in indices:
        dataset = manifest["datasets"][index]
        path = cache_dir / (dataset["name"] + ".pt")
        expected_sha = manifest["pinned_cache_sha256"][group][path.name]
        if sha(path) != expected_sha:
            raise ValueError("Raw feature cache hash changed: " + str(path))
        raw = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
        samples = raw["samples"]
        frame = pd.read_csv(files[index], usecols=["claim", "is_hallu"])
        if len(samples) != len(frame) or [s["claim"] for s in samples] != frame.claim.astype(str).str.strip().tolist() or not np.array_equal([s["label"] for s in samples], frame.is_hallu.to_numpy()):
            raise ValueError("Feature-cache claims or is_hallu labels differ from CSV in order: " + str(path))
        minimum_length = min(len(s["sequence_logprobs"]) for s in samples)
        if minimum_length < 5:
            raise ValueError("A cache exercises a retained short-sequence fallback; review scope before scoring: " + str(path))
        meta = raw.get("meta", {})
        if group == "mod1-transfer-cache":
            if any(meta.get(k) != v for k,v in dict(hidden_size=2048, num_layers=16, num_attention_heads=32, vocab_size=128256, modalities="logits", model_dtype="torch.float32").items()):
                raise ValueError("Unexpected target cache model metadata")
            source = torch.load(args.source_cache / path.name, mmap=True, weights_only=False, map_location="cpu")
            if len(source["samples"]) != len(samples) or not all(a["claim"]==b["claim"] and a["label"]==b["label"] and np.array_equal(a["input_ids"],b["input_ids"]) and np.array_equal(a["orig_input_ids"],b["orig_input_ids"]) for a,b in zip(source["samples"],samples)):
                raise ValueError("Matched source/target rows or chat/raw input token IDs differ")
            del source
        rows.append(dict(dataset=dataset["name"], raw_sha256=expected_sha, rows=len(samples),
                         claims_and_labels_match_csv_in_order=True, minimum_sequence_logprob_length=minimum_length, meta=meta))
        del raw, samples
        gc.collect()
    dump(output / (args.check_caches.replace("-", "_") + "_preflight.json"), dict(status="passed", datasets=rows,
        mode="read_only_cache_verification", held_out_scores_computed=False))


def prepare(args, output):
    inspection_path = ROOT / "results/overnight20261021/SCREEN_CANDIDATE_INSPECTION.json"
    inspection = json.loads(inspection_path.read_text())
    files = [p for p in sorted((ROOT/"data/claims").rglob("*.csv")) if "popqa_train" not in p.name]
    if len(files) != 9:
        raise ValueError("Expected exactly nine held-out CSVs")
    datasets = [dict(index=i, csv=relative(p), name=p.parent.name+"__"+p.stem) for i,p in enumerate(files)]
    archive = json.loads((ROOT / "results/feature_cache_validation.json").read_text())["archives"]
    cache_hashes = {group: {Path(path).name.removesuffix(".gz"): value["raw_sha256"]
        for path,value in archive.items() if Path(path).parent.name == group}
        for group in ["mod1-eval-cache", "mod1-transfer-cache"]}
    methods = []
    for policy, spec in POLICIES.items():
        row = inspection["policies"][policy]
        selected = row["top_generated"][0]
        if row["run_status"] != "finished" or row["requests"] != 30 or selected["candidate_hash"] != row["summary_selected_candidate"]:
            raise ValueError("Screen selection is not final and validation-frozen")
        path = ROOT / f"out/logs/research20261021/uncertainty_quantification_next_prompt_{policy}_2107/frozen_estimator.py"
        if sha(path) != spec["sha256"] or selected["source_file_sha256"] != spec["sha256"]:
            raise ValueError("Pinned winner hash differs: " + policy)
        source = path.read_text()
        if len(additive_terms(final_assignment(source,spec).value)) != len(spec["terms"]):
            raise ValueError("Unexpected final term count")
        definitions = [dict(name=name, expression=expr, definition=definition) for name,expr,definition in spec["terms"]]
        for index in [None] + list(range(len(spec["terms"]))):
            role = "full" if index is None else "drop_" + spec["terms"][index][0]
            folder = output / "frozen" / f"screen_{policy}__{role}"
            folder.mkdir(parents=True, exist_ok=True)
            modified, patch = (path.read_bytes(),None) if index is None else replace_term(source,spec,index)
            target = folder / "frozen_estimator.py"
            if target.exists() and target.read_bytes() != modified:
                raise ValueError("Refusing to overwrite an existing distinct frozen variant")
            target.write_bytes(modified)
            diff = "".join(difflib.unified_diff(source.splitlines(keepends=True),modified.decode().splitlines(keepends=True),fromfile=relative(path),tofile=relative(target)))
            (folder/"component.patch").write_text(diff)
            method = dict(method="new/"+folder.name, policy=policy, role=role, file=relative(target),
                sha256=sha(target), original_source_file=relative(path), original_source_sha256=spec["sha256"],
                source_candidate_hash=selected["candidate_hash"], source_selection="directed source PopQA validation only",
                final_score_expression=spec["expression"], components=definitions, dropped_component_index=index,
                dropped_component=None if index is None else spec["terms"][index][0], patch=patch,
                unchanged_full_control=index is None, fitted=False, coefficients_unchanged=True)
            methods.append(method)
    if len(methods) != 15:
        raise AssertionError("Expected four controls and eleven component drops")
    manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), study="all_four_screen_winner_fixed_component_ablations",
        status="prepared_not_scored", selection="All four completed screen source-validation winners; no exclusion using test quality",
        held_out_predictions_read_for_preparation=False, new_llm_or_gpu_calls=False,
        source_inspection_sha256=sha(inspection_path), preparation_script_sha256=sha(__file__),
        methods=methods, datasets=datasets, target_matched_indices=list(range(6)),
        data_sha256={relative(p):sha(p) for p in sorted((ROOT/"data/claims").rglob("*.csv"))},
        pinned_cache_sha256=cache_hashes, bootstrap_samples=500, bootstrap_unit="claim", scoring_chunk_size=512)
    dump(output/"manifest.json",manifest)
    verify(output,manifest)
    make_commands(args,output)
    print(json.dumps(dict(status="prepared_and_synthetic_checks_passed", methods=15, new_model_calls=0,
        actual_held_out_scoring_executed=False, directory=relative(output))))


def make_commands(args, output):
    output_rel = relative(output)
    env = ["env", "PYTHONPATH=src:.", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", ".venv/bin/python"]
    common = ["--data-root", "data/claims", "--frozen-glob", output_rel+"/frozen/*/frozen_estimator.py",
              "--method-prefix", "new/screen_", "--bootstrap", "500", "--bootstrap-unit", "claim"]
    preflight = ["scripts/prepare_screen_component_ablations.py", "--output", output_rel,
                 "--source-cache", str(args.source_cache), "--target-cache", str(args.target_cache)]
    source = env + ["scripts/evaluate_parallel.py", "--workers", "2", "--cache-dir", str(args.source_cache),
        "--output",output_rel+"/evaluations/source_nine", "--model-id", args.source_model] + common
    target = [env + ["scripts/evaluate_cached_claims.py", "--mode", "evaluate", "--dataset-index", str(i),
        "--cache-dir",str(args.target_cache), "--output",output_rel+"/evaluations/target_six", "--cpu-threads","1",
        "--model-id","unsloth/Llama-3.2-1B-Instruct", "--tokenizer-id",args.shared_tokenizer] + common for i in range(6)]
    commands = {"source_nine": [env+preflight+["--check-caches","source-nine"], source],
                "target_six": [env+preflight+["--check-caches","target-six"]]+target}
    dump(output/"commands.json",dict(executed=False, **commands))
    for name, rows in commands.items():
        (output/ ("run_"+name+".sh")).write_text("#!/usr/bin/env bash\nset -euo pipefail\n# Run from repository root. CPU only; explicit --mode evaluate in child evaluator.\n"+"\n".join(shlex.join(command) for command in rows)+"\n")
    text = """# Frozen screen component ablations

Prepared, not yet scored. All four completed 30-request screen winners are included, each frozen by directed source PopQA validation. There are four unchanged controls and eleven leave-one-additive-component-out variants. Test scores were not read to choose programs, terms, or coefficients. This is a mechanistic follow-up to inspecting the generated source, not a prompt-policy replication or an additional writer seed.

Every `full` program is byte-identical to its original frozen file. A drop replaces exactly one AST additive-term span of the final score with `0.0`; the full class, imports, features, constants, guards, order of operations outside that term, and evaluator interface remain unchanged. Each folder contains a unified `component.patch`; exact byte hashes, source candidate identities, term definitions, and patch spans are in `manifest.json`. Programs require only the same evaluator-supplied `Estimator` and typing names as their originals; they do not import other candidate files. No refitting, coefficient tuning, primary-registry change, or new model request is performed.

| Source policy | Frozen final score | Drops |
|---|---|---|
| Original | `0.7*normalized_slope + 0.3*normalized_inverse_tail_range` | slope; inverse tail range |
| Neutral | `NLL + 4*weighted_gap + 2*tail_gap_variance` | NLL; weighted gap; variance |
| Diverse | `NLL + 2*relative_gap_drift + 2.5*tail_gap_variance` | NLL; relative drift; variance |
| Compositional | `0.4*tail_surprisal_variance + 0.3*tail_surprisal_mean + 3*negative_position_correlation` | variance; tail mean; correlation |

Neutral/diverse `gap_t = max(top512_logits_t - logsumexp(top512_logits_t)) - token_logprob_t`. The top512 normalization is conditional on the stored candidates, not a full-vocabulary softmax. Rows `0..n-1` are used from the `n+1` top-k cache. `q=floor(n/4)` and the diverse last quarter begins at `3q`, so it can include a remainder. Original normalization and every shared feature remain present when one final term is dropped: e.g. removing explicit NLL does not remove all token log-probability information. The terms share inputs and are not orthogonal causal mechanisms.

**Retained limitations.** Diverse returns NLL before its final sum when `n<4`; all copies deliberately preserve that fallback, including `drop_total_nll`. Original returns zero for `n<5`. Preflight refuses caches containing `n<5` so the requested comparisons exercise the actual final sums for all four programs. No scoring function has been repaired: Pearson correlation can still produce NaN for constant sequences; the existing evaluator must report failures transparently. Shared denominator/feature coupling remains intact. These are ablations of the final additive contributions, not removal of all computations that contribute to a concept. Expensive removed features are still calculated, so runtime savings are not an outcome.

**Verification.** `verification.json` records byte-identical controls, compilation through the evaluator's actual candidate parser, exact full-prediction parity, independently traced final-term contributions, every drop identity, reconstruction `sum_j drop_j = (m-1)*full`, and independence from batch neighbors. Fixtures use variable lengths 8/13/31/48/67, a 768-token vocabulary with stored top512 and n+1 rows, and float32 plus float16 storage. Diverse short-sequence NLL fallback is also checked. These fixtures have no labels and do not estimate generalization.

**Execution.** Run `bash run_source_nine.sh` and `bash run_target_six.sh` from the repo root using their repository-relative paths. Scripts first verify every exported hash and CSV hash, historical raw feature-cache SHA256, cached claim/is_hallu row order, and minimum lengths. Target preflight also checks 1B architecture/float32 metadata and exact source/target chat/raw input-token parity. Historical model/tokenizer revision evidence remains in the pinned transfer protocol and cache-audit documents; raw cache hashes bind this follow-up to those exact bytes. No feature extraction is possible in these commands. Source uses all nine datasets; target uses exactly sorted CSV indices 0..5 (AVeriTeC, UHEAD, WildHallucinations, Cities, Common claims, Companies), explicitly six of nine. A cross-model macro must compare source on this same six-dataset subset. No target refit or reselection is allowed.

Both evaluations keep `is_hallu` labels, unchanged 512-row scoring chunks, no header trimming, and 500 shared claim-level bootstrap resamples. `new/screen_` scores exactly these 15 programs plus the shared NLL reference. Outputs are separate from primary evaluations. Report directed AUROC and the paper's per-dataset `max(AUC, 1-AUC)` convention, then unweighted dataset means; fold bootstrap draws by the same rule. Report every full-minus-drop paired difference, with matching claim-bootstrap replicate indices; no test-based choice of a winner. For transfer compare source and target on precisely the same six rows/datasets and preserved source coefficients. These multiple exploratory comparisons are descriptive; n=1 writer seed per policy does not support a seed-general prompt ranking.
"""
    (output/"DESIGN.md").write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--source-cache",type=Path,default=Path("data/restored/mod1-eval-cache"))
    ap.add_argument("--target-cache",type=Path,default=Path("data/restored/mod1-transfer-cache"))
    ap.add_argument("--source-model",default="/workspace/mod1-models/llama31")
    ap.add_argument("--shared-tokenizer",default="/workspace/mod1-models/llama31-tokenizer")
    ap.add_argument("--check-caches",choices=["source-nine","target-six"])
    args = ap.parse_args()
    os.chdir(ROOT)
    output = args.output.resolve()
    output.relative_to(ROOT)
    if args.check_caches:
        cache_preflight(args,output)
    else:
        prepare(args,output)


if __name__ == "__main__":
    main()
