"""CPU reproduction/data/environment gate for the next studies.

Keeps supplied is_hallu labels, historical artifacts, and cached arrays intact.
It does not load a language model or execute generated candidate code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from packing.evaluate.uncertainty_quantification.found_formulas import SequenceLogProb
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-zip", type=Path, default=Path("/root/.codex/attachments/295cdd20-86be-4ad4-9d2f-9e2341d05f35/datasets.zip"))
    ap.add_argument("--original-train", type=Path, default=Path("/root/.codex/attachments/5438d6d2-0e99-4acd-a190-7014620e7d6f/popqa_train_filtered_10shot.csv"))
    ap.add_argument("--output", type=Path, default=ROOT / "results/next_experiments/readiness.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    versions = {}
    for name in ["torch", "numpy", "scipy", "scikit-learn", "hydra-core", "transformers", "openai", "vllm"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    uv = shutil.which("uv")
    dependency_check = subprocess.run([uv, "pip", "check", "--python", str(ROOT / ".venv/bin/python")],
        capture_output=True, text=True, timeout=60) if uv else None
    environment = dict(python=platform.python_version(), packages=versions, cuda_available=torch.cuda.is_available(),
        dependency_check=None if dependency_check is None else dict(exit_code=dependency_check.returncode,
            output=(dependency_check.stdout+dependency_check.stderr).strip()),
        local_writer_serving_ready=torch.cuda.is_available() and versions["vllm"] is not None,
        caveat="CPU dependency compatibility does not certify the separate GPU serving environment")

    original_checks = []
    if args.datasets_zip.is_file():
        with zipfile.ZipFile(args.datasets_zip) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    local = ROOT / "data/claims" / name
                    expected = hashlib.sha256(z.read(name)).hexdigest()
                    original_checks.append(dict(file=name, source="supplied datasets.zip", sha256=expected,
                                                bytes_identical=local.is_file() and sha(local)==expected))
    train_csv = ROOT / "data/claims/popqa/popqa_train_filtered_10shot.csv"
    if args.original_train.is_file():
        original_checks.append(dict(file="popqa/popqa_train_filtered_10shot.csv", source="supplied standalone train CSV",
                                    sha256=sha(args.original_train), bytes_identical=sha(args.original_train)==sha(train_csv)))

    rows = []
    cohort = ROOT / "out/campaign/evaluation_with_4b"
    for csv in sorted((ROOT / "data/claims").rglob("*.csv")):
        if "popqa_train" in csv.name:
            continue
        relative = csv.relative_to(ROOT / "data/claims")
        score_path = cohort / (str(relative.with_suffix("")).replace("/", "__") + "_scores.npz")
        frame = pd.read_csv(csv, usecols=["claim", "is_hallu"])
        with np.load(score_path) as z:
            labels, scores = z["labels"], z["original/SequenceLogProb"]
            aligned = np.array_equal(labels, frame.is_hallu.to_numpy())
            rows.append(dict(dataset=str(relative), rows=len(labels), positives=int(labels.sum()),
                             saved_labels_match_csv_is_hallu=aligned,
                             auc_percent=100*float(roc_auc_score(labels, scores)),
                             csv_sha256=sha(csv), scores_sha256=sha(score_path)))
    macro = float(np.mean([r["auc_percent"] for r in rows]))

    raw_path = ROOT / "data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt"
    raw = torch.load(raw_path, mmap=True, weights_only=False, map_location="cpu")
    frame = pd.read_csv(train_csv, usecols=["claim", "is_hallu"])
    samples = raw["samples"]
    labels = np.array([s["label"] for s in samples])
    claims_match = len(samples)==len(frame) and all(str(s.get("claim"))==str(c) for s,c in zip(samples,frame.claim))
    shapes = Counter((len(s["sequence_logprobs"]), int(s["top_k_logits"].shape[0]), int(s["top_k_logits"].shape[1])) for s in samples)
    # This is the exact SequenceLogProb implementation used in the campaign;
    # float64 summation is a separate diagnostic, not a replacement baseline.
    scores = SequenceLogProb({"supervised": False})({"sequence_logprobs": [s["sequence_logprobs"] for s in samples]})
    train, val = _prepare_train_val_indices(raw, 0, .2, 441, "stratified")
    grouped_train, grouped_val = _prepare_train_val_indices(raw, 0, .2, 441, "claim_group")
    normalize = lambda text:" ".join(str(text).casefold().split())
    claims = np.array([normalize(s["claim"]) for s in samples])
    saved = np.load(ROOT / "out/logs/mod1/uncertainty_quantification_qwen35_9b_evo_1048/baseline_val_scores.npy")
    training = dict(rows=len(samples), raw_meta=raw.get("meta", {}), raw_sha256=sha(raw_path),
        labels_match_csv_is_hallu=np.array_equal(labels, frame.is_hallu.to_numpy()), claims_match_csv_in_order=claims_match,
        sequence_probability_input_dtypes=dict(Counter(str(np.asarray(s["sequence_logprobs"]).dtype) for s in samples)),
        rows_with_topk_length_equal_logprob_length=sum(count for (n,k,_),count in shapes.items() if n==k),
        rows_with_topk_length_one_greater=sum(count for (n,k,_),count in shapes.items() if k==n+1),
        rows_with_topk512=sum(count for (_,_,k),count in shapes.items() if k==512),
        train_rows=len(train), validation_rows=len(val), train_auc_percent=100*float(roc_auc_score(labels[train], scores[train])),
        validation_auc_percent=100*float(roc_auc_score(labels[val], scores[val])),
        saved_baseline_validation_scores_identical=np.array_equal(scores[val], saved),
        stratified_shared_normalized_claims=len(set(claims[train]) & set(claims[val])),
        grouped_shared_normalized_claims=len(set(claims[grouped_train]) & set(claims[grouped_val])),
        grouped_train_auc_percent=100*float(roc_auc_score(labels[grouped_train], scores[grouped_train])),
        grouped_validation_auc_percent=100*float(roc_auc_score(labels[grouped_val], scores[grouped_val])),
        grouped_train_rows=len(grouped_train), grouped_validation_rows=len(grouped_val))
    paper_audit = json.loads((ROOT / "results/paper_alignment.json").read_text())
    historical_hashes = {name:sha(ROOT/name)==expected
                         for name,expected in paper_audit["primary_artifact_sha256_before_and_after"].items()}
    folded_macro = float(np.mean([max(r["auc_percent"],100-r["auc_percent"]) for r in rows]))
    result = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), labels="is_hallu",
        no_gpu_or_model_calls=True, generated_candidate_code_executed=False,
        environment=environment, supplied_file_checks=original_checks,
        all_ten_supplied_csvs_identical=len(original_checks)==10 and all(r["bytes_identical"] for r in original_checks),
        benchmark_datasets=rows, evaluation_dataset_count=len(rows), files_including_training=len(rows)+1,
        macro_sequence_nll_auc_percent=macro, paper_sequence_probability_auc_percent=68.7,
        raw_vs_paper_reporting_difference_pp=68.7-macro,
        paper_direction_invariant_macro_auc_percent=folded_macro,
        paper_gap_resolved=round(folded_macro,1)==68.7,
        resolution="Author clarified per-dataset max(AUC,100-AUC) reporting; keep is_hallu unchanged",
        baseline_caveat="Saved-prediction replay; this verifies metric/labels but does not recreate historical paper features",
        training_cache=training, historical_primary_hashes_unchanged=historical_hashes,
        cpu_gate_passed=(len(rows)==9 and abs(macro-65.64388231516448)<1e-8
            and all(r["bytes_identical"] for r in original_checks)
            and all(r["saved_labels_match_csv_is_hallu"] for r in rows)
            and training["saved_baseline_validation_scores_identical"]
            and training["labels_match_csv_is_hallu"] and claims_match and all(historical_hashes.values())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False, default=str)+"\n")
    print(json.dumps({k:result[k] for k in ["cpu_gate_passed", "all_ten_supplied_csvs_identical",
                     "evaluation_dataset_count", "macro_sequence_nll_auc_percent", "paper_direction_invariant_macro_auc_percent", "paper_gap_resolved"]}, indent=2))
    print(json.dumps(training, indent=2, default=str))
    if not result["cpu_gate_passed"]:
        raise SystemExit("Readiness gate failed; inspect the saved audit before generation")


if __name__ == "__main__":
    main()
