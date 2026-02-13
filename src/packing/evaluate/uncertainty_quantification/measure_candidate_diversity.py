from __future__ import annotations

"""Measure diversity between candidate hallucination detectors.

This script:
- loads a precomputed eval_raw .pt (torch.load dict with keys: samples, meta)
- loads candidate programs from a candidates directory
- execs each candidate's Estimator subclass and scores the dataset
- saves the (n_candidates x n_samples) score matrix as CSV
- computes pairwise Spearman similarity matrix (via correlation of ranks)
- computes set-wise diversity on rank-normalized matrix: effective rank + k thresholds
- computes a line plot of adjacent-round similarity using the best-per-round (max Val AUC)

Run with:
  python -m packing.evaluate.uncertainty_quantification.measure_candidate_diversity \
    --eval-raw-path /path/to/eval_raw.pt \
    --candidates-dir /path/to/candidates \
    --output-dir /path/to/output
"""

import argparse
import gc
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import rankdata, spearmanr
from joblib import Parallel, delayed

from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.utils.functions import extract_class, separate_imports_from_func


@dataclass(frozen=True)
class CandidateProgram:
    candidate_id: str
    path: Path
    round_num: int
    function_num: int
    val_auc: float | None
    test_auc: float | None
    imports_str: str
    class_str: str


_CANDIDATE_FILENAME_RE = re.compile(r"candidate_round_(\d+)_function_(\d+)\.txt$")
_VAL_AUC_RE = re.compile(r"^\s*#\s*Val AUC:\s*([0-9]*\.?[0-9]+)\b")
_TEST_AUC_RE = re.compile(r"^\s*#\s*Test AUC:\s*([0-9]*\.?[0-9]+)\b")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_eval_raw(eval_raw_path: Path) -> Dict[str, Any]:
    assert eval_raw_path.is_file(), f"--eval-raw-path not found: {eval_raw_path}"
    eval_raw = torch.load(str(eval_raw_path), map_location="cpu", mmap=True, weights_only=False)
    assert isinstance(eval_raw, dict), f"eval_raw must be a dict, got {type(eval_raw)}"
    assert "samples" in eval_raw, "eval_raw dict must contain key 'samples'"
    assert isinstance(eval_raw["samples"], list), "eval_raw['samples'] must be a list"
    return eval_raw


def _parse_candidate_filename(path: Path) -> Tuple[int, int]:
    m = _CANDIDATE_FILENAME_RE.search(path.name)
    assert m is not None, f"Unrecognized candidate filename format: {path.name}"
    return int(m.group(1)), int(m.group(2))


def _parse_auc_headers(text: str) -> Tuple[float | None, float | None]:
    val_auc: float | None = None
    test_auc: float | None = None
    for ln in text.splitlines():
        m_val = _VAL_AUC_RE.match(ln)
        if m_val is not None:
            val_auc = float(m_val.group(1))
            continue
        m_test = _TEST_AUC_RE.match(ln)
        if m_test is not None:
            test_auc = float(m_test.group(1))
            continue
        if not ln.lstrip().startswith("#"):
            break
    return val_auc, test_auc


def _load_candidate_program(candidate_path: Path, *, class_name: str, base_class: str) -> CandidateProgram:
    assert candidate_path.is_file(), f"Candidate file not found: {candidate_path}"
    text = candidate_path.read_text(encoding="utf-8")

    round_num, function_num = _parse_candidate_filename(candidate_path)
    val_auc, test_auc = _parse_auc_headers(text)

    imports_str, _ = separate_imports_from_func(text)
    class_str = extract_class(text, class_name=class_name, base_class=base_class)
    assert class_str.strip(), f"Failed to extract class '{class_name}' from {candidate_path}"

    return CandidateProgram(
        candidate_id=candidate_path.stem,
        path=candidate_path,
        round_num=round_num,
        function_num=function_num,
        val_auc=val_auc,
        test_auc=test_auc,
        imports_str=imports_str,
        class_str=class_str,
    )


def _exec_and_instantiate_estimator(
    program: CandidateProgram,
    *,
    class_name: str,
    eval_meta: Mapping[str, Any],
    supervised: bool,
) -> Estimator:
    globals_dict: Dict[str, Any] = {
        "Estimator": Estimator,
        "np": np,
        "Dict": Dict,
        "Any": Any,
        "Tuple": Tuple,
        "List": List,
    }

    if program.imports_str.strip():
        exec(program.imports_str, globals_dict)

    local_dict: Dict[str, Any] = {}
    exec(program.class_str, globals_dict, local_dict)

    assert class_name in local_dict, f"Class {class_name!r} not found after exec() for {program.path}"

    EstimatorClass = local_dict[class_name]
    context = {"supervised": bool(supervised), "meta": {"eval_meta": dict(eval_meta)}}
    estimator = EstimatorClass(context)
    assert isinstance(estimator, Estimator), f"{class_name} must be an Estimator subclass, got {type(estimator)}"
    return estimator


def _score_candidate_on_eval_raw(
    program: CandidateProgram,
    *,
    class_name: str,
    eval_raw: Mapping[str, Any],
    input_regime: str,
    chunk_size: int,
    supervised: bool,
) -> np.ndarray:
    eval_samples = eval_raw["samples"]
    assert isinstance(eval_samples, list), "eval_raw['samples'] must be a list"
    assert chunk_size >= 1, f"chunk_size must be >=1, got {chunk_size}"

    estimator = _exec_and_instantiate_estimator(
        program,
        class_name=class_name,
        eval_meta=eval_raw.get("meta", {}),
        supervised=supervised,
    )

    scores_chunks: List[np.ndarray] = []
    train_stats_cache: Dict[str, Any] | None = None
    background_stats_cache: Dict[str, Any] | None = None

    for start in range(0, len(eval_samples), chunk_size):
        end = min(len(eval_samples), start + chunk_size)
        chunk_raw = {"samples": eval_samples[start:end], "meta": eval_raw.get("meta", {})}

        all_stats_chunk, _, train_stats_cache, background_stats_cache = make_all_stats(
            eval_raw=chunk_raw,
            train_raw=None,
            background_raw=None,
            supervised=bool(supervised),
            train_stats_cache=train_stats_cache,
            background_stats_cache=background_stats_cache,
            input_regime=input_regime,
        )

        chunk_scores = np.asarray(estimator(all_stats_chunk), dtype=np.float32)
        assert chunk_scores.ndim == 1, f"{program.candidate_id}: expected 1D scores, got shape {chunk_scores.shape}"
        assert chunk_scores.shape[0] == (end - start), f"{program.candidate_id}: expected {end-start} scores for chunk, got {chunk_scores.shape[0]}"
        scores_chunks.append(chunk_scores)

        del all_stats_chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = np.concatenate(scores_chunks, axis=0) if scores_chunks else np.zeros((0,), dtype=np.float32)
    assert scores.shape == (len(eval_samples),), f"{program.candidate_id}: expected shape {(len(eval_samples),)}, got {scores.shape}"
    return scores


def _rank_normalize_rows(scores_matrix: np.ndarray) -> np.ndarray:
    assert scores_matrix.ndim == 2, f"scores_matrix must be 2D, got shape {scores_matrix.shape}"
    n_candidates, n_samples = scores_matrix.shape
    assert n_samples > 1, f"Need n_samples>1 for rank normalization, got n_samples={n_samples}"
    ranks = np.vstack([rankdata(scores_matrix[i], method="average") for i in range(n_candidates)])
    return (ranks - 1.0) / float(n_samples - 1)


def _spearman_corr_matrix(scores_matrix: np.ndarray) -> np.ndarray:
    rank_matrix = _rank_normalize_rows(scores_matrix)
    corr = np.corrcoef(rank_matrix)
    np.fill_diagonal(corr, 1.0)
    return corr


def _effective_rank_and_k_thresholds(rank_norm_matrix: np.ndarray, thresholds: Sequence[float]) -> Tuple[float, Dict[float, int]]:
    assert rank_norm_matrix.ndim == 2, f"rank_norm_matrix must be 2D, got shape {rank_norm_matrix.shape}"

    _, s, _ = np.linalg.svd(rank_norm_matrix, full_matrices=False)
    weights = (s ** 2).astype(np.float64)
    denom = float(weights.sum())
    assert denom > 0.0, "SVD weights sum to zero; rank-norm matrix is all zeros?"

    p = weights / denom
    entropy = -float(np.sum([pi * math.log(pi) for pi in p if pi > 0.0]))
    effective_rank = float(math.exp(entropy))

    cum = np.cumsum(p)
    k_by_threshold = {float(t): int(np.searchsorted(cum, float(t), side="left") + 1) for t in thresholds}
    return effective_rank, k_by_threshold


def _select_best_candidate_per_round(programs: Sequence[CandidateProgram]) -> Dict[int, CandidateProgram]:
    by_round: Dict[int, List[CandidateProgram]] = {}
    for p in programs:
        by_round.setdefault(p.round_num, []).append(p)

    def _score_for_selection(p: CandidateProgram) -> float:
        if p.val_auc is not None:
            return float(p.val_auc)
        if p.test_auc is not None:
            return float(p.test_auc)
        return float("-inf")

    best_by_round: Dict[int, CandidateProgram] = {}
    for rnd, ps in by_round.items():
        ps_sorted = sorted(ps, key=_score_for_selection, reverse=True)
        assert ps_sorted, f"Empty programs list for round {rnd}"
        best_by_round[rnd] = ps_sorted[0]
    return best_by_round


def _plot_adjacent_round_spearman(rows: Sequence[Mapping[str, Any]], *, output_path: Path) -> None:
    if not rows:
        raise ValueError("No adjacent-round correlations to plot (need >=2 rounds with kept candidates).")

    x = [int(r["round_t1"]) for r in rows]
    y = [float(r["spearman"]) for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Round (T+1)")
    plt.ylabel("Spearman(best_round_T, best_round_T+1)")
    plt.title("Adjacent-round candidate similarity (best by Val AUC)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def _process_program(args_tuple: Tuple) -> Tuple[Any, Any, Any]:
    """Worker function to process a single program."""
    program, idx, total, class_name, eval_raw, input_regime, chunk_size = args_tuple
    logging.info(f"[{idx+1}/{total}] Scoring {program.candidate_id}")
    
    scores = _score_candidate_on_eval_raw(
        program,
        class_name=class_name,
        eval_raw=eval_raw,
        input_regime=input_regime,
        chunk_size=chunk_size,
        supervised=False,
    )
    
    if not np.isfinite(scores).all():
        return None, None, {"candidate_id": program.candidate_id, "path": str(program.path), "reason": "non_finite_scores"}
    
    return program, scores, None

def main() -> None:
    _setup_logging()

    ap = argparse.ArgumentParser(description="Measure diversity among candidate hallucination detectors.")
    ap.add_argument("--eval-raw-path", type=str, required=True, help="Path to eval_raw .pt (torch.load dict with keys: samples, meta).")
    ap.add_argument("--candidates-dir", type=str, required=True, help="Directory with candidate_round_*_function_*.txt files.")
    ap.add_argument("--output-dir", type=str, required=True, help="Directory to write CSV/JSON/PNG outputs.")
    ap.add_argument("--class-name", type=str, default="HallucinationDetector", help="Estimator class name to instantiate from each candidate program.")
    ap.add_argument("--input-regime", type=str, default="logits", choices=["full", "logits", "hidden_states", "attentions"])
    ap.add_argument("--chunk-size", type=int, default=128, help="Number of samples per chunk for make_all_stats + estimator call.")
    ap.add_argument("--max-candidates", type=int, default=None, help="Optional cap on number of candidates loaded (debugging).")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of eval samples used (debugging).")
    args = ap.parse_args()

    eval_raw_path = Path(args.eval_raw_path)
    candidates_dir = Path(args.candidates_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert candidates_dir.is_dir(), f"--candidates-dir is not a directory: {candidates_dir}"
    assert args.chunk_size >= 1, f"--chunk-size must be >=1, got {args.chunk_size}"
    assert args.max_candidates is None or args.max_candidates >= 1, "--max-candidates must be >=1 when provided"
    assert args.max_samples is None or args.max_samples >= 1, "--max-samples must be >=1 when provided"

    eval_raw = _load_eval_raw(eval_raw_path)
    if args.max_samples is not None:
        eval_raw = {"samples": eval_raw["samples"][: int(args.max_samples)], "meta": eval_raw.get("meta", {})}

    candidate_paths = sorted(candidates_dir.glob("candidate_round_*_function_*.txt"))
    assert candidate_paths, f"No candidate files found in {candidates_dir}"
    if args.max_candidates is not None:
        candidate_paths = candidate_paths[: int(args.max_candidates)]

    logging.info(f"Loaded eval_raw: n_samples={len(eval_raw['samples'])}")
    logging.info(f"Found {len(candidate_paths)} candidate files")

    programs_all = [_load_candidate_program(p, class_name=args.class_name, base_class="Estimator") for p in candidate_paths]

    # Prepare arguments for parallel processing
    args_list = [
        (program, i, len(programs_all), args.class_name, eval_raw, 
        str(args.input_regime), int(args.chunk_size))
        for i, program in enumerate(programs_all)
    ]

    # Process in parallel with configurable n_jobs
    # Use n_jobs=-1 to use all available cores, or set specific number
    results = Parallel(n_jobs=8, verbose=10, backend="loky")(
        delayed(_process_program)(arg) for arg in args_list
    )

    kept_programs: List[CandidateProgram] = []
    kept_scores: List[np.ndarray] = []
    skipped: List[Dict[str, Any]] = []

    for program, scores, skip_info in results:
        if skip_info is not None:
            skipped.append(skip_info)
        else:
            kept_programs.append(program)
            kept_scores.append(scores)

    assert kept_programs, "All candidates were skipped; no finite score vectors to analyze."

    candidate_ids = [p.candidate_id for p in kept_programs]
    id_to_row = {cid: idx for idx, cid in enumerate(candidate_ids)}
    scores_matrix = np.vstack(kept_scores).astype(np.float32)
    assert scores_matrix.shape == (len(candidate_ids), len(eval_raw["samples"]))

    # 1) Save raw score matrix
    df_scores = pd.DataFrame(scores_matrix, index=candidate_ids)
    df_scores.index.name = "candidate_id"
    df_scores.to_csv(output_dir / "scores_matrix.csv")

    # Save metadata / skipped
    df_meta = pd.DataFrame([
        {
            "candidate_id": p.candidate_id,
            "path": str(p.path),
            "round_num": p.round_num,
            "function_num": p.function_num,
            "val_auc": p.val_auc,
            "test_auc": p.test_auc,
        }
        for p in kept_programs
    ]).sort_values(["round_num", "function_num", "candidate_id"], ascending=True)
    df_meta.to_csv(output_dir / "candidates_metadata.csv", index=False)

    (output_dir / "skipped_candidates.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    # 2) Pairwise similarities (Spearman)
    corr = _spearman_corr_matrix(scores_matrix)
    pd.DataFrame(corr, index=candidate_ids, columns=candidate_ids).to_csv(output_dir / "spearman_candidates.csv")

    # 3) Set-wise diversity on rank-normalized matrix
    rank_norm = _rank_normalize_rows(scores_matrix)
    thresholds = [0.8, 0.9, 0.95, 0.99]
    effective_rank, k_by_threshold = _effective_rank_and_k_thresholds(rank_norm, thresholds)
    diversity_metrics = {
        "n_candidates": int(scores_matrix.shape[0]),
        "n_samples": int(scores_matrix.shape[1]),
        "effective_rank": float(effective_rank),
        "k_by_threshold": {str(k): int(v) for k, v in k_by_threshold.items()},
        "thresholds": [float(t) for t in thresholds],
        "input_regime": str(args.input_regime),
        "class_name": str(args.class_name),
    }
    (output_dir / "diversity_metrics.json").write_text(json.dumps(diversity_metrics, indent=2), encoding="utf-8")

    # 4) Adjacent-round plot using best-per-round (max Val AUC; fallback Test AUC)
    best_by_round = _select_best_candidate_per_round(kept_programs)
    rounds_sorted = sorted(best_by_round.keys())

    adjacent_rows: List[Dict[str, Any]] = []
    for r0, r1 in zip(rounds_sorted[:-1], rounds_sorted[1:]):
        p0 = best_by_round[r0]
        p1 = best_by_round[r1]
        i0 = id_to_row[p0.candidate_id]
        i1 = id_to_row[p1.candidate_id]
        rho = float(spearmanr(scores_matrix[i0], scores_matrix[i1]).correlation)
        adjacent_rows.append({
            "round_t": int(r0),
            "round_t1": int(r1),
            "candidate_id_t": p0.candidate_id,
            "candidate_id_t1": p1.candidate_id,
            "val_auc_t": p0.val_auc,
            "val_auc_t1": p1.val_auc,
            "spearman": rho,
        })

    pd.DataFrame(adjacent_rows).to_csv(output_dir / "spearman_round_adjacent.csv", index=False)
    _plot_adjacent_round_spearman(adjacent_rows, output_path=output_dir / "spearman_round_adjacent.png")

    logging.info(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
