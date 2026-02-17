"""Analyze long model generations in polygraph CSV datasets.

Rules:
- A dataset is anomalous if any generation length is > 1536 tokens.
- For each anomalous dataset, report:
  - count of rows with length > 1536
  - top-50 generation lengths
  - three generation samples with ranks 1, 25, and 50
"""

import argparse
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


LENGTH_THRESHOLD = 1536
FILTER_THRESHOLD = 2040
REQUIRED_COLUMNS = ("input", "model_generation")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze long model generations across polygraph CSV datasets."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/uncertainty_quantification/polygraph_generations",
        help="Directory with base-model polygraph CSV files.",
    )
    parser.add_argument(
        "--qwen-dir",
        type=str,
        default="data/uncertainty_quantification/polygraph_generations_qwen3_8b",
        help="Directory with Qwen3-8B polygraph CSV files.",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="VityaVitalich/Llama3.1-8b-instruct",
        help="Tokenizer model id for the base directory.",
    )
    parser.add_argument(
        "--qwen-model-id",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Tokenizer model id for the Qwen directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for tokenizer calls.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anomalous_generation_analysis.txt",
        help="Output analysis txt file path.",
    )
    parser.add_argument(
        "--do-filtering",
        action="store_true",
        help="If set, create length_filtered_* metrics directories and drop rows with long generations.",
    )
    parser.add_argument(
        "--filter-base-dir",
        type=str,
        default="data/uncertainty_quantification/polygraph_generations_metrics",
        help="Source base metrics directory for filtering.",
    )
    parser.add_argument(
        "--filter-qwen-dir",
        type=str,
        default="data/uncertainty_quantification/polygraph_generations_qwen3_8b_metrics",
        help="Source Qwen metrics directory for filtering.",
    )
    parser.add_argument(
        "--filter-base-output-dir",
        type=str,
        default="data/uncertainty_quantification/length_filtered_polygraph_generations_metrics",
        help="Destination base metrics directory for filtered CSVs.",
    )
    parser.add_argument(
        "--filter-qwen-output-dir",
        type=str,
        default="data/uncertainty_quantification/length_filtered_polygraph_generations_qwen3_8b_metrics",
        help="Destination Qwen metrics directory for filtered CSVs.",
    )
    parser.add_argument(
        "--filter-threshold",
        type=int,
        default=FILTER_THRESHOLD,
        help="Drop rows with model_generation token length greater than this value.",
    )
    return parser.parse_args()


def _batched(values: List[str], batch_size: int) -> Iterable[List[str]]:
    assert batch_size > 0, "batch_size must be positive"
    for idx in range(0, len(values), batch_size):
        yield values[idx : idx + batch_size]


def _token_lengths(texts: List[str], tokenizer, batch_size: int) -> np.ndarray:
    lengths: List[int] = []
    for batch in _batched(texts, batch_size):
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.int32)


def _read_generations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing_cols, f"{csv_path.name}: missing required columns: {missing_cols}"
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["model_generation"] = out["model_generation"].fillna("").astype(str)
    assert len(out) > 0, f"{csv_path.name}: file has no rows"
    return out


def _filter_directory(
    source_dir: Path,
    destination_dir: Path,
    tokenizer,
    batch_size: int,
    threshold: int,
    tag: str,
) -> Dict[str, object]:
    assert source_dir.exists(), f"Directory not found: {source_dir}"
    assert source_dir.is_dir(), f"Path is not a directory: {source_dir}"
    assert threshold > 0, "filter threshold must be positive"
    destination_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(source_dir.glob("*.csv"))
    assert csv_paths, f"No CSV files found in: {source_dir}"

    total_rows = 0
    total_dropped = 0
    dropped_per_dataset: Dict[str, int] = {}
    for source_csv_path in csv_paths:
        destination_csv_path = destination_dir / source_csv_path.name
        shutil.copy2(source_csv_path, destination_csv_path)

        df = pd.read_csv(destination_csv_path)
        assert "model_generation" in df.columns, (
            f"{destination_csv_path.name}: missing required column: model_generation"
        )
        generation_texts = df["model_generation"].fillna("").astype(str).tolist()
        lengths = _token_lengths(generation_texts, tokenizer, batch_size)
        keep_mask = lengths <= threshold
        filtered_df = df.loc[keep_mask].copy()
        filtered_df.to_csv(destination_csv_path, index=False)

        rows_total = int(len(df))
        rows_kept = int(len(filtered_df))
        rows_dropped = rows_total - rows_kept
        dropped_per_dataset[source_csv_path.name] = rows_dropped
        total_rows += rows_total
        total_dropped += rows_dropped
        print(
            f"[{tag}] {source_csv_path.name}: total={rows_total} dropped_gt_{threshold}={rows_dropped} kept={rows_kept}"
        )

    return {
        "files": len(csv_paths),
        "rows_total": total_rows,
        "rows_dropped": total_dropped,
        "rows_kept": total_rows - total_dropped,
        "dropped_per_dataset": dropped_per_dataset,
    }


def _write_filtering_note(
    directory: Path,
    threshold: int,
    model_name: str,
    model_id: str,
    dropped_per_dataset: Dict[str, int],
) -> None:
    assert threshold > 0, "filter threshold must be positive"
    assert model_id.strip(), "model_id must be non-empty"
    note_path = directory / "FILTERING_NOTE.md"
    dataset_lines = [f"- `{name}`: dropped `{count}` samples" for name, count in sorted(dropped_per_dataset.items())]
    text = (
        "This directory contains filtered datasets.\n\n"
        f"For all datasets, all samples with model_generation length > `filter-threshold`={threshold} tokens are removed. The threshold was set manually after reviewing anomalous generations. \n\n"
        f"Tokenizer note: native tokenizer for {model_name} is used for token counting.\n\n"
        f"Exact tokenizer model: `{model_id}`.\n\n"
        "Dropped samples per dataset:\n"
        + "\n".join(dataset_lines)
        + "\n"
    )
    note_path.write_text(text, encoding="utf-8")


def _top_indices(lengths: np.ndarray, k: int) -> np.ndarray:
    assert lengths.ndim == 1, "lengths must be a 1D array"
    assert k > 0, "k must be positive"
    n_rows = int(lengths.size)
    assert n_rows > 0, "lengths must be non-empty"
    use_k = min(k, n_rows)
    return np.argsort(-lengths, kind="mergesort")[:use_k]


def _build_section(
    dataset_tag: str,
    csv_path: Path,
    generations: List[str],
    lengths: np.ndarray,
) -> str:
    over_threshold = lengths > LENGTH_THRESHOLD
    anomalous_count = int(np.sum(over_threshold))
    top50_idx = _top_indices(lengths, 50)
    top50_lengths = [int(lengths[idx]) for idx in top50_idx]

    lines: List[str] = []
    lines.append(f"=== {dataset_tag} / {csv_path.name} ===")
    lines.append(f"rows_total: {len(generations)}")
    lines.append(f"count_gt_{LENGTH_THRESHOLD}: {anomalous_count}")
    lines.append("top_50_lengths_desc:")
    lines.append(", ".join(str(v) for v in top50_lengths))
    lines.append("")

    lines.append("[selected_longest_generations]")
    selected_ranks = (1, 25, 50)
    for rank in selected_ranks:
        if rank > len(top50_idx):
            lines.append(f"--- rank={rank} unavailable (dataset has fewer than {rank} rows) ---")
            lines.append("")
            continue
        row_idx = int(top50_idx[rank - 1])
        length_val = int(lengths[row_idx])
        generation_text = generations[row_idx].replace("\r\n", "\n").replace("\r", "\n")
        lines.append(f"--- rank={rank} row_index={row_idx} length={length_val} ---")
        lines.append(generation_text)
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def _analyze_directory(
    directory: Path,
    tokenizer,
    batch_size: int,
    dataset_tag: str,
) -> Tuple[List[str], int]:
    assert directory.exists(), f"Directory not found: {directory}"
    assert directory.is_dir(), f"Path is not a directory: {directory}"

    csv_paths = sorted(directory.glob("*.csv"))
    assert csv_paths, f"No CSV files found in: {directory}"

    sections: List[str] = []
    anomalous_datasets = 0
    for csv_path in csv_paths:
        df = _read_generations(csv_path)
        generations = df["model_generation"].tolist()
        lengths = _token_lengths(generations, tokenizer, batch_size)
        if int(np.max(lengths)) <= LENGTH_THRESHOLD:
            continue
        anomalous_datasets += 1
        section = _build_section(dataset_tag, csv_path, generations, lengths)
        sections.append(section)
    return sections, anomalous_datasets


def main() -> None:
    args = _parse_args()
    assert args.base_model_id.strip(), "base_model_id must be non-empty"
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.output.strip(), "output path must be non-empty"
    assert args.filter_threshold > 0, "filter_threshold must be positive"

    qwen_dir_arg = args.qwen_dir.strip()
    analyze_qwen = bool(qwen_dir_arg)
    filter_qwen_dir_arg = args.filter_qwen_dir.strip()
    filter_qwen = bool(filter_qwen_dir_arg)
    if analyze_qwen or filter_qwen:
        assert args.qwen_model_id.strip(), "qwen_model_id must be non-empty"

    base_dir = Path(args.base_dir)
    qwen_dir = Path(qwen_dir_arg) if analyze_qwen else None
    output_path = Path(args.output)

    tokenizers: Dict[str, object] = {
        "base": AutoTokenizer.from_pretrained(args.base_model_id),
    }
    if analyze_qwen or filter_qwen:
        tokenizers["qwen3_8b"] = AutoTokenizer.from_pretrained(args.qwen_model_id)

    if args.do_filtering:
        base_output_dir = Path(args.filter_base_output_dir)
        qwen_output_dir = Path(args.filter_qwen_output_dir)
        base_filter_summary = _filter_directory(
            source_dir=Path(args.filter_base_dir),
            destination_dir=base_output_dir,
            tokenizer=tokenizers["base"],
            batch_size=args.batch_size,
            threshold=args.filter_threshold,
            tag="base_metrics",
        )
        qwen_filter_summary = None
        if filter_qwen:
            qwen_filter_summary = _filter_directory(
                source_dir=Path(filter_qwen_dir_arg),
                destination_dir=qwen_output_dir,
                tokenizer=tokenizers["qwen3_8b"],
                batch_size=args.batch_size,
                threshold=args.filter_threshold,
                tag="qwen_metrics",
            )
        _write_filtering_note(
            base_output_dir,
            args.filter_threshold,
            "Llama",
            args.base_model_id,
            base_filter_summary["dropped_per_dataset"],
        )
        if qwen_filter_summary:
            _write_filtering_note(
                qwen_output_dir,
                args.filter_threshold,
                "Qwen",
                args.qwen_model_id,
                qwen_filter_summary["dropped_per_dataset"],
            )
        summary_parts = [
            f"base_dropped={base_filter_summary['rows_dropped']}",
        ]
        if qwen_filter_summary:
            summary_parts.append(f"qwen_dropped={qwen_filter_summary['rows_dropped']}")
        else:
            summary_parts.append("qwen_dropped=skipped")
        total_dropped = base_filter_summary["rows_dropped"]
        if qwen_filter_summary:
            total_dropped += qwen_filter_summary["rows_dropped"]
        print("Filtering summary: " + " ".join(summary_parts) + f" total_dropped={total_dropped}")
        return

    base_sections, base_count = _analyze_directory(
        directory=base_dir,
        tokenizer=tokenizers["base"],
        batch_size=args.batch_size,
        dataset_tag="polygraph_generations",
    )
    if analyze_qwen and qwen_dir is not None:
        qwen_sections, qwen_count = _analyze_directory(
            directory=qwen_dir,
            tokenizer=tokenizers["qwen3_8b"],
            batch_size=args.batch_size,
            dataset_tag="polygraph_generations_qwen3_8b",
        )
    else:
        qwen_sections, qwen_count = [], 0

    all_sections = base_sections + qwen_sections
    report_lines: List[str] = [
        "Long generation analysis",
        f"criterion: dataset is anomalous if any model_generation length > {LENGTH_THRESHOLD}",
        "",
        f"anomalous_datasets_in_polygraph_generations: {base_count}",
        (
            f"anomalous_datasets_in_polygraph_generations_qwen3_8b: {qwen_count}"
            if analyze_qwen
            else "anomalous_datasets_in_polygraph_generations_qwen3_8b: skipped (qwen-dir empty)"
        ),
        f"anomalous_datasets_total: {base_count + qwen_count}",
        "",
    ]
    report_lines.extend(all_sections)
    if not all_sections:
        report_lines.append("No anomalous datasets found.")

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote analysis to: {output_path}")


if __name__ == "__main__":
    main()
