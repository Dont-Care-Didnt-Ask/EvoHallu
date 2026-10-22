import csv
from itertools import product
from pathlib import Path

import numpy as np

from .config import DATASETS, DATASET_ORDER, METHOD_NAMES, MODELS


METRIC_FIELDS = ("model", "dataset", "view", "method", "rows", "positive", "full_dataset",
                 "auc_fixed", "auc_folded_compatibility", "status")
MACRO_FIELDS = ("mode", "model", "method", "macro_auc", "macro_gain_over_sequence_probability_pp")
MODES = {
    "full_folded": ("full", "auc_folded_compatibility"),
    "full_fixed": ("full", "auc_fixed"),
    "claim_fixed": ("claim", "auc_fixed"),
}
MODEL_LABELS = {"llama": "Llama 3.1", "gemma": "Gemma", "olmo": "OLMo"}
DATA_LABELS = ("PopQA", "AVeriTeC", "UHEAD", "Cities", "Companies", "CommonClaim",
               "CounterFact", "X-Fact", "WildHallucinations")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_complete(metrics):
    expected = set(product(MODELS, DATASET_ORDER, ("full", "claim"), METHOD_NAMES))
    actual = {(r["model"], r["dataset"], r["view"], r["method"]) for r in metrics}
    return len(metrics) == len(expected) and actual == expected and all(
        r["full_dataset"] and r["status"] == "passed"
        and r["rows"] == DATASETS[r["dataset"]]["rows"] for r in metrics
    )


def build_macro(metrics):
    if not is_complete(metrics):
        raise ValueError("Mean9 requires all three models, nine full datasets and 25 methods")
    index = {(r["model"], r["dataset"], r["view"], r["method"]): r for r in metrics}
    means = {
        (mode, model, method): float(np.mean([
            index[model, dataset, view, method][field] for dataset in DATASET_ORDER
        ]))
        for mode, (view, field) in MODES.items()
        for model in MODELS for method in METHOD_NAMES
    }
    return [
        {"mode": mode, "model": model, "method": method, "macro_auc": value,
         "macro_gain_over_sequence_probability_pp":
             100 * (value - means[mode, model, "SequenceLogProb"])}
        for (mode, model, method), value in means.items()
    ]


def markdown_table(header, rows):
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def write_report(metrics, output_dir):
    output_dir = Path(output_dir)
    write_csv(output_dir / "metrics_long.csv", METRIC_FIELDS, metrics)
    complete = is_complete(metrics)
    macro = build_macro(metrics) if complete else []
    write_csv(output_dir / "macro9.csv", MACRO_FIELDS, macro)
    if not complete:
        for name in ("average.md", *(f"tables_{mode}.md" for mode in MODES)):
            (output_dir / name).unlink(missing_ok=True)
        return macro

    means = {(r["mode"], r["model"], r["method"]): r["macro_auc"] for r in macro}
    average_rows = []
    for method in METHOD_NAMES:
        values = [means["full_folded", model, method] for model in MODELS]
        average_rows.append([method, *(
            f"**{100 * value:.2f}**" if value == max(values) else f"{100 * value:.2f}"
            for value in values
        )])
    note = "Full-chat folded AUROC (%). Average gives equal weight to nine datasets."
    (output_dir / "average.md").write_text(
        "# Average AUROC\n\n" + note + " Bold marks the maximum within each row.\n\n"
        + markdown_table(["Method", *MODEL_LABELS.values()], average_rows) + "\n",
        encoding="utf-8",
    )

    index = {(r["model"], r["dataset"], r["view"], r["method"]): r for r in metrics}
    for mode, (view, field) in MODES.items():
        sections = [f"# {mode}: AUROC (%)\n"]
        for method in METHOD_NAMES:
            rows = [[MODEL_LABELS[model], *(
                f"{100 * index[model, dataset, view, method][field]:.2f}"
                for dataset in DATASET_ORDER
            ), f"{100 * means[mode, model, method]:.2f}"] for model in MODELS]
            sections.append(f"## {method}\n\n" + markdown_table(
                ["Model", *DATA_LABELS, "Mean9"], rows
            ))
        (output_dir / f"tables_{mode}.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return macro
