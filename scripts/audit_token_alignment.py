import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices
from packing.evaluate.uncertainty_quantification.alignment_baselines import claim_span, claim_likelihoods, legacy_tail_crop_likelihoods

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt")
ap.add_argument("--output", type=Path, default=Path("results/token_alignment_audit.json"))
args = ap.parse_args()
raw = torch.load(args.raw, mmap=True, weights_only=False, map_location="cpu")
starts, suffix = collections.Counter(), collections.Counter()
bad, labels, values = [], [], collections.defaultdict(list)
for i, sample in enumerate(raw["samples"]):
    lp = np.asarray(sample["sequence_logprobs"], dtype=float)
    tokens, full = sample["orig_input_ids"], sample["input_ids"]
    try:
        start, end = claim_span(full, tokens)
    except ValueError as exc:
        bad.append({"row": i, "error": str(exc)})
        continue
    starts[start] += 1
    suffix[len(full)-end] += 1
    labels.append(sample["label"])
    values["template_included_nll"].append(-lp.sum())
    values["exact_claim_nll"].append(-claim_likelihoods(lp, tokens, full).sum())
    values["legacy_tail_crop_nll"].append(-legacy_tail_crop_likelihoods(lp, tokens).sum())
    values["last_input_token_surprisal"].append(-lp[-1])
    values["post_claim_surprisal"].append(-lp[end-1:].sum())
    values["claim_length"].append(len(tokens))
result = {"total": len(raw["samples"]), "start_counts": dict(starts), "suffix_token_counts": dict(suffix),
          "non_unique_or_unmatched": bad, "test_data_used": False, "protocols": {}}
if not bad:
    y = np.asarray(labels)
    for strategy in ["stratified", "claim_group"]:
        train, val = _prepare_train_val_indices(raw, 0, .2, 441, strategy)
        result["protocols"][strategy] = {name: {
            "train_auc_percent": 100 * roc_auc_score(y[train], np.asarray(pred)[train]),
            "validation_auc_percent": 100 * roc_auc_score(y[val], np.asarray(pred)[val]),
        } for name, pred in values.items()}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
