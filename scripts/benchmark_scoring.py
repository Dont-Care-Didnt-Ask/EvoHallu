import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scripts.evaluate_cached_claims import load_estimators
from packing.evaluate.uncertainty_quantification.features import make_all_stats

torch.set_num_threads(1)
logging.getLogger().setLevel(logging.ERROR)
raw = torch.load("data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt", mmap=True, weights_only=False)
samples = raw["samples"][:256]
stats, _, _, _ = make_all_stats({"samples": samples, "meta": raw["meta"]}, input_regime="logits")
stats["input_tokens"] = [s["input_ids"].tolist() for s in samples]
estimators, _ = load_estimators(sorted(Path("out/logs/mod1").glob("*/frozen_estimator.py")), raw["meta"])
rows = []
for name, estimator in estimators.items():
    started = time.perf_counter()
    try:
        scores = np.asarray(estimator(copy.deepcopy(stats)))
        row = {"method": name, "seconds_256": time.perf_counter()-started,
               "finite": bool(np.isfinite(scores).all())}
    except Exception as exc:
        row = {"method": name, "error": f"{type(exc).__name__}: {exc}"}
    rows.append(row)
total = sum(row.get("seconds_256", 0) for row in rows)
report = {"samples": 256, "methods": rows, "total_seconds_256": total,
          "projected_serial_seconds_97488": total * 97488 / 256,
          "note": "Projection from training features; excludes bootstrap and process startup."}
Path("out/campaign/scoring_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k:v for k,v in report.items() if k != "methods"}))
