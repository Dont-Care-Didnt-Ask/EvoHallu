"""Record frozen-source identity and inspect calls for RNG or external I/O.

This is a reproducibility audit, not a general Python sandbox or safety proof.
It never repairs a generated program or chooses a replacement winner.
"""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value)+"."+node.attr
    return ""


def main():
    records = []
    for path in sorted(Path("out/logs/mod1").glob("*/frozen_estimator.py")):
        text = path.read_text()
        tree = ast.parse(text)
        summary = json.loads((path.parent / "summary.json").read_text())
        calls = sorted({dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)})
        suspicious = [name for name in calls if name in {"open", "exec", "eval", "__import__", "np.load", "np.save", "np.savez"}
                      or name.startswith(("np.random.", "numpy.random.", "random.", "os.", "subprocess.", "requests.", "urllib.", "torch.rand"))]
        imports = sorted({node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
                         | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names})
        candidate_file = path.parent / "candidates.jsonl"
        selected = summary.get("selected_candidate")
        matching = None
        if candidate_file.exists() and selected != "baseline":
            candidates = [json.loads(line) for line in candidate_file.read_text().splitlines()]
            candidates = [c for c in candidates if c.get("code_hash") == selected]
            matching = any(c.get("imports", "")+"\n"+c["code"]+"\n" == text for c in candidates)
        row = {"run": path.parent.name, "file_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "selected_candidate": selected, "matches_selected_record": matching,
               "baseline_retained": selected == "baseline", "imports": imports,
               "rng_or_external_io_calls_found": suspicious,
               "excluded_from_quality_evaluation": (path.parent / "exclude_from_evaluation.json").exists()}
        records.append(row)
        if matching is False:
            raise ValueError("Frozen source does not match its selected candidate: " + str(path))
    result = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "estimators": records,
              "scope": "Static call/import inspection and exact equality to the recorded selected program; no program modifications"}
    Path("results/frozen_estimator_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"frozen_estimators": len(records), "source_mismatches": 0,
                      "runs_with_flagged_calls": [r["run"] for r in records if r["rng_or_external_io_calls_found"]]}))


if __name__ == "__main__":
    main()
