"""Evaluate independent dataset caches concurrently, sharing no estimator state."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/claims"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/dev/shm/mod1-eval-cache"))
    ap.add_argument("--output", type=Path, default=Path("out/campaign/evaluation"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--bootstrap-unit", choices=["claim", "row"], default="claim")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--extra-frozen-glob", action="append", default=[])
    ap.add_argument("--extra-fitted-models", action="append", type=Path, default=[])
    ap.add_argument("--model-id", default="/workspace/mod1-models/llama31")
    ap.add_argument("--tokenizer-id")
    ap.add_argument("--trim-header-suffix", type=int, default=0)
    ap.add_argument("--frozen-glob", default="out/logs/mod1/*/frozen_estimator.py")
    ap.add_argument("--method-prefix")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(args.data_root.rglob("*.csv")) if "popqa_train" not in p.name]
    if len(files) != 9:
        raise ValueError("Expected nine evaluation CSVs")
    names = [p.parent.name + "__" + p.stem for p in files]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def evaluate(index):
        name = names[index]
        result_path = args.output / (name + ".json")
        if args.resume and result_path.exists():
            print("Reusing completed dataset", name, flush=True)
            return
        command = [sys.executable, str(Path(__file__).with_name("evaluate_cached_claims.py").resolve()),
                   "--mode", "evaluate", "--data-root", str(args.data_root), "--cache-dir", str(args.cache_dir),
                   "--output", str(args.output), "--dataset-index", str(index), "--cpu-threads", "1",
                   "--bootstrap", str(args.bootstrap), "--bootstrap-unit", args.bootstrap_unit]
        command += ["--model-id", args.model_id]
        command += ["--frozen-glob", args.frozen_glob]
        if args.method_prefix:
            command += ["--method-prefix", args.method_prefix]
        if args.tokenizer_id:
            command += ["--tokenizer-id", args.tokenizer_id]
        if args.primary_only:
            command.append("--primary-only")
        for pattern in args.extra_frozen_glob:
            command += ["--extra-frozen-glob", pattern]
        for path in args.extra_fitted_models:
            command += ["--extra-fitted-models", str(path)]
        if args.trim_header_suffix:
            command += ["--trim-header-suffix", str(args.trim_header_suffix)]
        with (args.output / (name + ".log")).open("w") as stream:
            proc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"Evaluation failed for {name}; see {name}.log")
        print("Completed dataset", name, flush=True)

    # Begin with the largest caches, avoiding a long final tail.
    indices = sorted(range(9), key=lambda i: (args.cache_dir / (names[i] + ".pt")).stat().st_size, reverse=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(evaluate, indices))
    aggregate = {name: json.loads((args.output / (name + ".json")).read_text()) for name in names}
    (args.output / "all_datasets.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    print("All nine frozen-estimator evaluations completed", flush=True)


if __name__ == "__main__":
    main()
